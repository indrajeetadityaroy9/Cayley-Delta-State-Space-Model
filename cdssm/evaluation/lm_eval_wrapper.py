"""lm-evaluation-harness integration for CDSSM.

Wraps CDSSMLMHeadModel as an ``lm_eval.api.model.LM`` so checkpoints can be
evaluated with the standard Mamba/SSM zero-shot benchmark suite::

    python -m cdssm.eval --checkpoint checkpoints/best.pt \\
        --tasks wikitext,lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from cdssm.inference.predict import load_model_from_checkpoint

_DEVICE = torch.device("cuda")

# H100 80 GB / 26 vCPU defaults
_BATCH_SIZE = 64


class CDSSMEvalWrapper(LM):
    """Wraps a CDSSM checkpoint for lm-evaluation-harness (v0.4.2)."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        self._device = _DEVICE
        self._batch_size = _BATCH_SIZE

        self.model, self.tokenizer = load_model_from_checkpoint(checkpoint_path)
        self._max_length = self.model.config.context_length

    # Required LM properties

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    # Helpers

    def tok_encode(self, string: str) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _model_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass -> logits in float32."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(input_ids.to(self._device))
        return logits.float()

    # LM API: loglikelihood

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tok_encode(context) if context else []
            cont_ids = self.tok_encode(continuation)

            all_ids = (ctx_ids + cont_ids)[-self.max_length:]
            n_cont = min(len(cont_ids), len(all_ids))
            cont_start = len(all_ids) - n_cont

            input_ids = torch.tensor([all_ids], dtype=torch.long)
            logits = self._model_logits(input_ids)   # (1, T, V)
            log_probs = F.log_softmax(logits[0], dim=-1)

            total_ll = 0.0
            is_greedy = True
            for i in range(n_cont):
                pos = cont_start + i - 1
                tok = all_ids[cont_start + i]
                if pos >= 0:
                    total_ll += log_probs[pos, tok].item()
                    if logits[0, pos].argmax().item() != tok:
                        is_greedy = False

            results.append((total_ll, is_greedy))
        return results

    # LM API: loglikelihood_rolling  (used by `wikitext` perplexity)

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float]]:
        results: list[tuple[float]] = []
        for req in requests:
            (text,) = req.args
            token_ids = self.tok_encode(text)
            if len(token_ids) <= 1:
                results.append((0.0,))
                continue

            total_ll = 0.0
            # Non-overlapping windows without hidden-state carry-over:
            # position 0 of every chunk has no prior context, so skip it.
            for start in range(0, len(token_ids), self.max_length):
                end = min(start + self.max_length, len(token_ids))
                chunk = token_ids[start:end]
                if len(chunk) <= 1:
                    continue

                input_ids = torch.tensor([chunk], dtype=torch.long)
                logits = self._model_logits(input_ids)
                log_probs = F.log_softmax(logits[0], dim=-1)

                # log_probs[i-1] predicts chunk[i]; start from i=1 since
                # position 0 has no context in an independent forward pass
                for i in range(1, len(chunk)):
                    total_ll += log_probs[i - 1, chunk[i]].item()

            results.append((total_ll,))
        return results

    # LM API: generate_until

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results: list[str] = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            ctx_ids = self.tok_encode(context)
            if len(ctx_ids) > self.max_length - 1:
                ctx_ids = ctx_ids[-(self.max_length - 1):]

            generated: list[int] = list(ctx_ids)
            for _ in range(max_gen):
                window = generated[-self.max_length:]
                input_ids = torch.tensor([window], dtype=torch.long)
                logits = self._model_logits(input_ids)
                next_token = logits[0, -1].argmax().item()
                generated.append(next_token)

                decoded = self.tok_decode(generated[len(ctx_ids):])
                if any(s in decoded for s in until):
                    break

            result = self.tok_decode(generated[len(ctx_ids):])
            for s in until:
                if s in result:
                    result = result[: result.index(s)]
            results.append(result)
        return results
