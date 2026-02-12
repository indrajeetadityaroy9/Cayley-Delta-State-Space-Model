"""lm-evaluation-harness integration for KSSM.

Wraps KSSMLMHeadModel as an ``lm_eval.api.model.LM`` so checkpoints can be
evaluated with the standard Mamba/SSM zero-shot benchmark suite::

    python scripts/evaluate.py --checkpoint checkpoints/best.pt \\
        --tasks wikitext,lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from transformers import AutoTokenizer

from kssm.config.defaults import KSSMConfig
from kssm.models.language_model import KSSMLMHeadModel

# Fields computed in __post_init__ — must not be passed to the constructor.
_COMPUTED_FIELDS = frozenset({"d_inner", "n_heads", "head_dim", "ssm_norm_groups"})


class KSSMEvalWrapper(LM):
    """Wraps a KSSM checkpoint for lm-evaluation-harness (v0.4.2)."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        batch_size: int = 1,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = batch_size

        # ---- Load checkpoint ------------------------------------------------
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt["config"]

        # Reconstruct KSSMConfig (skip computed fields)
        init_kwargs = {
            k: v for k, v in cfg_dict.items()
            if k in KSSMConfig.__dataclass_fields__ and k not in _COMPUTED_FIELDS
        }
        config = KSSMConfig(**init_kwargs)
        self._max_length = max_length or config.context_length

        # Build model
        vocab_size = cfg_dict.get("vocab_size", 50257)
        self.model = KSSMLMHeadModel(config, vocab_size)

        # Handle torch.compile `_orig_mod.` prefix in state dicts
        state_dict = ckpt["model_state_dict"]
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(cleaned, strict=False)
        self.model.to(self._device).bfloat16().eval()

        # ---- Tokenizer ------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Required LM properties
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def tok_encode(self, string: str) -> list[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _model_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass → logits in float32."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(input_ids.to(self._device))
        return logits.float()

    # ------------------------------------------------------------------
    # LM API: loglikelihood
    # ------------------------------------------------------------------

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tok_encode(context) if context else []
            cont_ids = self.tok_encode(continuation)

            all_ids = (ctx_ids + cont_ids)[-self.max_length:]
            # How many continuation tokens survived the truncation
            n_cont = min(len(cont_ids), len(all_ids))
            cont_start = len(all_ids) - n_cont

            input_ids = torch.tensor([all_ids], dtype=torch.long)
            logits = self._model_logits(input_ids)   # (1, T, V)
            log_probs = F.log_softmax(logits[0], dim=-1)

            total_ll = 0.0
            is_greedy = True
            for i in range(n_cont):
                pos = cont_start + i - 1  # logit at pos predicts token at pos+1
                tok = all_ids[cont_start + i]
                if pos >= 0:
                    total_ll += log_probs[pos, tok].item()
                    if logits[0, pos].argmax().item() != tok:
                        is_greedy = False

            results.append((total_ll, is_greedy))
        return results

    # ------------------------------------------------------------------
    # LM API: loglikelihood_rolling  (used by `wikitext` perplexity)
    # ------------------------------------------------------------------

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Tuple[float]]:
        results: list[tuple[float]] = []
        for req in requests:
            (text,) = req.args
            token_ids = self.tok_encode(text)
            if len(token_ids) <= 1:
                results.append((0.0,))
                continue

            total_ll = 0.0
            # Sliding window with stride = max_length
            for start in range(0, len(token_ids), self.max_length):
                end = min(start + self.max_length, len(token_ids))
                chunk = token_ids[start:end]
                if len(chunk) <= 1:
                    continue

                input_ids = torch.tensor([chunk], dtype=torch.long)
                logits = self._model_logits(input_ids)
                log_probs = F.log_softmax(logits[0], dim=-1)

                # First chunk: predict from position 1 onward.
                # Subsequent chunks: predict all positions (context was prior chunk).
                pred_start = 1 if start == 0 else 0
                for i in range(pred_start, len(chunk)):
                    total_ll += log_probs[i - 1, chunk[i]].item()

            results.append((total_ll,))
        return results

    # ------------------------------------------------------------------
    # LM API: generate_until
    # ------------------------------------------------------------------

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
