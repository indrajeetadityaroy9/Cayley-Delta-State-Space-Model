"""lm-evaluation-harness integration for CDSSM.

Wraps CDSSMLMHeadModel as an ``lm_eval.api.model.LM`` so checkpoints can be
evaluated with the standard Mamba/SSM zero-shot benchmark suite::

    cdssm bench --config configs/experiment/bench_default.yaml
"""

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from cdssm.inference.predict import load_model_from_checkpoint


class CDSSMEvalWrapper(LM):
    """Wraps a CDSSM checkpoint for lm-evaluation-harness (v0.4.2)."""

    def __init__(self, checkpoint_path: str, batch_size: int = 64):
        super().__init__()
        self.model, self.tokenizer = load_model_from_checkpoint(checkpoint_path)
        self._device = torch.device("cuda")
        self._batch_size = batch_size
        self._max_length = self.model.config.context_length

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

    @torch.no_grad()
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for req in requests:
            context, continuation = req.args
            ctx_ids = self.tokenizer.encode(context, add_special_tokens=False) if context else []
            cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)

            all_ids = (ctx_ids + cont_ids)[-self._max_length:]
            n_cont = min(len(cont_ids), len(all_ids))
            cont_start = len(all_ids) - n_cont

            input_ids = torch.tensor([all_ids], dtype=torch.long)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids.cuda()).float()
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

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float]]:
        results: list[tuple[float]] = []
        for req in requests:
            (text,) = req.args
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) <= 1:
                results.append((0.0,))
                continue

            total_ll = 0.0
            for start in range(0, len(token_ids), self._max_length):
                end = min(start + self._max_length, len(token_ids))
                chunk = token_ids[start:end]
                if len(chunk) <= 1:
                    continue

                input_ids = torch.tensor([chunk], dtype=torch.long)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(input_ids.cuda()).float()
                log_probs = F.log_softmax(logits[0], dim=-1)

                for i in range(1, len(chunk)):
                    total_ll += log_probs[i - 1, chunk[i]].item()

            results.append((total_ll,))
        return results

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results: list[str] = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
            if len(ctx_ids) > self._max_length - 1:
                ctx_ids = ctx_ids[-(self._max_length - 1):]

            generated: list[int] = list(ctx_ids)
            for _ in range(max_gen):
                window = generated[-self._max_length:]
                input_ids = torch.tensor([window], dtype=torch.long)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(input_ids.cuda()).float()
                next_token = logits[0, -1].argmax().item()
                generated.append(next_token)

                decoded = self.tokenizer.decode(generated[len(ctx_ids):])
                if any(s in decoded for s in until):
                    break

            result = self.tokenizer.decode(generated[len(ctx_ids):])
            for s in until:
                if s in result:
                    result = result[: result.index(s)]
            results.append(result)
        return results
