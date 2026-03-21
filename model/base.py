from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseLLM(ABC):
    DEFAULT_MODEL_ID: str = ""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 5,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path or self.DEFAULT_MODEL_ID
        self.device = device or ("cuda"
                                 if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (
            torch.float16 if torch.cuda.is_available() else torch.float32)
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def score_change(self, prompt: str) -> float:
        ...

    def generate(self, prompt: str) -> str:
        if self.model is None:
            self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _get_yes_no_token_ids(self):
        yes_variants = ["Yes", "yes", " Yes", " yes"]
        no_variants = ["No", "no", " No", " no"]

        yes_ids = set()
        no_ids = set()
        for v in yes_variants:
            ids = self.tokenizer.encode(v, add_special_tokens=False)
            if ids:
                yes_ids.add(ids[0])
        for v in no_variants:
            ids = self.tokenizer.encode(v, add_special_tokens=False)
            if ids:
                no_ids.add(ids[0])

        return list(yes_ids), list(no_ids)

    def _score_from_logits(self, logits: torch.Tensor) -> float:
        yes_ids, no_ids = self._get_yes_no_token_ids()
        if not yes_ids or not no_ids:
            raise RuntimeError(
                "Could not resolve Yes/No token ids for this tokenizer.")

        probs = torch.softmax(logits.float(), dim=-1)
        p_yes = sum(probs[tid].item() for tid in yes_ids)
        p_no = sum(probs[tid].item() for tid in no_ids)

        if p_yes + p_no == 0:
            return 0.5
        return p_yes / (p_yes + p_no)

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name_or_path})"
