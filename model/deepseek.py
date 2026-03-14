import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.base import BaseLLM


class DeepSeekR1Model(BaseLLM):
    DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def score_change(self, prompt: str) -> float:
        if self.model is None:
            self.load()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        next_token_logits = outputs.logits[0, -1, :]
        return self._score_from_logits(next_token_logits)
