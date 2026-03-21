import os
from typing import Optional

import torch
from openai import OpenAI

from model.base import BaseLLM


class GPT4Model(BaseLLM):
    DEFAULT_MODEL_ID = "gpt-4.1-2025-04-14"

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = api_base
        self.client: Optional[OpenAI] = None

    def load(self) -> None:
        client_kwargs = {"api_key": self.api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        self.client = OpenAI(**client_kwargs)

    def score_change(self, prompt: str) -> float:
        if self.client is None:
            self.load()

        response = self.client.chat.completions.create(
            model=self.model_name_or_path,
            messages=[
                {
                    "role":
                    "system",
                    "content":
                    ("You are an expert linguist. Answer only Yes or No."),
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        )

        choice = response.choices[0]
        top_logprobs = choice.logprobs.content[0].top_logprobs

        import math

        p_yes, p_no = 0.0, 0.0
        yes_tokens = {"Yes", "yes", " Yes", " yes"}
        no_tokens = {"No", "no", " No", " no"}
        for entry in top_logprobs:
            prob = math.exp(entry.logprob)
            if entry.token in yes_tokens:
                p_yes += prob
            elif entry.token in no_tokens:
                p_no += prob

        if p_yes + p_no == 0:
            return 0.5
        return p_yes / (p_yes + p_no)

    def generate(self, prompt: str) -> str:
        if self.client is None:
            self.load()

        response = self.client.chat.completions.create(
            model=self.model_name_or_path,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content
