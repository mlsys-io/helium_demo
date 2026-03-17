import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

import yaml
from bench_programs.utils.common import random_shuffle
from bench_utils.mixin import BenchmarkMixin

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumResponse, HeliumSystemProfile
from helium.utils import identity


@dataclass
class PromptFormats:
    system_prompt: str
    first_prompt_fmt: str
    subsequent_prompt_fmt: str

    @classmethod
    def from_yaml(cls, fpath: Path) -> "PromptFormats":
        with open(fpath) as file:
            data = yaml.safe_load(file)
        return cls(**data)


def _aggregate_amazon_reviews(reviews: list[dict[str, Any]]) -> str:
    def _get_or_na(review: dict[str, Any], key: str) -> tuple[str, str]:
        return key, review.get(key, "N/A")

    keys = ["title", "rating", "text", "helpful_vote"]
    extracted_reviews = [
        dict(_get_or_na(review, key) for key in keys) for review in reviews
    ]
    chunk = json.dumps(extracted_reviews, ensure_ascii=False, indent=2)
    return chunk


OutputType = list[str]


class IterativeProgram(BenchmarkMixin, ABC):
    class OutputBuilder:
        def __init__(self) -> None:
            self._inner: list[tuple[int, str]] = []

        def add(self, index: int, content: str) -> None:
            self._inner.append((index, content))

        def update(self, items: Iterable[tuple[int, str]]) -> Self:
            for index, content in items:
                self.add(index, content)
            return self

        def build(self) -> OutputType:
            ret: list[str | None] = [None] * len(self._inner)
            for index, content in self._inner:
                ret[index] = content
            return cast(OutputType, ret)

    _PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
    PROMPT_FORMATS: dict[str, PromptFormats] = {
        "arxiv": PromptFormats.from_yaml(_PROMPT_DIR / "arxiv.yaml"),
        "amazon": PromptFormats.from_yaml(_PROMPT_DIR / "amazon.yaml"),
    }
    PREPROCESSORS: dict[str, Callable[[Any], str]] = {
        "amazon": _aggregate_amazon_reviews
    }

    async def run_async(
        self,
        dataset: Literal["arxiv", "amazon"],
        document_chunks: list[tuple[Any, ...]],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> list[str]:
        self.start_timer("run")
        summary, system_profile = await self._run(
            **self.prepare_kwargs(
                dataset=dataset,
                document_chunks=document_chunks,
                generation_config=generation_config,
                **kwargs,
            )
        )
        self.stop_timer()
        self.set_system_profile(system_profile)

        return summary

    async def precompute(
        self,
        dataset: Literal["arxiv", "amazon"],
        document_chunks: list[tuple[Any, ...]],
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        **kwargs,
    ) -> HeliumResponse:
        return await self._precompute(
            precompute_mode=precompute_mode,
            **self.prepare_kwargs(
                dataset=dataset,
                document_chunks=document_chunks,
                generation_config=generation_config,
                **kwargs,
            ),
        )

    def prepare_kwargs(
        self,
        dataset: Literal["arxiv", "amazon"],
        document_chunks: list[tuple[Any, ...]],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> dict[str, Any]:
        if dataset not in self.PROMPT_FORMATS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        prompt_formats = self.PROMPT_FORMATS[dataset]

        preprocessor = self.PREPROCESSORS.get(dataset, identity)
        pivotted_chunks = tuple(
            [preprocessor(chunk) for chunk in chunks]
            for chunks in zip(*document_chunks)
        )

        return dict(
            system_prompt=prompt_formats.system_prompt,
            first_prompt_fmt=prompt_formats.first_prompt_fmt,
            subsequent_prompt_fmt=prompt_formats.subsequent_prompt_fmt,
            document_chunks=pivotted_chunks,
            generation_config=generation_config,
            **kwargs,
        )

    def flatten_inputs(
        self, document_chunks: tuple[list[str], ...], shuffle: bool = True
    ) -> list[tuple[int, tuple[str, ...]]]:
        """
        Returns
        -------
        index : int
            The original index of the chunk.
        chunks : tuple[str, ...]
            The flattened chunks.
        """
        pivotted_chunks = (chunks for chunks in zip(*document_chunks))
        flattened = list(enumerate(pivotted_chunks))
        return random_shuffle(flattened) if shuffle else flattened

    @abstractmethod
    async def _run(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        document_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[OutputType, HeliumSystemProfile]:
        pass

    async def _precompute(
        self,
        system_prompt: str,
        first_prompt_fmt: str,
        subsequent_prompt_fmt: str,
        document_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        **kwargs,
    ) -> HeliumResponse:
        raise NotImplementedError("Precompute is not implemented.")
