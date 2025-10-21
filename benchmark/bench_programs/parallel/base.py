from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

import yaml
from bench_programs.utils.common import random_shuffle
from bench_utils.mixin import BenchmarkMixin

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumResponse, HeliumSystemProfile


@dataclass
class PromptFormats:
    role_instructions: dict[str, str]
    expert_system_prompt: str
    writer_system_prompt: str
    extraction_instruction: str
    partial_summary_instruction: str
    partial_report_instruction: str

    @classmethod
    def from_yaml(cls, fpath: Path) -> "PromptFormats":
        with open(fpath) as file:
            data = yaml.safe_load(file)
        return cls(**data)


OutputType = list[str]


class ParallelProgram(BenchmarkMixin, ABC):
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
        "amazon": PromptFormats.from_yaml(_PROMPT_DIR / "amazon.yaml")
    }

    async def run_async(
        self,
        dataset: Literal["amazon"],
        item_reviews: list[tuple[dict[str, Any], list[list[dict[str, Any]]]]],
        num_experts: int,
        num_review_chunks_per_item: int,
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> list[str]:
        self.start_timer("run")
        insight_reports, system_profile = await self._run(
            **self.prepare_kwargs(
                dataset=dataset,
                item_reviews=item_reviews,
                num_experts=num_experts,
                num_review_chunks_per_item=num_review_chunks_per_item,
                generation_config=generation_config,
                **kwargs,
            )
        )
        self.stop_timer()
        self.set_system_profile(system_profile)

        return insight_reports

    async def precompute(
        self,
        dataset: Literal["amazon"],
        item_reviews: list[tuple[dict[str, Any], list[list[dict[str, Any]]]]],
        num_experts: int,
        num_review_chunks_per_item: int,
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        **kwargs,
    ) -> HeliumResponse:
        return await self._precompute(
            precompute_mode=precompute_mode,
            **self.prepare_kwargs(
                dataset=dataset,
                item_reviews=item_reviews,
                num_experts=num_experts,
                num_review_chunks_per_item=num_review_chunks_per_item,
                generation_config=generation_config,
                **kwargs,
            ),
        )

    def prepare_kwargs(
        self,
        dataset: Literal["amazon"],
        item_reviews: list[tuple[dict[str, Any], list[list[dict[str, Any]]]]],
        num_experts: int,
        num_review_chunks_per_item: int,
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> dict[str, Any]:
        if dataset not in self.PROMPT_FORMATS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        prompt_formats = self.PROMPT_FORMATS[dataset]

        expert_system_prompt_fmt = prompt_formats.expert_system_prompt
        role_instructions = list(prompt_formats.role_instructions.items())[:num_experts]
        expert_system_prompts = [
            expert_system_prompt_fmt.format(role=role, instructions=instructions)
            for role, instructions in role_instructions
        ]

        writer_system_prompt = prompt_formats.writer_system_prompt
        extraction_instruction_fmt = prompt_formats.extraction_instruction

        partial_summary_instruction_fmt = prompt_formats.partial_summary_instruction
        extracted_insights_fmt = "\n".join(
            [f"Chunk {i}:\n{{}}" for i in range(1, num_review_chunks_per_item + 1)]
        )
        summary_instruction_fmt = partial_summary_instruction_fmt.format(
            item="{}", extracted_insights=extracted_insights_fmt
        )

        partial_report_instruction_fmt = prompt_formats.partial_report_instruction
        extracted_insights_fmt = "\n".join(
            [f"Insights from {role}:\n{{}}" for role, _ in role_instructions]
        )
        report_instruction_fmt = partial_report_instruction_fmt.format(
            item="{}", extracted_insights=extracted_insights_fmt
        )

        item_descriptions: list[str] = []
        review_chunks: tuple[list[str], ...] = tuple(
            [[] for _ in range(num_review_chunks_per_item)]
        )
        for item, reviews in item_reviews:
            item_descriptions.append(self._get_item_description(item))
            for i, review_chunk in enumerate(reviews):
                review_chunks[i].append(self._get_review_chunk(review_chunk))

        return dict(
            expert_system_prompts=expert_system_prompts,
            writer_system_prompt=writer_system_prompt,
            extraction_instruction_fmt=extraction_instruction_fmt,
            summary_instruction_fmt=summary_instruction_fmt,
            report_instruction_fmt=report_instruction_fmt,
            item_descriptions=item_descriptions,
            review_chunks=review_chunks,
            generation_config=generation_config,
        )

    def flatten_inputs(
        self,
        item_descriptions: list[str],
        review_chunks: tuple[list[str], ...],
        shuffle: bool = True,
    ) -> list[tuple[int, str, tuple[str, ...]]]:
        """
        Returns
        -------
        index : int
            The original index of the item.
        item_description : str
            The item description.
        review_chunks : tuple[str, ...]
            The review chunks.
        """
        pivotted_review_chunks = (chunks for chunks in zip(*review_chunks))
        flattened = [
            (index, item, chunks)
            for index, (item, chunks) in enumerate(
                zip(item_descriptions, pivotted_review_chunks)
            )
        ]
        return random_shuffle(flattened) if shuffle else flattened

    def _get_item_description(self, item: dict[str, Any]) -> str:
        def _get_or_na(key: str) -> str:
            return item.get(key, "N/A")

        features = item.get("features")
        description = item.get("description")
        categories = item.get("categories")
        details_dict: dict[str, str] | None = item.get("details")
        if details_dict is None:
            details = "N/A"
        else:
            details = "\n".join(f"  - {k}: {v}" for k, v in details_dict.items())

        return (
            f"- Category: {_get_or_na('main_category')}\n"
            f"- Title: {_get_or_na('title')}\n"
            f"- Average Rating: {_get_or_na('average_rating')}\n"
            f"- Rating Number: {_get_or_na('rating_number')}\n"
            f"- Total Review Count: {_get_or_na('review_count')}\n"
            f"- Features: {', '.join(features) if features else 'N/A'}\n"
            f"- Description: {', '.join(description) if description else 'N/A'}\n"
            f"- Price: {_get_or_na('price')}\n"
            f"- Store: {_get_or_na('store')}\n"
            f"- Categories: {', '.join(categories) if categories else 'N/A'}\n"
            f"- Details:\n{details}"
        )

    def _get_review_chunk(self, review_chunk: list[dict[str, Any]]) -> str:
        def _get_or_na(review: dict[str, Any], key: str) -> str:
            return review.get(key, "N/A")

        def _format_review(review: dict[str, Any]) -> str:
            return (
                f"  - Title: {_get_or_na(review, 'title')}\n"
                f"  - Rating: {_get_or_na(review, 'rating')}\n"
                f"  - Review: {_get_or_na(review, 'text')}\n"
                f"  - Helpful vote: {_get_or_na(review, 'helpful_vote')}"
            )

        return "\n".join(
            f"Review {i}:\n{_format_review(review)}"
            for i, review in enumerate(review_chunk, start=1)
        )

    @abstractmethod
    async def _run(
        self,
        expert_system_prompts: list[str],
        writer_system_prompt: str,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        item_descriptions: list[str],
        review_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[OutputType, HeliumSystemProfile]:
        pass

    async def _precompute(
        self,
        expert_system_prompts: list[str],
        writer_system_prompt: str,
        extraction_instruction_fmt: str,
        summary_instruction_fmt: str,
        report_instruction_fmt: str,
        item_descriptions: list[str],
        review_chunks: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        **kwargs,
    ) -> HeliumResponse:
        raise NotImplementedError("Precompute is not implemented.")
