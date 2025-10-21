import itertools
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, Self

from bench_programs.utils.common import random_shuffle
from bench_utils.mixin import BenchmarkMixin

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumResponse, HeliumSystemProfile


@dataclass
class PromptFormats:
    roles: list[str]
    role_prompt: str
    context_prompt: str | None
    question_prompt: str
    expert_system_prompt: str = (
        "You are a helpful AI assistant, roleplaying as an expert in your field."
    )
    summarizer_system_prompt: str = (
        "You are a helpful AI assistant, specializing in summarizing answers from "
        "multiple experts."
    )
    summary_prompt: str = (
        "Review the given question and the answers provided by different "
        "experts. Give your thoughts and summarize the answers into one final "
        'answer at the end of your response, preceding with "Answer: ".'
    )


OutputType = list[tuple[str, ...]]


class MapReduceProgram(BenchmarkMixin, ABC):
    class OutputBuilder:
        def __init__(self) -> None:
            self._inner: dict[tuple[int, int], str] = {}

        def add(self, context_idx: int, question_idx: int, response: str) -> None:
            self._inner[(context_idx, question_idx)] = response

        def update(self, items: Iterable[tuple[int, int, str]]) -> Self:
            for context_idx, question_idx, response in items:
                self.add(context_idx, question_idx, response)
            return self

        def build(self) -> OutputType:
            num_contexts = max(k[0] for k in self._inner) + 1
            num_questions = max(k[1] for k in self._inner) + 1
            ret = [
                tuple(
                    self._inner[(context_idx, question_idx)]
                    for question_idx in range(num_questions)
                )
                for context_idx in range(num_contexts)
            ]
            return ret

    PROMPT_FORMATS: dict[str, PromptFormats] = {
        "mmlu": PromptFormats(
            roles=[
                "a professor",
                "a doctor",
                "a mathematician",
                "a scientist",
                "a historian",
                "a philosopher",
                "an economist",
            ],
            role_prompt="Respond as if you are {}.",
            context_prompt=None,
            question_prompt=(
                "Can you answer the following question as accurately as possible? "
                "{}:\nA) {}.\nB) {}.\nC) {}.\nD) {}.\nExplain your answer, putting the "
                "answer in the form (X) at the end of your response. "
            ),
        ),
        "tatqa": PromptFormats(
            roles=[
                "a financial analyst",
                "a data scientist",
                "a statistician",
                "a business expert",
                "a market researcher",
                "a financial consultant",
                "a forensic accountant",
            ],
            role_prompt="Respond as if you are {}.",
            context_prompt=(
                "Please answer the given financial question based on the context.\n"
                "Context:\n{}\n\n"
            ),
            question_prompt=(
                "Question: {}\nExplain your thoughts step by step and summarize your "
                'answer at the end of your response, preceding with "Answer: ". '
            ),
        ),
    }

    def __init__(self) -> None:
        BenchmarkMixin.__init__(self)
        # context idx, question idx, agent idx -> conversation
        self._conversations: dict[tuple[int, int, int], list[dict[str, str]]] = {}

    async def run_async(
        self,
        dataset: Literal["mmlu", "tatqa"],
        contexts: list[str] | None,
        context_questions: tuple[list[str], ...],
        context_choices: tuple[list[list[str]], ...] | None,
        num_agents: int,
        different_roles: bool,
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> list[tuple[str | None, ...]]:
        """
        Returns
        -------
        list[tuple[str | None, ...]]
            A nested list/tuple in the following order: context -> question -> answer.
        """

        self.start_timer("run")
        responses_list, system_profile = await self._run(
            **self.prepare_kwargs(
                dataset,
                contexts,
                context_questions,
                context_choices,
                num_agents,
                different_roles,
                generation_config,
                **kwargs,
            )
        )
        self.stop_timer()
        self.set_system_profile(system_profile)

        return self._post_process_output(dataset, responses_list)

    async def precompute(
        self,
        dataset: Literal["mmlu", "tatqa"],
        contexts: list[str] | None,
        context_questions: tuple[list[str], ...],
        context_choices: tuple[list[list[str]], ...] | None,
        num_agents: int,
        different_roles: bool,
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        **kwargs,
    ) -> HeliumResponse:
        return await self._precompute(
            precompute_mode=precompute_mode,
            **self.prepare_kwargs(
                dataset,
                contexts,
                context_questions,
                context_choices,
                num_agents,
                different_roles,
                generation_config,
                **kwargs,
            ),
        )

    def prepare_kwargs(
        self,
        dataset: Literal["mmlu", "tatqa"],
        contexts: list[str] | None,
        context_questions: tuple[list[str], ...],
        context_choices: tuple[list[list[str]], ...] | None,
        num_agents: int,
        different_roles: bool,
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> dict[str, Any]:
        if dataset not in self.PROMPT_FORMATS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        prompt_formats = self.PROMPT_FORMATS[dataset]

        # Validate the arguments
        if context_choices is not None and len(context_choices) != len(
            context_questions
        ):
            raise ValueError(
                "Number of sets of choices must match number of questions."
            )

        # Create a role list
        if different_roles:
            num_available_roles = len(prompt_formats.roles)
            num_repetitions = num_agents // num_available_roles
            num_remaining_roles = num_agents % num_available_roles
            roles = prompt_formats.roles * num_repetitions
            roles += prompt_formats.roles[:num_remaining_roles]
        else:
            roles = None

        expert_system_prompt = prompt_formats.expert_system_prompt
        summarizer_system_prompt = prompt_formats.summarizer_system_prompt
        summary_prompt = prompt_formats.summary_prompt

        role_prompt_format = prompt_formats.role_prompt
        role_prompts = (
            [role_prompt_format.format(role) for role in roles]
            if roles is not None
            else None
        )

        context_prompt_format = prompt_formats.context_prompt
        context_prompts: list[str] | None
        if context_prompt_format is None:
            context_prompts = None
        else:
            if contexts is None:
                raise ValueError(f"Contexts must be provided for dataset {dataset}.")
            context_prompts = [
                context_prompt_format.format(context) for context in contexts
            ]

        question_prompt_format = prompt_formats.question_prompt
        context_question_prompts: tuple[list[str], ...]
        if context_choices is None:
            context_question_prompts = tuple(
                [question_prompt_format.format(q) for q in questions]
                for questions in context_questions
            )
        else:
            context_question_prompts = tuple(
                [
                    question_prompt_format.format(q, *cs)
                    for q, cs in zip(questions, question_choices)
                ]
                for questions, question_choices in zip(
                    context_questions, context_choices
                )
            )

        return dict(
            expert_system_prompt=expert_system_prompt,
            summarizer_system_prompt=summarizer_system_prompt,
            summary_prompt=summary_prompt,
            role_prompts=role_prompts,
            context_prompts=context_prompts,
            context_question_prompts=context_question_prompts,
            num_agents=num_agents,
            generation_config=generation_config,
            **kwargs,
        )

    @classmethod
    def build_user_prompt(
        cls, role_prompt: str | None, context_prompt: str | None, question_prompt: str
    ) -> str:
        user_prompt = ""
        if context_prompt is not None:
            user_prompt += context_prompt
        user_prompt += question_prompt
        if role_prompt is not None:
            user_prompt += role_prompt
        return user_prompt

    def flatten_inputs(
        self,
        context_prompts: list[str] | None,
        context_question_prompts: tuple[list[str], ...],
        shuffle: bool = True,
    ) -> list[tuple[int, int, str | None, str]]:
        """
        Returns
        -------
        content_idx: int
            The index of the context.
        question_idx: int
            The index of the question associated with the context.
        context_prompt: str | None
            The context prompt, or None if no context is provided.
        question_prompt: str
            The question prompt.
        """
        pivotted_context_questions = (qs for qs in zip(*context_question_prompts))
        flattened = [
            (context_idx, question_idx, context_prompt, question_prompt)
            for context_idx, (context_prompt, question_prompts) in enumerate(
                zip(
                    context_prompts or itertools.repeat(None),
                    pivotted_context_questions,
                )
            )
            for question_idx, question_prompt in enumerate(question_prompts)
        ]
        return random_shuffle(flattened) if shuffle else flattened

    @abstractmethod
    async def _run(
        self,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
        summary_prompt: str,
        role_prompts: list[str] | None,
        context_prompts: list[str] | None,
        context_question_prompts: tuple[list[str], ...],
        num_agents: int,
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[OutputType, HeliumSystemProfile]:
        pass

    async def _precompute(
        self,
        expert_system_prompt: str,
        summarizer_system_prompt: str,
        summary_prompt: str,
        role_prompts: list[str] | None,
        context_prompts: list[str] | None,
        context_question_prompts: tuple[list[str], ...],
        num_agents: int,
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        **kwargs,
    ) -> HeliumResponse:
        raise NotImplementedError("Precompute is not implemented.")

    def _post_process_output(
        self, dataset: Literal["mmlu", "tatqa"], output: OutputType
    ) -> list[tuple[str | None, ...]]:
        ret: list[tuple[str | None, ...]]
        match dataset:
            case "mmlu":
                ret = [
                    tuple(
                        self.parse_mcqa_answer(response)
                        for response in question_responses
                    )
                    for question_responses in output
                ]
            case "tatqa":
                ret = [question_responses for question_responses in output]
        return ret

    def parse_mcqa_answer(self, response: str) -> str | None:
        """Parses the multiple-choice answer from the response string"""
        match = re.search(r"\((\w)\)", response)
        if match is None:
            return None
        return match.group().upper()
