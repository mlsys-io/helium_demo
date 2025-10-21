import itertools
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from bench_programs.utils.common import random_shuffle
from bench_utils.mixin import BenchmarkMixin

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumResponse, HeliumSystemProfile


@dataclass
class PromptFormats:
    roles: list[str]
    system_prompt: str
    role_prompt: str
    context_prompt: str | None
    question_prompt: str
    revise_prompts: tuple[str, str] = (
        "Can you double check that your answer is correct. Put your final answer "
        "in the form (X) at the end of your response.",
        "Using the reasoning from other agents as additional advice, can you give "
        "an updated answer? Examine your solution and that other agents step by "
        "step. Put your answer in the form (X) at the end of your response.",
    )


OutputType = list[tuple[list[str], ...]]


class DebateProgram(BenchmarkMixin, ABC):
    class OutputBuilder:
        def __init__(self) -> None:
            self._inner: dict[tuple[int, int, int], str] = {}

        def add(
            self, context_idx: int, question_idx: int, agent_idx: int, response: str
        ) -> None:
            self._inner[(context_idx, question_idx, agent_idx)] = response

        def update(self, items: Iterable[tuple[int, int, int, str]]) -> Self:
            for context_idx, question_idx, agent_idx, response in items:
                self.add(context_idx, question_idx, agent_idx, response)
            return self

        def build(self) -> OutputType:
            num_contexts = max(k[0] for k in self._inner) + 1
            num_questions = max(k[1] for k in self._inner) + 1
            num_agents = max(k[2] for k in self._inner) + 1
            ret = [
                tuple(
                    [
                        self._inner[(context_idx, question_idx, agent_idx)]
                        for agent_idx in range(num_agents)
                    ]
                    for question_idx in range(num_questions)
                )
                for context_idx in range(num_contexts)
            ]
            return ret

        def to_dict(self) -> dict[tuple[int, int, int], str]:
            return self._inner.copy()

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
            system_prompt="You are a helpful AI Assistant.",
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
            system_prompt="You are a helpful AI Assistant.",
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
        # questions -> agents -> messages
        self._conversations: dict[tuple[int, int, int], list[dict[str, str]]] = {}

    async def run_async(
        self,
        dataset: Literal["mmlu", "tatqa"],
        contexts: list[str] | None,
        context_questions: tuple[list[str], ...],
        context_choices: tuple[list[list[str]], ...] | None,
        num_agents: int,
        num_rounds: int,
        different_roles: bool,
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> list[tuple[list[str | None], ...]]:
        """
        Returns
        -------
        list[tuple[list[str | None]]]
            A nested list/tuple in the following order: context -> question -> agent
            responses.
        """
        self.start_timer("run")
        responses_list, system_profile = await self._run(
            **self.prepare_kwargs(
                dataset,
                contexts,
                context_questions,
                context_choices,
                num_agents,
                num_rounds,
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
        num_rounds: int,
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
                num_rounds,
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
        num_rounds: int,
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

        system_prompt = prompt_formats.system_prompt

        role_prompt_format = prompt_formats.role_prompt
        role_prompts = (
            [
                role_prompt_format.format(role)
                for role in itertools.islice(
                    itertools.cycle(prompt_formats.roles), num_agents
                )
            ]
            if different_roles
            else None
        )
        if role_prompts is not None and len(role_prompts) != num_agents:
            raise ValueError(
                f"Number of role prompts ({len(role_prompts)}) does not match number "
                f"of agents ({num_agents})."
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

        revise_prompts = prompt_formats.revise_prompts

        return dict(
            system_prompt=system_prompt,
            role_prompts=role_prompts,
            context_prompts=context_prompts,
            context_question_prompts=context_question_prompts,
            revise_prompts=revise_prompts,
            num_agents=num_agents,
            num_rounds=num_rounds,
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
        system_prompt: str,
        role_prompts: list[str] | None,
        context_prompts: list[str] | None,
        context_question_prompts: tuple[list[str], ...],
        revise_prompts: tuple[str, str],
        num_agents: int,
        num_rounds: int,
        generation_config: GenerationConfig | None,
        dump_conversations: bool = False,
        **kwargs,
    ) -> tuple[OutputType, HeliumSystemProfile]:
        pass

    async def _precompute(
        self,
        system_prompt: str,
        role_prompts: list[str] | None,
        context_prompts: list[str] | None,
        context_question_prompts: tuple[list[str], ...],
        revise_prompts: tuple[str, str],
        num_agents: int,
        num_rounds: int,
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        dump_conversations: bool = False,
        **kwargs,
    ) -> HeliumResponse:
        raise NotImplementedError("Precompute not implemented")

    def _post_process_output(
        self, dataset: Literal["mmlu", "tatqa"], output: OutputType
    ) -> list[tuple[list[str | None], ...]]:
        # Postprocess final responses.
        ret: list[tuple[list[str | None], ...]]
        match dataset:
            case "mmlu":
                ret = [
                    tuple(
                        [
                            self.parse_mcqa_answer(agent_response)
                            for agent_response in question_responses
                        ]
                        for question_responses in context_responses
                    )
                    for context_responses in output
                ]
            case "tatqa":
                ret = [
                    tuple(
                        [agent_response for agent_response in question_responses]
                        for question_responses in context_responses
                    )
                    for context_responses in output
                ]
        return ret

    def parse_mcqa_answer(self, response: str) -> str | None:
        """Parses the multiple-choice answer from the response string"""
        match = re.search(r"\((\w)\)", response)
        if match is None:
            return None
        return match.group().upper()

    def _add_conversation(
        self,
        context_idx: int,
        question_idx: int,
        agent_idx: int,
        conversation: list[dict[str, str]],
    ) -> None:
        self._conversations[(context_idx, question_idx, agent_idx)] = conversation

    def _dump_conversations(self, dump_file: str | Path) -> None:
        dump_file = Path(dump_file)
        if dump_file.exists():
            print()
            if (
                input(
                    "Warning: Dump file already exists. Do you want to overwrite it? (y/n): "
                )
                .strip()
                .lower()
                != "y"
            ):
                return
        dump_file.parent.mkdir(parents=True, exist_ok=True)

        with dump_file.open("w", encoding="utf-8") as f:
            sorted_keys = sorted(self._conversations)
            for key in sorted_keys:
                conversation = self._conversations[key]
                context_idx, question_idx, agent_idx = key
                f.write(
                    f"Context {context_idx + 1}, Question {question_idx}, Conversation {agent_idx}\n"
                )
                for message in conversation:
                    role = message["role"]
                    content = message["content"]
                    f.write(f"{role}: {content}\n")
                f.write("\n")
        print(f"Conversations dumped to '{dump_file}'")
