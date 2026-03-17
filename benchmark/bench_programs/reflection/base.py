from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, Self

from bench_programs.utils.common import random_shuffle
from bench_utils.mixin import BenchmarkMixin

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumResponse, HeliumSystemProfile

FINANCIAL_ANALYST_SYSTEM_PROMPT = """You are a financial analysis agent specializing in interpreting earnings reports and financial statements. Your task is to answer specific financial questions based on the give context from financial reports.
When answering questions:
* Carefully read and analyze the provided financial information.
* Extract the relevant data points needed to answer the question from the table or text provided.
* Perform any necessary calculations.
* Remember to be precise in your calculations and clear in your step-by-step explanation. Maintain a professional and objective tone in your response.
* Use only the information provided in the context. Do not introduce external information.
* Provide the answer in the unit specified in the question (million, percentage, or billion). If no unit is specified, use the most appropriate unit based on the context and question."""
EXTRACTION_CRITIC_SYSTEM_PROMPT = """You are a meticulous financial analyst and critic. Your task is to review the response provided by another agent regarding financial calculations and provide feedback on its accuracy and completeness. Pay close attention to the following aspects:
* Question Comprehension: Does the response correctly understand the original question?
* Data Extraction: Are all relevant numbers accurately extracted from the provided text/tables?
Focus only on these two aspects. Do not evaluate calculations or provide additional analysis."""
CALCULATION_CRITIC_SYSTEM_PROMPT = """You are a meticulous financial analyst and critic. Your task is to review the response provided by another agent regarding financial calculations and provide feedback on its accuracy and completeness. Pay close attention to the following aspects:
* Calculation Steps: Confirm that all calculation steps are correct.
* Calculation Accuracy: Verify the accuracy of all calculations, including intermediate and final results.
* Unit Consistency: Ensure the final answer's unit matches what the question requires."""
FINANCIAL_ANALYST_INSTRUCTION = """Read the following texts and table with financial data from an earnings report carefully.
Present your answer in the following JSON format:
  {{
    "steps": ["show the calculation steps"],
    "Answer": "final numerical answer"
  }}
Below is the Context along with the Question."""
EXTRACTION_CRITIC_INSTRUCTION = "Review a given context, question, and the response provided by another agent. Then, you must reflect on the analysis and provide a detailed critique."
CALCULATION_CRITIC_INSTRUCTION = "Review a given context, question, the response provided by one agent, and the critic provided by another agent. Then, you must reflect on the analysis and provide a detailed critique."
FINAL_ANSWER_INSTRUCTION = "Review two critics of your answer provided by two different agents. Then, you must reflect on the analysis and provide a final answer."


@dataclass
class PromptFormats:
    financial_analyst: str
    extraction_critic: str
    calculation_critic: str
    final_answer: str
    system_prompts: tuple[str, str, str] = (
        FINANCIAL_ANALYST_SYSTEM_PROMPT,
        EXTRACTION_CRITIC_SYSTEM_PROMPT,
        CALCULATION_CRITIC_SYSTEM_PROMPT,
    )


OutputType = list[tuple[str, ...]]


class ReflectionProgram(BenchmarkMixin, ABC):
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
        "tatqa": PromptFormats(
            financial_analyst=FINANCIAL_ANALYST_INSTRUCTION
            + "\n\nContext: {context}\n\nQuestion: {question}",
            extraction_critic=EXTRACTION_CRITIC_INSTRUCTION
            + "\nContext and Question:\nContext: {context}\nQuestion: {question}\nResponse to Analyze:\n{response}",
            calculation_critic=CALCULATION_CRITIC_INSTRUCTION
            + "\nContext and Question:\nContext: {context}\nQuestion: {question}\nResponse to Analyze:\n{response}\nCritic from another agent:\n{critic}",
            final_answer=FINAL_ANSWER_INSTRUCTION
            + "\nData Extraction Critic:\n{extraction_critic}\nCalculation Critic:\n{calculation_critic}",
        ),
        "finqa": PromptFormats(
            financial_analyst=FINANCIAL_ANALYST_INSTRUCTION
            + "\n\nContext: {context}\n\nQuestion: {question}",
            extraction_critic=EXTRACTION_CRITIC_INSTRUCTION
            + "\nContext and Question:\nContext: {context}\nQuestion: {question}\nResponse to Analyze:\n{response}",
            calculation_critic=CALCULATION_CRITIC_INSTRUCTION
            + "\nContext and Question:\nContext: {context}\nQuestion: {question}\nResponse to Analyze:\n{response}\nCritic from another agent:\n{critic}",
            final_answer=FINAL_ANSWER_INSTRUCTION
            + "\nData Extraction Critic:\n{extraction_critic}\nCalculation Critic:\n{calculation_critic}",
        ),
    }

    def __init__(self) -> None:
        BenchmarkMixin.__init__(self)
        # question idx -> agent -> messages
        self._conversations: dict[int, dict[str, list[dict[str, str]]]] = {}

    async def run_async(
        self,
        dataset: Literal["finqa", "tatqa"],
        contexts: list[str],
        context_questions: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> list[tuple[str, ...]]:
        self.start_timer("run")
        responses_list, system_profile = await self._run(
            **self.prepare_kwargs(
                dataset,
                contexts,
                context_questions,
                generation_config,
                **kwargs,
            )
        )
        self.stop_timer()
        self.set_system_profile(system_profile)

        return responses_list

    async def precompute(
        self,
        dataset: Literal["finqa", "tatqa"],
        contexts: list[str],
        context_questions: tuple[list[str], ...],
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
                generation_config,
                **kwargs,
            ),
        )

    def prepare_kwargs(
        self,
        dataset: Literal["finqa", "tatqa"],
        contexts: list[str],
        context_questions: tuple[list[str], ...],
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> dict[str, Any]:
        if dataset not in self.PROMPT_FORMATS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        prompt_formats = self.PROMPT_FORMATS[dataset]

        return dict(
            contexts=contexts,
            context_questions=context_questions,
            system_prompts=prompt_formats.system_prompts,
            financial_analyst_fmt=prompt_formats.financial_analyst,
            extraction_critic_fmt=prompt_formats.extraction_critic,
            calculation_critic_fmt=prompt_formats.calculation_critic,
            final_answer_fmt=prompt_formats.final_answer,
            generation_config=generation_config,
            **kwargs,
        )

    def flatten_inputs(
        self,
        contexts: list[str],
        context_questions: tuple[list[str], ...],
        shuffle: bool = True,
    ) -> list[tuple[int, int, str, str]]:
        """
        Returns
        -------
        content_idx: int
            The index of the context.
        question_idx: int
            The index of the question associated with the context.
        context_prompt: str
            The context prompt.
        question_prompt: str
            The question prompt.
        """
        pivotted_context_questions = (qs for qs in zip(*context_questions))
        flattened = [
            (context_idx, question_idx, context_prompt, question_prompt)
            for context_idx, (context_prompt, question_prompts) in enumerate(
                zip(contexts, pivotted_context_questions)
            )
            for question_idx, question_prompt in enumerate(question_prompts)
        ]
        return random_shuffle(flattened) if shuffle else flattened

    @abstractmethod
    async def _run(
        self,
        contexts: list[str],
        context_questions: tuple[list[str], ...],
        system_prompts: tuple[str, str, str],
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> tuple[OutputType, HeliumSystemProfile]:
        pass

    async def _precompute(
        self,
        contexts: list[str],
        context_questions: tuple[list[str], ...],
        system_prompts: tuple[str, str, str],
        financial_analyst_fmt: str,
        extraction_critic_fmt: str,
        calculation_critic_fmt: str,
        final_answer_fmt: str,
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        **kwargs,
    ) -> HeliumResponse:
        raise NotImplementedError("Precompute is not implemented.")
