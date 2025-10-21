from collections import Counter
from typing import Literal

from helium import ops
from helium.common import GenerationConfig
from helium.frontend.agents import WrapperAgent
from helium.frontend.agents.examples import ChatAgent
from helium.frontend.programs import Program
from helium.runtime import HeliumServerConfig
from helium.runtime.protocol import HeliumRequestConfig


class MCQAProgram(Program):
    AGENT_SYSTEM_PROMPT: str = (
        "You are a student taking a multiple-choice quiz. "
        "You are asked the following question and choices. "
        "Please select the correct answer. "
        "You must answer the choice index (only number) in the following format:\n"
        "Answer: <index>\n"
    )
    COT_PROMPT_SUFFIX: str = (
        "You must think step by step and explain before giving your answer."
    )

    def __init__(
        self,
        name: str | None = None,
        num_agents: int = 1,
        method: Literal["direct", "cot"] = "direct",
        generation_config: GenerationConfig | None = None,
        server_config: HeliumServerConfig | None = None,
    ):
        super().__init__(name, server_config)
        self.num_agents = num_agents
        self.generation_config = generation_config
        self.method = method
        self.system_prompt = self._get_system_prompt()

    async def run_async(
        self,
        questions: list[str],
        choices: list[list[str]],
        config: HeliumRequestConfig | None = None,
    ) -> list[int]:
        if len(questions) != len(choices):
            raise ValueError("Number of questions and choices must match.")

        input_name = "user_inputs"
        program_agent = self.create_program_agent(input_name)

        agent_inputs = [
            "\n".join([q] + [f"{i}. {c}" for i, c in enumerate(cs)])
            for q, cs in zip(questions, choices)
        ]
        program_agent.compile(**{input_name: agent_inputs})
        response = await self.run_agents_async([program_agent], config)
        answers = self._get_most_common_answers(
            len(questions), list(response.outputs[self.name].values())
        )
        return answers

    def create_program_agent(self, input_name: str) -> WrapperAgent:
        input_op = ops.input_placeholder(name=input_name)
        agents: list[ChatAgent] = []
        messages = ChatAgent.create_messages(input_op, self.system_prompt)
        for i in range(self.num_agents):
            agent_name = f"agent_{i}"
            agents.append(
                ChatAgent(
                    input_op=input_op,
                    messages=messages,
                    output_name=agent_name,
                    name=agent_name,
                    generation_config=self.generation_config,
                    server_config=self.server_config,
                )
            )
        program_agent = WrapperAgent(
            agents, name=self.name, server_config=self.server_config
        )
        return program_agent

    def _get_system_prompt(self) -> str:
        if self.method == "direct":
            prompt_suffix = ""
        elif self.method == "cot":
            prompt_suffix = (
                "You must think step by step and explain before giving your answer."
            )
        else:
            raise ValueError(f"Invalid method: {self.method}")
        system_prompt = f"{self.AGENT_SYSTEM_PROMPT}\n{prompt_suffix}"
        return system_prompt

    def _get_most_common_answers(
        self, num_questions: int, agent_answers: list[list[str]]
    ) -> list[int]:
        counters: list[Counter[int]] = [Counter() for _ in range(num_questions)]
        for answers in agent_answers:
            for counter, ans in zip(counters, answers):
                if "Answer: " not in ans:
                    continue
                try:
                    parsed = int(ans.split("Answer: ")[-1])
                except ValueError:
                    continue
                counter[parsed] += 1
        outputs = []
        for counter in counters:
            most_common = counter.most_common(1)
            if len(most_common) == 0:
                outputs.append(-1)
            else:
                outputs.append(most_common[0][0])
        return outputs
