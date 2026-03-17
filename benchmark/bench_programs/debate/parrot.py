import asyncio
from collections.abc import Sequence

from bench_programs.debate.base import DebateProgram
from bench_programs.utils.parrot import (
    ParrotMixin,
    SemanticFunction,
    SemanticVariable,
    parrot_sampling_config,
    parrot_semantic_function,
    parrot_semantic_variable,
    parrot_start_benchmark,
    parrot_stop_benchmark,
)
from parrot import P

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class ParrotDebateProgram(ParrotMixin, DebateProgram):
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
    ) -> tuple[list[tuple[list[str], ...]], HeliumSystemProfile]:
        with self.get_vm() as vm:
            if generation_config is None:
                generation_config = GenerationConfig.from_env()

            # Start benchmarking
            parrot_start_benchmark(vm)

            # Prepare inputs
            flattened = self.flatten_inputs(context_prompts, context_question_prompts)

            # Start generation
            self.start_timer("generate")

            self.start_timer("prepare")
            sampling_config = parrot_sampling_config(generation_config)
            semantic_functions = self._create_semantic_functions(
                vm,
                system_prompt,
                role_prompts,
                revise_prompts,
                context_prompts is not None,
                num_agents,
                num_rounds,
                generation_config.model,
                sampling_config,
            )

            context_prompt_cache: dict[int, SemanticVariable] = {}
            input_semantic_vars: list[
                tuple[int, int, SemanticVariable | None, SemanticVariable]
            ] = []
            for context_idx, question_idx, context_prompt, question_prompt in flattened:
                if context_prompt is None:
                    context_var = None
                else:
                    context_var = context_prompt_cache.get(context_idx)
                    if context_var is None:
                        context_var = P.variable(content=context_prompt)
                        context_prompt_cache[context_idx] = context_var
                question_var = P.variable(content=question_prompt)
                input_semantic_vars.append(
                    (context_idx, question_idx, context_var, question_var)
                )
            semantic_variables = [
                (
                    context_idx,
                    question_idx,
                    self._create_semantic_variables(
                        semantic_functions,
                        num_agents,
                        num_rounds,
                        context_var,
                        question_var,
                    ),
                )
                for context_idx, question_idx, context_var, question_var in input_semantic_vars
            ]
            self.stop_timer()

            outputs = await asyncio.gather(
                *[
                    asyncio.gather(*[variable.aget() for variable in variables])
                    for _, _, variables in semantic_variables
                ]
            )

            # Stop generation
            self.stop_timer()

            # Stop benchmarking
            system_profile = parrot_stop_benchmark(vm)

        output_builder = self.OutputBuilder()
        for (context_idx, question_idx, _), results in zip(semantic_variables, outputs):
            for agent_idx, result in enumerate(results):
                output_builder.add(context_idx, question_idx, agent_idx, result)

        return output_builder.build(), system_profile

    def _get_revise_message(
        self,
        other_agent_names: list[str],
        revise_prompts: tuple[str, str],
    ) -> str:
        if len(other_agent_names) == 0:
            return revise_prompts[0]
        return "\n\n ".join(
            [
                "These are the solutions to the problem from other agents: ",
                *[
                    f"One agent solution: ```{{{{{other_agent}}}}}```"
                    for other_agent in other_agent_names
                ],
                revise_prompts[1],
            ]
        )

    def _create_semantic_functions(
        self,
        vm: P.VirtualMachine,
        system_prompt: str,
        role_prompts: list[str] | None,
        revise_prompts: tuple[str, str],
        has_contexts: bool,
        num_agents: int,
        num_rounds: int,
        model: str,
        sampling_config: P.SamplingConfig,
    ) -> dict[str, SemanticFunction]:
        # Initialize agent contexts
        maybe_role_prompts: Sequence[str | None]
        if role_prompts is None:
            maybe_role_prompts = [None] * num_agents
        else:
            maybe_role_prompts = role_prompts
        user_prompts = [
            self.build_user_prompt(
                role_prompt, "{{context}}" if has_contexts else None, "{{question}}"
            )
            for role_prompt in maybe_role_prompts
        ]
        system_message = {"role": "system", "content": system_prompt}
        agent_names: list[str] = []  # Agent names in the last round
        agent_contexts = [
            [system_message, {"role": "user", "content": user_prompt}]
            for user_prompt in user_prompts
        ]  # Agent contexts updated in place

        initial_kwargs = {"question": P.Input}
        if has_contexts:
            initial_kwargs["context"] = P.Input
        agent_kwargs_list: list[dict[str, type[P.Input]]] = [
            initial_kwargs.copy() for _ in range(num_agents)
        ]  # Agent kwargs updated in place

        semantic_functions: dict[str, SemanticFunction] = {}
        for round in range(num_rounds):
            if round > 0:
                new_agent_contexts: list[list[dict[str, str]]] = []
                for i, (agent_context, agent_kwargs) in enumerate(
                    zip(agent_contexts, agent_kwargs_list)
                ):
                    other_agent_names = agent_names[:i] + agent_names[i + 1 :]
                    new_message_content = self._get_revise_message(
                        other_agent_names, revise_prompts
                    )
                    new_agent_contexts.append(
                        [
                            *agent_context,
                            {"role": "user", "content": new_message_content},
                        ]
                    )
                    # Update agent kwargs to include inputs from other agents
                    for other_agent_name in other_agent_names:
                        agent_kwargs[other_agent_name] = P.Input
                # Update agent contexts
                agent_contexts = new_agent_contexts

            # Create agent names and semantic functions for this round
            agent_names = [self._get_agent_name(i, round) for i in range(num_agents)]
            for agent_name, agent_context, agent_kwargs in zip(
                agent_names, agent_contexts, agent_kwargs_list
            ):
                agent_context.append(
                    {"role": "assistant", "content": f"{{{{{agent_name}}}}}"}
                )
                agent_reply = parrot_semantic_function(
                    vm,
                    agent_name,
                    model,
                    agent_context,
                    **agent_kwargs | {agent_name: P.Output(sampling_config)},
                )
                # Update agent kwargs to include its own output
                agent_kwargs[agent_name] = P.Input
                # Store the semantic function
                semantic_functions[agent_name] = agent_reply

        return semantic_functions

    def _create_semantic_variables(
        self,
        semantic_functions: dict[str, SemanticFunction],
        num_agents: int,
        num_rounds: int,
        context_prompt: SemanticVariable | None,
        question_prompt: SemanticVariable,
    ) -> list[SemanticVariable]:
        agent_names: list[str] = []  # Agent names in the last round
        agent_vars: list[SemanticVariable] = []  # Agent variables in the last round

        initial_kwargs = {"question": question_prompt}
        if context_prompt is not None:
            initial_kwargs["context"] = context_prompt
        agent_kwargs_list = [
            initial_kwargs.copy() for _ in range(num_agents)
        ]  # Agent kwargs updated in place

        for round in range(num_rounds):
            if round > 0:
                for i, agent_kwargs in enumerate(agent_kwargs_list):
                    # Update agent kwargs to include inputs from other agents
                    agent_kwargs.update(
                        {
                            agent_name: agent_var
                            for j, (agent_name, agent_var) in enumerate(
                                zip(agent_names, agent_vars)
                            )
                            if j != i
                        }
                    )

            # Create agent names and semantic variables for this round
            agent_names = [self._get_agent_name(i, round) for i in range(num_agents)]
            new_agent_vars: list[SemanticVariable] = []
            for agent_name, agent_kwargs in zip(agent_names, agent_kwargs_list):
                agent_var = parrot_semantic_variable(
                    semantic_functions[agent_name], **agent_kwargs
                )
                # Update agent kwargs to include its own output
                agent_kwargs[agent_name] = agent_var
                # Store the semantic variable
                new_agent_vars.append(agent_var)
            agent_vars = new_agent_vars

        return agent_vars

    def _get_agent_name(self, agent_idx: int, round_idx: int) -> str:
        return f"agent_{agent_idx}_round_{round_idx}"
