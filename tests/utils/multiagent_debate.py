from typing import Any

from helium import graphs, ops


def build_graph(num_prompts: int) -> graphs.CompiledGraph:
    num_agents = 2
    num_rounds = 2
    questions = [f"TEST{i}" for i in range(1, num_prompts + 1)]
    system_prompt = "You are a helpful AI Assistant."
    input_op = ops.input_placeholder("questions")
    revise_prompts = (
        (
            "Can you double check that your answer is correct. Put your final answer "
            "in the form (X) at the end of your response."
        ),
        (
            "Using the reasoning from other agents as additional advice, can you give "
            "an updated answer? Examine your solution and that other agents step by "
            "step. Put your answer in the form (X) at the end of your response."
        ),
    )
    generation_config = None

    # First round
    initial_message_list = [
        [
            ops.OpMessage(role="system", content=system_prompt),
            ops.OpMessage(role="user", content=input_op),
        ]
        for _ in range(num_agents)
    ]
    history_list = [
        ops.llm_chat(message, generation_config, return_history=True)
        for message in initial_message_list
    ]

    if num_rounds == 1:
        return graphs.from_ops(
            [
                ops.as_output(f"agent_{i}", history)
                for i, history in enumerate(history_list)
            ]
        ).compile(questions=questions)

    # Debate rounds
    revise_prompt: ops.Op
    if num_agents == 1:
        revise_prompt = ops.data(revise_prompts[0])
        new_convo_list = [
            ops.append_message(history, revise_prompt) for history in history_list
        ]
    else:
        last_message_list = [ops.get_last_message(history) for history in history_list]
        new_convo_list = []
        for i, history in enumerate(history_list):
            other_agent_answers = last_message_list[:i] + last_message_list[i + 1 :]
            revise_prompt = ops.format_op(
                "\n\n ".join(
                    [
                        "These are the solutions to the problem from other agents: ",
                        *[
                            f"One agent solution: ```{{agent_{j}}}```"
                            for j in range(num_agents - 1)
                        ],
                        revise_prompts[1],
                    ]
                ),
                **{f"agent_{j}": ans for j, ans in enumerate(other_agent_answers)},
            )
            new_convo_list.append(ops.append_message(history, revise_prompt))
    revised_history_list = [
        ops.llm_chat(convo, generation_config, return_history=True)
        for convo in new_convo_list
    ]
    debate_loop = ops.loop(history_list, revised_history_list, num_rounds - 1)

    return graphs.from_ops(
        [
            ops.as_output(f"agent_{i}", agent_history)
            for i, agent_history in enumerate(debate_loop)
        ]
    ).compile(questions=questions)


def check_outputs(out: Any, num_prompts: int) -> None:
    expected = {
        "agent_0": [
            [
                {"role": "system", "content": "You are a helpful AI Assistant."},
                {"role": "user", "content": f"TEST{i}"},
                {"role": "assistant", "content": "MOCK"},
                {
                    "role": "user",
                    "content": "These are the solutions to the problem from other agents: \n\n One agent solution: ```MOCK```\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.",
                },
                {"role": "assistant", "content": "MOCK"},
            ]
            for i in range(1, num_prompts + 1)
        ],
        "agent_1": [
            [
                {"role": "system", "content": "You are a helpful AI Assistant."},
                {"role": "user", "content": f"TEST{i}"},
                {"role": "assistant", "content": "MOCK"},
                {
                    "role": "user",
                    "content": "These are the solutions to the problem from other agents: \n\n One agent solution: ```MOCK```\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.",
                },
                {"role": "assistant", "content": "MOCK"},
            ]
            for i in range(1, num_prompts + 1)
        ],
    }
    assert out == expected, f"Expected {expected}, got {out}"
