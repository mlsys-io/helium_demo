import random
import re
from typing import Any

from helium import graphs, ops


def _check_resp(inputs: tuple[ops.SingleDtype, ...]) -> str:
    s, answer = inputs
    assert isinstance(s, str) and isinstance(answer, str)
    guess = re.search(r"(\d+)", s)
    if guess and int(answer) < int(guess.group()):
        return "high"
    return "low"


def build_graph() -> graphs.Graph:
    num_agents = 2
    num_iter = 2

    random.seed(42)
    answers = random.choices(range(1, 101), k=num_agents)
    answer_data = ops.data([str(i) for i in answers])
    chat_instr = ops.message_data(
        [
            ops.OpMessage(role="system", content="You are a helpful assistant."),
            ops.OpMessage(
                role="user", content=["Guess a number between 1 and 100"] * num_agents
            ),
        ]
    )
    llm_chat = ops.llm_chat(chat_instr, return_history=True)
    llm_resp = ops.get_last_message(llm_chat)
    check = ops.lambda_op([llm_resp, answer_data], _check_resp)
    new_msg = ops.format_op("Too {check}. Try again.", check=check)
    new_instr = ops.append_message(llm_chat, new_msg)

    cmpl_prompt = ops.data("My name is")
    llm_cmpl = ops.llm_completion(cmpl_prompt, echo=True)

    chat, _, cmpl = ops.loop(
        [None, chat_instr, cmpl_prompt],
        [llm_chat, new_instr, llm_cmpl],
        num_iter=num_iter,
    )

    chat_out = ops.as_output("chat", chat)
    cmpl_out = ops.as_output("cmpl", cmpl)

    return graphs.from_ops([chat_out, cmpl_out])


def check_outputs(outputs: Any) -> None:
    assert len(outputs) == 2, "Output length mismatch"
    assert "chat" in outputs and "cmpl" in outputs, "Missing expected outputs"

    expected_completion = ["My name isMOCKMOCK"]
    assert (
        outputs["cmpl"] == expected_completion
    ), f"Expected {expected_completion}, got {outputs['cmpl']}"

    expected_chat = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Guess a number between 1 and 100"},
            {"role": "assistant", "content": "MOCK"},
            {"role": "user", "content": "Too low. Try again."},
            {"role": "assistant", "content": "MOCK"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Guess a number between 1 and 100"},
            {"role": "assistant", "content": "MOCK"},
            {"role": "user", "content": "Too low. Try again."},
            {"role": "assistant", "content": "MOCK"},
        ],
    ]
    assert (
        outputs["chat"] == expected_chat
    ), f"Expected {expected_chat}, got {outputs['chat']}"
