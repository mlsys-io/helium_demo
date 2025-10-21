from typing import Any

from helium import graphs, ops


def build_graph(num_generations: int) -> graphs.Graph:
    # 1. Generate a draft introduction.
    draft_instr = ops.message_data(
        [
            ops.OpMessage(
                role="system",
                content=(
                    "You are a non-native English speaker learning English. "
                    "You are to respond in basic and sometimes broken English."
                ),
            ),
            ops.OpMessage(
                role="user",
                content=["Write a paragraph to introduce yourself."] * num_generations,
            ),
        ]
    )
    draft_history = ops.llm_chat(draft_instr, return_history=True)

    # 2. Review the draft.
    draft = ops.get_last_message(draft_history)
    review_instr = ops.message_data(
        [
            ops.OpMessage(
                role="system",
                content=(
                    "You are an English teacher. You evaluate the user's writing "
                    "critically and respond with your suggestions for improvement."
                ),
            ),
            ops.OpMessage(role="user", content=draft),
        ]
    )
    review = ops.llm_chat(review_instr, return_history=False)

    # 3. Revise the draft based on the review.
    revise_msg = ops.format_op(
        "Below is the comment on your writing. Please revise your introduction "
        "accordingly and answer me with only that revised version:\n\n{review}",
        review=review,
    )
    revise_instr = ops.append_message(
        draft_history, ops.OpMessage(role="user", content=revise_msg)
    )
    final = ops.llm_chat(revise_instr, return_history=False)

    # 4. Collect outputs.
    draft_out = ops.as_output("draft", draft)
    review_out = ops.as_output("review", review)
    final_out = ops.as_output("final", final)

    return graphs.from_ops([draft_out, review_out, final_out])


def check_outputs(outputs: Any, num_generations: int) -> None:
    assert len(outputs) == 3, "Output length mismatch"
    expected = {
        "draft": ["MOCK"] * num_generations,
        "review": ["MOCK"] * num_generations,
        "final": ["MOCK"] * num_generations,
    }
    assert outputs == expected, f"Expected {expected}, got {outputs}"
