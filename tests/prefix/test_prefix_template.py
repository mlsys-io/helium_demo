from helium import graphs, ops
from helium.frontend.programs.examples import MCQAProgram
from helium.utils.prefix.radix_tree import (
    MessageDelimiter,
    Placeholder,
    PrefixType,
    prefix_message,
)


def check_prefix(prefix: PrefixType, expected: PrefixType):
    for item, expected_item in zip(prefix, expected):
        if isinstance(expected_item, (str, MessageDelimiter)):
            assert item == expected_item, f"Expected {expected_item} but got {item}"
        else:
            assert isinstance(item, Placeholder), f"Expected Placeholder but got {item}"
            assert item.op_id == expected_item.op_id


def assign_workers(graph: graphs.Graph) -> dict[str, str | None]:
    return {op.id: "worker" for op in graph.iter_ops()}


def test_format_op():
    user = ops.DataOp(["world"])
    name = ops.DataOp(["Elsa"])
    op = ops.format_op("Hello, {user}! My name is {name}.", user=user, name=name)
    op = ops.llm_completion(op)
    op = ops.as_output("out", op)
    graph = graphs.from_ops([op])
    tree = graph.build_radix_tree(assign_workers(graph))
    leaves = tree.leaves()

    assert len(leaves) == 1, "Expected 1 leaf node"

    prefix = leaves.pop().get_prefix()
    expected_prefix = (
        "Hello, ",
        Placeholder(user.id),
        "! My name is ",
        Placeholder(name.id),
        ".",
    )

    check_prefix(prefix, expected_prefix)


def test_mcqa_program():
    program = MCQAProgram(num_agents=5)
    agent = program.create_program_agent("user_inputs")
    tree = agent.graph.build_radix_tree(assign_workers(agent.graph))
    leaves = tree.leaves()

    assert len(leaves) == 1, "Expected 1 leaf node"

    user_message = next(iter(op.id for op in agent.graph.iter_ops(ops.InputOp)))
    prefix = leaves.pop().get_prefix()
    expected_prefix = (
        *prefix_message("system", (program.system_prompt,)),
        *prefix_message("user", (Placeholder(user_message),)),
    )

    check_prefix(prefix, expected_prefix)
