from typing import Any

from helium import graphs, ops


def build_graph() -> graphs.Graph:
    def func(s: tuple[ops.SingleDtype, ...]) -> str:
        assert len(s) == 1
        inp = s[0]
        assert isinstance(inp, str)
        return inp[:-1]

    pred = r"Hell$"

    data_op = ops.data(["Hello, there.", "Dude", "Hiasdfasdfasdfasdfasfddsafsadfasd"])
    lambda_op = ops.lambda_op([data_op], func)
    loop_out_ops = ops.while_loop([data_op], [lambda_op], pred)
    output_op = ops.as_output("output", loop_out_ops[0])

    return graphs.from_ops([output_op])


def check_outputs(outputs: Any) -> None:
    expected = {"output": ["Hell", "", "Hiasdfasdfasdfasdfasfdd"]}
    assert outputs == expected, f"Expected {expected}, got {outputs}"
