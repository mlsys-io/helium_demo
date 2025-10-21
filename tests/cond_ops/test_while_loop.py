import pytest

from helium import helium, ops


@pytest.mark.timeout(3)
def test_simple_while_loop(mock_helium_server):
    def func(s: tuple[ops.SingleDtype, ...]) -> str:
        assert len(s) == 1
        inp = s[0]
        assert isinstance(inp, str)
        return inp[:-1]

    pred = r"Hell$"

    data_op = ops.data(["Hello, there.", "Dude", "Hiasdfasdfasdfasdfasfddsafsadfasd"])
    lambda_op = ops.lambda_op([data_op], func)
    loop_out_ops = ops.while_loop([data_op], [lambda_op], pred)
    loop_out_op = loop_out_ops[0]

    out = helium.invoke(loop_out_op)
    expected = ["Hell", "", "Hiasdfasdfasdfasdfasfdd"]

    assert out == expected, f"Expected {expected}, got {out}"


@pytest.mark.timeout(3)
def test_simple_while_loop_2(mock_helium_server):
    test_simple_while_loop(mock_helium_server)
