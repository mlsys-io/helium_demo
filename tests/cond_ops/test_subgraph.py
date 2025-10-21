import pytest

from helium.ops import FormatOp, InputOp, Op, OutputOp
from helium.ops.cond_ops import _SubGraph as SubGraph


@pytest.fixture
def setup_ops() -> tuple:
    # Create mock operations
    input_op1 = InputOp(name="input1")
    input_op2 = InputOp(name="input2")
    output_op1 = OutputOp(name="output1", output=input_op1)
    output_op2 = OutputOp(name="output2", output=input_op2)
    merge_op = FormatOp("", input_op1=input_op1, input_op2=input_op2)

    # Mock graph and dependencies
    mock_graph = {
        input_op1.id: input_op1,
        input_op2.id: input_op2,
        output_op1.id: output_op1,
        output_op2.id: output_op2,
    }
    mock_dependencies: dict[Op, set[Op]] = {
        input_op1: {output_op1},
        input_op2: {output_op2},
    }

    return (
        input_op1,
        input_op2,
        output_op1,
        output_op2,
        merge_op,
        mock_graph,
        mock_dependencies,
    )


def test_initialization(setup_ops):
    input_op1, input_op2, output_op1, output_op2, *_ = setup_ops
    subgraph = SubGraph(
        input_ops=[input_op1, input_op2],
        output_ops=[output_op1, output_op2],
    )
    assert subgraph.input_ops == [input_op1, input_op2]
    assert subgraph.output_ops == [output_op1, output_op2]
    assert isinstance(subgraph.graph, dict)
    assert isinstance(subgraph.dependencies, dict)
    assert isinstance(subgraph.external_dependencies, dict)


def test_traverse_graph(setup_ops):
    input_op1, _, output_op1, *_ = setup_ops
    subgraph = SubGraph(input_ops=[input_op1], output_ops=[output_op1])
    graph = {}
    dependencies = {}
    subgraph._traverse_graph(output_op1, {input_op1}, graph, dependencies)

    assert output_op1.id in graph
    assert input_op1.id in graph
    assert input_op1 in dependencies
    assert output_op1 in dependencies[input_op1]


def test_prune_subgraph(setup_ops):
    input_op1, input_op2, output_op1, output_op2, _, mock_graph, mock_dependencies = (
        setup_ops
    )
    subgraph = SubGraph(input_ops=[input_op1], output_ops=[output_op1])
    subgraph.graph = mock_graph
    subgraph.dependencies = mock_dependencies
    subgraph.external_dependencies = {}

    subgraph._prune_subgraph(
        [input_op1],
        subgraph.graph,
        subgraph.dependencies,
        subgraph.external_dependencies,
    )
    assert input_op1.id in subgraph.graph
    assert output_op1.id in subgraph.graph
    assert input_op2.id not in subgraph.graph
    assert output_op2.id not in subgraph.graph


def test_build_subgraph(setup_ops):
    input_op1, _, output_op1, *_ = setup_ops
    subgraph = SubGraph(input_ops=[input_op1], output_ops=[output_op1])

    assert input_op1.id in subgraph.graph
    assert output_op1.id in subgraph.graph
    assert input_op1 in subgraph.dependencies
    assert output_op1 in subgraph.dependencies[input_op1]
    assert subgraph.external_dependencies == {}


def test_build_subgraph_with_merge_op(setup_ops):
    input_op1, input_op2, output_op1, output_op2, merge_op, *_ = setup_ops
    subgraph = SubGraph(input_ops=[input_op1], output_ops=[output_op1, merge_op])

    assert input_op1.id in subgraph.graph
    assert output_op1.id in subgraph.graph
    assert merge_op.id in subgraph.graph
    assert input_op2.id in subgraph.graph
    assert output_op2.id not in subgraph.graph

    # Check external dependencies for merge_op
    assert input_op2 in subgraph.external_dependencies[merge_op]
