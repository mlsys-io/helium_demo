import pytest

from helium import graphs, helium, ops
from helium.runtime import HeliumServer
from helium.runtime.protocol import (
    HeliumRequestConfig,
    QueryProfilingConfig,
    SystemProfilingConfig,
)


def raise_error(inputs) -> str:
    raise ValueError(f"This is a test error: {inputs[0]}")


@pytest.mark.timeout(3)
@pytest.mark.parametrize("system_profiling_config", [None, SystemProfilingConfig()])
@pytest.mark.parametrize("query_profiling_config", [None, QueryProfilingConfig()])
def test_simple_error(
    mock_helium_server: HeliumServer,
    system_profiling_config: SystemProfilingConfig | None,
    query_profiling_config: QueryProfilingConfig | None,
):
    raw_data = ["mock" for _ in range(100)]
    data_op = ops.data(raw_data)
    lambda_op = ops.lambda_op([data_op], raise_error)
    output_op = ops.as_output("output", lambda_op)
    graph = graphs.from_ops([output_op]).compile()

    request_config = HeliumRequestConfig(
        system_profiling_config=system_profiling_config,
        query_profiling_config=query_profiling_config,
    )
    response = helium.execute(graph, request_config)

    expected_outputs = {"output": None}
    assert (
        response.outputs == expected_outputs
    ), f"Expected {expected_outputs}, got {response.outputs}"

    assert response.error_info is not None, "Expected error info to be present"
    assert len(response.error_info) == 1, "Expected one error info entry"

    got_error_details = response.error_info[0]["details"]
    expected_error_details = "ValueError('This is a test error: mock')"
    assert (
        got_error_details == expected_error_details
    ), f"Expected {expected_error_details}, got {got_error_details}"


@pytest.mark.timeout(3)
@pytest.mark.parametrize("enable_cache_aware_scheduling", [True, False])
@pytest.mark.parametrize("enable_runtime_adjustment", [True, False])
@pytest.mark.parametrize("system_profiling_config", [None, SystemProfilingConfig()])
@pytest.mark.parametrize("query_profiling_config", [None, QueryProfilingConfig()])
def test_error_propagation(
    mock_helium_server: HeliumServer,
    enable_cache_aware_scheduling: bool,
    enable_runtime_adjustment: bool,
    system_profiling_config: SystemProfilingConfig | None,
    query_profiling_config: QueryProfilingConfig | None,
):
    raw_data = ["mock" for _ in range(100)]
    data_op = ops.data(raw_data)
    lambda_op = ops.lambda_op([data_op], raise_error)
    llm_op = ops.llm_completion(lambda_op)
    llm_out_op = ops.as_output("llm", llm_op)
    msg_op = ops.message_data([ops.OpMessage("assistant", lambda_op)])
    msg_out_op = ops.as_output("message", msg_op)
    llm_chat = ops.llm_chat(msg_op)
    llm_chat_out_op = ops.as_output("llm_chat", llm_chat)
    graph = graphs.from_ops([llm_out_op, msg_out_op, llm_chat_out_op]).compile()

    request_config = HeliumRequestConfig(
        enable_cache_aware_scheduling=enable_cache_aware_scheduling,
        enable_runtime_adjustment=enable_runtime_adjustment,
        system_profiling_config=system_profiling_config,
        query_profiling_config=query_profiling_config,
    )
    response = helium.execute(graph, request_config)

    expected_outputs = {"message": None, "llm": None, "llm_chat": None}
    assert (
        response.outputs == expected_outputs
    ), f"Expected {expected_outputs}, got {response.outputs}"

    assert response.error_info is not None, "Expected error info to be present"
    assert len(response.error_info) == 1, "Expected one error info entry"

    got_error_details = response.error_info[0]["details"]
    expected_error_details = "ValueError('This is a test error: mock')"
    assert (
        got_error_details == expected_error_details
    ), f"Expected {expected_error_details}, got {got_error_details}"
