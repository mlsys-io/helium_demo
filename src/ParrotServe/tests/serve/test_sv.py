from parrot.serve.graph import (
    RequestChain,
    ConstantFill,
    PlaceholderFill,
    PlaceholderGen,
)
from parrot.serve.graph.call_request import (
    SemanticCallMetadata,
    SemanticFunctionParameter,
)
from parrot.serve.variable_manager import SemanticVariableManager
from parrot.sampling_config import SamplingConfig


def test_content_hash():
    session_id = 0
    sv_content = "test"
    var_mgr = SemanticVariableManager(constant_prefix_var_timeout=10)
    var_mgr.register_local_var_space(session_id)
    var1 = var_mgr._get_local_var_by_content(session_id, sv_content)
    var2 = var_mgr._get_local_var_by_content(session_id, sv_content)
    assert var1 == var2


def test_request_chain_hash():
    var_mgr = SemanticVariableManager(constant_prefix_var_timeout=10)

    request_chain1 = RequestChain.from_nodes(
        nodes=[
            ConstantFill("Test1"),
            PlaceholderFill(
                parameter=SemanticFunctionParameter(
                    name="a", is_output=False, value="test"
                )
            ),
            ConstantFill("Test2"),
            PlaceholderGen(
                parameter=SemanticFunctionParameter(
                    name="b", is_output=True, sampling_config=SamplingConfig()
                )
            ),
        ]
    )
    request_chain2 = RequestChain.from_nodes(
        nodes=[
            ConstantFill("Test1"),
            PlaceholderFill(
                parameter=SemanticFunctionParameter(
                    name="a", is_output=False, value="test"
                )
            ),
            ConstantFill("Test2"),
            PlaceholderGen(
                parameter=SemanticFunctionParameter(
                    name="b", is_output=True, sampling_config=SamplingConfig()
                )
            ),
        ]
    )

    session_id = 0
    var_mgr.register_local_var_space(session_id)
    var_mgr.create_vars_for_semantic_request_chain(session_id, request_chain1)
    var_mgr.create_vars_for_semantic_request_chain(session_id, request_chain2)

    # Check the first chain
    print(request_chain1.pretty_print())

    # Check the second chain
    print(request_chain2.pretty_print())


if __name__ == "__main__":
    # test_content_hash()
    test_request_chain_hash()
