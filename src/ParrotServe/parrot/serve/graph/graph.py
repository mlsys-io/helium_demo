# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


from asyncio import Event
from typing import List, Dict, Set, Optional, Union

from parrot.exceptions import (
    parrot_assert,
    ParrotCoreUserError,
    ParrotCoreInternalError,
)
from parrot.utils import RecyclePool

from .perf_criteria import PerformanceCriteria
from .call_request import (
    TextChunk,
    PlaceholderChunk,
    SemanticFunctionParameter,
    SemanticCallMetadata,
    ChunkedSemanticCallRequest,
)
from .node import (
    BaseNode,
    SemanticNode,
    ConstantFill,
    PlaceholderFill,
    PlaceholderGen,
    NativeFuncNode,
)


"""Data structures for a set of nodes in Graph."""


class _CompletionChainIterator:

    def __init__(self, first_node: SemanticNode) -> None:
        self._cur_node = first_node

    def __iter__(self) -> "_CompletionChainIterator":
        return self

    def __next__(self) -> SemanticNode:
        if self._cur_node is None or (
            self._cur_node.has_edge_a_prev_node
            and self._cur_node.get_edge_a_prev_node().is_gen
        ):
            raise StopIteration
        else:
            ret = self._cur_node
            self._cur_node = self._cur_node.get_edge_a_next_node()
            return ret


class _CompletionChainFillIterator:

    def __init__(self, first_node: SemanticNode) -> None:
        self._cur_node = first_node

    def __iter__(self) -> "_CompletionChainFillIterator":
        return self

    def __next__(self) -> Union[ConstantFill, PlaceholderFill]:
        if self._cur_node.is_gen:
            raise StopIteration
        else:
            ret = self._cur_node
            self._cur_node = self._cur_node.get_edge_a_next_node()
            return ret


class CompChainGroup:
    """A ChainGroup is a set of parallel chains that point to the same consumer."""

    def __init__(self) -> None:
        self.chains: Set[CompletionChain] = set()


class CompletionChain:
    """A CompletionChain is the basic unit of scheduling (a.k.a Task).

    It contains several Fill primitives and one Gen primitive.

    Fill -> Fill -> Fill -> Gen.
    """

    def __init__(
        self,
        request_chain: "RequestChain",
        first_node: SemanticNode,
        gen_node: Optional[PlaceholderGen],
    ) -> None:
        self._request_chain = request_chain

        self.first_node = first_node
        self.gen_node = gen_node

        # Assign completion chain to nodes
        for node in self.iter():
            node.set_comp_chain(self)

        # Groups this chain belongs to.
        self.chain_groups: List[CompChainGroup] = []

    @property
    def request_id(self) -> int:
        return self._request_chain.request_id

    @property
    def session_id(self) -> int:
        return self._request_chain.session_id

    @property
    def sv_created(self) -> bool:
        return self._request_chain.sv_created

    @property
    def metadata(self) -> SemanticCallMetadata:
        return self._request_chain.metadata

    def pretty_print(self) -> str:
        """Pretty print it using Graph's pretty print APIs."""

        ret = "CompletionChain: Nodes: \n"
        for node in self.iter():
            ret += node.pretty_print()

        # ret += "Metadata: \n" + str(self.metadata) + "\n"

        return ret

    def iter(self) -> _CompletionChainIterator:
        return _CompletionChainIterator(self.first_node)

    def iter_fill(self) -> _CompletionChainFillIterator:
        return _CompletionChainFillIterator(self.first_node)


class _RequestChainIterator:

    def __init__(self, first_node: SemanticNode) -> None:
        self._cur_node = first_node

    def __iter__(self) -> "_RequestChainIterator":
        return self

    def __next__(self) -> SemanticNode:
        if self._cur_node is None:
            raise StopIteration
        else:
            ret = self._cur_node
            self._cur_node = self._cur_node.get_edge_a_next_node()
            return ret


class RequestChain:
    """RequestChain is a middle representation of the parsed request, in the form of a chain in
    the graph. It consists a list of Nodes (which is directly compatible in ComputeGraph).

    It's converted from ChunkedRequest (see sv/chunked_request.py).

    It can be inserted into a graph directly.
    """

    def __init__(
        self,
        request_id: int,
        session_id: int,
        first_node: SemanticNode,
        metadata: SemanticCallMetadata,
    ) -> None:
        self.request_id = request_id
        self.session_id = session_id
        self.first_node = first_node
        self.metadata = metadata
        self.comp_chains: List[CompletionChain] = []

        # Only valid after inserted into a graph.
        self._param_info: List[Dict] = []

        # Assign request chain to nodes
        # for node in self.iter():
        #     node.request_chain = self

    @property
    def sv_created(self) -> bool:
        return self.first_node.sv is not None

    @property
    def is_inserted(self) -> bool:
        return self.first_node.is_inserted

    def iter(self) -> _RequestChainIterator:
        return _RequestChainIterator(self.first_node)

    def __repr__(self) -> str:
        return f"RequestChain(first_node={self.first_node})"

    def pretty_print(self) -> str:
        """Pretty print it using Graph's pretty print APIs."""

        ret = "RequestChain: Nodes: \n"
        for node in self.iter():
            ret += node.pretty_print()

        ret += "Metadata: \n" + str(self.metadata) + "\n"

        return ret

    @classmethod
    def from_nodes(
        cls,
        nodes: List[SemanticNode],
        metadata: SemanticCallMetadata = SemanticCallMetadata.get_default(),
    ) -> "RequestChain":
        """Convert a list of nodes into a RequestChain.

        This function is ususally used in tests.
        """

        parrot_assert(
            len(nodes) > 0,
            "RequestChain creation failed: Empty nodes.",
        )

        request_chain = cls(
            request_id=0,  # Dummy value
            session_id=0,  # Dummy value
            first_node=nodes[0],
            metadata=metadata,
        )
        prev_node = nodes[0]
        completion_chain_first_node = nodes[0]

        for node in nodes[1:]:
            # Link edge type A with previous node.
            if prev_node is not None:
                node.link_edge_a_with(prev_node)
            prev_node = node

            # If current node is Gen, create a new CompletionChain.
            if node.is_gen:
                completion_chain = CompletionChain(
                    request_chain=request_chain,
                    first_node=completion_chain_first_node,
                    gen_node=node,
                )
                request_chain.comp_chains.append(completion_chain)
                completion_chain_first_node = node.get_edge_a_next_node()

        return request_chain

    @classmethod
    def from_chunked_request(
        cls, chunked_request: ChunkedSemanticCallRequest
    ) -> "RequestChain":
        """Convert a ChunkedRequest into a RequestChain."""

        prev_node: Optional[SemanticNode] = None

        for i, chunk in enumerate(chunked_request.body):
            is_gen: bool = False

            if isinstance(chunk, TextChunk):
                node = ConstantFill(constant_text=chunk.text)
            elif isinstance(chunk, PlaceholderChunk):
                parameter = chunked_request.parameters_map[chunk.name]
                if parameter.is_output:
                    node = PlaceholderGen(parameter=parameter)
                    is_gen = True
                else:
                    node = PlaceholderFill(parameter=parameter)
            else:
                raise ParrotCoreInternalError(ValueError("Unknown chunk type."))

            # Link edge type A with previous node.
            if prev_node is not None:
                node.link_edge_a_with(prev_node)
            prev_node = node

            # Record first node
            if i == 0:
                request_chain = cls(
                    request_id=chunked_request.request_id,
                    session_id=chunked_request.session_id,
                    first_node=node,
                    metadata=chunked_request.metadata,
                )
                completion_chain_first_node = node

            # If current node is Gen, create a new CompletionChain.
            if is_gen:
                completion_chain = CompletionChain(
                    request_chain=request_chain,
                    first_node=completion_chain_first_node,
                    gen_node=node,
                )
                request_chain.comp_chains.append(completion_chain)
                parrot_assert(
                    node.has_edge_a_prev_node, "Gen node should have a prev node."
                )
                completion_chain_first_node = node.get_edge_a_prev_node()

        return request_chain

    def get_param_info(self) -> List[Dict]:
        """Get the param info after inserted into a graph.

        Returns:
            List[Dict]: param info.
        """

        parrot_assert(
            self.is_inserted,
            "Get param_info failed: RequestChain has not been inserted into a graph.",
        )
        return self._param_info


class ComputeGraph:
    """Computational graph of LLM requests linked by Semantic Variables.

    It's made up of a list of nodes (And edges are maintained by nodes and SVs).

    It has several properties:
    1. It's a DAG (Directed Acyclic Graph) i.e. topologically sorted (if all requests are created valid).
       Thus, we can schedule it in a topological order.
    2. When scheduling, only chains are enterring and leaving the graph.
    3. Every node's in-degree is at most 2 (1 type A edge + 1 type B edge). Out-degree is not limited.
    """

    def __init__(self) -> None:
        self.nodes: Set[BaseNode] = set()
        self.chains: List[CompletionChain] = []

        self._node_id_pool = RecyclePool("Node Pool")

    def _insert_node(self, node: BaseNode) -> None:
        self.nodes.add(node)
        id_in_graph = self._node_id_pool.allocate()
        node.set_id_in_graph(id_in_graph)

        if isinstance(node, SemanticNode):
            # Link edge type B
            if node.is_gen:
                node.sv.assign_producer(node)
            else:
                node.sv.add_consumer(node)
        elif isinstance(node, NativeFuncNode):
            # NativeFuncNode
            for var in node.input_vars.values():
                var.add_consumer(node)

            for var in node.output_vars.values():
                var.assign_producer(node)
        else:
            raise ParrotCoreInternalError(ValueError("Unknown node type."))

    def insert_and_update_request_chain(self, request_chain: RequestChain) -> None:
        """Insert a RequestChain into the graph, and update its info.

        After inserted, "param info" (Parameter Information) can be fetched from this object.
        "param info" records the information of each parameter with its corresponding Semantic Variable.
        """

        parrot_assert(
            request_chain.sv_created,
            "Insert failed: SV should be created before inserting into a graph.",
        )

        parrot_assert(
            not request_chain.is_inserted,
            "Insert failed: RequestChain has been inserted into a graph.",
        )

        for node in request_chain.iter():
            self._insert_node(node)

            parrot_assert(node.sv is not None, "Insert failed: SV is not created.")
            if node.has_placeholder:
                parameter: SemanticFunctionParameter = node.placeholder_param

                # Maintain the param info
                # HACK: Access the private member directly
                request_chain._param_info.append(
                    {
                        "parameter_name": parameter.name,
                        "is_output": parameter.is_output,
                        "var_name": node.sv_name,
                        "var_id": node.var_id,
                    }
                )
        self.chains.extend(request_chain.comp_chains)

    def insert_native_func_node(self, native_func_node: NativeFuncNode) -> None:
        """Insert a NativeFuncNode into the graph."""

        self._insert_node(native_func_node)

        request = native_func_node.native_func

        for key, param in request.parameters_map.items():
            if param.has_value:
                continue

            if param.is_output:
                sv = native_func_node.output_vars[key]
            else:
                sv = native_func_node.input_vars[key]

            # Maintain the param info
            # HACK: Access the private member directly
            native_func_node._param_info.append(
                {
                    "parameter_name": param.name,
                    "is_output": param.is_output,
                    "var_name": sv.name,
                    "var_id": sv.id,
                }
            )

    def remove_completion_chain(self, completion_chain: CompletionChain) -> None:
        """Remove a CompletionChain from the graph. This is called when the task is finished."""

        # Remove chain
        self.chains.remove(completion_chain)
        for node in completion_chain.iter():
            # Remove node
            self.nodes.remove(node)
            self._node_id_pool.free(node.id_in_graph)
