from collections.abc import Callable
from typing import Any, Final, cast

from helium.common import Slice
from helium.runtime.llm import LLMProfilingInfo, LLMServiceInfo
from helium.utils.graph import calculate_node_depths, detect_sccs
from helium.utils.prefix.radix_tree import (
    TemplatedNode,
    TemplatedNodeDependency,
    TemplatedRadixTree,
)


def token_usage(
    alpha: float, profiling_info: LLMProfilingInfo, is_memory_limited: bool
) -> float:
    # TODO: Account for prefix sharing
    prompt_tokens = profiling_info.prompt_tokens_avg
    output_tokens = profiling_info.output_tokens_avg
    if is_memory_limited:
        u = output_tokens * prompt_tokens + (output_tokens * (output_tokens + 1)) / 2
    else:
        u = prompt_tokens + output_tokens
    return alpha * u


def precedence_delay(
    alpha: float, token_budget: int, profiling_info: LLMProfilingInfo
) -> float:
    return alpha * token_budget * profiling_info.output_tokens_avg


def adjust_token_step(
    cur_token_step: float,
    num_released_seqs: int,
    min_output_length: int,
    worker_info: LLMServiceInfo,
) -> float:
    return max(
        cur_token_step,
        (num_released_seqs // worker_info.max_num_reqs)
        * worker_info.alpha
        * worker_info.token_budget
        * min_output_length,
    )


class SchedulingNode:
    def __init__(
        self,
        node: TemplatedNode,
        node_depth_map: dict[str, int],
        scheduling_table: dict[str, float],
        token_step_map: dict[str, float],
        profiling_info_map: dict[str, LLMProfilingInfo],
        worker_info_map: dict[str, LLMServiceInfo],
    ) -> None:
        self.is_root: Final[bool] = node.is_root
        """Whether the node is the root of the scheduling tree."""
        worker = "root" if node.is_root else node.label
        assert isinstance(worker, str)
        self.worker: Final[str] = worker
        """The LLM worker that the node corresponds to."""
        self.has_placeholder: Final[bool] = (not node.is_leaf) and node.has_placeholder
        """Whether the node's prefix has a placeholder."""

        self.node_depth_map: Final[dict[str, int]] = node_depth_map
        """Mapping from node IDs to their depth in the LLM DAG."""
        self.scheduling_table: Final[dict[str, float]] = scheduling_table
        """Mapping from node IDs to the token step at which a dependent node can be scheduled."""
        self.profiling_info_map: Final[dict[str, LLMProfilingInfo]] = profiling_info_map
        """Mapping from node IDs to their profiling information."""
        self.token_step_map: Final[dict[str, float]] = token_step_map
        """
        Mapping from node IDs to the token step at which the node can be scheduled.
        It should only be used by a leaf node.
        """

        self.max_num_reqs: int
        """Maximum number of requests in a batch. Used to adjust the current token step."""
        self.max_num_batched_tokens: int
        """Maximum number of tokens in a batch of the inference engine. Used to
        calculate the cache saturation threshold."""
        self.cache_capacity: int
        """Cache capacity of the LLM worker in tokens."""
        self.alpha: float
        """Normalization constant for calculating the token step."""
        self.token_budget: int
        """Token budget of the LLM worker."""
        self.is_memory_limited: bool
        """Whether the scheduling is memory-limited. Otherwise, it is token budget-limited."""
        if self.is_root:
            self.max_num_reqs = 0
            self.max_num_batched_tokens = 0
            self.cache_capacity = 0
            self.alpha = 0
            self.token_budget = 0
            self.is_memory_limited = False
        else:
            worker_info = worker_info_map[worker]
            self.max_num_reqs = worker_info.max_num_reqs
            self.max_num_batched_tokens = worker_info.max_num_batched_tokens
            self.cache_capacity = worker_info.cache_capacity
            self.alpha = worker_info.alpha
            self.token_budget = worker_info.token_budget
            self.is_memory_limited = worker_info.is_memory_limited

        self._children: list[SchedulingNode] | None = None
        """List of children nodes."""
        self._inner: dict[str, set[str]] | None = None
        """
        Inner data of the node, which is a mapping from node IDs to a set of node 
        IDs that the former depend on.
        """

        self._sibling_dependents_map: dict[str, set[SchedulingNode]] | None = None
        """Mapping from the IDs of the descendant nodes to sibling dependents."""
        self._token_step_to_schedule: float = float("inf")
        """The minimum token step at which a descendant node can be scheduled."""
        self._non_blocking_node_depth: int = -1
        """
        If the node is a leaf, this is the maximum depth of the nodes that can be 
        scheduled without blocking. Otherwise, it should be set to the maximum depth 
        of the children nodes.
        """
        self._schedulable_node_depth: int = -1
        """
        If the node is a leaf, this is the maximum depth of the nodes that can be
        scheduled. Otherwise, it should be set to the maximum depth of the children 
        nodes.
        """

        # Initialize a leaf node
        if node.is_leaf:
            node_depth = -1
            # inner data and token step map
            self._inner = {}
            for node_id, deps in node.get_dependencies().items():
                self._inner[node_id] = {dep.node_id for dep in deps}
                if len(deps) == 0:
                    self.token_step_map[node_id] = 0
                    node_depth = max(node_depth, node_depth_map[node_id])
            # token step to schedule
            if node_depth >= 0:
                self._token_step_to_schedule = 0
            # node depths
            self._non_blocking_node_depth = self._schedulable_node_depth = node_depth

    @property
    def inner(self) -> dict[str, set[str]]:
        """
        Inner data of the node, which is a mapping from node IDs to a set of node
        IDs that the former depend on.

        Raises
        ------
        ValueError: If the node is not a leaf.
        """
        if self._inner is None:
            raise ValueError("Internal node has no inner data")
        return self._inner

    @property
    def children(self) -> list["SchedulingNode"]:
        """
        List of children nodes.

        Raises
        ------
        ValueError: If the children have not been set. They must be set by calling
        `set_children` before accessing this property.
        """
        assert self._children is not None, "Children have not been set"
        return self._children

    @property
    def sibling_dependents_map(self) -> dict[str, set["SchedulingNode"]]:
        """
        Mapping from node IDs to sibling dependents.

        Raises
        ------
        ValueError: If the node is a leaf or the sibling dependents map have not been set.
        They must be set by calling `set_sibling_dependents_map` before accessing this property.
        """
        assert (
            self._sibling_dependents_map is not None
        ), "Sibling dependents map have not been set"
        return self._sibling_dependents_map

    @property
    def sibling_dependents(self) -> set["SchedulingNode"]:
        """
        Sibling dependents of the node.

        Raises
        ------
        ValueError: If the node is a leaf or the sibling dependents map have not been set.
        They must be set by calling `set_sibling_dependents_map` before accessing this property.
        """
        return set(dep for deps in self.sibling_dependents_map.values() for dep in deps)

    @property
    def is_leaf(self) -> bool:
        return self._inner is not None

    @property
    def is_empty(self) -> bool:
        if self._inner is not None:
            # Leaf node
            return len(self._inner) == 0
        # Internal node
        return len(self.children) == 0

    @property
    def is_schedulable(self) -> bool:
        return self._schedulable_node_depth >= 0

    @property
    def is_non_blocking(self) -> bool:
        return self._non_blocking_node_depth >= 0

    def set_children(self, children: list["SchedulingNode"]) -> None:
        if self._children is not None:
            raise ValueError("Children have already been set")
        if self.is_leaf:
            raise ValueError("Leaf node cannot have children")
        self._children = children

    def set_sibling_dependents_map(
        self, dependents_map: dict[str, set["SchedulingNode"]]
    ) -> None:
        if self._sibling_dependents_map is not None:
            raise ValueError("Sibling dependents map have already been set")
        self._sibling_dependents_map = dependents_map

    def get_schedulable_node_ids(self) -> list[str]:
        """Returns the IDs of the nodes that can be scheduled."""
        return [node_id for node_id, deps in self.inner.items() if len(deps) == 0]

    def get_non_blocking_node_ids(
        self, cur_token_step: float, schedulable_node_ids: list[str] | None = None
    ) -> list[str]:
        """Returns the IDs of the nodes that can be scheduled without blocking."""
        if self.is_leaf:
            schedulable_nodes = (
                self.get_schedulable_node_ids()
                if schedulable_node_ids is None
                else schedulable_node_ids
            )
            return [
                node_id
                for node_id in schedulable_nodes
                if self.token_step_map[node_id] <= cur_token_step
            ]
        raise ValueError("Internal nodes do not have node IDs.")

    def schedule(self, cur_token_step: float, force: bool) -> list[str]:
        """Updates the scheduling table and returns the IDs of the nodes that are
        scheduled"""
        inner = self.inner
        to_schedule = (
            self.get_schedulable_node_ids()
            if force
            else self.get_non_blocking_node_ids(cur_token_step)
        )
        for node_id in to_schedule:
            del inner[node_id]
            self.scheduling_table[node_id] = cur_token_step + precedence_delay(
                self.alpha, self.token_budget, self.profiling_info_map[node_id]
            )
        return to_schedule

    def update_subtree(self, cur_token_step: float) -> None:
        """Updates the states of the subtree rooted at this node, given the current
        token step

        This method updates the token step to schedule, schedulable node depth, and
        non-blocking node depth of all nodes in the subtree after some nodes have
        been scheduled.
        """
        if self.is_leaf:
            for node_id, deps in self.inner.items():
                # Update token step to schedule
                token_step_to_schedule = self.token_step_map.get(node_id, -1)
                for dep in [dep for dep in deps if dep in self.scheduling_table]:
                    token_step_to_schedule = max(
                        token_step_to_schedule, self.scheduling_table[dep]
                    )
                    deps.remove(dep)  # Update the dependency
                self.token_step_map[node_id] = token_step_to_schedule

            schedulable_node_ids = self.get_schedulable_node_ids()
            if schedulable_node_ids:
                # There are schedulable nodes
                # Update the token step to schedule
                self._token_step_to_schedule = min(
                    [self.token_step_map[node_id] for node_id in schedulable_node_ids]
                )
                # Update the schedulable node depth
                self._schedulable_node_depth = max(
                    [self.node_depth_map[node_id] for node_id in schedulable_node_ids]
                )
                # Update the non-blocking node depth
                non_blocking_node_ids = self.get_non_blocking_node_ids(
                    cur_token_step, schedulable_node_ids
                )
                self._non_blocking_node_depth = (
                    max(
                        [
                            self.node_depth_map[node_id]
                            for node_id in non_blocking_node_ids
                        ]
                    )
                    if non_blocking_node_ids
                    else -1
                )
            else:
                # No schedulable nodes
                self._token_step_to_schedule = float("inf")
                self._schedulable_node_depth = -1
                self._non_blocking_node_depth = -1
        else:
            for child in self.children:
                child.update_subtree(cur_token_step)
            self._token_step_to_schedule = min(
                [child._token_step_to_schedule for child in self.children]
            )
            self._schedulable_node_depth = max(
                [child._schedulable_node_depth for child in self.children]
            )
            self._non_blocking_node_depth = max(
                [child._non_blocking_node_depth for child in self.children]
            )

    def sort_key_unforced(self, subtree_depth: int) -> Any:
        return (
            -self._non_blocking_node_depth,
            self._token_step_to_schedule,
            subtree_depth,
        )

    def sort_key_forced(self, subtree_depth: int) -> Any:
        return (
            self._token_step_to_schedule,
            -self._schedulable_node_depth,
            subtree_depth,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(inner={self._inner})"


class SchedulingTree:
    def __init__(
        self,
        radix_tree: TemplatedRadixTree,
        input_slice_map: dict[str, Slice],
        profiling_info_map: dict[str, LLMProfilingInfo],
        worker_info_map: dict[str, LLMServiceInfo],
    ):
        """
        Parameters
        ----------
        radix_tree: TemplatedRadixTree
            The radix tree to build the scheduling tree from.
        input_slice_map: dict[str, Slice]
            A mapping from node IDs to their input slices.
        profiling_info_map: dict[str, ProfilingInfo]
            A mapping from node IDs to their profiling information.
        worker_info_map: dict[str, LLMServiceInfo]
            A mapping from worker names to their LLM service information.
        """
        self.input_slice_map: Final[dict[str, Slice]] = input_slice_map
        self.workers: Final[set[str]] = cast(
            set[str], set(leaf.label for leaf in radix_tree.leaves())
        )
        self._node_dependencies: Final[dict[str, set[str]]] = {
            node_id: {dep.node_id for dep in deps}
            for leaf in radix_tree.leaves()
            for node_id, deps in leaf.get_dependencies().items()
        }
        self._node_depth_map: Final[dict[str, int]] = calculate_node_depths(
            self._node_dependencies
        )
        self._scheduling_table: Final[dict[str, float]] = {}
        self._token_step_map: Final[dict[str, float]] = {}
        self._profiling_info_map: Final[dict[str, LLMProfilingInfo]] = (
            profiling_info_map
        )
        self._worker_info_map: Final[dict[str, LLMServiceInfo]] = worker_info_map

        self._buffer: dict[str, list[list[str]]] = {
            worker: [] for worker in self.workers
        }
        self._worker_to_release: str | None = None
        self._to_release: list[str] = []

        self._num_seqs_to_release: int = 0
        self._token_steps_to_release: float = 0

        self._released_token_step: float = 0
        self._emitted_token_step: float = 0

        self._root = self._build(radix_tree)

    def schedule(self) -> dict[str, list[list[str]]]:
        freeable = self._schedule(self._root, batch_wise=False, force=False)
        # Repeat until all nodes are scheduled
        while not freeable:
            freeable = self._schedule(self._root, batch_wise=False, force=True)
        return self._get_and_reset_buffer()

    def _get_and_reset_buffer(self) -> dict[str, list[list[str]]]:
        buffer = self._buffer
        self._buffer = {worker: [] for worker in self.workers}
        return buffer

    def _build(self, radix_tree: TemplatedRadixTree) -> SchedulingNode:
        # Build the sibling dependencies (node -> dependencies)
        sibling_dependents_map: dict[TemplatedNode, dict[str, set[TemplatedNode]]] = {}
        _, root_deps = self._build_sibling_dependents_map(
            radix_tree.root, sibling_dependents_map
        )
        assert len(root_deps) == 0, "Root node should not have external dependencies"

        # Initialize all scheduling nodes
        scheduling_node_map: dict[TemplatedNode, SchedulingNode] = {
            node: SchedulingNode(
                node,
                self._node_depth_map,
                self._scheduling_table,
                self._token_step_map,
                self._profiling_info_map,
                self._worker_info_map,
            )
            for node in radix_tree.dfs()
        }

        # Build the scheduling tree
        radix_tree_root = radix_tree.root
        stack = [radix_tree_root]
        while stack:
            cur_templated_node = stack.pop()
            cur_scheduling_node = scheduling_node_map[cur_templated_node]
            if not cur_templated_node.is_root:
                cur_scheduling_node.set_sibling_dependents_map(
                    {
                        node_id: {scheduling_node_map[dep] for dep in deps}
                        for node_id, deps in sibling_dependents_map[
                            cur_templated_node
                        ].items()
                    }
                )
            if not cur_templated_node.is_leaf:
                children = cur_templated_node.get_children()
                cur_scheduling_node.set_children(
                    [scheduling_node_map[child] for child in children]
                )
                stack.extend(children)

        # Update the tree's scheduling states
        scheduling_tree_root = scheduling_node_map[radix_tree_root]
        scheduling_tree_root.update_subtree(0)
        return scheduling_tree_root

    def _build_sibling_dependents_map(
        self,
        node: TemplatedNode,
        sibling_dependents_map: dict[TemplatedNode, dict[str, set[TemplatedNode]]],
    ) -> tuple[set[TemplatedNode], dict[str, set[TemplatedNodeDependency]]]:
        """Recursively builds the sibling dependents of the node and its descendants

        Returns
        -------
        set[TemplatedNode]:
            Set of all the descendants of the node
        dict[str, set[TemplatedNodeDependency]]:
            Mapping from descendant node IDs to a set of external dependencies
        """
        # Set of all the descendants of the node
        children_descs: set[TemplatedNode]
        # node_id -> external dependencies
        children_deps: dict[str, set[TemplatedNodeDependency]]
        if node.is_leaf:
            # 1. Base case
            children_descs = {node}
            children_deps = node.get_dependencies()
            return children_descs, children_deps

        # Set of child nodes
        children = node.get_children()
        # node -> external dependencies
        children_deps_map: dict[
            TemplatedNode, dict[str, set[TemplatedNodeDependency]]
        ] = {}
        # node -> ancestor
        ancestor_map: dict[TemplatedNode, TemplatedNode] = {}
        children_descs = set()
        children_deps = {}

        # 2. Collect the external dependencies and descendants of the children
        for child in children:
            sibling_dependents_map[child] = {}
            child_descs, child_deps = self._build_sibling_dependents_map(
                child, sibling_dependents_map
            )
            children_deps_map[child] = child_deps
            children_descs.update(child_descs)
            for desc in child_descs:
                ancestor_map[desc] = child

        # 3. Build sibling dependents
        for child, child_deps in children_deps_map.items():
            for node_id, node_deps in child_deps.items():
                for node_dep in node_deps:
                    sibling = ancestor_map.get(node_dep.node)
                    if sibling is None:
                        # External dependency
                        if node_id in children_deps:
                            children_deps[node_id].add(node_dep)
                        else:
                            children_deps[node_id] = {node_dep}
                    else:
                        # Sibling dependency
                        dep_id = node_dep.node_id
                        sibling_dependents = sibling_dependents_map[sibling]
                        if dep_id in sibling_dependents:
                            sibling_dependents[dep_id].add(child)
                        else:
                            sibling_dependents[dep_id] = {child}

        # 4. Propagate the external dependencies to the parent
        return children_descs, children_deps

    def _get_cur_token_step(self) -> float:
        return self._released_token_step + self._emitted_token_step

    def _get_subtree_sort_key(
        self, subtree_depths: dict[SchedulingNode, int], force: bool
    ) -> Callable[[SchedulingNode], Any]:
        def _sort_key_forced(node: SchedulingNode) -> Any:
            return node.sort_key_forced(subtree_depths[node])

        def _sort_key_unforced(node: SchedulingNode) -> Any:
            return node.sort_key_unforced(subtree_depths[node])

        return _sort_key_forced if force else _sort_key_unforced

    def _assign_subtree_depths(
        self, nodes: list[SchedulingNode]
    ) -> dict[SchedulingNode, int]:
        # Detect SCCs in the sibling dependency graph
        sibling_dependents = {node: node.sibling_dependents for node in nodes}
        sccs = detect_sccs(sibling_dependents)

        # Build SCCs' external dependents
        node_scc_indices = {n: i for i, scc in enumerate(sccs) for n in scc}
        sccs_dependents: dict[int, set[int]] = {}
        for i, scc in enumerate(sccs):
            scc_dependents = set()
            for n in scc:
                for dep in sibling_dependents.get(n, ()):
                    scc_index = node_scc_indices[dep]
                    if scc_index != i:
                        scc_dependents.add(scc_index)
            sccs_dependents[i] = scc_dependents

        # Assign subtree depths based on the SCCs
        scc_depths = calculate_node_depths(sccs_dependents)
        subtree_depths = {
            node: depth for scc_i, depth in scc_depths.items() for node in sccs[scc_i]
        }

        return subtree_depths

    def _schedule(self, node: SchedulingNode, batch_wise: bool, force: bool) -> bool:
        """Schedules the subtree rooted at the given node

        Returns
        -------
        bool
            True if the subtree was completely scheduled and can be removed. False,
            otherwise.
        """
        if node.is_leaf:
            to_release = False
            cur_token_step = self._get_cur_token_step()
            to_schedule = node.schedule(cur_token_step, force)
            self._emit(node, to_schedule, force)
        else:
            to_release = not batch_wise
            batch_wise = batch_wise or node.has_placeholder
            children = node.children
            scheduled: bool = False
            nodes_to_schedule: list[SchedulingNode]
            while children:
                # Get non-blocking or schedulable nodes
                if force:
                    nodes_to_schedule = [
                        child for child in children if child.is_schedulable
                    ]
                else:
                    nodes_to_schedule = [
                        child for child in children if child.is_non_blocking
                    ]
                # Stop if there are no nodes to schedule
                if not nodes_to_schedule:
                    assert scheduled, "No progress made in scheduling"
                    break
                # Assign subtree depths for sorting
                subtree_depths = self._assign_subtree_depths(nodes_to_schedule)
                # Schedule the minimal subtree
                sort_key = self._get_subtree_sort_key(subtree_depths, force)
                node_to_schedule = min(nodes_to_schedule, key=sort_key)
                freeable = self._schedule(node_to_schedule, batch_wise, force)
                # Remove the subtree if it is freeable
                if freeable:
                    children.remove(node_to_schedule)
                # Update the remaining child subtrees
                cur_token_step = self._get_cur_token_step()
                for subtree in children:
                    subtree.update_subtree(cur_token_step)
                # Update the dependents map
                node_to_schedule._sibling_dependents_map = {
                    node_id: deps
                    for node_id, deps in node_to_schedule.sibling_dependents_map.items()
                    if node_id not in self._scheduling_table
                }
                # Force schedule only once
                force = False
                # Track scheduling progress
                scheduled = True
            # Update the remaining children
            node._children = children

        if to_release:
            # Release if scheduling subtree-wise
            self._release()

        return node.is_empty  # The subtree is freeable

    def _emit(self, node: SchedulingNode, node_ids: list[str], force: bool) -> None:
        """Emits the node IDs that can be scheduled to the release list"""
        if force:
            assert self._emitted_token_step == 0, "Only force schedule once"
            token_step_map = self._token_step_map
            self._released_token_step = max(
                token_step_map[node_id] for node_id in node_ids
            )
        alpha = node.alpha
        profiling_info_map = self._profiling_info_map
        for node_id in node_ids:
            # Check the worker
            if self._worker_to_release is None:
                self._worker_to_release = node.worker
            elif self._worker_to_release != node.worker:
                raise ValueError("Cannot emit nodes from different workers")
            self._to_release.append(node_id)
            node_token_usage = token_usage(
                alpha, profiling_info_map[node_id], node.is_memory_limited
            )
            slice_size = self.input_slice_map[node_id].length
            self._num_seqs_to_release += slice_size
            self._token_steps_to_release += node_token_usage * slice_size
            self._emitted_token_step += node_token_usage

    def _release(self) -> None:
        """Releases the node IDs that can be scheduled and independent node IDs"""
        if self._to_release:
            to_release = self._to_release
            self._to_release = []
            assert self._worker_to_release is not None
            # Release to the buffer
            worker = self._worker_to_release
            self._buffer[worker].append(to_release)
            # Update the token step
            worker_info = self._worker_info_map[worker]
            min_output_length = int(
                min(
                    self._profiling_info_map[node_id].output_tokens_avg
                    for node_id in to_release
                )
            )
            advance = self._token_steps_to_release
            self._released_token_step = adjust_token_step(
                self._released_token_step + advance,
                self._num_seqs_to_release,
                min_output_length,
                worker_info,
            )
            # Reset states
            self._num_seqs_to_release = 0
            self._token_steps_to_release = 0
            self._emitted_token_step = 0
            self._worker_to_release = None


def cache_aware_schedule(
    radix_tree: TemplatedRadixTree,
    input_slice_map: dict[str, Slice],
    profiling_info_map: dict[str, LLMProfilingInfo],
    worker_info_map: dict[str, LLMServiceInfo],
) -> dict[str, list[list[str]]]:
    scheduling_tree = SchedulingTree(
        radix_tree,
        input_slice_map,
        profiling_info_map,
        worker_info_map,
    )
    schedule = scheduling_tree.schedule()
    return schedule
