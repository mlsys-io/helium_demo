import enum
from collections.abc import Iterable
from typing import Hashable, cast

from helium.common import Message
from helium.utils.graph import reversed_topological_sort


class Placeholder:
    def __init__(self, op_id: str) -> None:
        self.op_id = op_id

    def __hash__(self) -> int:
        return hash(self.op_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Placeholder):
            return False
        return self.op_id == other.op_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(op_id={self.op_id})"


class MessageDelimiter(enum.Enum):
    ROLE = enum.auto()
    CONTENT = enum.auto()
    END = enum.auto()


TextKeyItem = str | Placeholder
MessageKeyItem = str | Placeholder | MessageDelimiter
KeyItem = TextKeyItem | MessageKeyItem
TextPrefixType = tuple[TextKeyItem, ...]
MessagePrefixType = tuple[MessageKeyItem, ...]
PrefixType = TextPrefixType | MessagePrefixType
StaticKeyItem = str | MessageDelimiter
StaticPrefixType = tuple[StaticKeyItem, ...]


def prefix_message(role: str, content: "TextPrefixType") -> MessagePrefixType:
    """Creates a PrefixMessage with the given role and content."""
    return (
        MessageDelimiter.ROLE,
        role,
        MessageDelimiter.CONTENT,
        *content,
        MessageDelimiter.END,
    )


def to_text_prefix(prefix: tuple[KeyItem, ...]) -> TextPrefixType:
    for item in prefix:
        if isinstance(item, MessageDelimiter):
            raise ValueError("Prefix contains message key items")
    return cast(TextPrefixType, prefix)


def to_message_prefix(prefix: tuple[KeyItem, ...]) -> MessagePrefixType:
    return cast(MessagePrefixType, prefix)


def to_raw_prefix(prefix: StaticPrefixType) -> str | list[Message]:
    if any(isinstance(item, MessageDelimiter) for item in prefix):
        return prefix_to_messages(prefix)
    return "".join(prefix)  # type: ignore


def prefix_to_messages(prefix: StaticPrefixType) -> list[Message]:
    messages: list[Message] = []
    i = 0
    while i < len(prefix):
        if prefix[i] != MessageDelimiter.ROLE:
            raise ValueError("Prefix does not start with role delimiter")
        if i + 1 >= len(prefix):
            raise ValueError("Prefix ends unexpectedly after role delimiter")
        role = prefix[i + 1]
        if not isinstance(role, str):
            raise ValueError("Role must be a string")
        i += 2
        if i >= len(prefix) or prefix[i] != MessageDelimiter.CONTENT:
            raise ValueError("Prefix does not contain content delimiter")
        i += 1
        content_parts: list[str] = []
        while i < len(prefix) and prefix[i] != MessageDelimiter.ROLE:
            part = prefix[i]
            if part == MessageDelimiter.END:
                i += 1
                break
            assert isinstance(part, str)
            content_parts.append(part)
            i += 1
        content = "".join(content_parts)
        messages.append(Message(role=role, content=content))
    return messages


def check_prefix_type_consistency(prefix: tuple[KeyItem, ...]) -> PrefixType:
    try:
        return to_text_prefix(prefix)
    except ValueError:
        pass
    return to_message_prefix(prefix)


def compress_prefix(prefix: PrefixType) -> PrefixType:
    compressed: tuple[KeyItem, ...] = ()

    prev_item = None
    for item in prefix:
        if isinstance(item, str):
            if isinstance(prev_item, str):
                prev_item += item
            else:
                prev_item = item
        else:
            compressed += (item,)
            prev_item = None

    return check_prefix_type_consistency(compressed)


def copy_text_prefix(prefix: TextPrefixType) -> TextPrefixType:
    prefix_list: list[TextKeyItem] = []
    for part in prefix:
        if isinstance(part, str):
            prefix_list.append(part)
        elif isinstance(part, Placeholder):
            prefix_list.append(Placeholder(part.op_id))
        else:
            raise TypeError(f"Unknown key item type: {type(part)}")
    return tuple(prefix_list)


def copy_prefix(prefix: PrefixType) -> PrefixType:
    prefix_list: list[KeyItem] = []
    for part in prefix:
        if isinstance(part, (str, MessageDelimiter)):
            prefix_list.append(part)
        else:
            prefix_list.append(Placeholder(part.op_id))
    return cast(PrefixType, tuple(prefix_list))


class TemplatedNodeDependency:
    __slots__ = "node_id", "node"

    def __init__(self, node_id: str, node: "TemplatedNode"):
        if not node.is_leaf:
            raise ValueError("Node must be a leaf node")
        self.node_id = node_id
        self.node = node

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(node_id={self.node_id})"


class TemplatedNode:
    __slots__ = "parent", "_children", "_key", "_inner", "_label"

    def __init__(
        self,
        parent: "TemplatedNode | None",
        key: PrefixType | None,
        label: Hashable | None = None,
    ):
        """
        Parameters
        ----------
        parent : TemplatedNode | None
            Parent node. If None, this node is a root node.
        key : PrefixType | None
            Key of the node. If None, this node is a leaf node.
        label : Hashable | None
            Label of the node for additional equality check.
        """
        self.parent = parent
        if parent is None and key is None:
            assert label is None, "Root node cannot have a label"
            key = ()
            self._label = None
        else:
            self._label = label

        self._children: dict[Hashable | None, "TemplatedNode"] | None
        self._key: PrefixType | None
        self._inner: dict[str, set[TemplatedNodeDependency]] | None
        if key is None:
            self._children = None
            self._key = None
            self._inner = {}
        else:
            self._children = {}
            self._key = key
            self._inner = None

    @property
    def key(self) -> PrefixType:
        """Key of the node."""
        if self._key is None:
            raise ValueError("Leaf node has no key")
        return self._key

    @property
    def children(self) -> dict[Hashable | None, "TemplatedNode"]:
        """Children of the node."""
        if self._children is None:
            raise ValueError("Leaf node has no children")
        return self._children

    @property
    def label(self) -> Hashable | None:
        return self._label

    @property
    def is_leaf(self) -> bool:
        return self._children is None

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def value_count(self) -> int:
        if self._inner is None:
            raise ValueError("Internal node has no value count")
        return len(self._inner)

    @property
    def has_placeholder(self) -> bool:
        if self._key is None:
            raise ValueError("Leaf node has no key")
        return any(isinstance(item, Placeholder) for item in self._key)

    def get_placeholders(self) -> list[Placeholder]:
        if self._key is None:
            raise ValueError("Leaf node has no key")
        placeholders: list[Placeholder] = [
            item for item in self._key if isinstance(item, Placeholder)
        ]
        return placeholders

    def get_children(self) -> set["TemplatedNode"]:
        return set(self.children.values())

    def get_dependencies(self) -> dict[str, set[TemplatedNodeDependency]]:
        if self._inner is None:
            raise ValueError("Internal node has no dependencies")
        return self._inner.copy()

    def get_node_ids(self) -> Iterable[str]:
        if self._inner is None:
            raise ValueError("Internal node has no node IDs")
        return self._inner.keys()

    def get_prefix(self) -> PrefixType:
        parent = self.parent
        key: tuple[KeyItem, ...] = () if self._key is None else self._key

        while parent is not None and parent.key != ():
            key = parent.key + key
            parent = parent.parent

        return check_prefix_type_consistency(key)

    @staticmethod
    def get_key_id(key: PrefixType | None, label: Hashable | None) -> Hashable | None:
        """Gets the id of the key

        Key id is the shorthand for the key.

        Parameters
        ----------
        key : PrefixType
            Key to get the id of.

        Returns
        -------
        KeyItem
            Id of the key.
        """
        if key is None:
            return None  # Should be the key of a leaf node

        if len(key) == 0:
            raise ValueError("Key is empty")

        k = key[0]
        if isinstance(k, (Placeholder, MessageDelimiter)):
            return k, label
        if len(k) == 0:
            raise ValueError("Key is empty")
        return k[0], label

    def get_leaf(self) -> "TemplatedNode | None":
        return self.children.get(None)

    def split_to(self, new_node: "TemplatedNode", key: PrefixType) -> None:
        if self._key is None:
            raise ValueError("Leaf node cannot be split")
        self._key = key
        self.parent = new_node

    def add_child(self, child: "TemplatedNode") -> None:
        key_id = self.get_key_id(child._key, child._label)
        if key_id in self.children:
            raise ValueError("Child already exists")
        self.children[key_id] = child

    def change_child(self, child: "TemplatedNode") -> None:
        key_id = self.get_key_id(child._key, child._label)
        if key_id not in self.children:
            raise ValueError("Child does not exist")
        self.children[key_id] = child

    def remove_child(self, child: "TemplatedNode") -> None:
        key_id = self.get_key_id(child._key, child._label)
        if key_id not in self.children:
            raise ValueError("Child does not exist")
        self.children.pop(key_id)

    def remove(self, node_id: str) -> None:
        if self._inner is None:
            raise ValueError("Internal node has no dependencies")
        if node_id not in self._inner:
            raise KeyError("Node ID does not exist")
        self._inner.pop(node_id)

    def add(
        self,
        node_id: str,
        dependencies: set["TemplatedNodeDependency"] | None = None,
    ) -> None:
        if self._inner is None:
            raise ValueError("Internal node cannot have dependencies")
        if node_id in self._inner:
            raise KeyError("Node ID already exists")
        self._inner[node_id] = set() if dependencies is None else dependencies

    def clear(self) -> None:
        if self._inner is None:
            raise ValueError("Internal node cannot be cleared")
        self._inner.clear()

    def is_removable(self) -> bool:
        if self._inner is None:
            return len(self.children) == 0  # Internal node
        return len(self._inner) == 0  # Leaf node

    def dfs(self) -> Iterable["TemplatedNode"]:
        """Depth-first search traversal of the tree starting from this node."""
        stack: list["TemplatedNode"] = [self]
        while stack:
            node = stack.pop()
            yield node
            if not node.is_leaf:
                stack.extend(node.children.values())

    def leaves(self) -> list["TemplatedNode"]:
        return [node for node in self.dfs() if node.is_leaf]

    def split_placeholder(self) -> "TemplatedNode":
        """Splits the node at the first placeholder, creating a new node as the parent
        of the current node
        """
        if self._key is None:
            raise ValueError("Leaf node cannot be split")

        # Find the last placeholder
        first_placeholder_index = -1
        for i, item in enumerate(self._key):
            if isinstance(item, Placeholder):
                first_placeholder_index = i
                break

        if first_placeholder_index == -1:
            raise ValueError("No placeholder found in the key")

        if first_placeholder_index == 0:
            # The node starts with a placeholder, do nothing.
            assert self.parent is not None
            return self.parent

        # Create a new node with the key up to the last placeholder
        new_key = self._key[:first_placeholder_index]
        new_node = TemplatedNode(self.parent, new_key, self._label)
        assert self.parent is not None
        self.parent.remove_child(self)
        self.parent.add_child(new_node)
        self.split_to(new_node, self._key[first_placeholder_index:])
        new_node.add_child(self)

        return new_node

    def copy_subtree(
        self,
    ) -> tuple["TemplatedNode", dict["TemplatedNode", "TemplatedNode"]]:
        """Creates a copy of the subtree rooted at this node

        Note that this method copies the node IDs of the original nodes but ignores
        the data dependencies.

        Returns
        -------
        TemplatedNode
            New node with the same key and children as this node.
        dict[TemplatedNode, TemplatedNode]
            Mapping of the original nodes to the new nodes in the copied subtree.
        """
        node_map: dict[TemplatedNode, TemplatedNode] = {}
        new_node = TemplatedNode(
            self.parent,
            None if self._key is None else copy_prefix(self._key),
            self._label,
        )
        self._copy_subtree_to(new_node, node_map)
        return new_node, node_map

    def _copy_subtree_to(
        self,
        new_node: "TemplatedNode",
        node_map: dict["TemplatedNode", "TemplatedNode"],
    ) -> None:
        node_map[self] = new_node
        if new_node.is_leaf:
            new_node._inner = {node_id: set() for node_id in self.get_node_ids()}
        else:
            for child in self.get_children():
                new_child = TemplatedNode(
                    new_node,
                    None if child._key is None else copy_prefix(child._key),
                    child._label,
                )
                new_node.add_child(new_child)
                child._copy_subtree_to(new_child, node_map)

    def __repr__(self) -> str:
        return f"Node(key={self._key}, label={self._label}, inner={self._inner})"


class TemplatedRadixTree:
    def __init__(self) -> None:
        self.root = TemplatedNode(None, (), None)
        self.node_map: dict[str, TemplatedNode] = {}

    def refresh(self) -> None:
        self.node_map = {
            node_id: leaf
            for leaf in self.leaves_dfs()
            for node_id in leaf.get_node_ids()
        }

    def add(
        self,
        key: PrefixType,
        label: Hashable | None,
        node_id: str,
        dependencies: set[TemplatedNodeDependency] | None = None,
    ) -> TemplatedNodeDependency:
        if self._is_key_empty(key):
            raise ValueError("Key is empty")
        if node_id in self.node_map:
            raise KeyError("Node ID already exists")

        dep = self._add_to_node(self.root, key, label, node_id, dependencies)
        self.node_map[node_id] = dep.node
        return dep

    def remove(self, node_id: str) -> None:
        if node_id not in self.node_map:
            raise KeyError("Node ID does not exist")

        node = self.node_map.pop(node_id)
        key = node.get_prefix()
        self._remove_from_node(self.root, key, node.label, node_id)

    def merge(self, other: "TemplatedRadixTree") -> None:
        templated_node_deps: dict[str, TemplatedNodeDependency] = {}
        node_id_deps: dict[str, Iterable[str]] = {
            node_id: {dep.node_id for dep in node_deps}
            for leaf in other.leaves()
            for node_id, node_deps in leaf.get_dependencies().items()
        }
        # Sort the leaves by topological order to ensure that dependencies are
        # processed in the correct order
        sorted_node_ids = reversed_topological_sort(node_id_deps)
        for node_id in sorted_node_ids:
            node = other.node_map[node_id]
            deps = node_id_deps[node_id]
            templated_node_dep = self.add(
                node.get_prefix(),
                node.label,
                node_id,
                {templated_node_deps[dep] for dep in deps},
            )
            templated_node_deps[node_id] = templated_node_dep

    @staticmethod
    def _is_key_empty(key: PrefixType) -> bool:
        return len(key) == 0

    def _add_to_node(
        self,
        node: TemplatedNode,
        key: PrefixType,
        label: Hashable | None,
        node_id: str,
        dependencies: set[TemplatedNodeDependency] | None = None,
    ) -> TemplatedNodeDependency:
        cur_node = node
        cur_key = key

        while True:
            assert not self._is_key_empty(key)

            # Assume that key does not overlap with node.key
            key_id = TemplatedNode.get_key_id(cur_key, label)

            if key_id not in cur_node.children:
                # No child with the given key. Create a new node.
                new_node = TemplatedNode(cur_node, cur_key, label)
                leaf_node = TemplatedNode(new_node, None, label)
                leaf_node.add(node_id, dependencies)
                new_node.add_child(leaf_node)
                cur_node.add_child(new_node)
                return TemplatedNodeDependency(node_id, leaf_node)

            child = cur_node.children[key_id]
            prefix_len, compare_flag = self.compare_prefix(cur_key, child.key)
            if compare_flag > 0:
                # Key already exists
                leaf = child.get_leaf()
                if leaf is None:
                    leaf = TemplatedNode(child, None, label)
                    child.add_child(leaf)
                leaf.add(node_id, dependencies)
                return TemplatedNodeDependency(node_id, leaf)
            _, key_second_part = self._slice_key(cur_key, prefix_len)
            if compare_flag == 0:
                # Key contains child.key
                cur_node = child
                cur_key = key_second_part
                continue
            # Keys diverge. Need to split.
            new_node = self._split_node(child, prefix_len)
            cur_node.change_child(new_node)
            cur_node = new_node
            cur_key = key_second_part

    def _remove_from_node(
        self, node: TemplatedNode, key: PrefixType, label: Hashable | None, node_id: str
    ) -> None:
        cur_node = node
        cur_key = key

        # For cleaning up the branch
        node_stack: list[tuple[TemplatedNode, Hashable | None]] = []
        last_node: TemplatedNode | None = None

        while True:
            assert not self._is_key_empty(cur_key)

            # Assume that key does not overlap with node.key
            key_id = TemplatedNode.get_key_id(cur_key, label)

            node_stack.append((cur_node, key_id))

            if key_id not in cur_node.children:
                raise KeyError("Key does not exist:", cur_key)

            child = cur_node.children[key_id]
            prefix_len, compare_flag = self.compare_prefix(cur_key, child.key)
            if compare_flag > 0:
                # Match found
                leaf = child.get_leaf()
                if leaf is None:
                    raise KeyError("Key does not exist", cur_key)
                leaf.remove(node_id)
                node_stack.append((child, None))
                last_node = leaf
                break
            # Remove from the child's subtree
            _, key_second_part = self._slice_key(cur_key, prefix_len)
            cur_node = child
            cur_key = key_second_part

        # Clean up the branch
        assert last_node is not None
        cur_node = last_node
        while node_stack:
            child = cur_node
            cur_node, key_id = node_stack.pop()
            if child.is_removable():
                cur_node.children.pop(key_id)
            else:
                break

    def compare_prefix(
        self, key: PrefixType, child_key: PrefixType
    ) -> tuple[list[int], int]:
        """
        Returns
        -------
        list[int]
            Lengths of the common prefix
        int
            Flag indicating the equality of the keys:
            1: key and child_key are equal
            0: key contains child_key
            -1: otherwise
        """
        prefix_len: list[int] = []

        if len(key) == len(child_key):
            flag = 1
        elif len(key) > len(child_key):
            flag = 0
        else:
            flag = -1

        for k1, k2 in zip(key, child_key):
            if isinstance(k1, (Placeholder, MessageDelimiter)) or isinstance(
                k2, (Placeholder, MessageDelimiter)
            ):
                # Compare placeholders or message delimiters
                if k1 != k2:
                    flag = -1  # Keys diverge
                    break
                prefix_len.append(1)  # Placeholder or message delimiter
            else:
                # Compare strings
                count = 0
                for c1, c2 in zip(k1, k2):
                    if c1 != c2:
                        break
                    count += 1
                prefix_len.append(count)
                if count == len(k1) and count == len(k2):
                    continue  # Keys are equal

                if count < len(k1) or count < len(k2):
                    if count < len(k2):
                        flag = -1  # Keys diverge
                    else:
                        flag = 0  # key contains child_key
                if count < len(k1):
                    if count < len(k2):
                        flag = -1  # Keys diverge
                    else:
                        flag = 0  # key contains child_key
                elif count < len(k2):
                    flag = -1  # Keys diverge
                break

        return prefix_len, flag

    def _split_node(self, node: TemplatedNode, prefix_len: list[int]) -> TemplatedNode:
        # new_node -> node
        key_first_part, key_second_part = self._slice_key(node.key, prefix_len)
        new_node = TemplatedNode(node.parent, key_first_part, node.label)
        node.split_to(new_node, key_second_part)
        new_node.add_child(node)
        return new_node

    def _slice_key(
        self, key: PrefixType, prefix_len: list[int]
    ) -> tuple[PrefixType, PrefixType]:
        if len(prefix_len) == 0:
            raise ValueError("Prefix length is empty")

        i = len(prefix_len) - 1
        last_common_part = key[i]
        if isinstance(last_common_part, (Placeholder, MessageDelimiter)):
            key_first_part = key[: i + 1]
            key_second_part = key[i + 1 :]
        else:
            last_prefix_len = prefix_len[i]
            new_slice = last_common_part[last_prefix_len:]
            if len(new_slice) == 0:
                key_first_part = key[: i + 1]
                key_second_part = key[i + 1 :]
            else:
                key_first_part = tuple([*key[:i], last_common_part[:last_prefix_len]])
                key_second_part = tuple([new_slice, *key[i + 1 :]])
        return cast(tuple[PrefixType, PrefixType], (key_first_part, key_second_part))

    def get_static_prefixes(self) -> dict[str, StaticPrefixType]:
        """
        Returns
        -------
        dict[str, set[StaticPrefixType]]
            Mapping of op IDs to their static prefixes.
        """
        static_prefixes: dict[str, StaticPrefixType] = {}
        for leaf in self.leaves():
            for node_id in leaf.get_node_ids():
                static_prefix: list[StaticKeyItem] = []
                for item in leaf.get_prefix():
                    if isinstance(item, str):
                        static_prefix.append(item)
                    elif isinstance(item, MessageDelimiter):
                        static_prefix.append(item)
                    else:
                        break
                static_prefixes[node_id] = tuple(static_prefix)
        return static_prefixes

    def leaves(self) -> set[TemplatedNode]:
        return set(leaf for leaf in self.node_map.values())

    def leaves_dfs(self) -> list[TemplatedNode]:
        return [node for node in self.dfs() if node.is_leaf]

    def dfs(self) -> Iterable[TemplatedNode]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            if not node.is_leaf:
                stack.extend(node.children.values())

    def __str__(self) -> str:
        out_str: str = ""

        def _node_to_str(node: TemplatedNode, indent: str) -> None:
            nonlocal out_str

            if node.parent is None:
                out_str += indent + "ROOT\n"
            else:
                out_str += indent + repr(node) + "\n"

            if not node.is_leaf:
                for child in node.children.values():
                    _node_to_str(child, indent + "  ")

        _node_to_str(self.root, "")
        return out_str

    def show(self, **kwargs) -> None:
        print(str(self), **kwargs)


class StringNode:
    __slots__ = "children", "parent", "key", "_inner"

    children: dict[str, "StringNode"]
    parent: "StringNode | None"
    key: str

    def __init__(self, parent: "StringNode | None", key: str):
        self.children = {}
        self.parent = parent
        self.key = key
        self._inner: list[str] = []

    @property
    def value_count(self) -> int:
        return len(self._inner)

    @property
    def is_leaf(self) -> bool:
        return self.value_count > 0

    def get_children(self) -> set["StringNode"]:
        return set(self.children.values())

    def get_node_ids(self) -> Iterable[str]:
        return self._inner

    def get_prefix(self) -> str:
        parent = self.parent
        key = self.key

        while parent is not None:
            key = parent.key + key
            parent = parent.parent

        return key

    @staticmethod
    def get_key_id(key: str) -> str:
        if len(key) == 0:
            raise ValueError("Key is empty")
        return key[0]

    def add(self, node_id: str) -> None:
        if node_id in self._inner:
            raise KeyError("Node ID already exists")
        self._inner.append(node_id)

    def add_child(self, child: "StringNode") -> None:
        key_id = self.get_key_id(child.key)
        if key_id in self.children:
            raise ValueError("Child already exists")
        self.children[key_id] = child

    def change_child(self, child: "StringNode") -> None:
        key_id = self.get_key_id(child.key)
        if key_id not in self.children:
            raise ValueError("Child does not exist")
        self.children[key_id] = child

    def remove(self, node_id: str) -> None:
        if node_id not in self._inner:
            raise KeyError("Node ID does not exist")
        self._inner.remove(node_id)

    def split_to(self, new_node: "StringNode", key: str) -> None:
        self.parent = new_node
        self.key = key

    def is_removable(self) -> bool:
        return (
            self.parent is not None
            and len(self.children) == 0
            and self.value_count == 0
        )

    def __repr__(self) -> str:
        return f"StringNode(key={self.key}, inner={self._inner})"


class StringRadixTree:
    def __init__(self) -> None:
        self.root = StringNode(None, "")
        self.node_map: dict[str, StringNode] = {}

    def add(self, key: str, node_id: str) -> None:
        if node_id in self.node_map:
            raise KeyError("Node ID already exists")

        node = self._add_to_node(self.root, key, node_id)
        self.node_map[node_id] = node

    def remove(self, node_id: str) -> None:
        if node_id not in self.node_map:
            raise KeyError("Node ID does not exist")

        node = self.node_map.pop(node_id)
        key = node.get_prefix()
        self._remove_from_node(self.root, key, node_id)

    def _add_to_node(self, node: StringNode, key: str, node_id: str) -> StringNode:
        cur_node = node
        cur_key = key

        while True:
            # Assume that key does not overlap with node.key
            key_id = StringNode.get_key_id(cur_key)

            if key_id not in cur_node.children:
                # No child with the given key. Create a new node.
                new_node = StringNode(cur_node, cur_key)
                new_node.add(node_id)
                cur_node.add_child(new_node)
                return new_node

            child = cur_node.children[key_id]
            prefix_len, compare_flag = self.compare_prefix(cur_key, child.key)
            if compare_flag > 0:
                # Key already exists
                child.add(node_id)
                return child
            _, key_second_part = self._slice_key(cur_key, prefix_len)
            if compare_flag == 0:
                # Key contains child.key
                cur_node = child
                cur_key = key_second_part
                continue
            # Keys diverge. Need to split.
            new_node = self._split_node(child, prefix_len)
            cur_node.change_child(new_node)
            cur_node = new_node
            cur_key = key_second_part

    def _remove_from_node(self, node: StringNode, key: str, node_id: str) -> None:
        cur_node = node
        cur_key = key

        node_stack: list[tuple[StringNode, str]] = []
        last_node: StringNode | None = None

        while True:
            # Assume that key does not overlap with node.key
            key_id = StringNode.get_key_id(cur_key)

            node_stack.append((cur_node, key_id))

            if key_id not in cur_node.children:
                raise KeyError("Key does not exist:", cur_key)

            child = cur_node.children[key_id]
            prefix_len, compare_flag = self.compare_prefix(cur_key, child.key)
            if compare_flag > 0:
                # Match found
                if child.value_count == 0:
                    raise KeyError("Key does not exist", cur_key)
                child.remove(node_id)
                last_node = child
                break
            # Remove from the child's subtree
            _, key_second_part = self._slice_key(cur_key, prefix_len)
            cur_node = child
            cur_key = key_second_part

        # Clean up the branch
        assert last_node is not None
        cur_node = last_node
        while node_stack:
            child = cur_node
            cur_node, key_id = node_stack.pop()
            if child.is_removable():
                cur_node.children.pop(key_id)
            else:
                break

    def compare_prefix(self, key: str, child_key: str) -> tuple[int, int]:
        """
        Returns
        -------
        list[int]
            Lengths of the common prefix
        int
            Flag indicating the equality of the keys:
            1: key and child_key are equal
            0: key contains child_key
            -1: otherwise
        """
        count = 0
        for c1, c2 in zip(key, child_key):
            if c1 != c2:
                break
            count += 1

        if count == len(key) == len(child_key):
            flag = 1
        elif count == len(child_key) and count < len(key):
            flag = 0
        else:
            flag = -1

        return count, flag

    def _split_node(self, node: StringNode, prefix_len: int) -> StringNode:
        # new_node -> node
        key_first_part, key_second_part = self._slice_key(node.key, prefix_len)
        new_node = StringNode(node.parent, key_first_part)
        node.split_to(new_node, key_second_part)
        new_node.add_child(node)
        return new_node

    def _slice_key(self, key: str, prefix_len: int) -> tuple[str, str]:
        return key[:prefix_len], key[prefix_len:]

    def leaves(self) -> set[StringNode]:
        return set(leaf for leaf in self.node_map.values())

    def leaves_dfs(self) -> list[StringNode]:
        return [node for node in self.dfs() if node.is_leaf]

    def dfs(self) -> Iterable[StringNode]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def __str__(self) -> str:
        out_str: str = ""

        def _node_to_str(node: StringNode, indent: str) -> None:
            nonlocal out_str

            if node.parent is None:
                out_str += indent + f"ROOT (value_count={node.value_count})\n"
            else:
                out_str += indent + repr(node) + "\n"
            for child in node.children.values():
                _node_to_str(child, indent + "  ")

        _node_to_str(self.root, "")
        return out_str

    def show(self, **kwargs) -> None:
        print(str(self), **kwargs)
