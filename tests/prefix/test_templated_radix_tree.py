import pytest

from helium import utils
from helium.utils.prefix.radix_tree import (
    Placeholder,
    PrefixType,
    TemplatedNode,
    TemplatedNodeDependency,
    TemplatedRadixTree,
    compress_prefix,
)


def _add_key_to_tree(
    tree: TemplatedRadixTree, key: PrefixType, node_id: str | None = None
) -> TemplatedNodeDependency:
    if node_id is None:
        node_id = utils.unique_id()
    return tree.add(key, None, node_id)


def test_add_empty_key():
    tree = TemplatedRadixTree()
    with pytest.raises(ValueError):
        _add_key_to_tree(tree, ())


def test_add_single_key():
    tree = TemplatedRadixTree()
    key = ("abcd",)
    key_id = ("a", None)
    _add_key_to_tree(tree, key)

    # Expect the child to be stored under the id of the first character ("a")
    assert key_id in tree.root.children, "key not found in root children"

    node = tree.root.children[key_id]
    assert node.key == key, "node key is incorrect"
    leaf = node.get_leaf()
    assert leaf is not None, "expected a leaf node"
    assert leaf.value_count == 1, "node value_count is incorrect"


def test_add_duplicate_key():
    tree = TemplatedRadixTree()
    key = ("test",)
    key_id = ("t", None)

    _add_key_to_tree(tree, key)
    _add_key_to_tree(tree, key)

    node = tree.root.children.get(key_id)
    assert node is not None, "expected child node with key ('t', None)"
    leaf = node.get_leaf()
    assert leaf is not None, "expected a leaf node"
    assert (
        leaf.value_count == 2
    ), f"expected value_count to be 2, got {leaf.value_count}"


def test_add_duplicate_node_ids():
    tree = TemplatedRadixTree()
    node_id = utils.unique_id()

    _add_key_to_tree(tree, ("key1",), node_id)

    with pytest.raises(KeyError):
        _add_key_to_tree(tree, ("key2",), node_id)


def test_remove_key():
    tree = TemplatedRadixTree()
    key = ("remove",)
    dep = _add_key_to_tree(tree, key)
    tree.remove(dep.node_id)

    assert ("r", None) not in tree.root.children, "key still exists after removal"


def test_remove_nonexistent_key():
    tree = TemplatedRadixTree()
    node_id = "nonexistent"

    with pytest.raises(KeyError):
        tree.remove(node_id)


def test_compare_prefix_equal():
    tree = TemplatedRadixTree()
    key = ("abc",)
    child_key = ("abc",)

    prefix, flag = tree.compare_prefix(key, child_key)

    assert flag == 1, f"expected flag to be 1, got {flag}"
    assert prefix == [3], f"expected prefix [3], got {prefix}"


def test_compare_prefix_key_contains_child():
    tree = TemplatedRadixTree()
    key = ("abcd",)
    child_key = ("abc",)

    prefix, flag = tree.compare_prefix(key, child_key)

    assert flag == 0, f"expected flag to be 0, got {flag}"
    assert prefix == [3], f"expected prefix [3], got {prefix}"


def test_compare_prefix_diverge():
    tree = TemplatedRadixTree()
    key = ("man",)
    child_key = ("mad",)

    prefix, flag = tree.compare_prefix(key, child_key)

    assert flag == -1, f"expected flag to be -1, got {flag}"
    assert prefix == [2], f"expected prefix [2], got {prefix}"


def test_slice_key_nonempty_new_slice():
    tree = TemplatedRadixTree()
    key = ("abcd",)
    prefix = [3]

    first_part, second_part = tree._slice_key(key, prefix)

    assert first_part == ("abc",), f"expected ('abc',), got {first_part}"
    assert second_part == ("d",), f"expected ('d',), got {second_part}"


def test_slice_key_empty_new_slice():
    tree = TemplatedRadixTree()
    key = ("abc",)
    prefix = [3]

    first_part, second_part = tree._slice_key(key, prefix)

    assert first_part == ("abc",), f"expected ('abc',), got {first_part}"
    assert second_part == (), f"expected (), got {second_part}"


def test_placeholder_equality():
    ph1 = Placeholder("op_id1")
    ph2 = Placeholder("op_id1")
    ph3 = Placeholder("op_id2")

    assert ph1 == ph2, "placeholders with same op_id should be equal"
    assert ph1 != ph3, "placeholders with different op_id should not be equal"
    assert hash(ph1) == hash(ph2), "hash of equal placeholders should be the same"


def test_tree_node_add_child_duplicate():
    parent = TemplatedNode(None, ("a",))
    child1 = TemplatedNode(parent, ("abc",))
    parent.add_child(child1)

    with pytest.raises(ValueError):
        child2 = TemplatedNode(parent, ("abc",))
        parent.add_child(child2)


def test_tree_node_change_child_nonexistent():
    parent = TemplatedNode(None, ("a",))
    child = TemplatedNode(parent, ("abc",))

    with pytest.raises(ValueError):
        parent.change_child(child)


def test_tree_node_is_removable():
    parent = TemplatedNode(None, ("parent",))
    node = TemplatedNode(parent, ("child",))

    assert (
        node.is_removable()
    ), "node should be removable if it has no children or value_count"


def test_add_split_node():
    tree = TemplatedRadixTree()
    key1 = ("man",)
    key2 = ("mad",)
    key_id = ("m", None)

    _add_key_to_tree(tree, key1)
    _add_key_to_tree(tree, key2)

    node = tree.root.children.get(key_id)
    assert node, "missing split node"

    assert node.key == ("ma",), f"expected ('ma',), got {node.key}"

    child_keys = set(node.children.keys())
    assert child_keys == {
        ("n", None),
        ("d", None),
    }, f"expected children keys {{('n', None), ('d', None)}}, got {child_keys}"


def test_remove_duplicate_key_decrements_count():
    tree = TemplatedRadixTree()
    key = ("dup",)

    dep = _add_key_to_tree(tree, key)
    _add_key_to_tree(tree, key)

    node = tree.root.children.get(("d", None))
    assert node, "expected child node with key ('d', None)"
    leaf = node.get_leaf()
    assert leaf is not None, "expected a leaf node"
    assert leaf.value_count == 2, f"expected initial count 2, got {leaf.value_count}"

    tree.remove(dep.node_id)
    assert (
        leaf.value_count == 1
    ), f"expected value_count 1 after removal, got {leaf.value_count}"


def test_remove_nested_key_single_node_decrements_count():
    tree = TemplatedRadixTree()
    key = ("abc", "def")

    dep1 = _add_key_to_tree(tree, key)
    dep2 = _add_key_to_tree(tree, key)

    # Ensure there's exactly one child at root.
    assert (
        len(tree.root.children) == 1
    ), f"expected 1 child at root, got {len(tree.root.children)}"

    node = next(iter(tree.root.children.values()))

    # Verify that the node's key is exactly ("abc", "def") and value_count is 2
    assert node.key == key, f"expected node.key to be {key}, got {node.key}"
    leaf = node.get_leaf()
    assert leaf is not None, "expected a leaf node"
    assert (
        leaf.value_count == 2
    ), f"expected node.value_count to be 2, got {leaf.value_count}"

    # Remove one occurrence
    tree.remove(dep1.node_id)
    assert (
        leaf.value_count == 1
    ), f"expected node.value_count to be 1, got {leaf.value_count}"

    # Remove the second occurrence
    tree.remove(dep2.node_id)
    assert not tree.root.children, "expected no children after removing key completely"


def test_add_mixed_key():
    """
    Test adding a key that mixes a string and a Placeholder.
    Key: ("hello", Placeholder("op_id"), "world")
    Expected: A single node is created under root keyed by "h".
    """
    tree = TemplatedRadixTree()
    ph = Placeholder("op_id")
    key = ("hello", ph, "world")
    key_id = ("h", None)

    _add_key_to_tree(tree, key)

    assert key_id in tree.root.children, "missing root child with key id 'h'"

    node = tree.root.children[key_id]
    assert node.key == key, f"expected node.key {key}, got {node.key}"
    leaf = node.get_leaf()
    assert leaf is not None, "expected a leaf node"
    assert leaf.value_count == 1, f"expected node.value_count 1, got {leaf.value_count}"


def test_add_two_mixed_keys_with_different_placeholders():
    """
    Test adding two keys that share a common string prefix but have different Placeholder values.
    """
    tree = TemplatedRadixTree()
    ph1 = Placeholder("op_id1")
    ph2 = Placeholder("op_id2")

    key1 = ("hello", ph1, "world")
    key2 = ("hello", ph2, "world")
    key_id = ("h", None)

    _add_key_to_tree(tree, key1)
    _add_key_to_tree(tree, key2)

    assert key_id in tree.root.children, "missing root child 'h'"

    split_node = tree.root.children[key_id]
    # The split node should have two children (one for ph1, one for ph2).
    assert (
        len(split_node.children) == 2
    ), f"expected 2 children in split node, got {len(split_node.children)}"

    child_keys = set(split_node.children.keys())
    assert child_keys == {
        (ph1, None),
        (ph2, None),
    }, f"expected child keys {{(ph1, None), (ph2, None)}}, got {child_keys}"


def test_add_and_remove_mixed_key():
    """
    Test adding and then removing a key that starts with a Placeholder.
    """
    tree = TemplatedRadixTree()
    ph = Placeholder("op_id")
    key = (ph, "data", "end")
    key_id = (ph, None)

    dep = _add_key_to_tree(tree, key)

    assert key_id in tree.root.children, "missing root child with placeholder key id"

    node = tree.root.children[key_id]
    assert node.key == key, f"expected node.key {key}, got {node.key}"
    leaf = node.get_leaf()
    assert leaf is not None, "expected a leaf node"
    assert leaf.value_count == 1, f"expected value_count 1, got {leaf.value_count}"

    tree.remove(dep.node_id)
    assert ph not in tree.root.children, "node still exists after removal"


def test_duplicate_mixed_key_increments_count():
    """
    Test adding the same mixed key twice to ensure that value_count increments.
    """
    tree = TemplatedRadixTree()
    ph = Placeholder("op_id")
    key = ("prefix", ph, "suffix")
    key_id = ("p", None)

    _add_key_to_tree(tree, key)
    _add_key_to_tree(tree, key)

    assert key_id in tree.root.children, "missing root child for key 'prefix'"

    node = tree.root.children[key_id]
    assert node.key == key, f"expected node.key {key}, got {node.key}"
    leaf = node.get_leaf()
    assert leaf is not None, "expected a leaf node"
    assert leaf.value_count == 2, f"expected value_count 2, got {leaf.value_count}"


def test_add_remove_multiple_mixed_keys():
    """
    Test adding multiple mixed keys and then removing them in arbitrary order.
    After all removals, the tree should be empty.
    """
    tree = TemplatedRadixTree()
    ph = Placeholder("op_id")
    key1 = ("start", ph, "middle", "end1")
    key2 = ("start", ph, "middle", "end2")
    key3 = ("start", ph, "middle", "end3")

    dep1 = _add_key_to_tree(tree, key1)
    dep2 = _add_key_to_tree(tree, key2)
    dep3 = _add_key_to_tree(tree, key3)

    assert tree.root.children, "tree is unexpectedly empty after adding keys"

    # Remove in arbitrary order
    tree.remove(dep2.node_id)
    tree.remove(dep1.node_id)
    tree.remove(dep3.node_id)

    assert not tree.root.children, "tree is not empty after removing all keys"


def test_get_leaves():
    tree = TemplatedRadixTree()
    key1 = ("leaf1",)
    key2 = ("branch", "leaf2")
    key3 = ("branch", "leaf3")
    key4 = ("another", "branch", "leaf4")

    _add_key_to_tree(tree, key1)
    _add_key_to_tree(tree, key2)
    _add_key_to_tree(tree, key3)
    _add_key_to_tree(tree, key4)

    leaves = tree.leaves()
    assert len(leaves) == 4, f"expected 4 leaves, got {len(leaves)}"

    for leaf in leaves:
        assert leaf.is_leaf, f"Node {leaf} returned by leaves_dfs is not a leaf"

    leaf_prefixes = {compress_prefix(leaf.get_prefix()) for leaf in leaves}
    expected_prefixes = {compress_prefix(k) for k in [key1, key2, key3, key4]}
    assert (
        leaf_prefixes == expected_prefixes
    ), f"expected leaf keys {expected_prefixes}, got {leaf_prefixes}"


def test_leaves_dfs():
    tree = TemplatedRadixTree()
    key1 = ("leaf1",)
    key2 = ("branch", "leaf2")
    key3 = ("branch", "leaf3")
    key4 = ("another", "branch", "leaf4")

    _add_key_to_tree(tree, key1)
    _add_key_to_tree(tree, key2)
    _add_key_to_tree(tree, key3)
    _add_key_to_tree(tree, key4)

    leaves = tree.leaves_dfs()
    assert len(leaves) == 4, f"Expected 4 leaves, got {len(leaves)}"

    for leaf in leaves:
        assert leaf.is_leaf, f"Node {leaf} returned by leaves_dfs is not a leaf"

    leaf_prefixes = {compress_prefix(leaf.get_prefix()) for leaf in leaves}
    expected_prefixes = {compress_prefix(k) for k in [key1, key2, key3, key4]}
    assert (
        leaf_prefixes == expected_prefixes
    ), f"Expected leaf prefixes {expected_prefixes}, got {leaf_prefixes}"
