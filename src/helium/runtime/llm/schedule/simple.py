import random
from collections.abc import Iterable


def op_wise_schedule(op_ids: list[str]) -> Iterable[list[str]]:
    return [[op_id] for op_id in op_ids]


def batch_wise_schedule(op_ids: list[str]) -> Iterable[list[str]]:
    new_op_ids = op_ids.copy()
    random.shuffle(new_op_ids)
    return [new_op_ids]
