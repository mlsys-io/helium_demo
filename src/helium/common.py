from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any, Final, Hashable, overload

from helium import envs
from helium.utils import partition


@dataclass(slots=True)
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class Slice:
    __slots__ = ("range_or_indices", "is_range", "length")

    def __init__(self, range_or_indices: tuple[int, int] | list[int]):
        self.range_or_indices: Final[tuple[int, int] | list[int]] = range_or_indices
        if isinstance(range_or_indices, tuple):
            self.is_range = True
            self.length = range_or_indices[1] - range_or_indices[0]
        else:
            self.is_range = False
            self.length = len(range_or_indices)

    def as_slice(self) -> slice:
        if isinstance(self.range_or_indices, list):
            raise ValueError("Cannot convert indices to slice")
        start, stop = self.range_or_indices
        return slice(start, stop)

    def as_tuple(self) -> tuple[int, int]:
        if isinstance(self.range_or_indices, list):
            raise ValueError("Cannot convert indices to tuple")
        return self.range_or_indices

    def serialize(self) -> dict[str, Any]:
        if isinstance(self.range_or_indices, tuple):
            return {
                "type": "range",
                "start": self.range_or_indices[0],
                "stop": self.range_or_indices[1],
            }
        return {"type": "indices", "indices": self.range_or_indices}

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "Slice":
        if data["type"] == "range":
            return cls((data["start"], data["stop"]))
        return cls(data["indices"])

    def to_indices(self) -> list[int]:
        return list(self)

    def to_hashable(self) -> Hashable:
        if isinstance(self.range_or_indices, tuple):
            return ("range", self.range_or_indices[0], self.range_or_indices[1])
        return tuple(sorted(self.range_or_indices))

    @classmethod
    def merge(cls, slices: Iterable["Slice"]) -> "Slice":
        if all(s.is_range for s in slices):
            # Combine contiguous ranges
            sorted_slices = sorted([s.as_tuple() for s in slices])
            initial, counter = sorted_slices[0]
            for start, stop in sorted_slices[1:]:
                if start != counter:
                    raise ValueError(
                        f"Input slices are not contiguous. {sorted_slices}"
                    )
                counter = stop
            return Slice((initial, counter))
        # Combine non-contiguous indices
        all_indices = []
        for s in slices:
            all_indices.extend(s.to_indices())
        sorted_indices = sorted(set(all_indices))
        if len(sorted_indices) != len(all_indices):
            raise ValueError("Input slices have overlapping indices.")
        if sorted_indices[-1] - sorted_indices[0] + 1 == len(sorted_indices):
            # Contiguous range
            return Slice((sorted_indices[0], sorted_indices[-1] + 1))
        # Non-contiguous indices
        return Slice(sorted_indices)

    def partition(self, partition_count: int) -> list["Slice"]:
        if partition_count <= 1:
            return [self]
        partitions: list[Slice]
        if isinstance(self.range_or_indices, tuple):
            partitions = []
            start, stop = self.range_or_indices
            input_size = stop - start
            partition_size = input_size // partition_count
            remainder = input_size % partition_count
            counter = start
            while counter < stop:
                # Calculate the size of the current partition
                size = partition_size + (1 if remainder > 0 else 0)
                remainder -= 1
                # Advance the counter
                part_start = counter
                counter = min(counter + size, stop)
                partitions.append(Slice((part_start, counter)))
        else:
            partitions = [
                Slice(indices)
                for indices in partition(self.range_or_indices, partition_count)
            ]
        return partitions

    def difference(self, other: "Slice") -> "Slice":
        """Returns a new Slice that is the difference between this slice and another slice."""
        if other not in self:
            raise ValueError("Other slice is not a subset of this slice.")
        if self == other:
            return Slice((0, 0))  # Empty slice
        if isinstance(self.range_or_indices, tuple) and isinstance(
            other.range_or_indices, tuple
        ):
            s_start, s_stop = self.range_or_indices
            o_start, o_stop = other.range_or_indices
            if s_start == o_start:
                return Slice((o_stop, s_stop))
            if s_stop == o_stop:
                return Slice((s_start, o_start))
        diff_indices = sorted(set(self) - set(other))
        return Slice(diff_indices)

    def intersect(self, other: "Slice") -> "Slice":
        """Returns a new Slice that is the intersection between this slice and another slice."""
        if isinstance(self.range_or_indices, tuple) and isinstance(
            other.range_or_indices, tuple
        ):
            s_start, s_stop = self.range_or_indices
            o_start, o_stop = other.range_or_indices
            if s_start >= o_stop or o_start >= s_stop:
                return Slice((0, 0))  # Empty slice
            return Slice((max(s_start, o_start), min(s_stop, o_stop)))
        intersect_indices = sorted(set(self) & set(other))
        if not intersect_indices:
            return Slice((0, 0))  # Empty slice
        return Slice(intersect_indices)

    def __iter__(self) -> Iterator[int]:
        if isinstance(self.range_or_indices, list):
            return iter(self.range_or_indices)
        return iter(range(self.range_or_indices[0], self.range_or_indices[1]))

    __hash__ = None  # type: ignore

    def __repr__(self):
        return f"Slice(range_or_indices={self.range_or_indices})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Slice):
            if isinstance(self.range_or_indices, tuple):
                if isinstance(other.range_or_indices, tuple):
                    return self.range_or_indices == other.range_or_indices
                return self.to_indices() == other.range_or_indices
            return self.range_or_indices == other.to_indices()
        return False

    @overload
    def __contains__(self, other: int) -> bool: ...

    @overload
    def __contains__(self, other: "Slice") -> bool: ...

    def __contains__(self, other: "int | Slice") -> bool:
        if isinstance(other, int):
            if isinstance(self.range_or_indices, list):
                return other in self.range_or_indices
            start, stop = self.range_or_indices
            return start <= other < stop
        if isinstance(self.range_or_indices, list):
            return all(o in self.range_or_indices for o in other.to_indices())
        start, stop = self.range_or_indices
        if isinstance(other.range_or_indices, list):
            return all(start <= o < stop for o in other.range_or_indices)
        o_start, o_stop = other.range_or_indices
        return start <= o_start and stop >= o_stop


@dataclass
class GenerationConfig:
    model: str
    base_url: str
    api_key: str = "EMPTY"
    frequency_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = 1
    presence_penalty: float | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = False
    stream_options: Any = None
    temperature: float | None = None
    top_p: float | None = None
    ignore_eos: bool = False
    llm_service: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def openai_kwargs(self) -> dict[str, Any]:
        kwargs = self.to_dict()
        kwargs.pop("llm_service")
        kwargs.pop("api_key")
        kwargs.pop("base_url")
        kwargs.pop("ignore_eos")
        return kwargs

    def vllm_kwargs(self) -> dict[str, Any]:
        kwargs = self.to_dict()
        kwargs.pop("llm_service")
        kwargs.pop("api_key")
        kwargs.pop("base_url")
        return kwargs

    @classmethod
    def from_env(cls, **kwargs) -> "GenerationConfig":
        env_config: dict[str, Any] = dict(
            llm_service=envs.LLM_SERVICE,
            model=envs.LLM_MODEL,
            api_key=envs.LLM_API_KEY,
            base_url=envs.LLM_BASE_URL,
            max_tokens=envs.LLM_MAX_TOKENS,
        )
        env_config.update(kwargs)

        return cls(**env_config)
