import time
from abc import ABC, abstractmethod
from typing import Any, Literal

from helium.runtime.protocol import HeliumResponse, HeliumSystemProfile


class BenchmarkMixin(ABC):
    def __init__(self) -> None:
        self._start_times: dict[str, list[float]] = {}
        self._end_times: dict[str, list[float]] = {}
        self._keys: list[str] = []
        self._system_profile: HeliumSystemProfile | None = None

    @abstractmethod
    async def run_async(self, *args, **kwargs) -> Any:
        """Runs the benchmark"""
        pass

    async def precompute(
        self, *args, precompute_mode: Literal["none", "only", "both"], **kwargs
    ) -> HeliumResponse:
        """Precomputes the benchmark"""
        raise NotImplementedError("Precompute not implemented")

    def start_timer(self, key: str) -> None:
        """Starts a timer for a given key"""
        if key not in self._start_times:
            self._start_times[key] = []
        self._start_times[key].append(time.time())
        self._keys.append(key)

    def stop_timer(self) -> None:
        """Stops the last started timer"""
        key = self._keys.pop()
        if key not in self._end_times:
            self._end_times[key] = []
        self._end_times[key].append(time.time())

    def get_elapsed_times(self) -> dict[str, list[float]]:
        """Gets elapsed times for each key"""
        self.validate_timer()
        return {
            key: [
                end_time - start_time
                for start_time, end_time in zip(start_times, self._end_times[key])
            ]
            for key, start_times in self._start_times.items()
        }

    def get_total_elapsed_time(self) -> dict[str, float]:
        """Gets total elapsed time for each key"""
        return {key: sum(times) for key, times in self.get_elapsed_times().items()}

    def validate_timer(self) -> None:
        """Validates that the timer states are consistent"""
        for key, start_times in self._start_times.items():
            if key not in self._end_times:
                raise ValueError(f"Timer for key '{key}' has not been stopped.")
            if len(start_times) != len(self._end_times[key]):
                raise ValueError(
                    f"Timer for key '{key}' has been started {len(start_times)} times "
                    f"but stopped {len(self._end_times[key])} times."
                )

    def reset_timer(self) -> None:
        """Resets the timer states"""
        self._start_times = {}
        self._end_times = {}
        self._keys = []

    def set_system_profile(self, system_profile: HeliumSystemProfile) -> None:
        """Sets profiling results"""
        if self._system_profile is not None:
            raise ValueError("Profiling results have already been set.")
        self._system_profile = system_profile

    def get_and_reset_system_profile(self) -> HeliumSystemProfile:
        """Gets and resets profiling results"""
        if self._system_profile is None:
            raise ValueError("Profiling results have not been set.")
        system_profile = self._system_profile
        self._system_profile = None
        return system_profile
