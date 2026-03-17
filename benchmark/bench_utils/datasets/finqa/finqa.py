import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal, TypedDict

import requests


class FinQADataEntry(TypedDict):
    pre_text: list[str]
    table: list[list[str]]
    post_text: list[str]
    question: str
    answer: str


class FinQADataset:
    DATA_URL = {
        "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
        "dev": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
        "test": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
    }

    def __init__(
        self,
        data_dir: Path | None = None,
        split: Literal["train", "dev", "test"] = "test",
        seed: int | None = 42,
    ) -> None:
        self._split = split
        self._seed = seed

        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "data"
        data_path = data_dir / f"{self._split}.json"
        if not data_path.exists():
            print("Dataset not found. Downloading...")
            self.download(data_path)
            print("Download complete.")

        self._data = self._load_data(data_path)

    def download(self, data_path: Path) -> None:
        url = self.DATA_URL[self._split]
        response = requests.get(url)
        response.raise_for_status()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "wb") as f:
            f.write(response.content)

    def iter_context_qa_pairs(
        self, num_contexts: int
    ) -> Iterable[tuple[str, str, str | None]]:
        if num_contexts > len(self._data):
            raise ValueError(
                f"Requested count {num_contexts} exceeds available data size {len(self._data)}."
            )

        for entry in self._data[:num_contexts]:
            pre_text = " ".join(entry["pre_text"])
            table = "\n".join(f"|{'|'.join(row)}|" for row in entry["table"])
            post_text = " ".join(entry["post_text"])
            context = f"{pre_text}\n\n{table}\n{post_text}"
            yield context, entry["question"], entry["answer"]

    @property
    def split(self) -> str:
        return self._split

    def _load_data(self, data_path: Path) -> list[FinQADataEntry]:
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data: list[FinQADataEntry] = [
            {
                "pre_text": item["pre_text"],
                "table": item["table"],
                "post_text": item["post_text"],
                "question": item["qa"]["question"],
                "answer": item["qa"]["answer"],
            }
            for item in raw
        ]

        seed = self._seed
        if seed is not None:
            random.seed(seed)
        random.shuffle(data)

        return data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[FinQADataEntry]:
        return iter(self._data)

    def __getitem__(self, idx: int) -> FinQADataEntry:
        return self._data[idx]
