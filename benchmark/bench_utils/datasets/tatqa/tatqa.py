import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal, TypedDict

import requests


class TatQADataEntry(TypedDict):
    table: dict[str, Any]
    paragraphs: list[dict[str, Any]]
    questions: list[dict[str, Any]]


class TatQADataset:
    MAX_NUM_QUESTIONS = 6
    DATA_URL = {
        "train": "https://raw.githubusercontent.com/NExTplusplus/tat-qa/master/dataset_raw/tatqa_dataset_train.json",
        "dev": "https://raw.githubusercontent.com/NExTplusplus/tat-qa/master/dataset_raw/tatqa_dataset_dev.json",
        "test": "https://raw.githubusercontent.com/NExTplusplus/tat-qa/master/dataset_raw/tatqa_dataset_test.json",
    }

    def __init__(
        self,
        data_dir: Path | None = None,
        split: Literal["train", "dev", "test"] = "test",
        num_questions: int | None = None,
        seed: int | None = 42,
    ) -> None:
        if num_questions is None:
            num_questions = self.MAX_NUM_QUESTIONS
        elif num_questions < 1 or num_questions > self.MAX_NUM_QUESTIONS:
            raise ValueError(
                f"num_questions must be between 1 and {self.MAX_NUM_QUESTIONS}."
            )

        self._split = split
        self._num_questions = num_questions
        self._seed = seed

        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "data"
        data_path = data_dir / f"tatqa_dataset_{self._split}.json"
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
    ) -> Iterable[tuple[str, tuple[str, ...], tuple[Any, ...] | None]]:
        if num_contexts > len(self._data):
            raise ValueError(
                f"Requested count {num_contexts} exceeds available data size {len(self._data)}."
            )

        for entry in self._data[:num_contexts]:
            # Format paragraphs
            paragraphs: list[dict[str, Any]] = sorted(
                entry["paragraphs"], key=lambda para: para["order"]
            )
            para_context = "\n".join(para["text"] for para in paragraphs)
            # Format tabular data
            table: list[list[str]] = entry["table"]["table"]
            table_context = "\n".join(f"|{'|'.join(row)}|" for row in table)
            # Combine context
            context = f"{para_context}\n\n{table_context}"

            # Extract questions and answers
            question_dicts: list[dict[str, Any]] = sorted(
                entry["questions"], key=lambda q: q["order"]
            )[: self._num_questions]
            questions = tuple([q["question"] for q in question_dicts])
            answers = (
                None
                if self.split == "test"
                else tuple([q["answer"] for q in question_dicts])
            )

            yield context, questions, answers

    @property
    def split(self) -> str:
        return self._split

    def _load_data(self, data_path: Path) -> list[TatQADataEntry]:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        seed = self._seed
        if seed is not None:
            random.seed(seed)
        random.shuffle(data)

        return data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[TatQADataEntry]:
        return iter(self._data)

    def __getitem__(self, idx: int) -> TatQADataEntry:
        return self._data[idx]
