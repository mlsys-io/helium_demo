"""
Adapted from https://github.com/metauto-ai/GPTSwarm.
"""

import os
import random
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pandas as pd
import requests


class MMLUDataset:
    DATA_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

    def __init__(
        self,
        data_dir: Path | None = None,
        split: Literal["dev", "val", "test"] = "test",
        seed: int | None = 42,
    ) -> None:
        self._split = split
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "data"
        if not data_dir.exists():
            print("Dataset not found. Downloading...")
            self.download(data_dir)
            print("Download complete.")
        self._total_df: pd.DataFrame = self._load_data(data_dir / split, seed)

    def download(self, data_dir: Path) -> None:
        tar_path = data_dir.with_suffix(".tar")
        if not tar_path.exists():
            r = requests.get(self.DATA_URL, allow_redirects=True)
            with open(tar_path, "wb") as f:
                f.write(r.content)
        tar = tarfile.open(tar_path)
        tar.extractall(data_dir.parent)
        tar.close()
        data_dir.parent.joinpath("data").rename(data_dir)
        os.remove(tar_path)

    @staticmethod
    def _load_data(data_path: Path, seed: int | None) -> pd.DataFrame:
        csv_paths = sorted(data_path.glob("*.csv"))

        names = ["question", "A", "B", "C", "D", "correct_answer"]

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None, names=names)
            total_df = pd.concat([total_df, single_df])

        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        index = total_df.index.tolist()
        if seed is not None:
            random.seed(seed)
        random.shuffle(index)
        total_df = total_df.reindex(index)

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __iter__(self) -> Iterator[tuple[str, list[str]]]:
        for _, record in self._total_df.iterrows():
            question, *choices, _ = record
            yield question, choices
