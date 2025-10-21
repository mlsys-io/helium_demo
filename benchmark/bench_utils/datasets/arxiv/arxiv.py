import random
import shutil
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import gdown
import jsonlines


class ArxivDataset:
    GDRIVE_FILE_ID = "1b3rmCSIoh6VhD4HKWjI4HOW-cSwcwbeC"

    def __init__(
        self,
        data_dir: Path | None = None,
        split: Literal["val", "test"] = "test",
        num_tokens_range: tuple[int, int] = (6000, 8000),
        seed: int | None = 42,
    ) -> None:
        self._split = split
        self._num_tokens_range = num_tokens_range
        self._seed = seed

        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "data"
        data_path = data_dir / f"{self._split}.jsonl"
        if not data_path.exists():
            print("Filtered data not found. Downloading and preprocessing...")
            self.download(data_dir)
            print("DONE")

        self._data_path = data_path

    def download(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        # Download the raw zip file
        raw_data_path = data_dir / "raw.zip"
        gdown.download(id=self.GDRIVE_FILE_ID, output=str(raw_data_path))
        # Extract the zip file
        extracted_dir = data_dir / "arxiv-dataset"
        print(f"Extracting data into '{extracted_dir}'...")
        with zipfile.ZipFile(raw_data_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        raw_data_path.unlink()
        shutil.rmtree(data_dir / "__MACOSX", ignore_errors=True)
        # Filter the data
        print("Filtering data...")
        for split in ("val", "test"):
            self._filter_data(
                extracted_dir / f"{split}.txt",
                data_dir / f"{split}.jsonl",
            )
        shutil.rmtree(extracted_dir)

    def _filter_data(self, data_path: Path, dest_path: Path) -> None:
        min_tokens, max_tokens = self._num_tokens_range
        with (
            jsonlines.open(data_path) as reader,
            jsonlines.open(dest_path, mode="w") as writer,
        ):
            for article in reader:
                tokens = " ".join(article["article_text"]).split()
                if min_tokens <= len(tokens) <= max_tokens:
                    writer.write(article)

    def iter_articles(
        self, num_articles: int, num_chunks_per_article: int
    ) -> Iterable[tuple[str, ...]]:
        random.seed(self._seed)
        with jsonlines.open(self._data_path) as reader:
            articles = list(reader)
        random.shuffle(articles)

        for article in articles[:num_articles]:
            tokens = " ".join(article["article_text"]).split()
            num_tokens = len(tokens)
            chunk_size = num_tokens // num_chunks_per_article
            remainder = num_tokens % num_chunks_per_article
            chunks = []
            counter = 0
            while counter < num_tokens:
                # Calculate the actual chunk size
                size = chunk_size + (1 if remainder > 0 else 0)
                remainder -= 1
                # Advance the counter
                start = counter
                counter += size
                # Append the chunk
                chunks.append(" ".join(tokens[start:counter]))
            yield tuple(chunks)
