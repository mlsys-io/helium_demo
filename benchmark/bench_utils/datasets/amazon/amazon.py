import gzip
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jsonlines
import requests


class AmazonReviewsDataset:
    REVIEW_URL_FMT = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{}.jsonl.gz"
    META_URL_FMT = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{}.jsonl.gz"

    def __init__(
        self,
        data_dir: Path | None = None,
        category: str = "Health_and_Personal_Care",
        split: str = "test",
        min_reviews: int = 60,
        dev_size: int | float = 30,
        seed: int | None = 42,
    ) -> None:
        self._category = category
        self._split = split
        self._seed = seed

        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "data"

        filtered_review_data_path = data_dir / f"{category}.{split}.jsonl"
        filtered_meta_data_path = data_dir / f"meta_{category}.{split}.jsonl"
        if not (
            filtered_review_data_path.exists() and filtered_meta_data_path.exists()
        ):
            print("Filtered data not found. Downloading and preprocessing...")

            review_data_path = data_dir / f"{category}.jsonl"
            url = self.REVIEW_URL_FMT.format(category)
            self._download(url, review_data_path)

            meta_data_path = data_dir / f"meta_{category}.jsonl"
            url = self.META_URL_FMT.format(category)
            self._download(url, meta_data_path)

            self._preprocess(review_data_path, meta_data_path, min_reviews, dev_size)

            print("DONE")

        self._review_data_path = filtered_review_data_path
        self._meta_data_path = filtered_meta_data_path

    def _download(self, url: str, data_path: Path) -> None:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url)
        response.raise_for_status()
        with open(data_path, "wb") as f:
            f.write(gzip.decompress(response.content))

    def _preprocess(
        self,
        review_data_path: Path,
        meta_data_path: Path,
        min_reviews: int = 60,
        dev_size: int | float = 30,
    ) -> None:
        def preprocess_partition(parent_asins: set[str], split: str) -> None:
            # Filter reviews
            filtered_file_path = review_data_path.with_suffix(f".{split}.jsonl")
            with (
                jsonlines.open(review_data_path) as reader,
                jsonlines.open(filtered_file_path, "w") as writer,
            ):
                for review in reader:
                    if review["parent_asin"] in parent_asins:
                        writer.write(review)
            # Filter metadata
            filtered_file_path = meta_data_path.with_suffix(f".{split}.jsonl")
            with (
                jsonlines.open(meta_data_path) as reader,
                jsonlines.open(filtered_file_path, "w") as writer,
            ):
                for item in reader:
                    asin = item["parent_asin"]
                    if asin in parent_asins:
                        item["review_count"] = review_counter[asin]
                        writer.write(item)

        random.seed(self._seed)
        with jsonlines.open(review_data_path) as reader:
            review_counter = Counter(review["parent_asin"] for review in reader)
        parent_asins = [
            asin for asin, count in review_counter.items() if count >= min_reviews
        ]
        random.shuffle(parent_asins)

        # Partition into dev and test sets
        dev_size = (
            dev_size if isinstance(dev_size, int) else int(len(parent_asins) * dev_size)
        )
        preprocess_partition(set(parent_asins[:dev_size]), "dev")
        preprocess_partition(set(parent_asins[dev_size:]), "test")

        # Remove original files
        review_data_path.unlink()
        meta_data_path.unlink()

    def iter_item_reviews(
        self, num_items: int, num_reviews_per_item: int
    ) -> Iterable[tuple[dict[str, Any], list[dict[str, Any]]]]:
        random.seed(self._seed)

        meta_list: list[dict[str, Any]]
        with jsonlines.open(self._meta_data_path) as reader:
            meta_list = [
                item for item in reader if item["review_count"] >= num_reviews_per_item
            ]
        random.shuffle(meta_list)

        asin_dict = {item["parent_asin"]: item for item in meta_list}
        meta_list = []
        review_dict: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with jsonlines.open(self._review_data_path) as reader:
            for review in reader:
                asin = review["parent_asin"]
                if asin in asin_dict:
                    review_dict[asin].append(review)
                    if len(review_dict[asin]) >= num_reviews_per_item:
                        item = asin_dict.pop(asin)
                        meta_list.append(item)
                if len(meta_list) >= num_items:
                    break
            else:
                raise ValueError(
                    f"Not enough items in category '{self._category}' to sample "
                    f"{num_items} items."
                )

        for item in meta_list:
            yield item, review_dict[item["parent_asin"]]
