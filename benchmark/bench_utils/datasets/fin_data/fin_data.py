import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from helium.utils import partition


@dataclass
class FinancialData:
    stock_symbols: list[str]
    funda_company_profiles: list[str]
    funda_balance_sheets: list[str]
    funda_income_stmts: list[str]
    funda_cashflow_stmts: list[str]
    funda_insider_sentiments: list[str]
    funda_insider_transactions: list[str] | None
    market_stock_prices: list[str]
    market_stock_stats: list[str]
    news_combined_chunked: list[list[str]]
    social_reddit_post_chunked: list[list[str]]


def _to_table_string(df: pd.DataFrame) -> str:
    table: list[list] = []
    table.append(df.columns.tolist())
    for _, row in df.iterrows():
        table.append(row.tolist())
    table_str = "\n".join(f"|{'|'.join([str(item) for item in row])}|" for row in table)
    return table_str


class FinancialDataset:
    def __init__(
        self,
        data_dir: Path | None = None,
        date: str = "2024-06-02",
        include_insider_transactions: bool = False,
        seed: int | None = 42,
    ) -> None:
        self._date = date
        self._seed = seed
        self._include_insider_transactions = include_insider_transactions

        if data_dir is None:
            data_dir = Path(__file__).resolve().parent / "data"

        if not data_dir.exists():
            self._prepare_data(data_dir, include_insider_transactions)

        data_path = data_dir / date
        if not data_path.exists():
            raise FileNotFoundError(f"Data for date {date} not found in {data_dir}.")
        self._data_path = data_path

        self.stock_list: list[str] | None
        stock_list_path = data_dir / "stock_list.txt"
        if stock_list_path.exists():
            with open(stock_list_path) as f:
                self.stock_list = [line.strip() for line in f]
        else:
            self.stock_list = None

    def _prepare_data(self, data_dir: Path, include_insider_transactions: bool) -> None:
        data_dir.mkdir(exist_ok=True)
        data_path = data_dir.parent / "fin_data.zip"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file {data_path} not found.")
        # Extract the zip file
        print("Extracting data...")
        with zipfile.ZipFile(data_path, "r") as zip_ref:
            zip_ref.extractall(data_dir.parent)
        (data_dir.parent / "fin_data").rename(data_dir)

        stock_list: set[str] | None = None
        for date_dir in data_dir.iterdir():
            if not date_dir.is_dir():
                continue
            for info_dir in date_dir.iterdir():
                if not info_dir.is_dir():
                    continue
                if (
                    not include_insider_transactions
                    and info_dir.name == "funda-insider_transactions"
                ):
                    continue
                cur_stocks = {stock_file.stem for stock_file in info_dir.glob("*.csv")}
                if stock_list is None:
                    stock_list = cur_stocks
                else:
                    stock_list &= cur_stocks
        stock_list = stock_list or set()
        print(f"Available stocks: {len(stock_list)}")

        stock_metadata_path = data_dir / "stock_metadata.json"
        if not stock_metadata_path.exists():
            stock_metadata: dict[str, dict[str, dict]] = {}
            for date_dir in data_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                date = date_dir.name
                for info_dir in date_dir.iterdir():
                    if not info_dir.is_dir():
                        continue
                    if (
                        not include_insider_transactions
                        and info_dir.name == "funda-insider_transactions"
                    ):
                        continue
                    info_name = info_dir.name
                    for stock in stock_list:
                        try:
                            df = pd.read_csv(info_dir / f"{stock}.csv")
                        except pd.errors.EmptyDataError:
                            df = pd.DataFrame()
                        if date in stock_metadata:
                            if stock in stock_metadata[date]:
                                if info_name in stock_metadata[date][stock]:
                                    stock_metadata[date][stock][info_name] += len(df)
                                else:
                                    stock_metadata[date][stock][info_name] = len(df)
                            else:
                                stock_metadata[date][stock] = {info_name: len(df)}
                        else:
                            stock_metadata[date] = {stock: {info_name: len(df)}}

            with open(stock_metadata_path, "w") as f:
                json.dump(stock_metadata, f, indent=2)
        print("DONE")

    def _sort_stocks(self, max_news: int, max_social_posts: int) -> list[str]:
        metadata_path = self._data_path.parent / "stock_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file {metadata_path} not found.")
        with open(metadata_path, "r") as f:
            stock_metadata = json.load(f)
        date_metadata = stock_metadata[self._date]

        def stock_key(stock: str) -> tuple:
            meta = date_metadata.get(stock, {})
            social_score = meta.get("social-reddit_posts", 0) / max_social_posts
            if social_score == 0:
                social_score = float("-inf")
            news_score = meta.get("news-combined", 0) / max_news
            if news_score == 0:
                news_score = float("-inf")
            sort_key = (
                -(min(social_score, 1) + min(news_score, 1)),
                -social_score,
                -news_score,
                -meta.get("funda-insider_sentiments", 0),
                -meta.get("funda-balance_sheets", 0),
                -meta.get("funda-income_stmts", 0),
                -meta.get("funda-cashflow_stmts", 0),
                -meta.get("market-stock_prices", 0),
                -meta.get("market-stock_stats", 0),
                (
                    -meta.get("funda-insider_transactions", 0)
                    if self._include_insider_transactions
                    else 0
                ),
            )
            return sort_key

        sorted_stocks = sorted(date_metadata.keys(), key=stock_key)
        return sorted_stocks

    def get_data(
        self,
        num_stocks: int,
        num_news_chunks: int,
        max_news: int,
        num_social_chunks: int,
        max_social_posts: int,
        max_news_len: int = 256,
        max_social_len: int = 256,
        stocks: list[str] | None = None,
    ) -> FinancialData:
        if stocks is None:
            stocks = (
                self._sort_stocks(max_news, max_social_posts)
                if self.stock_list is None
                else self.stock_list
            )
        stocks = stocks[:num_stocks]

        info_key_map = {
            "funda_company_profiles": "funda-profile",
            "funda_balance_sheets": "funda-balance_sheet",
            "funda_income_stmts": "funda-income_stmt",
            "funda_cashflow_stmts": "funda-cashflow",
            "funda_insider_sentiments": "funda-insider_sentiment",
            "funda_insider_transactions": "funda-insider_transactions",
            "market_stock_prices": "market-yfin_data",
            "market_stock_stats": "market-stock_stats",
            "news_combined_chunked": "news-combined",
            "social_reddit_post_chunked": "social-reddit_posts",
        }

        stock_symbols = stocks

        # Get company profiles
        funda_company_profiles: list[str] = []
        profile_keys = {
            "name": "Name",
            "description": "Description",
            "city": "City",
            "country": "Country",
            "finnhubIndustry": "Industry",
            "employeeTotal": "Employees",
            "marketCapCurrency": "Market Cap Currency",
            "marketCapitalization": "Market Cap",
            "floatingShare": "Floating Shares",
            "shareOutstanding": "Shares Outstanding",
            "ipo": "IPO Date",
        }
        for stock in stocks:
            path = (
                self._data_path
                / info_key_map["funda_company_profiles"]
                / f"{stock}.csv"
            )
            df = pd.read_csv(path)
            series = df[list(profile_keys)].rename(columns=profile_keys).squeeze()
            assert isinstance(series, pd.Series)
            profile_str = "\n".join(f"{k}: {v}" for k, v in series.items())
            funda_company_profiles.append(profile_str)

        # Get balance sheets
        funda_balance_sheets: list[str] = []
        for stock in stocks:
            path = (
                self._data_path / info_key_map["funda_balance_sheets"] / f"{stock}.csv"
            )
            df = pd.read_csv(path)
            df = df.sort_values("period", ascending=False)
            df_str = _to_table_string(df)
            funda_balance_sheets.append(df_str)

        # Get income statements
        funda_income_stmts: list[str] = []
        for stock in stocks:
            path = self._data_path / info_key_map["funda_income_stmts"] / f"{stock}.csv"
            df = pd.read_csv(path)
            df = df.sort_values("period", ascending=False)
            df_str = _to_table_string(df)
            funda_income_stmts.append(df_str)

        # Get cashflow statements
        funda_cashflow_stmts: list[str] = []
        for stock in stocks:
            path = (
                self._data_path / info_key_map["funda_cashflow_stmts"] / f"{stock}.csv"
            )
            df = pd.read_csv(path)
            df = df.sort_values("period", ascending=False)
            df_str = _to_table_string(df)
            funda_cashflow_stmts.append(df_str)

        # Get insider sentiments
        funda_insider_sentiments: list[str] = []
        for stock in stocks:
            path = (
                self._data_path
                / info_key_map["funda_insider_sentiments"]
                / f"{stock}.csv"
            )
            df = pd.read_csv(path)
            df = df.sort_values(["year", "month"], ascending=False)
            df_str = _to_table_string(df)
            funda_insider_sentiments.append(df_str)

        funda_insider_transactions: list[str] | None
        if self._include_insider_transactions:
            # Get insider transactions
            funda_insider_transactions = []
            for stock in stocks:
                path = (
                    self._data_path
                    / info_key_map["funda_insider_transactions"]
                    / f"{stock}.csv"
                )
                df = pd.read_csv(path)
                df = df.sort_values("transactionDate", ascending=False)
                df_str = _to_table_string(df)
                funda_insider_transactions.append(df_str)
        else:
            funda_insider_transactions = None

        # Get stock prices
        market_stock_prices: list[str] = []
        for stock in stocks:
            path = (
                self._data_path / info_key_map["market_stock_prices"] / f"{stock}.csv"
            )
            df = pd.read_csv(path)
            df = df.sort_values("Date", ascending=False)
            df_str = _to_table_string(df)
            market_stock_prices.append(df_str)

        # Get stock stats
        market_stock_stats: list[str] = []
        for stock in stocks:
            path = self._data_path / info_key_map["market_stock_stats"] / f"{stock}.csv"
            df = pd.read_csv(path)
            df = df.sort_values("date", ascending=False)
            df_str = _to_table_string(df)
            market_stock_stats.append(df_str)

        # Get news combined
        news_combined_chunked: list[list[str]] = []
        for stock in stocks:
            path = (
                self._data_path / info_key_map["news_combined_chunked"] / f"{stock}.csv"
            )
            df = pd.read_csv(path)
            df = df.sort_values("timestamp", ascending=False)
            rows = [row for _, row in df.iterrows()][:max_news]
            chunks = list(partition(rows, n=num_news_chunks))
            formatted_chunks: list[str] = []
            for chunk in chunks:
                formatted_chunk = [
                    f"Headline: {row['headline']}\n"
                    f"Date: {row['date']}\n"
                    f"Source: {row['source']}\n"
                    f"Summary: {' '.join(row['summary'].split()[:max_news_len])}"
                    for row in chunk
                    if isinstance(row["summary"], str)
                ]
                formatted_chunks.append("\n\n".join(formatted_chunk))
            news_combined_chunked.append(formatted_chunks)

        # Get social reddit posts
        social_reddit_post_chunked: list[list[str]] = []
        for stock in stocks:
            path = (
                self._data_path
                / info_key_map["social_reddit_post_chunked"]
                / f"{stock}.csv"
            )
            df = pd.read_csv(path)
            df = df.sort_values("upvotes", ascending=False)
            rows = [row for _, row in df.iterrows()][:max_social_posts]
            chunks = list(partition(rows, n=num_social_chunks))
            formatted_chunks = []
            for chunk in chunks:
                formatted_chunk = [
                    f"Title: {row['title']}\n"
                    f"Content: {' '.join(row['content'].split()[:max_social_len])}\n"
                    f"Score: {row['upvotes']}"
                    for row in chunk
                ]
                formatted_chunks.append("\n\n".join(formatted_chunk))
            social_reddit_post_chunked.append(formatted_chunks)

        return FinancialData(
            stock_symbols=stock_symbols,
            funda_company_profiles=funda_company_profiles,
            funda_balance_sheets=funda_balance_sheets,
            funda_income_stmts=funda_income_stmts,
            funda_cashflow_stmts=funda_cashflow_stmts,
            funda_insider_sentiments=funda_insider_sentiments,
            funda_insider_transactions=funda_insider_transactions,
            market_stock_prices=market_stock_prices,
            market_stock_stats=market_stock_stats,
            news_combined_chunked=news_combined_chunked,
            social_reddit_post_chunked=social_reddit_post_chunked,
        )
