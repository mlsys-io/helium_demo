import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Self, TypedDict, cast

import yaml
from bench_programs.utils.common import random_shuffle
from bench_utils.mixin import BenchmarkMixin

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumResponse, HeliumSystemProfile

OutputType = list[str]

"""
# Prompt templates used in the Trading Agents benchmark.
* Fundamentals Analyst prompts
    - extraction_instruction_fmt(stock, company_profile, doc_name, doc_desc, doc)
    - summary_instruction_fmt(stock, company_profile, **{doc_ref: doc_summary})
* Other Analysts prompts
    - extraction_instruction_fmt(stock, doc_name, doc)
    - summary_instruction_fmt(stock, **{doc_ref: doc_summary})
* Bull Researcher prompts
    - first_round_fmt(stock, fundamentals_report, market_report, news_report, social_media_report)
    - subsequent_round_fmt(other_response)
* Bear Researcher prompts
    - first_round_fmt(stock, fundamentals_report, market_report, news_report, social_media_report)
    - subsequent_round_fmt(other_response)
* Research Manager prompts
    - user_prompt_fmt(stock, debate)
* Trader prompts
    - user_prompt_fmt(stock, investment_plan)
* Risk Analyst prompts
    - first_round_fmt(stock, fundamentals_report, market_report, news_report, social_media_report, trader_decision)
    - subsequent_round_fmt(**{agent_name: agent_response})
* Judge prompts
    - user_prompt_fmt(stock, debate)
* Fund Manager prompts
    - user_prompt_fmt(stock, **{trader_name: recommendation})
"""


@dataclass
class PromptFormats:
    doc_descriptions: dict[str, str]
    # System prompts
    fundamentals_system_prompt: str
    market_system_prompt: str
    news_system_prompt: str
    social_media_system_prompt: str
    bull_system_prompt: str
    bear_system_prompt: str
    research_manager_system_prompt: str
    trader_system_prompts: dict[str, str]
    risky_analyst_system_prompt: str
    safe_analyst_system_prompt: str
    neutral_analyst_system_prompt: str
    judge_system_prompt: str
    fund_manager_system_prompt: str
    # Prompt templates
    fundamentals_extraction_fmt: str
    fundamentals_summary_fmt: str
    market_extraction_fmt: str
    market_summary_fmt: str
    news_extraction_fmt: str
    news_summary_fmt: str
    social_extraction_fmt: str
    social_summary_fmt: str
    researcher_first_round_fmt: str
    researcher_subsequent_round_fmt: str
    researcher_manager_user_prompt_fmt: str
    trader_user_prompt_fmt: str
    risk_analyst_first_round_fmt: str
    risk_analyst_subsequent_round_fmt: str
    judge_user_prompt_fmt: str
    fund_manager_user_prompt_fmt: str

    @classmethod
    def from_yaml(cls, data_dir: Path) -> "PromptFormats":
        kwargs = {}
        for fpath in data_dir.rglob("*.yaml"):
            with open(fpath) as file:
                kwargs.update(yaml.safe_load(file))
        return cls(**kwargs)


class TradingData(TypedDict):
    stock_symbol: str
    funda_company_profile: str
    funda_balance_sheet: str
    funda_income_stmt: str
    funda_cashflow_stmt: str
    funda_insider_sentiment: str
    funda_insider_transaction: str | None
    market_stock_price: str
    market_stock_stat: str
    news_combined_chunks: list[str]
    social_reddit_post_chunks: list[str]


@dataclass(slots=True)
class TradingInput:
    # Configuration
    num_news_chunks: int
    num_social_chunks: int
    num_debate_rounds: int
    generation_config: GenerationConfig | None
    doc_descriptions: dict[str, str]
    # System prompts
    fundamentals_system_prompt: str
    market_system_prompt: str
    news_system_prompt: str
    social_media_system_prompt: str
    bull_system_prompt: str
    bear_system_prompt: str
    research_manager_system_prompt: str
    trader_system_prompts: dict[str, str]
    risky_analyst_system_prompt: str
    safe_analyst_system_prompt: str
    neutral_analyst_system_prompt: str
    judge_system_prompt: str
    fund_manager_system_prompt: str
    # Prompt templates
    fundamentals_extraction_fmt: str
    fundamentals_summary_fmt: str
    market_extraction_fmt: str
    market_summary_fmt: str
    news_extraction_fmt: str
    news_summary_fmt: str
    social_extraction_fmt: str
    social_summary_fmt: str
    researcher_first_round_fmt: str
    researcher_subsequent_round_fmt: str
    researcher_manager_user_prompt_fmt: str
    trader_user_prompt_fmt: str
    risk_analyst_first_round_fmt: str
    risk_analyst_subsequent_round_fmt: str
    judge_user_prompt_fmt: str
    fund_manager_user_prompt_fmt: str
    # Data
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

    @classmethod
    def create(
        cls,
        prompt_formats: PromptFormats,
        stock_symbols: list[str],
        funda_company_profiles: list[str],
        funda_balance_sheets: list[str],
        funda_income_stmts: list[str],
        funda_cashflow_stmts: list[str],
        funda_insider_sentiments: list[str],
        funda_insider_transactions: list[str] | None,
        market_stock_prices: list[str],
        market_stock_stats: list[str],
        news_combined_chunked: list[list[str]],
        social_reddit_post_chunked: list[list[str]],
        num_news_chunks: int,
        num_social_chunks: int,
        num_debate_rounds: int,
        generation_config: GenerationConfig | None,
    ) -> "TradingInput":
        return cls(
            **asdict(prompt_formats),
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
            num_news_chunks=num_news_chunks,
            num_social_chunks=num_social_chunks,
            num_debate_rounds=num_debate_rounds,
            generation_config=generation_config,
        )

    def shuffle(self) -> list[int]:
        indices = list(range(len(self.stock_symbols)))
        to_shuffle: list[list] = [
            indices,
            self.stock_symbols,
            self.funda_company_profiles,
            self.funda_balance_sheets,
            self.funda_income_stmts,
            self.funda_cashflow_stmts,
            self.funda_insider_sentiments,
            self.market_stock_prices,
            self.market_stock_stats,
            self.news_combined_chunked,
            self.social_reddit_post_chunked,
        ]
        if self.funda_insider_transactions is not None:
            to_shuffle.append(self.funda_insider_transactions)

        for lst in to_shuffle:
            random_shuffle(lst, inplace=True)

        return indices

    def iter_data(self) -> Iterable[TradingData]:
        for (
            stock_symbol,
            funda_company_profile,
            funda_balance_sheet,
            funda_income_stmt,
            funda_cashflow_stmt,
            funda_insider_sentiment,
            funda_insider_transaction,
            market_stock_price,
            market_stock_stat,
            news_combined_chunks,
            social_reddit_post_chunks,
        ) in zip(
            self.stock_symbols,
            self.funda_company_profiles,
            self.funda_balance_sheets,
            self.funda_income_stmts,
            self.funda_cashflow_stmts,
            self.funda_insider_sentiments,
            (
                itertools.repeat(None)
                if self.funda_insider_transactions is None
                else self.funda_insider_transactions
            ),
            self.market_stock_prices,
            self.market_stock_stats,
            self.news_combined_chunked,
            self.social_reddit_post_chunked,
        ):
            yield TradingData(
                stock_symbol=stock_symbol,
                funda_company_profile=funda_company_profile,
                funda_balance_sheet=funda_balance_sheet,
                funda_income_stmt=funda_income_stmt,
                funda_cashflow_stmt=funda_cashflow_stmt,
                funda_insider_sentiment=funda_insider_sentiment,
                funda_insider_transaction=funda_insider_transaction,
                market_stock_price=market_stock_price,
                market_stock_stat=market_stock_stat,
                news_combined_chunks=news_combined_chunks,
                social_reddit_post_chunks=social_reddit_post_chunks,
            )


class TradingProgram(BenchmarkMixin, ABC):
    class OutputBuilder:
        def __init__(self) -> None:
            self._inner: list[tuple[int, str]] = []

        def add(self, index: int, content: str) -> None:
            self._inner.append((index, content))

        def update(self, items: Iterable[tuple[int, str]]) -> Self:
            for index, content in items:
                self.add(index, content)
            return self

        def build(self) -> OutputType:
            ret: list[str | None] = [None] * len(self._inner)
            for index, content in self._inner:
                ret[index] = content
            return cast(OutputType, ret)

    _PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
    PROMPT_FORMATS: dict[str, PromptFormats] = {
        "fin_data": PromptFormats.from_yaml(_PROMPT_DIR / "fin_data"),
        "fin_data-4b": PromptFormats.from_yaml(_PROMPT_DIR / "fin_data-4b"),
        "fin_data-12b": PromptFormats.from_yaml(_PROMPT_DIR / "fin_data-12b"),
        "fin_data-16b": PromptFormats.from_yaml(_PROMPT_DIR / "fin_data-16b"),
    }

    async def run_async(
        self,
        dataset: Literal["fin_data"],
        stock_symbols: list[str],
        funda_company_profiles: list[str],
        funda_balance_sheets: list[str],
        funda_income_stmts: list[str],
        funda_cashflow_stmts: list[str],
        funda_insider_sentiments: list[str],
        funda_insider_transactions: list[str] | None,
        market_stock_prices: list[str],
        market_stock_stats: list[str],
        news_combined_chunked: list[list[str]],
        social_reddit_post_chunked: list[list[str]],
        num_news_chunks: int,
        num_social_chunks: int,
        num_debate_rounds: int,
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> OutputType:
        self.start_timer("run")
        invest_reports, system_profile = await self._run(
            **self.prepare_kwargs(
                dataset=dataset,
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
                num_news_chunks=num_news_chunks,
                num_social_chunks=num_social_chunks,
                num_debate_rounds=num_debate_rounds,
                generation_config=generation_config,
                **kwargs,
            )
        )
        self.stop_timer()
        self.set_system_profile(system_profile)

        return invest_reports

    async def precompute(
        self,
        dataset: Literal["fin_data"],
        stock_symbols: list[str],
        funda_company_profiles: list[str],
        funda_balance_sheets: list[str],
        funda_income_stmts: list[str],
        funda_cashflow_stmts: list[str],
        funda_insider_sentiments: list[str],
        funda_insider_transactions: list[str] | None,
        market_stock_prices: list[str],
        market_stock_stats: list[str],
        news_combined_chunked: list[list[str]],
        social_reddit_post_chunked: list[list[str]],
        num_news_chunks: int,
        num_social_chunks: int,
        num_debate_rounds: int,
        generation_config: GenerationConfig | None,
        precompute_mode: Literal["none", "only", "both"],
        **kwargs,
    ) -> HeliumResponse:
        return await self._precompute(
            precompute_mode=precompute_mode,
            **self.prepare_kwargs(
                dataset=dataset,
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
                num_news_chunks=num_news_chunks,
                num_social_chunks=num_social_chunks,
                num_debate_rounds=num_debate_rounds,
                generation_config=generation_config,
                **kwargs,
            ),
        )

    def prepare_kwargs(
        self,
        dataset: Literal["fin_data"],
        stock_symbols: list[str],
        funda_company_profiles: list[str],
        funda_balance_sheets: list[str],
        funda_income_stmts: list[str],
        funda_cashflow_stmts: list[str],
        funda_insider_sentiments: list[str],
        funda_insider_transactions: list[str] | None,
        market_stock_prices: list[str],
        market_stock_stats: list[str],
        news_combined_chunked: list[list[str]],
        social_reddit_post_chunked: list[list[str]],
        num_news_chunks: int,
        num_social_chunks: int,
        num_debate_rounds: int,
        generation_config: GenerationConfig | None,
        **kwargs,
    ) -> dict[str, Any]:
        if dataset not in self.PROMPT_FORMATS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        prompt_formats = self.PROMPT_FORMATS[dataset]
        return dict(
            trading_input=TradingInput.create(
                prompt_formats=prompt_formats,
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
                num_news_chunks=num_news_chunks,
                num_social_chunks=num_social_chunks,
                num_debate_rounds=num_debate_rounds,
                generation_config=generation_config,
            )
        )

    @abstractmethod
    async def _run(
        self, trading_input: TradingInput
    ) -> tuple[OutputType, HeliumSystemProfile]:
        pass

    async def _precompute(
        self,
        trading_input: TradingInput,
        precompute_mode: Literal["none", "only", "both"],
    ) -> HeliumResponse:
        raise NotImplementedError("Precompute is not implemented.")
