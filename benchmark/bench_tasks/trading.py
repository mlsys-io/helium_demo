from pathlib import Path
from typing import Any, Protocol

from bench_programs.trading.agentscope import ASTradingProgram
from bench_programs.trading.autogen import AutoGenTradingProgram
from bench_programs.trading.helium import HeliumTradingProgram, TradingAgent
from bench_programs.trading.langgraph import LangGraphTradingProgram
from bench_programs.trading.opwise import OpWiseTradingProgram
from bench_programs.trading.parrot import ParrotTradingProgram
from bench_programs.trading.querywise import QueryWiseTradingProgram
from bench_tasks.base import BenchmarkConfig, BenchmarkTask
from bench_utils.datasets.fin_data import FinancialDataset
from bench_utils.runner.base import BenchmarkRunner, RunnerConfig

from helium.common import GenerationConfig
from helium.runtime import HeliumServerConfig
from helium.runtime.protocol import (
    HeliumRequestConfig,
    QueryProfilingConfig,
    SystemProfilingConfig,
)


class TradingDataShape(Protocol):
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


class TradingBenchmarkConfig(BenchmarkConfig):
    """
    Configuration specific to Trading Agents benchmark tasks.
    """

    def __init__(
        self,
        num_trials: int,
        helium_profiling: bool,
        enable_cache_aware_scheduling: bool,
        enable_runtime_adjustment: bool,
        enable_query_profiling: bool,
        helium_prebuilt_file: str | Path | None,
        runner_config: RunnerConfig,
        dataset_name: str,
        num_stocks: int,
        num_news_chunks: int,
        max_news: int,
        num_social_chunks: int,
        max_social_posts: int,
        num_debate_rounds: int = 2,
        include_insider_transactions: bool = False,
        split: str = "2024-06-02",
        dev_size: int = 30,
    ) -> None:
        super().__init__(
            system=runner_config.system,
            num_trials=num_trials,
            helium_profiling=helium_profiling,
            enable_cache_aware_scheduling=enable_cache_aware_scheduling,
            enable_runtime_adjustment=enable_runtime_adjustment,
            enable_query_profiling=enable_query_profiling,
            helium_prebuilt_file=helium_prebuilt_file,
        )

        llm_server_config = runner_config.llm_server_config
        kwargs: dict[str, Any] = {
            "model": llm_server_config.model,
            "base_url": f"http://{llm_server_config.host}:{llm_server_config.port}/v1",
            "temperature": 0,
            "max_tokens": 512,
        }
        self.generation_config = GenerationConfig.from_env(**kwargs)

        self.num_stocks = num_stocks
        self.num_news_chunks = num_news_chunks
        self.max_news = max_news
        self.num_social_chunks = num_social_chunks
        self.max_social_posts = max_social_posts
        self.num_debate_rounds = num_debate_rounds
        self.include_insider_transactions = include_insider_transactions

        self.dataset_name = dataset_name
        # Load the dataset
        if helium_profiling:
            num_stocks = dev_size
            split = "2024-06-01"
        data: TradingDataShape
        if dataset_name.startswith("fin_data"):
            dataset = FinancialDataset(
                date=split,
                include_insider_transactions=include_insider_transactions,
            )
            data = dataset.get_data(
                num_stocks=num_stocks,
                num_news_chunks=num_news_chunks,
                max_news=max_news,
                num_social_chunks=num_social_chunks,
                max_social_posts=max_social_posts,
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.data = data

    @property
    def input_size(self) -> int:
        return self.num_stocks


class TradingBenchmarkTask(BenchmarkTask):
    """
    Benchmark task based on Trading Agents
    """

    def __init__(
        self,
        config: TradingBenchmarkConfig,
        runner: BenchmarkRunner,
        name_suffix: str | None = None,
    ) -> None:
        workload = f"trading-{config.dataset_name}-{config.num_debate_rounds}r"
        if name_suffix is not None:
            workload += f"-{name_suffix}"
        self.config: TradingBenchmarkConfig
        super().__init__(workload=workload, config=config, runner=runner)

    def create_program(self) -> Any:
        system_name = self.config.system.lower()
        runner_config = self.runner.config
        if system_name.startswith("querywise"):
            return QueryWiseTradingProgram()
        elif system_name.startswith("opwise"):
            return OpWiseTradingProgram()
        elif system_name.startswith("autogen"):
            return AutoGenTradingProgram()
        elif system_name.startswith("langgraph"):
            return LangGraphTradingProgram()
        elif system_name.startswith("agentscope"):
            return ASTradingProgram()
        elif system_name.startswith("parrot"):
            llm_service_config = runner_config.llm_server_config
            return ParrotTradingProgram(
                llm_service_config.host, llm_service_config.port
            )
        elif system_name.startswith("helium"):
            if self.config.helium_prebuilt_file:
                agent, query_profile = TradingAgent.load(
                    self.config.helium_prebuilt_file
                )
                query_profile_map = (
                    None if query_profile is None else {agent.name: query_profile}
                )
            else:
                agent = query_profile_map = None
            helium_server_config = HeliumServerConfig(is_local=True, benchmarking=True)
            helium_request_config = HeliumRequestConfig(
                enable_cache_aware_scheduling=self.config.enable_cache_aware_scheduling,
                enable_runtime_adjustment=self.config.enable_runtime_adjustment,
                system_profiling_config=SystemProfilingConfig(),
                query_profiling_config=(
                    QueryProfilingConfig(query_profile_map=query_profile_map)
                    if self.config.enable_query_profiling
                    else None
                ),
            )
            return HeliumTradingProgram(
                helium_request_config, helium_server_config, agent
            )
        else:
            raise ValueError(f"Unsupported system: {self.config.system}")

    def get_run_configurations(
        self,
    ) -> tuple[list[str], list[Any], list[list[dict]]]:
        config = self.config
        run_names = [self.system_name]
        kwargs = [
            [
                {
                    "dataset": config.dataset_name,
                    "stock_symbols": config.data.stock_symbols,
                    "funda_company_profiles": config.data.funda_company_profiles,
                    "funda_balance_sheets": config.data.funda_balance_sheets,
                    "funda_income_stmts": config.data.funda_income_stmts,
                    "funda_cashflow_stmts": config.data.funda_cashflow_stmts,
                    "funda_insider_sentiments": config.data.funda_insider_sentiments,
                    "funda_insider_transactions": config.data.funda_insider_transactions,
                    "market_stock_prices": config.data.market_stock_prices,
                    "market_stock_stats": config.data.market_stock_stats,
                    "news_combined_chunked": config.data.news_combined_chunked,
                    "social_reddit_post_chunked": config.data.social_reddit_post_chunked,
                    "num_news_chunks": config.num_news_chunks,
                    "num_social_chunks": config.num_social_chunks,
                    "num_debate_rounds": config.num_debate_rounds,
                    "generation_config": config.generation_config,
                }
            ]
            * config.num_trials
        ]
        return run_names, [self.create_program()], kwargs
