import asyncio
from typing import Any

from autogen import AssistantAgent
from bench_programs.trading.base import (
    OutputType,
    TradingData,
    TradingInput,
    TradingProgram,
)
from bench_programs.utils.autogen import autogen_generate_async, autogen_get_llm_config
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark

from helium.runtime.protocol import HeliumSystemProfile


class AutoGenTradingProgram(TradingProgram):
    async def _run(
        self, trading_input: TradingInput
    ) -> tuple[OutputType, HeliumSystemProfile]:
        base_url, llm_config = autogen_get_llm_config(trading_input.generation_config)
        indices = trading_input.shuffle()

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        tasks = []
        for original_idx, data in zip(indices, trading_input.iter_data()):
            task = self._run_single_stock(trading_input, data, llm_config, original_idx)
            tasks.append(task)
        outputs = await asyncio.gather(*tasks)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        investment_recommendations = self.OutputBuilder().update(outputs).build()

        return investment_recommendations, system_profile

    async def _run_single_stock(
        self,
        trading_input: TradingInput,
        data: TradingData,
        llm_config: dict[str, Any],
        index: int,
    ) -> tuple[int, str]:
        # Create agents
        fundamentals_agent = AssistantAgent(
            "fundamentals_analyst",
            system_message=trading_input.fundamentals_system_prompt,
            llm_config=llm_config,
        )
        market_agent = AssistantAgent(
            "market_analyst",
            system_message=trading_input.market_system_prompt,
            llm_config=llm_config,
        )
        news_agent = AssistantAgent(
            "news_analyst",
            system_message=trading_input.news_system_prompt,
            llm_config=llm_config,
        )
        social_media_agent = AssistantAgent(
            "social_media_analyst",
            system_message=trading_input.social_media_system_prompt,
            llm_config=llm_config,
        )
        bull_researcher = AssistantAgent(
            "bull_researcher",
            system_message=trading_input.bull_system_prompt,
            llm_config=llm_config,
        )
        bear_researcher = AssistantAgent(
            "bear_researcher",
            system_message=trading_input.bear_system_prompt,
            llm_config=llm_config,
        )
        research_manager = AssistantAgent(
            "research_manager",
            system_message=trading_input.research_manager_system_prompt,
            llm_config=llm_config,
        )

        trader_agents = {
            name: AssistantAgent(
                f"trader_{name}", system_message=prompt, llm_config=llm_config
            )
            for name, prompt in trading_input.trader_system_prompts.items()
        }

        risky_analyst = AssistantAgent(
            "risky_analyst",
            system_message=trading_input.risky_analyst_system_prompt,
            llm_config=llm_config,
        )
        safe_analyst = AssistantAgent(
            "safe_analyst",
            system_message=trading_input.safe_analyst_system_prompt,
            llm_config=llm_config,
        )
        neutral_analyst = AssistantAgent(
            "neutral_analyst",
            system_message=trading_input.neutral_analyst_system_prompt,
            llm_config=llm_config,
        )
        judge_agent = AssistantAgent(
            "judge",
            system_message=trading_input.judge_system_prompt,
            llm_config=llm_config,
        )
        fund_manager = AssistantAgent(
            "fund_manager",
            system_message=trading_input.fund_manager_system_prompt,
            llm_config=llm_config,
        )

        # Stage 1: Analyst reports
        fundamentals_report = await self._fundamentals_analyst(
            fundamentals_agent,
            trading_input,
            data["stock_symbol"],
            data["funda_company_profile"],
            data["funda_balance_sheet"],
            data["funda_income_stmt"],
            data["funda_cashflow_stmt"],
            data["funda_insider_sentiment"],
            (
                None
                if data["funda_insider_transaction"] is None
                else data["funda_insider_transaction"]
            ),
        )

        market_report = await self._market_analyst(
            market_agent,
            trading_input,
            data["stock_symbol"],
            data["market_stock_price"],
            data["market_stock_stat"],
        )

        news_report = await self._news_analyst(
            news_agent,
            trading_input,
            data["stock_symbol"],
            data["news_combined_chunks"],
        )

        social_media_report = await self._social_media_analyst(
            social_media_agent,
            trading_input,
            data["stock_symbol"],
            data["social_reddit_post_chunks"],
        )

        # Stage 2: Researcher debate
        investment_plan = await self._researcher_debate(
            bull_researcher,
            bear_researcher,
            research_manager,
            trading_input,
            data["stock_symbol"],
            fundamentals_report,
            market_report,
            news_report,
            social_media_report,
            trading_input.num_debate_rounds,
        )

        # Stage 3: Trader + Risk Analyst chains
        investment_recommendations = {}
        for trader_name, trader_agent in trader_agents.items():
            investment_recommendation = await self._trader_analyst_chain(
                trader_agent,
                risky_analyst,
                safe_analyst,
                neutral_analyst,
                judge_agent,
                trading_input,
                data["stock_symbol"],
                investment_plan,
                fundamentals_report,
                market_report,
                news_report,
                social_media_report,
                trading_input.num_debate_rounds,
            )
            investment_recommendations[trader_name] = investment_recommendation

        # Stage 4: Fund manager final decision
        final_recommendation = await self._fund_manager(
            fund_manager,
            trading_input,
            data["stock_symbol"],
            investment_recommendations,
        )

        return index, final_recommendation

    async def _summarize_documents(
        self,
        agent: AssistantAgent,
        trading_input: TradingInput,
        extraction_fmt: str,
        stock_symbol: str,
        doc_list: list[tuple[str, str]],
        **common_kwargs,
    ) -> list[tuple[str, str]]:
        doc_summaries = []

        # Extract insights from each document
        for doc_name, doc_content in doc_list:
            # Two-stage formatting: first doc_name and doc_desc, then remaining args
            formatted_template = extraction_fmt.format(
                doc_name=doc_name,
                doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
            )
            extraction_message = [
                {
                    "role": "user",
                    "content": formatted_template.format(
                        stock=stock_symbol,
                        doc=doc_content,
                        **common_kwargs,
                    ),
                }
            ]
            doc_summary = await autogen_generate_async(agent, extraction_message)
            doc_summaries.append((doc_name.replace(" ", "_"), doc_summary))

        return doc_summaries

    async def _aggregate_summaries(
        self,
        agent: AssistantAgent,
        stock_symbol: str,
        doc_summaries: list[tuple[str, str]],
        summary_fmt: str,
        **common_kwargs,
    ) -> str:
        summary_dict = dict(doc_summaries)
        summary_message = [
            {
                "role": "user",
                "content": summary_fmt.format(
                    stock=stock_symbol,
                    **summary_dict,
                    **common_kwargs,
                ),
            }
        ]
        return await autogen_generate_async(agent, summary_message)

    async def _fundamentals_analyst(
        self,
        agent: AssistantAgent,
        trading_input: TradingInput,
        stock_symbol: str,
        company_profile: str,
        balance_sheet: str,
        income_stmt: str,
        cashflow_stmt: str,
        insider_sentiment: str,
        insider_transactions: str | None,
    ) -> str:
        doc_list = [
            ("balance sheet", balance_sheet),
            ("income statement", income_stmt),
            ("cashflow statement", cashflow_stmt),
            ("insider sentiment", insider_sentiment),
        ]
        if insider_transactions is not None:
            doc_list.append(("insider transactions", insider_transactions))

        doc_summaries = await self._summarize_documents(
            agent,
            trading_input,
            trading_input.fundamentals_extraction_fmt,
            stock_symbol,
            doc_list,
            company_profile=company_profile,
        )

        return await self._aggregate_summaries(
            agent,
            stock_symbol,
            doc_summaries,
            trading_input.fundamentals_summary_fmt,
            company_profile=company_profile,
        )

    async def _market_analyst(
        self,
        agent: AssistantAgent,
        trading_input: TradingInput,
        stock_symbol: str,
        stock_price: str,
        stock_stats: str,
    ) -> str:
        doc_list = [
            ("stock price data", stock_price),
            ("stock stats indicators", stock_stats),
        ]

        doc_summaries = await self._summarize_documents(
            agent,
            trading_input,
            trading_input.market_extraction_fmt,
            stock_symbol,
            doc_list,
        )

        return await self._aggregate_summaries(
            agent,
            stock_symbol,
            doc_summaries,
            trading_input.market_summary_fmt,
        )

    async def _news_analyst(
        self,
        agent: AssistantAgent,
        trading_input: TradingInput,
        stock_symbol: str,
        news_chunks: list[str],
    ) -> str:
        doc_list = [("news chunk", chunk) for chunk in news_chunks]

        doc_summaries = await self._summarize_documents(
            agent,
            trading_input,
            trading_input.news_extraction_fmt,
            stock_symbol,
            doc_list,
        )

        # Rename chunks for aggregation
        doc_summaries = [
            (f"chunk_{i}", summary) for i, (_, summary) in enumerate(doc_summaries)
        ]

        return await self._aggregate_summaries(
            agent,
            stock_symbol,
            doc_summaries,
            trading_input.news_summary_fmt,
        )

    async def _social_media_analyst(
        self,
        agent: AssistantAgent,
        trading_input: TradingInput,
        stock_symbol: str,
        social_chunks: list[str],
    ) -> str:
        doc_list = [("post chunk", chunk) for chunk in social_chunks]

        doc_summaries = await self._summarize_documents(
            agent,
            trading_input,
            trading_input.social_extraction_fmt,
            stock_symbol,
            doc_list,
        )

        # Rename chunks for aggregation
        doc_summaries = [
            (f"chunk_{i}", summary) for i, (_, summary) in enumerate(doc_summaries)
        ]

        return await self._aggregate_summaries(
            agent,
            stock_symbol,
            doc_summaries,
            trading_input.social_summary_fmt,
        )

    async def _researcher_debate(
        self,
        bull_researcher: AssistantAgent,
        bear_researcher: AssistantAgent,
        research_manager: AssistantAgent,
        trading_input: TradingInput,
        stock_symbol: str,
        fundamentals_report: str,
        market_report: str,
        news_report: str,
        social_media_report: str,
        num_rounds: int,
    ) -> str:
        assert num_rounds >= 1

        # First round
        first_round_content = trading_input.researcher_first_round_fmt.format(
            stock=stock_symbol,
            fundamentals_report=fundamentals_report,
            market_report=market_report,
            news_report=news_report,
            social_media_report=social_media_report,
        )

        bull_messages = [{"role": "user", "content": first_round_content}]
        bear_messages = [{"role": "user", "content": first_round_content}]

        bull_response = await autogen_generate_async(bull_researcher, bull_messages)
        bear_response = await autogen_generate_async(bear_researcher, bear_messages)

        bull_messages.append({"role": "assistant", "content": bull_response})
        bear_messages.append({"role": "assistant", "content": bear_response})

        # Subsequent rounds
        for _ in range(1, num_rounds):
            # Bull responds to bear
            bull_content = trading_input.researcher_subsequent_round_fmt.format(
                other_response=bear_response
            )
            bull_messages.append({"role": "user", "content": bull_content})
            bull_response = await autogen_generate_async(bull_researcher, bull_messages)
            bull_messages.append({"role": "assistant", "content": bull_response})

            # Bear responds to bull
            bear_content = trading_input.researcher_subsequent_round_fmt.format(
                other_response=bull_response
            )
            bear_messages.append({"role": "user", "content": bear_content})
            bear_response = await autogen_generate_async(bear_researcher, bear_messages)
            bear_messages.append({"role": "assistant", "content": bear_response})

        # Convert debate to text
        debate_lines = []
        for bull_msg, bear_msg in zip(bull_messages, bear_messages):
            if bull_msg["role"] == "assistant":
                debate_lines.append(f"**Bull Researcher:** {bull_msg['content']}")
            if bear_msg["role"] == "assistant":
                debate_lines.append(f"**Bear Researcher:** {bear_msg['content']}")

        debate_text = "\n".join(debate_lines)

        # Research manager makes final decision
        manager_message = [
            {
                "role": "user",
                "content": trading_input.researcher_manager_user_prompt_fmt.format(
                    stock=stock_symbol, debate=debate_text
                ),
            }
        ]

        return await autogen_generate_async(research_manager, manager_message)

    async def _trader_analyst_chain(
        self,
        trader: AssistantAgent,
        risky_analyst: AssistantAgent,
        safe_analyst: AssistantAgent,
        neutral_analyst: AssistantAgent,
        judge: AssistantAgent,
        trading_input: TradingInput,
        stock_symbol: str,
        investment_plan: str,
        fundamentals_report: str,
        market_report: str,
        news_report: str,
        social_media_report: str,
        num_rounds: int,
    ) -> str:
        # Trader makes initial decision
        trader_message = [
            {
                "role": "user",
                "content": trading_input.trader_user_prompt_fmt.format(
                    stock=stock_symbol, investment_plan=investment_plan
                ),
            }
        ]
        trader_decision = await autogen_generate_async(trader, trader_message)

        # Risk analyst debate
        return await self._risk_analyst_debate(
            risky_analyst,
            safe_analyst,
            neutral_analyst,
            judge,
            trading_input,
            stock_symbol,
            fundamentals_report,
            market_report,
            news_report,
            social_media_report,
            trader_decision,
            num_rounds,
        )

    async def _risk_analyst_debate(
        self,
        risky_analyst: AssistantAgent,
        safe_analyst: AssistantAgent,
        neutral_analyst: AssistantAgent,
        judge: AssistantAgent,
        trading_input: TradingInput,
        stock_symbol: str,
        fundamentals_report: str,
        market_report: str,
        news_report: str,
        social_media_report: str,
        trader_decision: str,
        num_rounds: int,
    ) -> str:
        assert num_rounds >= 1

        analysts = [
            ("risky", risky_analyst),
            ("safe", safe_analyst),
            ("neutral", neutral_analyst),
        ]

        # First round
        first_round_content = trading_input.risk_analyst_first_round_fmt.format(
            stock=stock_symbol,
            fundamentals_report=fundamentals_report,
            market_report=market_report,
            news_report=news_report,
            social_media_report=social_media_report,
            trader_decision=trader_decision,
        )

        analyst_histories = {}
        analyst_responses = {}

        for name, analyst in analysts:
            messages = [{"role": "user", "content": first_round_content}]
            response = await autogen_generate_async(analyst, messages)
            messages.append({"role": "assistant", "content": response})
            analyst_histories[name] = messages
            analyst_responses[name] = response

        # Subsequent rounds
        for _ in range(1, num_rounds):
            new_responses = {}
            for name, analyst in analysts:
                # Get responses from all analysts (including current one)
                other_responses = analyst_responses.copy()

                subsequent_content = (
                    trading_input.risk_analyst_subsequent_round_fmt.format(
                        **other_responses
                    )
                )

                analyst_histories[name].append(
                    {"role": "user", "content": subsequent_content}
                )
                response = await autogen_generate_async(
                    analyst, analyst_histories[name]
                )
                analyst_histories[name].append(
                    {"role": "assistant", "content": response}
                )
                new_responses[name] = response

            analyst_responses = new_responses

        # Convert debate to text
        debate_lines = []
        analyst_names = ["risky", "safe", "neutral"]
        all_histories = [analyst_histories[name] for name in analyst_names]
        for agent_messages in zip(*all_histories):
            for agent_name, message in zip(analyst_names, agent_messages):
                if message["role"] == "assistant":
                    debate_lines.append(
                        f"**{agent_name.capitalize()} Risk Analyst:** {message['content']}"
                    )

        debate_text = "\n".join(debate_lines)

        # Judge makes final decision
        judge_message = [
            {
                "role": "user",
                "content": trading_input.judge_user_prompt_fmt.format(
                    stock=stock_symbol, debate=debate_text
                ),
            }
        ]

        return await autogen_generate_async(judge, judge_message)

    async def _fund_manager(
        self,
        fund_manager: AssistantAgent,
        trading_input: TradingInput,
        stock_symbol: str,
        investment_recommendations: dict[str, str],
    ) -> str:
        manager_message = [
            {
                "role": "user",
                "content": trading_input.fund_manager_user_prompt_fmt.format(
                    stock=stock_symbol, **investment_recommendations
                ),
            }
        ]

        return await autogen_generate_async(fund_manager, manager_message)
