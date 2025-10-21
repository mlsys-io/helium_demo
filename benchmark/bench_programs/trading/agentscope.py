from typing import cast

from bench_programs.trading.base import (
    OutputType,
    TradingData,
    TradingInput,
    TradingProgram,
)
from bench_programs.utils.agentscope import (
    AgentScopeAgent,
    FormatMsg,
    Msg,
    PlaceholderMsg,
    RpcObject,
    agentscope_call_agent,
    agentscope_reinit_from_config,
)
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class ASTradingProgram(TradingProgram):
    async def _run(
        self, trading_input: TradingInput
    ) -> tuple[OutputType, HeliumSystemProfile]:
        self.start_timer("prepare")
        llm_config = agentscope_reinit_from_config(trading_input.generation_config)
        base_url = llm_config.base_url

        # Create agents
        fundamentals_agent = AgentScopeAgent.dist(
            "fundamentals_analyst", trading_input.fundamentals_system_prompt
        )
        market_agent = AgentScopeAgent.dist(
            "market_analyst", trading_input.market_system_prompt
        )
        news_agent = AgentScopeAgent.dist(
            "news_analyst", trading_input.news_system_prompt
        )
        social_media_agent = AgentScopeAgent.dist(
            "social_media_analyst", trading_input.social_media_system_prompt
        )
        bull_researcher = AgentScopeAgent.dist(
            "bull_researcher", trading_input.bull_system_prompt
        )
        bear_researcher = AgentScopeAgent.dist(
            "bear_researcher", trading_input.bear_system_prompt
        )
        research_manager = AgentScopeAgent.dist(
            "research_manager", trading_input.research_manager_system_prompt
        )

        # Create trader agents
        trader_agents: dict[str, RpcObject] = {}
        for trader_name, trader_prompt in trading_input.trader_system_prompts.items():
            trader_agents[trader_name] = AgentScopeAgent.dist(
                f"trader_{trader_name}", trader_prompt
            )

        # Create risk analyst agents (per trader)
        risk_analyst_agents: dict[str, dict[str, RpcObject]] = {}
        for trader_name in trading_input.trader_system_prompts.keys():
            risk_analyst_agents[trader_name] = {
                "risky": AgentScopeAgent.dist(
                    f"risky_analyst_{trader_name}",
                    trading_input.risky_analyst_system_prompt,
                ),
                "safe": AgentScopeAgent.dist(
                    f"safe_analyst_{trader_name}",
                    trading_input.safe_analyst_system_prompt,
                ),
                "neutral": AgentScopeAgent.dist(
                    f"neutral_analyst_{trader_name}",
                    trading_input.neutral_analyst_system_prompt,
                ),
            }

        # Create judge agents (per trader)
        judge_agents = {}
        for trader_name in trading_input.trader_system_prompts.keys():
            judge_agents[trader_name] = AgentScopeAgent.dist(
                f"judge_{trader_name}", trading_input.judge_system_prompt
            )

        # Create fund manager
        fund_manager_agent = AgentScopeAgent.dist(
            "fund_manager", trading_input.fund_manager_system_prompt
        )

        # Collect all agents for cleanup
        all_agents = [
            fundamentals_agent,
            market_agent,
            news_agent,
            social_media_agent,
            bull_researcher,
            bear_researcher,
            research_manager,
            fund_manager_agent,
            *trader_agents.values(),
        ]

        for trader_analysts in risk_analyst_agents.values():
            all_agents.extend(trader_analysts.values())
        all_agents.extend(judge_agents.values())

        self.stop_timer()

        # Shuffle inputs
        indices = trading_input.shuffle()

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs = [
            self._run_single_workflow(
                original_idx,
                data,
                trading_input,
                fundamentals_agent,
                market_agent,
                news_agent,
                social_media_agent,
                bull_researcher,
                bear_researcher,
                research_manager,
                trader_agents,
                risk_analyst_agents,
                judge_agents,
                fund_manager_agent,
                llm_config,
            )
            for original_idx, data in zip(indices, trading_input.iter_data())
        ]
        output_builder = self.OutputBuilder()
        for index, final_recommendation in outputs:
            output_builder.add(index, final_recommendation.content)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        # Clean up agents
        AgentScopeAgent.stop_all(all_agents)

        return output_builder.build(), system_profile

    def _run_single_workflow(
        self,
        original_idx: int,
        data: TradingData,
        trading_input: TradingInput,
        fundamentals_agent: RpcObject,
        market_agent: RpcObject,
        news_agent: RpcObject,
        social_media_agent: RpcObject,
        bull_researcher: RpcObject,
        bear_researcher: RpcObject,
        research_manager: RpcObject,
        trader_agents: dict[str, RpcObject],
        risk_analyst_agents: dict[str, dict[str, RpcObject]],
        judge_agents: dict[str, RpcObject],
        fund_manager_agent: RpcObject,
        generation_config: GenerationConfig,
    ) -> tuple[int, PlaceholderMsg]:
        # Prepare input data
        stock_symbol = data["stock_symbol"]
        company_profile = data["funda_company_profile"]
        balance_sheet = data["funda_balance_sheet"]
        income_stmt = data["funda_income_stmt"]
        cashflow_stmt = data["funda_cashflow_stmt"]
        insider_sentiment = data["funda_insider_sentiment"]
        insider_transactions = (
            None
            if data["funda_insider_transaction"] is None
            else data["funda_insider_transaction"]
        )
        stock_price = data["market_stock_price"]
        stock_stats = data["market_stock_stat"]
        news_chunks = data["news_combined_chunks"]
        social_chunks = data["social_reddit_post_chunks"]

        # Stage 1: Analysts
        analyst_reports = self._run_analysts(
            stock_symbol,
            company_profile,
            balance_sheet,
            income_stmt,
            cashflow_stmt,
            insider_sentiment,
            insider_transactions,
            stock_price,
            stock_stats,
            news_chunks,
            social_chunks,
            fundamentals_agent,
            market_agent,
            news_agent,
            social_media_agent,
            trading_input,
            generation_config,
        )

        # Stage 2: Researcher Debate
        investment_plan = self._run_researcher_debate(
            stock_symbol,
            analyst_reports,
            bull_researcher,
            bear_researcher,
            research_manager,
            trading_input,
            generation_config,
        )

        # Stage 3: Traders & Analysts Chains
        trader_recommendations = self._run_traders_and_risk_analysts(
            stock_symbol,
            analyst_reports,
            investment_plan,
            trader_agents,
            risk_analyst_agents,
            judge_agents,
            trading_input,
            generation_config,
        )

        # Stage 4: Fund Manager
        final_recommendation = self._run_fund_manager(
            stock_symbol,
            trader_recommendations,
            fund_manager_agent,
            trading_input,
            generation_config,
        )

        return original_idx, final_recommendation

    def _run_analysts(
        self,
        stock_symbol: str,
        company_profile: str,
        balance_sheet: str,
        income_stmt: str,
        cashflow_stmt: str,
        insider_sentiment: str,
        insider_transactions: str | None,
        stock_price: str,
        stock_stats: str,
        news_chunks: list[str],
        social_chunks: list[str],
        fundamentals_agent: RpcObject,
        market_agent: RpcObject,
        news_agent: RpcObject,
        social_media_agent: RpcObject,
        trading_input: TradingInput,
        generation_config: GenerationConfig,
    ) -> dict[str, PlaceholderMsg]:
        """Run all analyst stages to produce reports"""
        analyst_reports = {}

        # Fundamentals analyst
        doc_list = [
            ("balance sheet", balance_sheet),
            ("income statement", income_stmt),
            ("cashflow statement", cashflow_stmt),
            ("insider sentiment", insider_sentiment),
        ]
        if insider_transactions is not None:
            doc_list.append(("insider transactions", insider_transactions))
        # Extract documents
        doc_summaries: dict[str, PlaceholderMsg] = {}
        for doc_name, doc_content in doc_list:
            formatted_template = trading_input.fundamentals_extraction_fmt.format(
                doc_name=doc_name,
                doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
            )
            extraction_message = [
                Msg(
                    "user",
                    formatted_template.format(
                        stock=stock_symbol,
                        doc=doc_content,
                        company_profile=company_profile,
                    ),
                    "user",
                )
            ]
            doc_summaries[doc_name.replace(" ", "_")] = agentscope_call_agent(
                fundamentals_agent, extraction_message, generation_config
            )
        # Aggregate summaries
        summary_message = [
            FormatMsg(
                "user",
                trading_input.fundamentals_summary_fmt,
                stock=stock_symbol,
                company_profile=company_profile,
                **doc_summaries,
            )
        ]
        analyst_reports["fundamentals"] = agentscope_call_agent(
            fundamentals_agent, summary_message, generation_config
        )

        # Market analyst
        doc_list = [
            ("stock price data", stock_price),
            ("stock stats indicators", stock_stats),
        ]
        # Extract documents
        doc_summaries = {}
        for doc_name, doc_content in doc_list:
            formatted_template = trading_input.market_extraction_fmt.format(
                doc_name=doc_name,
                doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
            )
            extraction_message = [
                Msg(
                    "user",
                    formatted_template.format(
                        stock=stock_symbol,
                        doc=doc_content,
                    ),
                    "user",
                )
            ]
            doc_summaries[doc_name.replace(" ", "_")] = agentscope_call_agent(
                market_agent, extraction_message, generation_config
            )
        # Aggregate summaries
        summary_message = [
            FormatMsg(
                "user",
                trading_input.market_summary_fmt,
                stock=stock_symbol,
                **doc_summaries,
            )
        ]
        analyst_reports["market"] = agentscope_call_agent(
            market_agent, summary_message, generation_config
        )

        # News analyst
        # Extract documents
        doc_summaries = {}
        for i, chunk in enumerate(news_chunks):
            formatted_template = trading_input.news_extraction_fmt.format(
                doc_name="news chunk",
                doc_desc=trading_input.doc_descriptions.get("news chunk", ""),
            )
            extraction_message = [
                Msg(
                    "user",
                    formatted_template.format(
                        stock=stock_symbol,
                        doc=chunk,
                    ),
                    "user",
                )
            ]
            doc_summaries[f"chunk_{i}"] = agentscope_call_agent(
                news_agent, extraction_message, generation_config
            )
        # Aggregate summaries
        summary_message = [
            FormatMsg(
                "user",
                trading_input.news_summary_fmt,
                stock=stock_symbol,
                **doc_summaries,
            )
        ]
        analyst_reports["news"] = agentscope_call_agent(
            news_agent, summary_message, generation_config
        )

        # Social media analyst
        # Extract documents
        doc_summaries = {}
        for i, chunk in enumerate(social_chunks):
            formatted_template = trading_input.social_extraction_fmt.format(
                doc_name="post chunk",
                doc_desc=trading_input.doc_descriptions.get("post chunk", ""),
            )
            extraction_message = [
                Msg(
                    "user",
                    formatted_template.format(
                        stock=stock_symbol,
                        doc=chunk,
                    ),
                    "user",
                )
            ]
            doc_summaries[f"chunk_{i}"] = agentscope_call_agent(
                social_media_agent, extraction_message, generation_config
            )
        # Aggregate summaries
        summary_message = [
            FormatMsg(
                "user",
                trading_input.social_summary_fmt,
                stock=stock_symbol,
                **doc_summaries,
            )
        ]
        analyst_reports["social_media"] = agentscope_call_agent(
            social_media_agent, summary_message, generation_config
        )

        return analyst_reports

    def _run_researcher_debate(
        self,
        stock_symbol: str,
        analyst_reports: dict[str, PlaceholderMsg],
        bull_researcher: RpcObject,
        bear_researcher: RpcObject,
        research_manager: RpcObject,
        trading_input: TradingInput,
        generation_config: GenerationConfig,
    ) -> PlaceholderMsg:
        """Run the researcher debate to produce investment plan"""
        # First rounds
        first_round_message = [
            FormatMsg(
                "user",
                trading_input.researcher_first_round_fmt,
                stock=stock_symbol,
                fundamentals_report=analyst_reports["fundamentals"],
                market_report=analyst_reports["market"],
                news_report=analyst_reports["news"],
                social_media_report=analyst_reports["social_media"],
            )
        ]

        bull_response = agentscope_call_agent(
            bull_researcher, first_round_message, generation_config
        )
        bear_response = agentscope_call_agent(
            bear_researcher, first_round_message, generation_config
        )

        bull_history: list[Msg | FormatMsg | PlaceholderMsg] = [
            first_round_message[0],
            bull_response,
        ]
        bear_history: list[Msg | FormatMsg | PlaceholderMsg] = [
            first_round_message[0],
            bear_response,
        ]
        # Subsequent rounds
        for _ in range(1, trading_input.num_debate_rounds):
            bull_msg = FormatMsg(
                "user",
                trading_input.researcher_subsequent_round_fmt,
                other_response=bear_response,
            )
            bull_response = agentscope_call_agent(
                bull_researcher, bull_history + [bull_msg], generation_config
            )
            bull_history.extend([bull_msg, bull_response])

            bear_msg = FormatMsg(
                "user",
                trading_input.researcher_subsequent_round_fmt,
                other_response=bull_response,
            )
            bear_response = agentscope_call_agent(
                bear_researcher, bear_history + [bear_msg], generation_config
            )
            bear_history.extend([bear_msg, bear_response])

        # Extract assistant responses (chronological per side)
        bull_responses = [
            msg for msg in bull_history if isinstance(msg, PlaceholderMsg)
        ]
        bear_responses = [
            msg for msg in bear_history if isinstance(msg, PlaceholderMsg)
        ]

        # Interleave responses per round for manager
        debate_lines: list[str] = []
        interleaved: list[PlaceholderMsg] = []
        for bull_resp, bear_resp in zip(bull_responses, bear_responses):
            debate_lines.append("**Bull Researcher:** {}")
            debate_lines.append("**Bear Researcher:** {}")
            interleaved.extend([bull_resp, bear_resp])
        debate_format = "\n".join(debate_lines)

        manager_message = [
            FormatMsg(
                "user",
                trading_input.researcher_manager_user_prompt_fmt.format(
                    stock="{}", debate=debate_format
                ),
                stock_symbol,
                *interleaved,
            )
        ]
        investment_plan = agentscope_call_agent(
            research_manager, manager_message, generation_config
        )
        return investment_plan

    def _run_traders_and_risk_analysts(
        self,
        stock_symbol: str,
        analyst_reports: dict[str, PlaceholderMsg],
        investment_plan: PlaceholderMsg,
        trader_agents: dict[str, RpcObject],
        risk_analyst_agents: dict[str, dict[str, RpcObject]],
        judge_agents: dict[str, RpcObject],
        trading_input: TradingInput,
        generation_config: GenerationConfig,
    ) -> dict[str, PlaceholderMsg]:
        """Run traders and their corresponding risk analyst debates"""
        trader_recommendations = {}

        for trader_name, trader_agent in trader_agents.items():
            # Trader makes decision
            trader_message = [
                FormatMsg(
                    "user",
                    trading_input.trader_user_prompt_fmt,
                    stock=stock_symbol,
                    investment_plan=investment_plan,
                )
            ]
            trader_decision = agentscope_call_agent(
                trader_agent, trader_message, generation_config
            )

            # First rounds
            analyst_types = ["risky", "safe", "neutral"]
            analyst_histories: dict[str, list[Msg | FormatMsg | PlaceholderMsg]] = {}
            first_round_message = FormatMsg(
                "user",
                trading_input.risk_analyst_first_round_fmt,
                stock=stock_symbol,
                fundamentals_report=analyst_reports["fundamentals"],
                market_report=analyst_reports["market"],
                news_report=analyst_reports["news"],
                social_media_report=analyst_reports["social_media"],
                trader_decision=trader_decision,
            )
            for analyst_type in analyst_types:
                analyst_response = agentscope_call_agent(
                    risk_analyst_agents[trader_name][analyst_type],
                    [first_round_message],
                    generation_config,
                )
                analyst_histories[analyst_type] = [
                    first_round_message,
                    analyst_response,
                ]

            # Subsequent rounds
            for _ in range(1, trading_input.num_debate_rounds):
                for analyst_type in analyst_types:
                    analyst_msg = FormatMsg(
                        "user",
                        trading_input.risk_analyst_subsequent_round_fmt,
                        risky=cast(PlaceholderMsg, analyst_histories["risky"][-1]),
                        safe=cast(PlaceholderMsg, analyst_histories["safe"][-1]),
                        neutral=cast(PlaceholderMsg, analyst_histories["neutral"][-1]),
                    )
                    analyst_response = agentscope_call_agent(
                        risk_analyst_agents[trader_name][analyst_type],
                        analyst_histories[analyst_type] + [analyst_msg],
                        generation_config,
                    )
                    analyst_histories[analyst_type].extend(
                        [analyst_msg, analyst_response]
                    )

            # Collect assistant responses per analyst preserving round order
            risky_resps = [
                m for m in analyst_histories["risky"] if isinstance(m, PlaceholderMsg)
            ]
            safe_resps = [
                m for m in analyst_histories["safe"] if isinstance(m, PlaceholderMsg)
            ]
            neutral_resps = [
                m for m in analyst_histories["neutral"] if isinstance(m, PlaceholderMsg)
            ]
            debate_lines: list[str] = []
            interleaved: list[PlaceholderMsg] = []
            for risky_resp, safe_resp, neutral_resp in zip(
                risky_resps, safe_resps, neutral_resps
            ):
                debate_lines.append("**Risky Risk Analyst:** {}")
                debate_lines.append("**Safe Risk Analyst:** {}")
                debate_lines.append("**Neutral Risk Analyst:** {}")
                interleaved.extend([risky_resp, safe_resp, neutral_resp])
            debate_format = "\n".join(debate_lines)

            judge_message = [
                FormatMsg(
                    "user",
                    trading_input.judge_user_prompt_fmt.format(
                        stock="{}", debate=debate_format
                    ),
                    stock_symbol,
                    *interleaved,
                )
            ]
            trader_recommendations[trader_name] = agentscope_call_agent(
                judge_agents[trader_name], judge_message, generation_config
            )

        return trader_recommendations

    def _run_fund_manager(
        self,
        stock_symbol: str,
        trader_recommendations: dict[str, PlaceholderMsg],
        fund_manager_agent: RpcObject,
        trading_input: TradingInput,
        generation_config: GenerationConfig,
    ) -> PlaceholderMsg:
        """Run fund manager to produce final recommendation"""
        manager_message = [
            FormatMsg(
                "user",
                trading_input.fund_manager_user_prompt_fmt,
                stock=stock_symbol,
                **trader_recommendations,
            )
        ]
        final_recommendation = agentscope_call_agent(
            fund_manager_agent, manager_message, generation_config
        )
        return final_recommendation
