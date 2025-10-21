import asyncio
from typing import Any

from bench_programs.trading.base import OutputType, TradingInput, TradingProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.openai import (
    BATCH_SIZE,
    openai_generate_async,
    openai_iter_batch,
    prepare_openai,
)

from helium.common import Message
from helium.runtime.protocol import HeliumSystemProfile


class QueryWiseTradingProgram(TradingProgram):
    async def _run(
        self, trading_input: TradingInput
    ) -> tuple[OutputType, HeliumSystemProfile]:
        client, generation_config = prepare_openai(trading_input.generation_config)
        generation_kwargs = generation_config.openai_kwargs()
        base_url = generation_config.base_url

        # Shuffle inputs
        indices = trading_input.shuffle()

        # Prepare inputs
        inputs = []
        for original_idx, data in zip(indices, trading_input.iter_data()):
            state = {
                "index": original_idx,
                "stock_symbol": data["stock_symbol"],
                "company_profile": data["funda_company_profile"],
                "balance_sheet": data["funda_balance_sheet"],
                "income_stmt": data["funda_income_stmt"],
                "cashflow_stmt": data["funda_cashflow_stmt"],
                "insider_sentiment": data["funda_insider_sentiment"],
                "insider_transactions": (
                    None
                    if data["funda_insider_transaction"] is None
                    else data["funda_insider_transaction"]
                ),
                "stock_price": data["market_stock_price"],
                "stock_stats": data["market_stock_stat"],
                "news_chunks": data["news_combined_chunks"],
                "social_chunks": data["social_reddit_post_chunks"],
            }
            inputs.append(state)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        output_builder = self.OutputBuilder()
        for batch in openai_iter_batch(inputs, BATCH_SIZE):
            batch_outputs = await asyncio.gather(
                *[
                    self._run_single_workflow(
                        client, generation_kwargs, input_data, trading_input
                    )
                    for input_data in batch
                ]
            )
            for output in batch_outputs:
                output_builder.add(output["index"], output["final_recommendation"])
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        return output_builder.build(), system_profile

    async def _run_single_workflow(
        self,
        client: Any,
        generation_kwargs: dict[str, Any],
        input_data: dict[str, Any],
        trading_input: TradingInput,
    ) -> dict[str, Any]:
        # Stage 1: Analysts
        analyst_reports = await self._run_analysts(
            client, generation_kwargs, input_data, trading_input
        )

        # Stage 2: Researcher debate
        investment_plan = await self._run_researcher_debate(
            client, generation_kwargs, input_data, analyst_reports, trading_input
        )

        # Stage 3 & 4: Traders + Risk Analysts
        trader_recommendations = await self._run_traders_and_risk_analysts(
            client,
            generation_kwargs,
            input_data,
            analyst_reports,
            investment_plan,
            trading_input,
        )

        # Stage 5: Fund Manager
        final_recommendation = await self._run_fund_manager(
            client, generation_kwargs, input_data, trader_recommendations, trading_input
        )

        return {
            "index": input_data["index"],
            "final_recommendation": final_recommendation,
        }

    async def _run_analysts(
        self,
        client: Any,
        generation_kwargs: dict[str, Any],
        input_data: dict[str, Any],
        trading_input: TradingInput,
    ) -> dict[str, str]:
        """Run all analyst stages to produce reports"""
        analyst_reports = {}

        # Fundamentals analyst
        doc_list = [
            ("balance sheet", input_data["balance_sheet"]),
            ("income statement", input_data["income_stmt"]),
            ("cashflow statement", input_data["cashflow_stmt"]),
            ("insider sentiment", input_data["insider_sentiment"]),
        ]
        if input_data["insider_transactions"]:
            doc_list.append(
                ("insider transactions", input_data["insider_transactions"])
            )

        # Summarize documents
        doc_summaries = []
        for doc_name, doc_content in doc_list:
            formatted_template = trading_input.fundamentals_extraction_fmt.format(
                doc_name=doc_name,
                doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
            )
            messages = [
                Message(
                    role="system", content=trading_input.fundamentals_system_prompt
                ),
                Message(
                    role="user",
                    content=formatted_template.format(
                        stock=input_data["stock_symbol"],
                        doc=doc_content,
                        company_profile=input_data["company_profile"],
                    ),
                ),
            ]
            response = await openai_generate_async(client, messages, generation_kwargs)
            doc_summaries.append((doc_name.replace(" ", "_"), response))

        # Aggregate summaries
        summary_dict = dict(doc_summaries)
        messages = [
            Message(role="system", content=trading_input.fundamentals_system_prompt),
            Message(
                role="user",
                content=trading_input.fundamentals_summary_fmt.format(
                    stock=input_data["stock_symbol"],
                    company_profile=input_data["company_profile"],
                    **summary_dict,
                ),
            ),
        ]
        response = await openai_generate_async(client, messages, generation_kwargs)
        analyst_reports["fundamentals"] = response

        # Market analyst
        doc_list = [
            ("stock price data", input_data["stock_price"]),
            ("stock stats indicators", input_data["stock_stats"]),
        ]

        # Summarize documents
        doc_summaries = []
        for doc_name, doc_content in doc_list:
            formatted_template = trading_input.market_extraction_fmt.format(
                doc_name=doc_name,
                doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
            )
            messages = [
                Message(role="system", content=trading_input.market_system_prompt),
                Message(
                    role="user",
                    content=formatted_template.format(
                        stock=input_data["stock_symbol"],
                        doc=doc_content,
                    ),
                ),
            ]
            response = await openai_generate_async(client, messages, generation_kwargs)
            doc_summaries.append((doc_name.replace(" ", "_"), response))

        # Aggregate summaries
        summary_dict = dict(doc_summaries)
        messages = [
            Message(role="system", content=trading_input.market_system_prompt),
            Message(
                role="user",
                content=trading_input.market_summary_fmt.format(
                    stock=input_data["stock_symbol"],
                    **summary_dict,
                ),
            ),
        ]
        response = await openai_generate_async(client, messages, generation_kwargs)
        analyst_reports["market"] = response

        # News analyst
        doc_list = [("news chunk", chunk) for chunk in input_data["news_chunks"]]

        # Summarize documents
        doc_summaries = []
        for doc_name, doc_content in doc_list:
            formatted_template = trading_input.news_extraction_fmt.format(
                doc_name=doc_name,
                doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
            )
            messages = [
                Message(role="system", content=trading_input.news_system_prompt),
                Message(
                    role="user",
                    content=formatted_template.format(
                        stock=input_data["stock_symbol"],
                        doc=doc_content,
                    ),
                ),
            ]
            response = await openai_generate_async(client, messages, generation_kwargs)
            doc_summaries.append((f"chunk_{len(doc_summaries)}", response))

        # Aggregate summaries
        summary_dict = dict(doc_summaries)
        messages = [
            Message(role="system", content=trading_input.news_system_prompt),
            Message(
                role="user",
                content=trading_input.news_summary_fmt.format(
                    stock=input_data["stock_symbol"],
                    **summary_dict,
                ),
            ),
        ]
        response = await openai_generate_async(client, messages, generation_kwargs)
        analyst_reports["news"] = response

        # Social media analyst
        doc_list = [("post chunk", chunk) for chunk in input_data["social_chunks"]]

        # Summarize documents
        doc_summaries = []
        for doc_name, doc_content in doc_list:
            formatted_template = trading_input.social_extraction_fmt.format(
                doc_name=doc_name,
                doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
            )
            messages = [
                Message(
                    role="system", content=trading_input.social_media_system_prompt
                ),
                Message(
                    role="user",
                    content=formatted_template.format(
                        stock=input_data["stock_symbol"],
                        doc=doc_content,
                    ),
                ),
            ]
            response = await openai_generate_async(client, messages, generation_kwargs)
            doc_summaries.append((f"chunk_{len(doc_summaries)}", response))

        # Aggregate summaries
        summary_dict = dict(doc_summaries)
        messages = [
            Message(role="system", content=trading_input.social_media_system_prompt),
            Message(
                role="user",
                content=trading_input.social_summary_fmt.format(
                    stock=input_data["stock_symbol"],
                    **summary_dict,
                ),
            ),
        ]
        response = await openai_generate_async(client, messages, generation_kwargs)
        analyst_reports["social_media"] = response

        return analyst_reports

    async def _run_researcher_debate(
        self,
        client: Any,
        generation_kwargs: dict[str, Any],
        input_data: dict[str, Any],
        analyst_reports: dict[str, str],
        trading_input: TradingInput,
    ) -> str:
        """Run the researcher debate to produce investment plan"""
        # First round
        first_round_msg_content = trading_input.researcher_first_round_fmt.format(
            stock=input_data["stock_symbol"],
            fundamentals_report=analyst_reports["fundamentals"],
            market_report=analyst_reports["market"],
            news_report=analyst_reports["news"],
            social_media_report=analyst_reports["social_media"],
        )

        bull_messages = [
            Message(role="system", content=trading_input.bull_system_prompt),
            Message(role="user", content=first_round_msg_content),
        ]
        bear_messages = [
            Message(role="system", content=trading_input.bear_system_prompt),
            Message(role="user", content=first_round_msg_content),
        ]

        bull_response = await openai_generate_async(
            client, bull_messages, generation_kwargs
        )
        bear_response = await openai_generate_async(
            client, bear_messages, generation_kwargs
        )

        bull_messages.append(Message(role="assistant", content=bull_response))
        bear_messages.append(Message(role="assistant", content=bear_response))

        # Subsequent rounds
        for _ in range(1, trading_input.num_debate_rounds):
            # Bull responds to bear's last response
            bull_messages.append(
                Message(
                    role="user",
                    content=trading_input.researcher_subsequent_round_fmt.format(
                        other_response=bear_response
                    ),
                )
            )
            bull_response = await openai_generate_async(
                client, bull_messages, generation_kwargs
            )
            bull_messages.append(Message(role="assistant", content=bull_response))

            # Bear responds to bull's last response
            bear_messages.append(
                Message(
                    role="user",
                    content=trading_input.researcher_subsequent_round_fmt.format(
                        other_response=bull_response
                    ),
                )
            )
            bear_response = await openai_generate_async(
                client, bear_messages, generation_kwargs
            )
            bear_messages.append(Message(role="assistant", content=bear_response))

        # Convert debate to text
        debate_lines = []
        # Extract assistant messages from both histories - use explicit role checks
        for bull_msg, bear_msg in zip(bull_messages, bear_messages):
            if bull_msg.role == "assistant":
                debate_lines.append(f"**Bull Researcher:** {bull_msg.content}")
            if bear_msg.role == "assistant":
                debate_lines.append(f"**Bear Researcher:** {bear_msg.content}")

        debate_text = "\n".join(debate_lines)

        # Research manager summarizes
        manager_messages = [
            Message(
                role="system", content=trading_input.research_manager_system_prompt
            ),
            Message(
                role="user",
                content=trading_input.researcher_manager_user_prompt_fmt.format(
                    stock=input_data["stock_symbol"], debate=debate_text
                ),
            ),
        ]
        manager_response = await openai_generate_async(
            client, manager_messages, generation_kwargs
        )

        return manager_response

    async def _run_traders_and_risk_analysts(
        self,
        client: Any,
        generation_kwargs: dict[str, Any],
        input_data: dict[str, Any],
        analyst_reports: dict[str, str],
        investment_plan: str,
        trading_input: TradingInput,
    ) -> dict[str, str]:
        """Run traders and their corresponding risk analyst debates"""
        trader_recommendations = {}

        for trader_name, trader_prompt in trading_input.trader_system_prompts.items():
            # Trader makes decision
            trader_messages = [
                Message(role="system", content=trader_prompt),
                Message(
                    role="user",
                    content=trading_input.trader_user_prompt_fmt.format(
                        stock=input_data["stock_symbol"],
                        investment_plan=investment_plan,
                    ),
                ),
            ]
            trader_response = await openai_generate_async(
                client, trader_messages, generation_kwargs
            )
            trader_decision = trader_response

            # Risk analysts debate
            analyst_histories = {}

            # First round - all analysts respond to trader decision
            first_round_msg_content = trading_input.risk_analyst_first_round_fmt.format(
                stock=input_data["stock_symbol"],
                fundamentals_report=analyst_reports["fundamentals"],
                market_report=analyst_reports["market"],
                news_report=analyst_reports["news"],
                social_media_report=analyst_reports["social_media"],
                trader_decision=trader_decision,
            )

            analyst_prompts = {
                "risky": trading_input.risky_analyst_system_prompt,
                "safe": trading_input.safe_analyst_system_prompt,
                "neutral": trading_input.neutral_analyst_system_prompt,
            }

            for analyst_type, analyst_prompt in analyst_prompts.items():
                analyst_messages = [
                    Message(role="system", content=analyst_prompt),
                    Message(role="user", content=first_round_msg_content),
                ]
                analyst_response = await openai_generate_async(
                    client, analyst_messages, generation_kwargs
                )
                analyst_messages.append(
                    Message(role="assistant", content=analyst_response)
                )
                analyst_histories[analyst_type] = analyst_messages

            # Subsequent rounds
            for _ in range(1, trading_input.num_debate_rounds):
                # Get all analyst responses from previous round
                other_responses = {
                    "risky": analyst_histories["risky"][-1].content,
                    "safe": analyst_histories["safe"][-1].content,
                    "neutral": analyst_histories["neutral"][-1].content,
                }

                # Each analyst responds to all other analysts
                for analyst_type in ["risky", "safe", "neutral"]:
                    analyst_messages = analyst_histories[analyst_type]
                    analyst_messages.append(
                        Message(
                            role="user",
                            content=trading_input.risk_analyst_subsequent_round_fmt.format(
                                **other_responses
                            ),
                        )
                    )
                    analyst_response = await openai_generate_async(
                        client, analyst_messages, generation_kwargs
                    )
                    analyst_messages.append(
                        Message(role="assistant", content=analyst_response)
                    )

            # Convert debate to text for judge
            debate_lines = []
            risk_analysts = ["risky", "safe", "neutral"]
            all_histories = [analyst_histories[name] for name in risk_analysts]

            for agent_messages in zip(*all_histories):
                for agent_name, message in zip(risk_analysts, agent_messages):
                    if message.role == "assistant":
                        debate_lines.append(
                            f"**{agent_name.capitalize()} Risk Analyst:** {message.content}"
                        )

            debate_text = "\n".join(debate_lines)

            # Judge makes final recommendation
            judge_messages = [
                Message(role="system", content=trading_input.judge_system_prompt),
                Message(
                    role="user",
                    content=trading_input.judge_user_prompt_fmt.format(
                        stock=input_data["stock_symbol"], debate=debate_text
                    ),
                ),
            ]
            judge_response = await openai_generate_async(
                client, judge_messages, generation_kwargs
            )
            trader_recommendations[trader_name] = judge_response

        return trader_recommendations

    async def _run_fund_manager(
        self,
        client: Any,
        generation_kwargs: dict[str, Any],
        input_data: dict[str, Any],
        trader_recommendations: dict[str, str],
        trading_input: TradingInput,
    ) -> str:
        """Run fund manager to produce final recommendation"""
        manager_messages = [
            Message(role="system", content=trading_input.fund_manager_system_prompt),
            Message(
                role="user",
                content=trading_input.fund_manager_user_prompt_fmt.format(
                    stock=input_data["stock_symbol"],
                    **trader_recommendations,
                ),
            ),
        ]
        manager_response = await openai_generate_async(
            client, manager_messages, generation_kwargs
        )
        return manager_response
