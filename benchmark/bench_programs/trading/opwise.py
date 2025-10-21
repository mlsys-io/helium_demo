from typing import Any

from bench_programs.trading.base import OutputType, TradingInput, TradingProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.openai import openai_generate_async, prepare_openai
from bench_programs.utils.opwise import WorkflowDAG

from helium.common import Message
from helium.runtime.protocol import HeliumSystemProfile


class OpWiseTradingProgram(TradingProgram):
    async def _run(
        self, trading_input: TradingInput
    ) -> tuple[OutputType, HeliumSystemProfile]:
        client, generation_config = prepare_openai(trading_input.generation_config)
        generation_kwargs = generation_config.openai_kwargs()
        base_url = generation_config.base_url

        indices = trading_input.shuffle()

        inputs: list[dict[str, Any]] = []
        # Assume uniform chunk counts across batch (as per user instruction)
        for original_idx, data in zip(indices, trading_input.iter_data()):
            news_chunks = data["news_combined_chunks"]
            social_chunks = data["social_reddit_post_chunks"]
            inputs.append(
                {
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
                    "news_chunks": news_chunks,
                    "social_chunks": social_chunks,
                    # Intermediate stores
                    "fundamentals_extractions": {},
                    "market_extractions": {},
                    "news_extractions": {},
                    "social_extractions": {},
                    # Stage outputs
                    "fundamentals_report": None,
                    "market_report": None,
                    "news_report": None,
                    "social_media_report": None,
                    "bull_history": [],
                    "bear_history": [],
                    "investment_plan": None,
                    "trader_decisions": {},
                    "risk_histories": {},  # (trader, role) -> history list[Message]
                    "trader_recommendations": {},
                    "final_recommendation": None,
                }
            )

        workflow = self._build_workflow(client, generation_kwargs, trading_input)

        try_start_benchmark(base_url)

        self.start_timer("generate")
        states = await workflow.execute(inputs)
        self.stop_timer()

        system_profile = try_stop_benchmark(base_url)

        output_builder = self.OutputBuilder()
        for state in states:
            output_builder.add(state["index"], state["final_recommendation"])  # type: ignore[arg-type]

        return output_builder.build(), system_profile

    def _build_workflow(
        self,
        client: Any,
        generation_kwargs: dict[str, Any],
        trading_input: TradingInput,
    ) -> WorkflowDAG:
        wf = WorkflowDAG()

        # Fundamentals Analyst nodes
        funda_docs = [
            ("balance_sheet", "balance sheet"),
            ("income_stmt", "income statement"),
            ("cashflow_stmt", "cashflow statement"),
            ("insider_sentiment", "insider sentiment"),
        ]
        if trading_input.funda_insider_transactions is not None:
            funda_docs.append(("insider_transactions", "insider transactions"))

        def make_funda_extract(attr: str, doc_name: str):
            async def node(state: dict) -> dict:
                template = trading_input.fundamentals_extraction_fmt.format(
                    doc_name=doc_name,
                    doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
                )
                messages = [
                    Message("system", trading_input.fundamentals_system_prompt),
                    Message(
                        "user",
                        template.format(
                            stock=state["stock_symbol"],
                            doc=state[attr],
                            company_profile=state["company_profile"],
                        ),
                    ),
                ]
                resp = await openai_generate_async(client, messages, generation_kwargs)
                key = doc_name.replace(" ", "_")
                state["fundamentals_extractions"][key] = resp
                return state

            return node

        # Summary node
        async def fundamentals_summary(state: dict) -> dict:
            messages = [
                Message("system", trading_input.fundamentals_system_prompt),
                Message(
                    "user",
                    trading_input.fundamentals_summary_fmt.format(
                        stock=state["stock_symbol"],
                        company_profile=state["company_profile"],
                        **state["fundamentals_extractions"],
                    ),
                ),
            ]
            state["fundamentals_report"] = await openai_generate_async(
                client, messages, generation_kwargs
            )
            return state

        # Market Analyst nodes
        market_docs = [
            ("stock_price", "stock price data"),
            ("stock_stats", "stock stats indicators"),
        ]

        def make_market_extract(attr: str, doc_name: str):
            async def node(state: dict) -> dict:
                template = trading_input.market_extraction_fmt.format(
                    doc_name=doc_name,
                    doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
                )
                messages = [
                    Message("system", trading_input.market_system_prompt),
                    Message(
                        "user",
                        template.format(
                            stock=state["stock_symbol"],
                            doc=state[attr],
                        ),
                    ),
                ]
                resp = await openai_generate_async(client, messages, generation_kwargs)
                key = doc_name.replace(" ", "_")
                state["market_extractions"][key] = resp
                return state

            return node

        async def market_summary(state: dict) -> dict:
            messages = [
                Message("system", trading_input.market_system_prompt),
                Message(
                    "user",
                    trading_input.market_summary_fmt.format(
                        stock=state["stock_symbol"],
                        **state["market_extractions"],
                    ),
                ),
            ]
            state["market_report"] = await openai_generate_async(
                client, messages, generation_kwargs
            )
            return state

        # News Analyst nodes
        def make_news_extract(idx: int):
            async def node(state: dict) -> dict:
                if idx >= len(state["news_chunks"]):
                    return state
                chunk = state["news_chunks"][idx]
                template = trading_input.news_extraction_fmt.format(
                    doc_name="news chunk",
                    doc_desc=trading_input.doc_descriptions.get("news chunk", ""),
                )
                messages = [
                    Message("system", trading_input.news_system_prompt),
                    Message(
                        "user",
                        template.format(
                            stock=state["stock_symbol"],
                            doc=chunk,
                        ),
                    ),
                ]
                resp = await openai_generate_async(client, messages, generation_kwargs)
                state["news_extractions"][f"chunk_{idx}"] = resp
                return state

            return node

        async def news_summary(state: dict) -> dict:
            messages = [
                Message("system", trading_input.news_system_prompt),
                Message(
                    "user",
                    trading_input.news_summary_fmt.format(
                        stock=state["stock_symbol"],
                        **state["news_extractions"],
                    ),
                ),
            ]
            state["news_report"] = await openai_generate_async(
                client, messages, generation_kwargs
            )
            return state

        # Social Analyst nodes
        def make_social_extract(idx: int):
            async def node(state: dict) -> dict:
                if idx >= len(state["social_chunks"]):
                    return state
                chunk = state["social_chunks"][idx]
                template = trading_input.social_extraction_fmt.format(
                    doc_name="post chunk",
                    doc_desc=trading_input.doc_descriptions.get("post chunk", ""),
                )
                messages = [
                    Message("system", trading_input.social_media_system_prompt),
                    Message(
                        "user",
                        template.format(
                            stock=state["stock_symbol"],
                            doc=chunk,
                        ),
                    ),
                ]
                resp = await openai_generate_async(client, messages, generation_kwargs)
                state["social_extractions"][f"chunk_{idx}"] = resp
                return state

            return node

        async def social_summary(state: dict) -> dict:
            messages = [
                Message("system", trading_input.social_media_system_prompt),
                Message(
                    "user",
                    trading_input.social_summary_fmt.format(
                        stock=state["stock_symbol"],
                        **state["social_extractions"],
                    ),
                ),
            ]
            state["social_media_report"] = await openai_generate_async(
                client, messages, generation_kwargs
            )
            return state

        # Researcher Debate nodes
        async def bull_first(state: dict) -> dict:
            content = trading_input.researcher_first_round_fmt.format(
                stock=state["stock_symbol"],
                fundamentals_report=state["fundamentals_report"],
                market_report=state["market_report"],
                news_report=state["news_report"],
                social_media_report=state["social_media_report"],
            )
            messages = [
                Message("system", trading_input.bull_system_prompt),
                Message("user", content),
            ]
            resp = await openai_generate_async(client, messages, generation_kwargs)
            state["bull_history"] = messages + [Message("assistant", resp)]
            return state

        async def bear_first(state: dict) -> dict:
            content = trading_input.researcher_first_round_fmt.format(
                stock=state["stock_symbol"],
                fundamentals_report=state["fundamentals_report"],
                market_report=state["market_report"],
                news_report=state["news_report"],
                social_media_report=state["social_media_report"],
            )
            messages = [
                Message("system", trading_input.bear_system_prompt),
                Message("user", content),
            ]
            resp = await openai_generate_async(client, messages, generation_kwargs)
            state["bear_history"] = messages + [Message("assistant", resp)]
            return state

        def make_bull_round():
            async def node(state: dict) -> dict:
                other_resp = state["bear_history"][-1].content
                state["bull_history"].append(
                    Message(
                        "user",
                        trading_input.researcher_subsequent_round_fmt.format(
                            other_response=other_resp
                        ),
                    )
                )
                resp = await openai_generate_async(
                    client, state["bull_history"], generation_kwargs
                )
                state["bull_history"].append(Message("assistant", resp))
                return state

            return node

        def make_bear_round():
            async def node(state: dict) -> dict:
                other_resp = state["bull_history"][-1].content
                state["bear_history"].append(
                    Message(
                        "user",
                        trading_input.researcher_subsequent_round_fmt.format(
                            other_response=other_resp
                        ),
                    )
                )
                resp = await openai_generate_async(
                    client, state["bear_history"], generation_kwargs
                )
                state["bear_history"].append(Message("assistant", resp))
                return state

            return node

        async def research_manager(state: dict) -> dict:
            lines: list[str] = []
            for bull_msg, bear_msg in zip(state["bull_history"], state["bear_history"]):
                if bull_msg.role == "assistant":
                    lines.append(f"**Bull Researcher:** {bull_msg.content}")
                if bear_msg.role == "assistant":
                    lines.append(f"**Bear Researcher:** {bear_msg.content}")
            debate = "\n".join(lines)
            messages = [
                Message("system", trading_input.research_manager_system_prompt),
                Message(
                    "user",
                    trading_input.researcher_manager_user_prompt_fmt.format(
                        stock=state["stock_symbol"], debate=debate
                    ),
                ),
            ]
            state["investment_plan"] = await openai_generate_async(
                client, messages, generation_kwargs
            )
            return state

        # Trader & Risk Analyst chains
        def make_trader_decision(trader_name: str, system_prompt: str):
            async def node(state: dict) -> dict:
                messages = [
                    Message("system", system_prompt),
                    Message(
                        "user",
                        trading_input.trader_user_prompt_fmt.format(
                            stock=state["stock_symbol"],
                            investment_plan=state["investment_plan"],
                        ),
                    ),
                ]
                resp = await openai_generate_async(client, messages, generation_kwargs)
                state["trader_decisions"][trader_name] = resp
                return state

            return node

        def make_risk_first(trader_name: str, role: str, system_prompt: str):
            async def node(state: dict) -> dict:
                decision = state["trader_decisions"][trader_name]
                content = trading_input.risk_analyst_first_round_fmt.format(
                    stock=state["stock_symbol"],
                    fundamentals_report=state["fundamentals_report"],
                    market_report=state["market_report"],
                    news_report=state["news_report"],
                    social_media_report=state["social_media_report"],
                    trader_decision=decision,
                )
                messages = [
                    Message("system", system_prompt),
                    Message("user", content),
                ]
                resp = await openai_generate_async(client, messages, generation_kwargs)
                state["risk_histories"][(trader_name, role)] = messages + [
                    Message("assistant", resp)
                ]
                return state

            return node

        def make_risk_round(trader_name: str, role: str):
            async def node(state: dict) -> dict:
                roles = ["risky", "safe", "neutral"]
                prev = {
                    r: state["risk_histories"][(trader_name, r)][-1].content
                    for r in roles
                }
                history = state["risk_histories"][(trader_name, role)]
                history.append(
                    Message(
                        "user",
                        trading_input.risk_analyst_subsequent_round_fmt.format(**prev),
                    )
                )
                resp = await openai_generate_async(client, history, generation_kwargs)
                history.append(Message("assistant", resp))
                return state

            return node

        def make_risk_judge(trader_name: str):
            async def node(state: dict) -> dict:
                roles = ["risky", "safe", "neutral"]
                lines: list[str] = []
                grouped = list(
                    zip(*[state["risk_histories"][(trader_name, r)] for r in roles])
                )
                for group in grouped:
                    for role, message in zip(roles, group):
                        if message.role == "assistant":
                            lines.append(
                                f"**{role.capitalize()} Risk Analyst:** {message.content}"
                            )
                debate = "\n".join(lines)
                messages = [
                    Message("system", trading_input.judge_system_prompt),
                    Message(
                        "user",
                        trading_input.judge_user_prompt_fmt.format(
                            stock=state["stock_symbol"], debate=debate
                        ),
                    ),
                ]
                resp = await openai_generate_async(client, messages, generation_kwargs)
                state["trader_recommendations"][trader_name] = resp
                return state

            return node

        async def fund_manager(state: dict) -> dict:
            messages = [
                Message("system", trading_input.fund_manager_system_prompt),
                Message(
                    "user",
                    trading_input.fund_manager_user_prompt_fmt.format(
                        stock=state["stock_symbol"],
                        **state["trader_recommendations"],
                    ),
                ),
            ]
            state["final_recommendation"] = await openai_generate_async(
                client, messages, generation_kwargs
            )
            return state

        # Assemble the DAG
        previous: str | None = None
        # Fundamentals
        for attr, doc_name in funda_docs:
            node_name = f"funda_extract_{attr}"
            wf.add_node(node_name, make_funda_extract(attr, doc_name))
            if previous is not None:
                wf.add_edge(previous, node_name)
            previous = node_name
        wf.add_node("funda_summary", fundamentals_summary)
        if previous is not None:
            wf.add_edge(previous, "funda_summary")
        previous = "funda_summary"
        # Market
        for attr, doc_name in market_docs:
            node_name = f"market_extract_{attr}"
            wf.add_node(node_name, make_market_extract(attr, doc_name))
            wf.add_edge(previous, node_name)
            previous = node_name
        wf.add_node("market_summary", market_summary)
        wf.add_edge(previous, "market_summary")
        previous = "market_summary"
        # News chunks
        for idx in range(trading_input.num_news_chunks):
            node_name = f"news_extract_{idx}"
            wf.add_node(node_name, make_news_extract(idx))
            wf.add_edge(previous, node_name)
            previous = node_name
        wf.add_node("news_summary", news_summary)
        wf.add_edge(previous, "news_summary")
        previous = "news_summary"
        # Social chunks
        for idx in range(trading_input.num_social_chunks):
            node_name = f"social_extract_{idx}"
            wf.add_node(node_name, make_social_extract(idx))
            wf.add_edge(previous, node_name)
            previous = node_name
        wf.add_node("social_summary", social_summary)
        wf.add_edge(previous, "social_summary")
        previous = "social_summary"
        # Researchers
        wf.add_node("bull_first", bull_first)
        wf.add_edge(previous, "bull_first")
        wf.add_node("bear_first", bear_first)
        wf.add_edge("bull_first", "bear_first")
        previous = "bear_first"
        for round_idx in range(1, trading_input.num_debate_rounds):
            bull_node = f"bull_round_{round_idx}"
            bear_node = f"bear_round_{round_idx}"
            wf.add_node(bull_node, make_bull_round())
            wf.add_edge(previous, bull_node)
            wf.add_node(bear_node, make_bear_round())
            wf.add_edge(bull_node, bear_node)
            previous = bear_node
        wf.add_node("research_manager", research_manager)
        wf.add_edge(previous, "research_manager")
        previous = "research_manager"
        # Traders & risk analysts
        for trader_name, system_prompt in trading_input.trader_system_prompts.items():
            decision_node = f"trader_decision_{trader_name}"
            wf.add_node(decision_node, make_trader_decision(trader_name, system_prompt))
            wf.add_edge(previous, decision_node)
            previous = decision_node
            # risk first round
            for role, prompt in [
                ("risky", trading_input.risky_analyst_system_prompt),
                ("safe", trading_input.safe_analyst_system_prompt),
                ("neutral", trading_input.neutral_analyst_system_prompt),
            ]:
                node = f"risk_{trader_name}_{role}_first"
                wf.add_node(node, make_risk_first(trader_name, role, prompt))
                wf.add_edge(previous, node)
                previous = node
            for round_idx in range(1, trading_input.num_debate_rounds):
                for role in ["risky", "safe", "neutral"]:
                    node = f"risk_{trader_name}_{role}_round_{round_idx}"
                    wf.add_node(node, make_risk_round(trader_name, role))
                    wf.add_edge(previous, node)
                    previous = node
            judge_node = f"judge_{trader_name}"
            wf.add_node(judge_node, make_risk_judge(trader_name))
            wf.add_edge(previous, judge_node)
            previous = judge_node
        # Fund manager
        wf.add_node("fund_manager", fund_manager)
        wf.add_edge(previous, "fund_manager")

        return wf
