from typing import Annotated, Any, Literal, TypedDict, TypeVar

from bench_programs.trading.base import OutputType, TradingInput, TradingProgram
from bench_programs.utils.common import try_start_benchmark, try_stop_benchmark
from bench_programs.utils.langgraph import get_langgraph_openai_client
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile

T = TypeVar("T")

SCHEDULING_METHOD: Literal["random", "query-wise"] = "random"


def update_dict(
    left: dict[str, T], right: tuple[str, T] | dict[str, T]
) -> dict[str, T]:
    if isinstance(right, dict):
        return right
    key, value = right
    left[key] = value
    return left


class TradingWorkflowState(TypedDict):
    # Metadata
    index: int
    stock_symbol: str

    # Input data
    company_profile: str
    balance_sheet: str
    income_stmt: str
    cashflow_stmt: str
    insider_sentiment: str
    insider_transactions: str | None
    stock_price: str
    stock_stats: str
    news_chunks: list[str]
    social_chunks: list[str]

    # Extraction accumulators (Stage 1 intermediate)
    fundamentals_extractions: Annotated[dict[str, str], update_dict]
    market_extractions: Annotated[dict[str, str], update_dict]
    news_extractions: Annotated[dict[str, str], update_dict]
    social_extractions: Annotated[dict[str, str], update_dict]

    # Stage 1: Analyst reports (final summaries)
    analyst_reports: Annotated[dict[str, str], update_dict]

    # Stage 2: Researcher debate
    investment_plan: str
    researcher_histories: Annotated[dict[str, list[BaseMessage]], update_dict]

    # Stage 3: Trader decisions (separate from final recommendations)
    trader_decisions: Annotated[dict[str, str], update_dict]

    # Stage 4: Risk analyst debates and final recommendations
    trader_recommendations: Annotated[dict[str, str], update_dict]

    # Stage 5: Risk analyst histories (optional, for debugging)
    risk_analyst_histories: Annotated[dict[str, list[BaseMessage]], update_dict]

    # Final result
    final_recommendation: str


class LangGraphTradingProgram(TradingProgram):
    async def _run(
        self, trading_input: TradingInput
    ) -> tuple[OutputType, HeliumSystemProfile]:
        generation_config = (
            GenerationConfig.from_env()
            if trading_input.generation_config is None
            else trading_input.generation_config
        )

        base_url = generation_config.base_url

        workflow = self._build_workflow(trading_input, generation_config).compile()

        # Prepare inputs
        indices = trading_input.shuffle()
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
                # new extraction stores
                "fundamentals_extractions": {},
                "market_extractions": {},
                "news_extractions": {},
                "social_extractions": {},
                # summaries
                "analyst_reports": {},
                "investment_plan": "",
                "researcher_histories": {},
                "trader_decisions": {},
                "trader_recommendations": {},
                "risk_analyst_histories": {},
                "final_recommendation": "",
            }
            inputs.append(state)

        # Start benchmarking
        try_start_benchmark(base_url)

        self.start_timer("generate")
        outputs: list[dict[str, Any]]
        if SCHEDULING_METHOD == "random":
            outputs = await workflow.abatch(inputs)
        else:
            outputs = []
            for inp in inputs:
                out = await workflow.ainvoke(inp)
                outputs.append(out)
        self.stop_timer()

        # Stop benchmarking
        system_profile = try_stop_benchmark(base_url)

        output_builder = self.OutputBuilder()
        for output in outputs:
            output_builder.add(output["index"], output["final_recommendation"])

        return output_builder.build(), system_profile

    def _build_workflow(
        self, trading_input: TradingInput, generation_config: GenerationConfig
    ) -> StateGraph:
        llm = get_langgraph_openai_client(generation_config)

        # Fundamentals Analyst nodes
        def make_fundamentals_extraction(attr: str, doc_name: str):
            async def node(state: TradingWorkflowState) -> dict[str, Any]:
                template = trading_input.fundamentals_extraction_fmt.format(
                    doc_name=doc_name,
                    doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
                )
                user_content = template.format(
                    stock=state["stock_symbol"],
                    doc=state[attr],  # type: ignore
                    company_profile=state["company_profile"],
                )
                msgs = [
                    SystemMessage(trading_input.fundamentals_system_prompt),
                    HumanMessage(user_content),
                ]
                resp = await llm.ainvoke(msgs)
                content = (
                    resp.content if isinstance(resp.content, str) else str(resp.content)
                )
                key = doc_name.replace(" ", "_")
                return {"fundamentals_extractions": (key, content)}

            return node

        async def fundamentals_summary(state: TradingWorkflowState) -> dict[str, Any]:
            summary_user = trading_input.fundamentals_summary_fmt.format(
                stock=state["stock_symbol"],
                company_profile=state["company_profile"],
                **state["fundamentals_extractions"],
            )
            msgs = [
                SystemMessage(trading_input.fundamentals_system_prompt),
                HumanMessage(summary_user),
            ]
            resp = await llm.ainvoke(msgs)
            return {"analyst_reports": ("fundamentals", resp.content)}

        # Market Analyst nodes
        def make_market_extraction(attr: str, doc_name: str):
            async def node(state: TradingWorkflowState) -> dict[str, Any]:
                template = trading_input.market_extraction_fmt.format(
                    doc_name=doc_name,
                    doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
                )
                user_content = template.format(
                    stock=state["stock_symbol"], doc=state[attr]  # type: ignore
                )
                msgs = [
                    SystemMessage(trading_input.market_system_prompt),
                    HumanMessage(user_content),
                ]
                resp = await llm.ainvoke(msgs)
                key = doc_name.replace(" ", "_")
                return {
                    "market_extractions": (
                        key,
                        (
                            resp.content
                            if isinstance(resp.content, str)
                            else str(resp.content)
                        ),
                    )
                }

            return node

        async def market_summary(state: TradingWorkflowState) -> dict[str, Any]:
            user_content = trading_input.market_summary_fmt.format(
                stock=state["stock_symbol"], **state["market_extractions"]
            )
            msgs = [
                SystemMessage(trading_input.market_system_prompt),
                HumanMessage(user_content),
            ]
            resp = await llm.ainvoke(msgs)
            return {"analyst_reports": ("market", resp.content)}

        # News Analyst nodes
        def make_news_extraction(idx: int):
            async def node(state: TradingWorkflowState) -> dict[str, Any]:
                if idx >= len(state["news_chunks"]):
                    return {"news_extractions": {}}
                doc_name = "news chunk"
                template = trading_input.news_extraction_fmt.format(
                    doc_name=doc_name,
                    doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
                )
                user_content = template.format(
                    stock=state["stock_symbol"], doc=state["news_chunks"][idx]
                )
                msgs = [
                    SystemMessage(trading_input.news_system_prompt),
                    HumanMessage(user_content),
                ]
                resp = await llm.ainvoke(msgs)
                return {
                    "news_extractions": (
                        f"chunk_{idx}",
                        (
                            resp.content
                            if isinstance(resp.content, str)
                            else str(resp.content)
                        ),
                    )
                }

            return node

        async def news_summary(state: TradingWorkflowState) -> dict[str, Any]:
            user_content = trading_input.news_summary_fmt.format(
                stock=state["stock_symbol"], **state["news_extractions"]
            )
            msgs = [
                SystemMessage(trading_input.news_system_prompt),
                HumanMessage(user_content),
            ]
            resp = await llm.ainvoke(msgs)
            return {"analyst_reports": ("news", resp.content)}

        # Social Analyst nodes
        def make_social_extraction(idx: int):
            async def node(state: TradingWorkflowState) -> dict[str, Any]:
                if idx >= len(state["social_chunks"]):
                    return {"social_extractions": {}}
                doc_name = "post chunk"
                template = trading_input.social_extraction_fmt.format(
                    doc_name=doc_name,
                    doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
                )
                user_content = template.format(
                    stock=state["stock_symbol"], doc=state["social_chunks"][idx]
                )
                msgs = [
                    SystemMessage(trading_input.social_media_system_prompt),
                    HumanMessage(user_content),
                ]
                resp = await llm.ainvoke(msgs)
                return {
                    "social_extractions": (
                        f"chunk_{idx}",
                        (
                            resp.content
                            if isinstance(resp.content, str)
                            else str(resp.content)
                        ),
                    )
                }

            return node

        async def social_summary(state: TradingWorkflowState) -> dict[str, Any]:
            user_content = trading_input.social_summary_fmt.format(
                stock=state["stock_symbol"], **state["social_extractions"]
            )
            msgs = [
                SystemMessage(trading_input.social_media_system_prompt),
                HumanMessage(user_content),
            ]
            resp = await llm.ainvoke(msgs)
            return {"analyst_reports": ("social_media", resp.content)}

        # Researcher Debate nodes
        def make_researcher_first_round(label: str, system_prompt: str):
            async def first_round(state: TradingWorkflowState) -> dict[str, Any]:
                message = [
                    SystemMessage(system_prompt),
                    HumanMessage(
                        trading_input.researcher_first_round_fmt.format(
                            stock=state["stock_symbol"],
                            fundamentals_report=state["analyst_reports"][
                                "fundamentals"
                            ],
                            market_report=state["analyst_reports"]["market"],
                            news_report=state["analyst_reports"]["news"],
                            social_media_report=state["analyst_reports"][
                                "social_media"
                            ],
                        )
                    ),
                ]
                response = await llm.ainvoke(message)
                history = message + [response]
                return {"researcher_histories": (label, history)}

            return first_round

        bull_researcher_first_round = make_researcher_first_round(
            "bull", trading_input.bull_system_prompt
        )
        bear_researcher_first_round = make_researcher_first_round(
            "bear", trading_input.bear_system_prompt
        )

        def create_researcher_round():
            async def bull_researcher_round(
                state: TradingWorkflowState,
            ) -> dict[str, Any]:
                bear_response = state["researcher_histories"]["bear"][-1].content
                bull_history = state["researcher_histories"]["bull"].copy()
                user_message = HumanMessage(
                    trading_input.researcher_subsequent_round_fmt.format(
                        other_response=bear_response
                    )
                )
                bull_history.append(user_message)
                response = await llm.ainvoke(bull_history)
                bull_history.append(response)
                return {"researcher_histories": ("bull", bull_history)}

            async def bear_researcher_round(
                state: TradingWorkflowState,
            ) -> dict[str, Any]:
                bull_response = state["researcher_histories"]["bull"][-1].content
                bear_history = state["researcher_histories"]["bear"].copy()
                user_message = HumanMessage(
                    trading_input.researcher_subsequent_round_fmt.format(
                        other_response=bull_response
                    )
                )
                bear_history.append(user_message)
                response = await llm.ainvoke(bear_history)
                bear_history.append(response)
                return {"researcher_histories": ("bear", bear_history)}

            return bull_researcher_round, bear_researcher_round

        async def research_manager(state: TradingWorkflowState) -> dict[str, Any]:
            bull_history = state["researcher_histories"]["bull"]
            bear_history = state["researcher_histories"]["bear"]
            debate_lines = []
            for bull_msg, bear_msg in zip(bull_history, bear_history):
                if isinstance(bull_msg, AIMessage):
                    debate_lines.append(f"**Bull Researcher:** {bull_msg.content}")
                if isinstance(bear_msg, AIMessage):
                    debate_lines.append(f"**Bear Researcher:** {bear_msg.content}")
            debate_text = "\n".join(debate_lines)
            message = [
                SystemMessage(trading_input.research_manager_system_prompt),
                HumanMessage(
                    trading_input.researcher_manager_user_prompt_fmt.format(
                        stock=state["stock_symbol"], debate=debate_text
                    )
                ),
            ]
            response = await llm.ainvoke(message)
            return {"investment_plan": response.content}

        # Trader & Risk Analyst chains
        def create_trader_node(trader_name: str, trader_system_prompt: str):
            async def trader_node(state: TradingWorkflowState) -> dict[str, Any]:
                message = [
                    SystemMessage(trader_system_prompt),
                    HumanMessage(
                        trading_input.trader_user_prompt_fmt.format(
                            stock=state["stock_symbol"],
                            investment_plan=state["investment_plan"],
                        )
                    ),
                ]
                response = await llm.ainvoke(message)
                return {"trader_decisions": (trader_name, response.content)}

            return trader_node

        def create_risk_analyst_first_round_nodes(trader_name: str):
            analyst_configs = [
                ("risky", trading_input.risky_analyst_system_prompt),
                ("safe", trading_input.safe_analyst_system_prompt),
                ("neutral", trading_input.neutral_analyst_system_prompt),
            ]

            def create_first_round_func(analyst_type: str, system_prompt: str):
                async def risk_analyst_first_round(
                    state: TradingWorkflowState,
                ) -> dict[str, Any]:
                    trader_decision = state["trader_decisions"][trader_name]
                    message = [
                        SystemMessage(system_prompt),
                        HumanMessage(
                            trading_input.risk_analyst_first_round_fmt.format(
                                stock=state["stock_symbol"],
                                fundamentals_report=state["analyst_reports"][
                                    "fundamentals"
                                ],
                                market_report=state["analyst_reports"]["market"],
                                news_report=state["analyst_reports"]["news"],
                                social_media_report=state["analyst_reports"][
                                    "social_media"
                                ],
                                trader_decision=trader_decision,
                            )
                        ),
                    ]
                    response = await llm.ainvoke(message)
                    history = message + [response]
                    return {
                        "risk_analyst_histories": (
                            f"{trader_name}_{analyst_type}",
                            history,
                        )
                    }

                return risk_analyst_first_round

            return tuple(
                create_first_round_func(analyst_type, system_prompt)
                for analyst_type, system_prompt in analyst_configs
            )

        def create_risk_analyst_round_nodes(trader_name: str):
            analyst_types = ["risky", "safe", "neutral"]

            def create_round_func(analyst_type: str):
                async def risk_analyst_round(
                    state: TradingWorkflowState,
                ) -> dict[str, Any]:
                    histories = state["risk_analyst_histories"]
                    other_responses = {
                        "risky": histories[f"{trader_name}_risky"][-1].content,
                        "safe": histories[f"{trader_name}_safe"][-1].content,
                        "neutral": histories[f"{trader_name}_neutral"][-1].content,
                    }
                    current_history = histories[f"{trader_name}_{analyst_type}"].copy()
                    user_message = HumanMessage(
                        trading_input.risk_analyst_subsequent_round_fmt.format(
                            **other_responses
                        )
                    )
                    current_history.append(user_message)
                    response = await llm.ainvoke(current_history)
                    current_history.append(response)
                    return {
                        "risk_analyst_histories": (
                            f"{trader_name}_{analyst_type}",
                            current_history,
                        )
                    }

                return risk_analyst_round

            return tuple(
                create_round_func(analyst_type) for analyst_type in analyst_types
            )

        def create_risk_analyst_judge_node(trader_name: str):
            async def risk_analyst_judge(state: TradingWorkflowState) -> dict[str, Any]:
                risk_analysts = ["risky", "safe", "neutral"]
                histories = state["risk_analyst_histories"]
                analyst_histories = {
                    analyst_type: histories[f"{trader_name}_{analyst_type}"]
                    for analyst_type in risk_analysts
                }
                debate_lines = []
                all_histories = [analyst_histories[name] for name in risk_analysts]
                for agent_messages in zip(*all_histories):
                    for agent_name, message in zip(risk_analysts, agent_messages):
                        if isinstance(message, AIMessage):
                            debate_lines.append(
                                f"**{agent_name.capitalize()} Risk Analyst:** {message.content}"
                            )
                debate_text = "\n".join(debate_lines)
                judge_message = [
                    SystemMessage(trading_input.judge_system_prompt),
                    HumanMessage(
                        trading_input.judge_user_prompt_fmt.format(
                            stock=state["stock_symbol"], debate=debate_text
                        )
                    ),
                ]
                judge_response = await llm.ainvoke(judge_message)
                return {"trader_recommendations": (trader_name, judge_response.content)}

            return risk_analyst_judge

        async def fund_manager(state: TradingWorkflowState) -> dict[str, Any]:
            message = [
                SystemMessage(trading_input.fund_manager_system_prompt),
                HumanMessage(
                    trading_input.fund_manager_user_prompt_fmt.format(
                        stock=state["stock_symbol"],
                        **state["trader_recommendations"],
                    )
                ),
            ]
            response = await llm.ainvoke(message)
            return {"final_recommendation": response.content}

        #  Build workflow graph
        workflow_builder = StateGraph(TradingWorkflowState)

        # Fundamentals extraction nodes
        funda_docs = [
            ("balance_sheet", "balance sheet"),
            ("income_stmt", "income statement"),
            ("cashflow_stmt", "cashflow statement"),
            ("insider_sentiment", "insider sentiment"),
        ]
        if trading_input.funda_insider_transactions is not None:
            funda_docs.append(("insider_transactions", "insider transactions"))
        funda_nodes: list[str] = []
        for attr, doc_name in funda_docs:
            node_name = f"fundamentals_extract_{attr}"
            workflow_builder.add_node(
                node_name, make_fundamentals_extraction(attr, doc_name)
            )
            workflow_builder.add_edge(START, node_name)
            funda_nodes.append(node_name)
        workflow_builder.add_node("fundamentals_summary", fundamentals_summary)
        for n in funda_nodes:
            workflow_builder.add_edge(n, "fundamentals_summary")

        # Market extraction nodes
        market_docs = [
            ("stock_price", "stock price data"),
            ("stock_stats", "stock stats indicators"),
        ]
        market_nodes: list[str] = []
        for attr, doc_name in market_docs:
            node_name = f"market_extract_{attr}"
            workflow_builder.add_node(node_name, make_market_extraction(attr, doc_name))
            workflow_builder.add_edge(START, node_name)
            market_nodes.append(node_name)
        workflow_builder.add_node("market_summary", market_summary)
        for n in market_nodes:
            workflow_builder.add_edge(n, "market_summary")

        # News extraction nodes
        news_nodes: list[str] = []
        for idx in range(trading_input.num_news_chunks):
            node_name = f"news_extract_{idx}"
            workflow_builder.add_node(node_name, make_news_extraction(idx))
            workflow_builder.add_edge(START, node_name)
            news_nodes.append(node_name)
        workflow_builder.add_node("news_summary", news_summary)
        for n in news_nodes:
            workflow_builder.add_edge(n, "news_summary")

        # Social extraction nodes
        social_nodes: list[str] = []
        for idx in range(trading_input.num_social_chunks):
            node_name = f"social_extract_{idx}"
            workflow_builder.add_node(node_name, make_social_extraction(idx))
            workflow_builder.add_edge(START, node_name)
            social_nodes.append(node_name)
        workflow_builder.add_node("social_summary", social_summary)
        for n in social_nodes:
            workflow_builder.add_edge(n, "social_summary")

        # Researcher first round depends
        workflow_builder.add_node("bull_researcher_first", bull_researcher_first_round)
        workflow_builder.add_node("bear_researcher_first", bear_researcher_first_round)
        for summary_node in [
            "fundamentals_summary",
            "market_summary",
            "news_summary",
            "social_summary",
        ]:
            workflow_builder.add_edge(summary_node, "bull_researcher_first")
            workflow_builder.add_edge(summary_node, "bear_researcher_first")

        # Subsequent researcher rounds
        prev_bull_node = "bull_researcher_first"
        prev_bear_node = "bear_researcher_first"
        for round_num in range(1, trading_input.num_debate_rounds):
            bull_round_func, bear_round_func = create_researcher_round()
            bull_node_name = f"bull_researcher_round_{round_num}"
            bear_node_name = f"bear_researcher_round_{round_num}"
            workflow_builder.add_node(bull_node_name, bull_round_func)
            workflow_builder.add_node(bear_node_name, bear_round_func)
            workflow_builder.add_edge(prev_bull_node, bull_node_name)
            workflow_builder.add_edge(prev_bear_node, bear_node_name)
            workflow_builder.add_edge(prev_bear_node, bull_node_name)
            workflow_builder.add_edge(prev_bull_node, bear_node_name)
            prev_bull_node = bull_node_name
            prev_bear_node = bear_node_name

        workflow_builder.add_node("research_manager", research_manager)
        workflow_builder.add_edge(prev_bull_node, "research_manager")
        workflow_builder.add_edge(prev_bear_node, "research_manager")

        # Traders
        trader_nodes: list[str] = []
        for (
            trader_name,
            trader_system_prompt,
        ) in trading_input.trader_system_prompts.items():
            node_name = f"trader_{trader_name}"
            workflow_builder.add_node(
                node_name, create_trader_node(trader_name, trader_system_prompt)
            )
            workflow_builder.add_edge("research_manager", node_name)
            trader_nodes.append(node_name)

        # Risk analysts per trader
        judge_nodes: list[str] = []
        for trader_name in trading_input.trader_system_prompts.keys():
            risky_first, safe_first, neutral_first = (
                create_risk_analyst_first_round_nodes(trader_name)
            )
            risky_first_node = f"risk_analyst_{trader_name}_risky_first"
            safe_first_node = f"risk_analyst_{trader_name}_safe_first"
            neutral_first_node = f"risk_analyst_{trader_name}_neutral_first"
            workflow_builder.add_node(risky_first_node, risky_first)
            workflow_builder.add_node(safe_first_node, safe_first)
            workflow_builder.add_node(neutral_first_node, neutral_first)
            for t_node in [f"trader_{trader_name}"]:
                workflow_builder.add_edge(t_node, risky_first_node)
                workflow_builder.add_edge(t_node, safe_first_node)
                workflow_builder.add_edge(t_node, neutral_first_node)
            prev_risky = risky_first_node
            prev_safe = safe_first_node
            prev_neutral = neutral_first_node
            for round_num in range(1, trading_input.num_debate_rounds):
                risky_round, safe_round, neutral_round = (
                    create_risk_analyst_round_nodes(trader_name)
                )
                risky_round_node = f"risk_analyst_{trader_name}_risky_round_{round_num}"
                safe_round_node = f"risk_analyst_{trader_name}_safe_round_{round_num}"
                neutral_round_node = (
                    f"risk_analyst_{trader_name}_neutral_round_{round_num}"
                )
                workflow_builder.add_node(risky_round_node, risky_round)
                workflow_builder.add_node(safe_round_node, safe_round)
                workflow_builder.add_node(neutral_round_node, neutral_round)
                # all-to-all from previous round
                for src in [prev_risky, prev_safe, prev_neutral]:
                    workflow_builder.add_edge(src, risky_round_node)
                    workflow_builder.add_edge(src, safe_round_node)
                    workflow_builder.add_edge(src, neutral_round_node)
                prev_risky = risky_round_node
                prev_safe = safe_round_node
                prev_neutral = neutral_round_node
            judge_node = f"risk_analyst_{trader_name}_judge"
            workflow_builder.add_node(
                judge_node, create_risk_analyst_judge_node(trader_name)
            )
            for src in [prev_risky, prev_safe, prev_neutral]:
                workflow_builder.add_edge(src, judge_node)
            judge_nodes.append(judge_node)

        # Fund manager
        workflow_builder.add_node("fund_manager", fund_manager)
        for jn in judge_nodes:
            workflow_builder.add_edge(jn, "fund_manager")

        return workflow_builder
