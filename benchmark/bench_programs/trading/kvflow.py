from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Annotated, Any, TypeVar

from bench_programs.trading.base import OutputType, TradingInput, TradingProgram
from bench_programs.utils.kvflow import (
    KVFlowStepGraph,
    KVFlowStepGraphUpdater,
    llm_ainvoke,
    precompute_static_prompts,
)
from bench_programs.utils.langgraph import get_langgraph_openai_client
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile

T = TypeVar("T")


def update_dict(
    left: dict[str, T], right: tuple[str, T] | dict[str, T]
) -> dict[str, T]:
    if isinstance(right, dict):
        return right
    key, value = right
    left[key] = value
    return left


class TradingWorkflowState(TypedDict):
    index: int
    stock_symbol: str

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

    fundamentals_extractions: Annotated[dict[str, str], update_dict]
    market_extractions: Annotated[dict[str, str], update_dict]
    news_extractions: Annotated[dict[str, str], update_dict]
    social_extractions: Annotated[dict[str, str], update_dict]

    analyst_reports: Annotated[dict[str, str], update_dict]

    investment_plan: str
    researcher_histories: Annotated[dict[str, list[BaseMessage]], update_dict]

    trader_decisions: Annotated[dict[str, str], update_dict]
    trader_recommendations: Annotated[dict[str, str], update_dict]
    risk_analyst_histories: Annotated[dict[str, list[BaseMessage]], update_dict]

    final_recommendation: str


@dataclass
class AgentIds:
    fundamentals_extract_agent_ids: dict[str, str]
    fundamentals_summary_agent_id: str
    market_extract_agent_ids: dict[str, str]
    market_summary_agent_id: str
    news_extract_agent_ids: list[str]
    news_summary_agent_id: str
    social_extract_agent_ids: list[str]
    social_summary_agent_id: str

    bull_first_agent_id: str
    bear_first_agent_id: str
    bull_round_agent_ids: list[str]
    bear_round_agent_ids: list[str]
    research_manager_agent_id: str

    trader_agent_ids: dict[str, str]

    risk_first_agent_ids_by_trader: dict[str, dict[str, str]]
    risk_round_agent_ids_by_trader: dict[str, dict[str, list[str]]]
    risk_judge_agent_ids_by_trader: dict[str, str]

    fund_manager_agent_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_str(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


class KVFlowTradingProgram(TradingProgram):
    async def _run(
        self,
        trading_input: TradingInput,
        start_benchmark: Callable[[], Awaitable[None]] | None = None,
        stop_benchmark: Callable[[], Awaitable[HeliumSystemProfile]] | None = None,
        update_agent_step_graph: (
            Callable[[dict[str, Any], dict[int, list[str]], int], Awaitable[None]]
            | None
        ) = None,
        get_worker_generation_configs: (
            Callable[[GenerationConfig], list[GenerationConfig]] | None
        ) = None,
        **kwargs,
    ) -> tuple[OutputType, HeliumSystemProfile]:
        assert start_benchmark is not None
        assert stop_benchmark is not None
        assert update_agent_step_graph is not None

        generation_config = (
            GenerationConfig.from_env()
            if trading_input.generation_config is None
            else trading_input.generation_config
        )

        step_graph_updater, agent_ids = await self._prepare_agent_step_graph(
            trading_input=trading_input,
            update_agent_step_graph=update_agent_step_graph,
        )

        self.start_timer("precompute")
        assert get_worker_generation_configs is not None
        await self._precompute_static_prompts(
            agent_ids=agent_ids,
            trading_input=trading_input,
            generation_configs=get_worker_generation_configs(generation_config),
        )
        self.stop_timer()

        workflow = (
            await self._build_workflow(
                trading_input=trading_input,
                agent_ids=agent_ids,
                updater=step_graph_updater,
                generation_config=generation_config,
            )
        ).compile()

        indices = trading_input.shuffle()
        inputs: list[TradingWorkflowState] = []
        for original_idx, data in zip(indices, trading_input.iter_data()):
            inputs.append(
                {
                    "index": original_idx,
                    "stock_symbol": data["stock_symbol"],
                    "company_profile": data["funda_company_profile"],
                    "balance_sheet": data["funda_balance_sheet"],
                    "income_stmt": data["funda_income_stmt"],
                    "cashflow_stmt": data["funda_cashflow_stmt"],
                    "insider_sentiment": data["funda_insider_sentiment"],
                    "insider_transactions": data["funda_insider_transaction"],
                    "stock_price": data["market_stock_price"],
                    "stock_stats": data["market_stock_stat"],
                    "news_chunks": data["news_combined_chunks"],
                    "social_chunks": data["social_reddit_post_chunks"],
                    "fundamentals_extractions": {},
                    "market_extractions": {},
                    "news_extractions": {},
                    "social_extractions": {},
                    "analyst_reports": {},
                    "investment_plan": "",
                    "researcher_histories": {},
                    "trader_decisions": {},
                    "trader_recommendations": {},
                    "risk_analyst_histories": {},
                    "final_recommendation": "",
                }
            )

        await step_graph_updater.reset(total_items=len(inputs))
        await start_benchmark()

        self.start_timer("generate")
        outputs: list[dict[str, Any]] = await workflow.abatch(inputs)  # type: ignore[arg-type]
        self.stop_timer()

        system_profile = await stop_benchmark()

        output_builder = self.OutputBuilder()
        for output in outputs:
            output_builder.add(output["index"], output["final_recommendation"])
        return output_builder.build(), system_profile

    async def _build_workflow(
        self,
        trading_input: TradingInput,
        agent_ids: AgentIds,
        updater: KVFlowStepGraphUpdater,
        generation_config: GenerationConfig,
    ) -> StateGraph:
        llm = get_langgraph_openai_client(generation_config)

        def make_fundamentals_extraction(agent_id: str, attr: str, doc_name: str):
            async def node(state: TradingWorkflowState) -> dict[str, Any]:
                async with updater.node(agent_id):
                    template = trading_input.fundamentals_extraction_fmt.format(
                        doc_name=doc_name,
                        doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
                    )
                    user_content = template.format(
                        stock=state["stock_symbol"],
                        doc=state[attr],  # type: ignore[literal-required]
                        company_profile=state["company_profile"],
                    )
                    msgs = [
                        SystemMessage(trading_input.fundamentals_system_prompt),
                        HumanMessage(user_content),
                    ]
                    resp = await llm_ainvoke(llm, msgs, agent_id)
                    key = doc_name.replace(" ", "_")
                    return {"fundamentals_extractions": (key, _as_str(resp.content))}

            return node

        async def fundamentals_summary(state: TradingWorkflowState) -> dict[str, Any]:
            agent_id = agent_ids.fundamentals_summary_agent_id
            async with updater.node(agent_id):
                user_content = trading_input.fundamentals_summary_fmt.format(
                    stock=state["stock_symbol"],
                    company_profile=state["company_profile"],
                    **state["fundamentals_extractions"],
                )
                msgs = [
                    SystemMessage(trading_input.fundamentals_system_prompt),
                    HumanMessage(user_content),
                ]
                resp = await llm_ainvoke(llm, msgs, agent_id)
                return {"analyst_reports": ("fundamentals", _as_str(resp.content))}

        def make_market_extraction(agent_id: str, attr: str, doc_name: str):
            async def node(state: TradingWorkflowState) -> dict[str, Any]:
                async with updater.node(agent_id):
                    template = trading_input.market_extraction_fmt.format(
                        doc_name=doc_name,
                        doc_desc=trading_input.doc_descriptions.get(doc_name, ""),
                    )
                    user_content = template.format(
                        stock=state["stock_symbol"],
                        doc=state[attr],  # type: ignore[literal-required]
                    )
                    msgs = [
                        SystemMessage(trading_input.market_system_prompt),
                        HumanMessage(user_content),
                    ]
                    resp = await llm_ainvoke(llm, msgs, agent_id)
                    key = doc_name.replace(" ", "_")
                    return {"market_extractions": (key, _as_str(resp.content))}

            return node

        async def market_summary(state: TradingWorkflowState) -> dict[str, Any]:
            agent_id = agent_ids.market_summary_agent_id
            async with updater.node(agent_id):
                user_content = trading_input.market_summary_fmt.format(
                    stock=state["stock_symbol"], **state["market_extractions"]
                )
                msgs = [
                    SystemMessage(trading_input.market_system_prompt),
                    HumanMessage(user_content),
                ]
                resp = await llm_ainvoke(llm, msgs, agent_id)
                return {"analyst_reports": ("market", _as_str(resp.content))}

        def make_news_extraction(agent_id: str, idx: int):
            async def node(state: TradingWorkflowState) -> dict[str, Any]:
                async with updater.node(agent_id):
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
                    resp = await llm_ainvoke(llm, msgs, agent_id)
                    return {"news_extractions": (f"chunk_{idx}", _as_str(resp.content))}

            return node

        async def news_summary(state: TradingWorkflowState) -> dict[str, Any]:
            agent_id = agent_ids.news_summary_agent_id
            async with updater.node(agent_id):
                user_content = trading_input.news_summary_fmt.format(
                    stock=state["stock_symbol"], **state["news_extractions"]
                )
                msgs = [
                    SystemMessage(trading_input.news_system_prompt),
                    HumanMessage(user_content),
                ]
                resp = await llm_ainvoke(llm, msgs, agent_id)
                return {"analyst_reports": ("news", _as_str(resp.content))}

        def make_social_extraction(agent_id: str, idx: int):
            async def node(state: TradingWorkflowState) -> dict[str, Any]:
                async with updater.node(agent_id):
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
                    resp = await llm_ainvoke(llm, msgs, agent_id)
                    return {
                        "social_extractions": (f"chunk_{idx}", _as_str(resp.content))
                    }

            return node

        async def social_summary(state: TradingWorkflowState) -> dict[str, Any]:
            agent_id = agent_ids.social_summary_agent_id
            async with updater.node(agent_id):
                user_content = trading_input.social_summary_fmt.format(
                    stock=state["stock_symbol"], **state["social_extractions"]
                )
                msgs = [
                    SystemMessage(trading_input.social_media_system_prompt),
                    HumanMessage(user_content),
                ]
                resp = await llm_ainvoke(llm, msgs, agent_id)
                return {"analyst_reports": ("social_media", _as_str(resp.content))}

        def make_researcher_first_round(agent_id: str, label: str, system_prompt: str):
            async def first_round(state: TradingWorkflowState) -> dict[str, Any]:
                async with updater.node(agent_id):
                    msgs = [
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
                    response = await llm_ainvoke(llm, msgs, agent_id)
                    history = msgs + [response]
                    return {"researcher_histories": (label, history)}

            return first_round

        bull_researcher_first_round = make_researcher_first_round(
            agent_ids.bull_first_agent_id, "bull", trading_input.bull_system_prompt
        )
        bear_researcher_first_round = make_researcher_first_round(
            agent_ids.bear_first_agent_id, "bear", trading_input.bear_system_prompt
        )

        def create_researcher_round(bull_agent_id: str, bear_agent_id: str):
            async def bull_researcher_round(
                state: TradingWorkflowState,
            ) -> dict[str, Any]:
                async with updater.node(bull_agent_id):
                    bear_response = state["researcher_histories"]["bear"][-1].content
                    bull_history = state["researcher_histories"]["bull"].copy()
                    bull_history.append(
                        HumanMessage(
                            trading_input.researcher_subsequent_round_fmt.format(
                                other_response=bear_response
                            )
                        )
                    )
                    response = await llm_ainvoke(llm, bull_history, bull_agent_id)
                    bull_history.append(response)
                    return {"researcher_histories": ("bull", bull_history)}

            async def bear_researcher_round(
                state: TradingWorkflowState,
            ) -> dict[str, Any]:
                async with updater.node(bear_agent_id):
                    bull_response = state["researcher_histories"]["bull"][-1].content
                    bear_history = state["researcher_histories"]["bear"].copy()
                    bear_history.append(
                        HumanMessage(
                            trading_input.researcher_subsequent_round_fmt.format(
                                other_response=bull_response
                            )
                        )
                    )
                    response = await llm_ainvoke(llm, bear_history, bear_agent_id)
                    bear_history.append(response)
                    return {"researcher_histories": ("bear", bear_history)}

            return bull_researcher_round, bear_researcher_round

        async def research_manager(state: TradingWorkflowState) -> dict[str, Any]:
            agent_id = agent_ids.research_manager_agent_id
            async with updater.node(agent_id):
                bull_history = state["researcher_histories"]["bull"]
                bear_history = state["researcher_histories"]["bear"]
                debate_lines: list[str] = []
                for agent_messages in zip(bull_history, bear_history):
                    for agent_name, message in zip(["bull", "bear"], agent_messages):
                        if isinstance(message, AIMessage):
                            debate_lines.append(
                                f"**{agent_name.capitalize()} Researcher:** {message.content}"
                            )
                debate_text = "\n".join(debate_lines)
                msgs = [
                    SystemMessage(trading_input.research_manager_system_prompt),
                    HumanMessage(
                        trading_input.researcher_manager_user_prompt_fmt.format(
                            stock=state["stock_symbol"], debate=debate_text
                        )
                    ),
                ]
                resp = await llm_ainvoke(llm, msgs, agent_id)
                return {"investment_plan": _as_str(resp.content)}

        def create_trader_node(
            trader_name: str, trader_system_prompt: str, agent_id: str
        ):
            async def trader_node(state: TradingWorkflowState) -> dict[str, Any]:
                async with updater.node(agent_id):
                    msgs = [
                        SystemMessage(trader_system_prompt),
                        HumanMessage(
                            trading_input.trader_user_prompt_fmt.format(
                                stock=state["stock_symbol"],
                                investment_plan=state["investment_plan"],
                            )
                        ),
                    ]
                    resp = await llm_ainvoke(llm, msgs, agent_id)
                    return {"trader_decisions": (trader_name, _as_str(resp.content))}

            return trader_node

        def create_risk_analyst_first_round_nodes(trader_name: str):
            analyst_configs = [
                ("risky", trading_input.risky_analyst_system_prompt),
                ("safe", trading_input.safe_analyst_system_prompt),
                ("neutral", trading_input.neutral_analyst_system_prompt),
            ]

            def create_first_round_func(
                analyst_type: str, system_prompt: str, agent_id: str
            ):
                async def risk_analyst_first_round(
                    state: TradingWorkflowState,
                ) -> dict[str, Any]:
                    async with updater.node(agent_id):
                        trader_decision = state["trader_decisions"][trader_name]
                        msgs = [
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
                        resp = await llm_ainvoke(llm, msgs, agent_id)
                        history = msgs + [resp]
                        return {
                            "risk_analyst_histories": (
                                f"{trader_name}_{analyst_type}",
                                history,
                            )
                        }

                return risk_analyst_first_round

            return tuple(
                create_first_round_func(
                    analyst_type,
                    system_prompt,
                    agent_ids.risk_first_agent_ids_by_trader[trader_name][analyst_type],
                )
                for analyst_type, system_prompt in analyst_configs
            )

        def create_risk_analyst_round_nodes(
            trader_name: str, agent_ids_by_type: dict[str, str]
        ):
            analyst_types = ["risky", "safe", "neutral"]

            def create_round_func(analyst_type: str):
                async def risk_analyst_round(
                    state: TradingWorkflowState,
                ) -> dict[str, Any]:
                    agent_id = agent_ids_by_type[analyst_type]
                    async with updater.node(agent_id):
                        histories = state["risk_analyst_histories"]
                        other_responses = {
                            "risky": histories[f"{trader_name}_risky"][-1].content,
                            "safe": histories[f"{trader_name}_safe"][-1].content,
                            "neutral": histories[f"{trader_name}_neutral"][-1].content,
                        }
                        current_history = histories[
                            f"{trader_name}_{analyst_type}"
                        ].copy()
                        current_history.append(
                            HumanMessage(
                                trading_input.risk_analyst_subsequent_round_fmt.format(
                                    **other_responses
                                )
                            )
                        )
                        resp = await llm_ainvoke(llm, current_history, agent_id)
                        current_history.append(resp)
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

        def create_risk_analyst_judge_node(trader_name: str, agent_id: str):
            async def risk_analyst_judge(state: TradingWorkflowState) -> dict[str, Any]:
                async with updater.node(agent_id):
                    risk_analysts = ["risky", "safe", "neutral"]
                    histories = state["risk_analyst_histories"]
                    analyst_histories = {
                        analyst_type: histories[f"{trader_name}_{analyst_type}"]
                        for analyst_type in risk_analysts
                    }
                    debate_lines: list[str] = []
                    all_histories = [analyst_histories[name] for name in risk_analysts]
                    for agent_messages in zip(*all_histories):
                        for agent_name, message in zip(risk_analysts, agent_messages):
                            if isinstance(message, AIMessage):
                                debate_lines.append(
                                    f"**{agent_name.capitalize()} Risk Analyst:** {message.content}"
                                )
                    debate_text = "\n".join(debate_lines)
                    msgs = [
                        SystemMessage(trading_input.judge_system_prompt),
                        HumanMessage(
                            trading_input.judge_user_prompt_fmt.format(
                                stock=state["stock_symbol"], debate=debate_text
                            )
                        ),
                    ]
                    resp = await llm_ainvoke(llm, msgs, agent_id)
                    return {
                        "trader_recommendations": (trader_name, _as_str(resp.content))
                    }

            return risk_analyst_judge

        async def fund_manager(state: TradingWorkflowState) -> dict[str, Any]:
            agent_id = agent_ids.fund_manager_agent_id
            async with updater.node(agent_id):
                msgs = [
                    SystemMessage(trading_input.fund_manager_system_prompt),
                    HumanMessage(
                        trading_input.fund_manager_user_prompt_fmt.format(
                            stock=state["stock_symbol"],
                            **state["trader_recommendations"],
                        )
                    ),
                ]
                resp = await llm_ainvoke(llm, msgs, agent_id)
                return {"final_recommendation": _as_str(resp.content)}

        workflow_builder = StateGraph(TradingWorkflowState)

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
            node_name = agent_ids.fundamentals_extract_agent_ids[attr]
            workflow_builder.add_node(
                node_name, make_fundamentals_extraction(node_name, attr, doc_name)
            )
            workflow_builder.add_edge(START, node_name)
            funda_nodes.append(node_name)
        workflow_builder.add_node(
            agent_ids.fundamentals_summary_agent_id, fundamentals_summary
        )
        for n in funda_nodes:
            workflow_builder.add_edge(n, agent_ids.fundamentals_summary_agent_id)

        market_docs = [
            ("stock_price", "stock price data"),
            ("stock_stats", "stock stats indicators"),
        ]
        market_nodes: list[str] = []
        for attr, doc_name in market_docs:
            node_name = agent_ids.market_extract_agent_ids[attr]
            workflow_builder.add_node(
                node_name, make_market_extraction(node_name, attr, doc_name)
            )
            workflow_builder.add_edge(START, node_name)
            market_nodes.append(node_name)
        workflow_builder.add_node(agent_ids.market_summary_agent_id, market_summary)
        for n in market_nodes:
            workflow_builder.add_edge(n, agent_ids.market_summary_agent_id)

        news_nodes: list[str] = []
        for idx in range(trading_input.num_news_chunks):
            node_name = agent_ids.news_extract_agent_ids[idx]
            workflow_builder.add_node(node_name, make_news_extraction(node_name, idx))
            workflow_builder.add_edge(START, node_name)
            news_nodes.append(node_name)
        workflow_builder.add_node(agent_ids.news_summary_agent_id, news_summary)
        for n in news_nodes:
            workflow_builder.add_edge(n, agent_ids.news_summary_agent_id)

        social_nodes: list[str] = []
        for idx in range(trading_input.num_social_chunks):
            node_name = agent_ids.social_extract_agent_ids[idx]
            workflow_builder.add_node(node_name, make_social_extraction(node_name, idx))
            workflow_builder.add_edge(START, node_name)
            social_nodes.append(node_name)
        workflow_builder.add_node(agent_ids.social_summary_agent_id, social_summary)
        for n in social_nodes:
            workflow_builder.add_edge(n, agent_ids.social_summary_agent_id)

        workflow_builder.add_node(
            agent_ids.bull_first_agent_id, bull_researcher_first_round
        )
        workflow_builder.add_node(
            agent_ids.bear_first_agent_id, bear_researcher_first_round
        )
        for summary_node in [
            agent_ids.fundamentals_summary_agent_id,
            agent_ids.market_summary_agent_id,
            agent_ids.news_summary_agent_id,
            agent_ids.social_summary_agent_id,
        ]:
            workflow_builder.add_edge(summary_node, agent_ids.bull_first_agent_id)
            workflow_builder.add_edge(summary_node, agent_ids.bear_first_agent_id)

        prev_bull_node = agent_ids.bull_first_agent_id
        prev_bear_node = agent_ids.bear_first_agent_id
        for round_num in range(1, trading_input.num_debate_rounds):
            bull_node_name = agent_ids.bull_round_agent_ids[round_num - 1]
            bear_node_name = agent_ids.bear_round_agent_ids[round_num - 1]
            bull_round_func, bear_round_func = create_researcher_round(
                bull_node_name, bear_node_name
            )
            workflow_builder.add_node(bull_node_name, bull_round_func)
            workflow_builder.add_node(bear_node_name, bear_round_func)
            workflow_builder.add_edge(prev_bull_node, bull_node_name)
            workflow_builder.add_edge(prev_bear_node, bear_node_name)
            workflow_builder.add_edge(prev_bear_node, bull_node_name)
            workflow_builder.add_edge(prev_bull_node, bear_node_name)
            prev_bull_node = bull_node_name
            prev_bear_node = bear_node_name

        workflow_builder.add_node(agent_ids.research_manager_agent_id, research_manager)
        workflow_builder.add_edge(prev_bull_node, agent_ids.research_manager_agent_id)
        workflow_builder.add_edge(prev_bear_node, agent_ids.research_manager_agent_id)

        trader_nodes: list[str] = []
        for (
            trader_name,
            trader_system_prompt,
        ) in trading_input.trader_system_prompts.items():
            node_name = agent_ids.trader_agent_ids[trader_name]
            workflow_builder.add_node(
                node_name,
                create_trader_node(trader_name, trader_system_prompt, node_name),
            )
            workflow_builder.add_edge(agent_ids.research_manager_agent_id, node_name)
            trader_nodes.append(node_name)

        judge_nodes: list[str] = []
        for trader_name in trading_input.trader_system_prompts.keys():
            risky_first, safe_first, neutral_first = (
                create_risk_analyst_first_round_nodes(trader_name)
            )
            risky_first_node = agent_ids.risk_first_agent_ids_by_trader[trader_name][
                "risky"
            ]
            safe_first_node = agent_ids.risk_first_agent_ids_by_trader[trader_name][
                "safe"
            ]
            neutral_first_node = agent_ids.risk_first_agent_ids_by_trader[trader_name][
                "neutral"
            ]
            workflow_builder.add_node(risky_first_node, risky_first)
            workflow_builder.add_node(safe_first_node, safe_first)
            workflow_builder.add_node(neutral_first_node, neutral_first)
            trader_node = agent_ids.trader_agent_ids[trader_name]
            for dst in [risky_first_node, safe_first_node, neutral_first_node]:
                workflow_builder.add_edge(trader_node, dst)

            prev_risky = risky_first_node
            prev_safe = safe_first_node
            prev_neutral = neutral_first_node
            for round_num in range(1, trading_input.num_debate_rounds):
                round_ids = {
                    "risky": agent_ids.risk_round_agent_ids_by_trader[trader_name][
                        "risky"
                    ][round_num - 1],
                    "safe": agent_ids.risk_round_agent_ids_by_trader[trader_name][
                        "safe"
                    ][round_num - 1],
                    "neutral": agent_ids.risk_round_agent_ids_by_trader[trader_name][
                        "neutral"
                    ][round_num - 1],
                }
                risky_round, safe_round, neutral_round = (
                    create_risk_analyst_round_nodes(trader_name, round_ids)
                )
                risky_round_node = round_ids["risky"]
                safe_round_node = round_ids["safe"]
                neutral_round_node = round_ids["neutral"]
                workflow_builder.add_node(risky_round_node, risky_round)
                workflow_builder.add_node(safe_round_node, safe_round)
                workflow_builder.add_node(neutral_round_node, neutral_round)
                for src in [prev_risky, prev_safe, prev_neutral]:
                    workflow_builder.add_edge(src, risky_round_node)
                    workflow_builder.add_edge(src, safe_round_node)
                    workflow_builder.add_edge(src, neutral_round_node)
                prev_risky = risky_round_node
                prev_safe = safe_round_node
                prev_neutral = neutral_round_node

            judge_node = agent_ids.risk_judge_agent_ids_by_trader[trader_name]
            workflow_builder.add_node(
                judge_node, create_risk_analyst_judge_node(trader_name, judge_node)
            )
            for src in [prev_risky, prev_safe, prev_neutral]:
                workflow_builder.add_edge(src, judge_node)
            judge_nodes.append(judge_node)

        workflow_builder.add_node(agent_ids.fund_manager_agent_id, fund_manager)
        for jn in judge_nodes:
            workflow_builder.add_edge(jn, agent_ids.fund_manager_agent_id)

        return workflow_builder

    def _build_agent_ids(self, trading_input: TradingInput) -> AgentIds:
        fundamentals_extract_agent_ids = {
            "balance_sheet": "fundamentals_extract_balance_sheet",
            "income_stmt": "fundamentals_extract_income_stmt",
            "cashflow_stmt": "fundamentals_extract_cashflow_stmt",
            "insider_sentiment": "fundamentals_extract_insider_sentiment",
        }
        if trading_input.funda_insider_transactions is not None:
            fundamentals_extract_agent_ids["insider_transactions"] = (
                "fundamentals_extract_insider_transactions"
            )

        market_extract_agent_ids = {
            "stock_price": "market_extract_stock_price",
            "stock_stats": "market_extract_stock_stats",
        }

        news_extract_agent_ids = [
            f"news_extract_{i}" for i in range(trading_input.num_news_chunks)
        ]
        social_extract_agent_ids = [
            f"social_extract_{i}" for i in range(trading_input.num_social_chunks)
        ]

        bull_round_agent_ids = [
            f"bull_researcher_round_{r}"
            for r in range(1, trading_input.num_debate_rounds)
        ]
        bear_round_agent_ids = [
            f"bear_researcher_round_{r}"
            for r in range(1, trading_input.num_debate_rounds)
        ]

        trader_agent_ids = {
            name: f"trader_{name}" for name in trading_input.trader_system_prompts
        }

        risk_first: dict[str, dict[str, str]] = {}
        risk_round: dict[str, dict[str, list[str]]] = {}
        risk_judge: dict[str, str] = {}
        for trader_name in trading_input.trader_system_prompts:
            risk_first[trader_name] = {
                "risky": f"risk_analyst_{trader_name}_risky_first",
                "safe": f"risk_analyst_{trader_name}_safe_first",
                "neutral": f"risk_analyst_{trader_name}_neutral_first",
            }
            risk_round[trader_name] = {
                "risky": [
                    f"risk_analyst_{trader_name}_risky_round_{r}"
                    for r in range(1, trading_input.num_debate_rounds)
                ],
                "safe": [
                    f"risk_analyst_{trader_name}_safe_round_{r}"
                    for r in range(1, trading_input.num_debate_rounds)
                ],
                "neutral": [
                    f"risk_analyst_{trader_name}_neutral_round_{r}"
                    for r in range(1, trading_input.num_debate_rounds)
                ],
            }
            risk_judge[trader_name] = f"risk_analyst_{trader_name}_judge"

        return AgentIds(
            fundamentals_extract_agent_ids=fundamentals_extract_agent_ids,
            fundamentals_summary_agent_id="fundamentals_summary",
            market_extract_agent_ids=market_extract_agent_ids,
            market_summary_agent_id="market_summary",
            news_extract_agent_ids=news_extract_agent_ids,
            news_summary_agent_id="news_summary",
            social_extract_agent_ids=social_extract_agent_ids,
            social_summary_agent_id="social_summary",
            bull_first_agent_id="bull_researcher_first",
            bear_first_agent_id="bear_researcher_first",
            bull_round_agent_ids=bull_round_agent_ids,
            bear_round_agent_ids=bear_round_agent_ids,
            research_manager_agent_id="research_manager",
            trader_agent_ids=trader_agent_ids,
            risk_first_agent_ids_by_trader=risk_first,
            risk_round_agent_ids_by_trader=risk_round,
            risk_judge_agent_ids_by_trader=risk_judge,
            fund_manager_agent_id="fund_manager",
        )

    async def _prepare_agent_step_graph(
        self,
        trading_input: TradingInput,
        update_agent_step_graph: Callable[
            [dict[str, Any], dict[int, list[str]], int], Awaitable[None]
        ],
    ) -> tuple[KVFlowStepGraphUpdater, AgentIds]:
        agent_ids = self._build_agent_ids(trading_input)
        step_graph = self._build_agent_step_graph(trading_input, **agent_ids.to_dict())
        updater = KVFlowStepGraphUpdater(
            step_graph=step_graph,
            send_update=update_agent_step_graph,
        )
        return updater, agent_ids

    def _build_agent_step_graph(
        self, trading_input: TradingInput, **agent_ids: Any
    ) -> KVFlowStepGraph:
        ids = AgentIds(**agent_ids)
        edges: dict[str, set[str]] = {}

        for a in ids.fundamentals_extract_agent_ids.values():
            edges.setdefault(a, set()).add(ids.fundamentals_summary_agent_id)
        for a in ids.market_extract_agent_ids.values():
            edges.setdefault(a, set()).add(ids.market_summary_agent_id)
        for a in ids.news_extract_agent_ids:
            edges.setdefault(a, set()).add(ids.news_summary_agent_id)
        for a in ids.social_extract_agent_ids:
            edges.setdefault(a, set()).add(ids.social_summary_agent_id)

        for s in [
            ids.fundamentals_summary_agent_id,
            ids.market_summary_agent_id,
            ids.news_summary_agent_id,
            ids.social_summary_agent_id,
        ]:
            edges.setdefault(s, set()).update(
                {ids.bull_first_agent_id, ids.bear_first_agent_id}
            )

        prev_bull = ids.bull_first_agent_id
        prev_bear = ids.bear_first_agent_id
        for r in range(1, trading_input.num_debate_rounds):
            bull = ids.bull_round_agent_ids[r - 1]
            bear = ids.bear_round_agent_ids[r - 1]
            edges.setdefault(prev_bull, set()).update({bull, bear})
            edges.setdefault(prev_bear, set()).update({bull, bear})
            prev_bull, prev_bear = bull, bear

        edges.setdefault(prev_bull, set()).add(ids.research_manager_agent_id)
        edges.setdefault(prev_bear, set()).add(ids.research_manager_agent_id)

        for t in ids.trader_agent_ids.values():
            edges.setdefault(ids.research_manager_agent_id, set()).add(t)

        for trader_name, trader_node_id in ids.trader_agent_ids.items():
            for rid in ids.risk_first_agent_ids_by_trader[trader_name].values():
                edges.setdefault(trader_node_id, set()).add(rid)

        for trader_name in ids.trader_agent_ids:
            prev = ids.risk_first_agent_ids_by_trader[trader_name].copy()
            for rnd in range(1, trading_input.num_debate_rounds):
                curr = {
                    role: ids.risk_round_agent_ids_by_trader[trader_name][role][rnd - 1]
                    for role in ["risky", "safe", "neutral"]
                }
                for src in prev.values():
                    edges.setdefault(src, set()).update(curr.values())
                prev = curr

            judge = ids.risk_judge_agent_ids_by_trader[trader_name]
            for src in prev.values():
                edges.setdefault(src, set()).add(judge)
            edges.setdefault(judge, set()).add(ids.fund_manager_agent_id)

        return KVFlowStepGraph(
            edges=edges,
            all_agents=set(edges.keys())
            | {
                ids.fundamentals_summary_agent_id,
                ids.market_summary_agent_id,
                ids.news_summary_agent_id,
                ids.social_summary_agent_id,
                ids.bull_first_agent_id,
                ids.bear_first_agent_id,
                *ids.bull_round_agent_ids,
                *ids.bear_round_agent_ids,
                ids.research_manager_agent_id,
                *ids.trader_agent_ids.values(),
                *[
                    a
                    for d in ids.risk_first_agent_ids_by_trader.values()
                    for a in d.values()
                ],
                *[
                    a
                    for d in ids.risk_round_agent_ids_by_trader.values()
                    for lst in d.values()
                    for a in lst
                ],
                *ids.risk_judge_agent_ids_by_trader.values(),
                ids.fund_manager_agent_id,
            },
        )

    def _build_static_prompts(
        self, trading_input: TradingInput, agent_ids: AgentIds
    ) -> dict[str, list[BaseMessage]]:
        static: dict[str, list[BaseMessage]] = {}
        static[agent_ids.fundamentals_summary_agent_id] = [
            SystemMessage(trading_input.fundamentals_system_prompt)
        ]
        for a in agent_ids.fundamentals_extract_agent_ids.values():
            static[a] = [SystemMessage(trading_input.fundamentals_system_prompt)]

        static[agent_ids.market_summary_agent_id] = [
            SystemMessage(trading_input.market_system_prompt)
        ]
        for a in agent_ids.market_extract_agent_ids.values():
            static[a] = [SystemMessage(trading_input.market_system_prompt)]

        static[agent_ids.news_summary_agent_id] = [
            SystemMessage(trading_input.news_system_prompt)
        ]
        for a in agent_ids.news_extract_agent_ids:
            static[a] = [SystemMessage(trading_input.news_system_prompt)]

        static[agent_ids.social_summary_agent_id] = [
            SystemMessage(trading_input.social_media_system_prompt)
        ]
        for a in agent_ids.social_extract_agent_ids:
            static[a] = [SystemMessage(trading_input.social_media_system_prompt)]

        static[agent_ids.bull_first_agent_id] = [
            SystemMessage(trading_input.bull_system_prompt)
        ]
        static[agent_ids.bear_first_agent_id] = [
            SystemMessage(trading_input.bear_system_prompt)
        ]
        for a in agent_ids.bull_round_agent_ids:
            static[a] = [SystemMessage(trading_input.bull_system_prompt)]
        for a in agent_ids.bear_round_agent_ids:
            static[a] = [SystemMessage(trading_input.bear_system_prompt)]
        static[agent_ids.research_manager_agent_id] = [
            SystemMessage(trading_input.research_manager_system_prompt)
        ]

        for trader_name, system_prompt in trading_input.trader_system_prompts.items():
            static[agent_ids.trader_agent_ids[trader_name]] = [
                SystemMessage(system_prompt)
            ]

        for trader_name in trading_input.trader_system_prompts:
            for role, system_prompt in [
                ("risky", trading_input.risky_analyst_system_prompt),
                ("safe", trading_input.safe_analyst_system_prompt),
                ("neutral", trading_input.neutral_analyst_system_prompt),
            ]:
                static[agent_ids.risk_first_agent_ids_by_trader[trader_name][role]] = [
                    SystemMessage(system_prompt)
                ]
                for a in agent_ids.risk_round_agent_ids_by_trader[trader_name][role]:
                    static[a] = [SystemMessage(system_prompt)]

            static[agent_ids.risk_judge_agent_ids_by_trader[trader_name]] = [
                SystemMessage(trading_input.judge_system_prompt)
            ]

        static[agent_ids.fund_manager_agent_id] = [
            SystemMessage(trading_input.fund_manager_system_prompt)
        ]
        return static

    async def _precompute_static_prompts(
        self,
        agent_ids: AgentIds,
        trading_input: TradingInput,
        generation_configs: list[GenerationConfig],
    ) -> None:
        static_prompts = self._build_static_prompts(trading_input, agent_ids)
        await precompute_static_prompts(static_prompts, generation_configs)
