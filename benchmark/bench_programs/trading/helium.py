from typing import Literal

from bench_programs.trading.base import OutputType, TradingInput, TradingProgram

from helium import ops
from helium.common import GenerationConfig, Message
from helium.frontend.agents import Agent
from helium.frontend.programs import Program as HeliumProgram
from helium.runtime import HeliumServerConfig
from helium.runtime.protocol import (
    HeliumRequestConfig,
    HeliumResponse,
    HeliumSystemProfile,
)


def _summarize_documents(
    system_prompt: str,
    extraction_instruction_fmt: str,
    stock_op: ops.Op,
    doc_list: list[tuple[str, ops.Op]],
    generation_config: GenerationConfig | None,
    doc_descriptions: dict[str, str],
    **common_kwargs: ops.Op,
) -> list[tuple[str, ops.Op]]:
    doc_summaries: list[tuple[str, ops.Op]] = []
    # Summarize documents
    for doc_name, doc_op in doc_list:
        messages = [
            ops.OpMessage(role="system", content=system_prompt),
            ops.OpMessage(
                role="user",
                content=ops.format_op(
                    extraction_instruction_fmt.format(
                        doc_name=doc_name, doc_desc=doc_descriptions.get(doc_name, "")
                    ),
                    stock=stock_op,
                    doc=doc_op,
                    **common_kwargs,
                ),
            ),
        ]
        doc_summary = ops.llm_chat(messages, generation_config)
        doc_summaries.append((doc_name.replace(" ", "_"), doc_summary))
    return doc_summaries


def _aggregate_summaries(
    system_prompt: str,
    summary_instruction_fmt: str,
    stock_op: ops.Op,
    doc_summaries: list[tuple[str, ops.Op]],
    generation_config: GenerationConfig | None,
    **common_kwargs: ops.Op,
) -> ops.Op:
    """
    doc_summaries: list of (doc_ref, doc_name, doc_summary)
    """
    # Gather all summaries
    summary_messages = [
        ops.OpMessage(role="system", content=system_prompt),
        ops.OpMessage(
            role="user",
            content=ops.format_op(
                summary_instruction_fmt,
                stock=stock_op,
                **dict(doc_summaries),
                **common_kwargs,
            ),
        ),
    ]
    summary_op = ops.llm_chat(summary_messages, generation_config)
    return summary_op


def fundamentals_analyst(
    system_prompt: str,
    extraction_instruction_fmt: str,
    summary_instruction_fmt: str,
    stock_op: ops.Op,
    company_profile_op: ops.Op,
    balance_sheet_op: ops.Op,
    income_stmt_op: ops.Op,
    cashflow_op: ops.Op,
    insider_sentiment_op: ops.Op,
    insider_transactions_op: ops.Op | None,
    generation_config: GenerationConfig | None,
    doc_descriptions: dict[str, str],
) -> ops.Op:
    doc_list = [
        ("balance sheet", balance_sheet_op),
        ("income statement", income_stmt_op),
        ("cashflow statement", cashflow_op),
        ("insider sentiment", insider_sentiment_op),
    ]
    if insider_transactions_op is not None:
        doc_list.append(("insider transactions", insider_transactions_op))

    doc_summaries = _summarize_documents(
        system_prompt,
        extraction_instruction_fmt,
        stock_op,
        doc_list,
        generation_config,
        doc_descriptions,
        company_profile=company_profile_op,
    )
    summary_op = _aggregate_summaries(
        system_prompt,
        summary_instruction_fmt,
        stock_op,
        doc_summaries,
        generation_config,
        company_profile=company_profile_op,
    )
    return summary_op


def market_analyst(
    system_prompt: str,
    extraction_instruction_fmt: str,
    summary_instruction_fmt: str,
    stock_op: ops.Op,
    stock_price_op: ops.Op,
    stock_stats_op: ops.Op,
    generation_config: GenerationConfig | None,
    doc_descriptions: dict[str, str],
) -> ops.Op:
    doc_list = [
        ("stock price data", stock_price_op),
        ("stock stats indicators", stock_stats_op),
    ]
    doc_summaries = _summarize_documents(
        system_prompt,
        extraction_instruction_fmt,
        stock_op,
        doc_list,
        generation_config,
        doc_descriptions,
    )
    summary_op = _aggregate_summaries(
        system_prompt,
        summary_instruction_fmt,
        stock_op,
        doc_summaries,
        generation_config,
    )
    return summary_op


def news_analyst(
    system_prompt: str,
    extraction_instruction_fmt: str,
    summary_instruction_fmt: str,
    stock_op: ops.Op,
    news_chunk_ops: list[ops.Op],
    generation_config: GenerationConfig | None,
    doc_descriptions: dict[str, str],
) -> ops.Op:
    doc_list = [("news chunk", chunk_op) for chunk_op in news_chunk_ops]
    doc_summaries = _summarize_documents(
        system_prompt,
        extraction_instruction_fmt,
        stock_op,
        doc_list,
        generation_config,
        doc_descriptions,
    )
    doc_summaries = [
        (f"chunk_{i}", doc_summary) for i, (_, doc_summary) in enumerate(doc_summaries)
    ]
    summary_op = _aggregate_summaries(
        system_prompt,
        summary_instruction_fmt,
        stock_op,
        doc_summaries,
        generation_config,
    )
    return summary_op


def social_media_analyst(
    system_prompt: str,
    extraction_instruction_fmt: str,
    summary_instruction_fmt: str,
    stock_op: ops.Op,
    social_media_chunk_ops: list[ops.Op],
    generation_config: GenerationConfig | None,
    doc_descriptions: dict[str, str],
) -> ops.Op:
    doc_list = [("post chunk", chunk_op) for chunk_op in social_media_chunk_ops]
    doc_summaries = _summarize_documents(
        system_prompt,
        extraction_instruction_fmt,
        stock_op,
        doc_list,
        generation_config,
        doc_descriptions,
    )
    doc_summaries = [
        (f"chunk_{i}", doc_summary) for i, (_, doc_summary) in enumerate(doc_summaries)
    ]
    summary_op = _aggregate_summaries(
        system_prompt,
        summary_instruction_fmt,
        stock_op,
        doc_summaries,
        generation_config,
    )
    return summary_op


def researcher_first_round(
    system_prompt: str,
    first_round_fmt: str,
    stock_op: ops.Op,
    fundamentals_report_op: ops.Op,
    market_report_op: ops.Op,
    news_report_op: ops.Op,
    social_media_report_op: ops.Op,
    generation_config: GenerationConfig | None,
    set_cacheable: bool,
) -> ops.Op:
    messages = [
        ops.OpMessage(role="system", content=system_prompt),
        ops.OpMessage(
            role="user",
            content=ops.format_op(
                first_round_fmt,
                stock=stock_op,
                fundamentals_report=fundamentals_report_op,
                market_report=market_report_op,
                news_report=news_report_op,
                social_media_report=social_media_report_op,
            ),
        ),
    ]
    response_op = ops.llm_chat(
        messages, generation_config, return_history=True, cacheable=set_cacheable
    )
    return response_op


def researcher_subsequent_round(
    subsequent_round_fmt: str,
    cur_history: ops.Op,
    other_history: ops.Op,
    generation_config: GenerationConfig | None,
) -> ops.Op:
    other_response = ops.get_last_message(other_history)
    messages = ops.append_message(
        cur_history,
        content=ops.format_op(subsequent_round_fmt, other_response=other_response),
        role="user",
    )
    response_op = ops.llm_chat(messages, generation_config, return_history=True)
    return response_op


def _convert_history_to_text(args: tuple[ops.SingleDtype, ...]) -> str:
    bull_history, bear_history = args
    assert isinstance(bull_history, list)
    assert isinstance(bear_history, list)
    history_lines: list[str] = []
    for bull_msg, bear_msg in zip(bull_history, bear_history):
        if bull_msg.role == "assistant":
            history_lines.append(f"**Bull Researcher:** {bull_msg.content}")
        if bear_msg.role == "assistant":
            history_lines.append(f"**Bear Researcher:** {bear_msg.content}")
    return "\n".join(history_lines)


def research_manager(
    system_prompt: str,
    user_prompt_fmt: str,
    stock_op: ops.Op,
    bull_history: ops.Op,
    bear_history: ops.Op,
    generation_config: GenerationConfig | None,
) -> ops.Op:
    debate = ops.lambda_op([bull_history, bear_history], _convert_history_to_text)
    messages = [
        ops.OpMessage(role="system", content=system_prompt),
        ops.OpMessage(
            role="user",
            content=ops.format_op(user_prompt_fmt, stock=stock_op, debate=debate),
        ),
    ]
    investment_plan = ops.llm_chat(messages, generation_config)
    return investment_plan


def researcher_debate(
    bull_system_prompt: str,
    bear_system_prompt: str,
    manager_system_prompt: str,
    first_round_fmt: str,
    subsequent_round_fmt: str,
    manager_user_prompt_fmt: str,
    stock_op: ops.Op,
    fundamentals_report_op: ops.Op,
    market_report_op: ops.Op,
    news_report_op: ops.Op,
    social_media_report_op: ops.Op,
    num_rounds: int,
    generation_config: GenerationConfig | None,
    set_cacheable: bool,
) -> ops.Op:
    assert num_rounds >= 1
    # First round
    bull_history = researcher_first_round(
        bull_system_prompt,
        first_round_fmt,
        stock_op,
        fundamentals_report_op,
        market_report_op,
        news_report_op,
        social_media_report_op,
        generation_config,
        set_cacheable,
    )
    bear_history = researcher_first_round(
        bear_system_prompt,
        first_round_fmt,
        stock_op,
        fundamentals_report_op,
        market_report_op,
        news_report_op,
        social_media_report_op,
        generation_config,
        set_cacheable,
    )
    # Subsequent rounds
    for _ in range(1, num_rounds):
        new_bull_history = researcher_subsequent_round(
            subsequent_round_fmt, bull_history, bear_history, generation_config
        )
        new_bear_history = researcher_subsequent_round(
            subsequent_round_fmt, bear_history, bull_history, generation_config
        )
        bull_history = new_bull_history
        bear_history = new_bear_history
    # Return the investment plan
    investment_plan = research_manager(
        manager_system_prompt,
        manager_user_prompt_fmt,
        stock_op,
        bull_history,
        bear_history,
        generation_config,
    )
    return investment_plan


def risk_analyst_first_round(
    system_prompt: str,
    first_round_fmt: str,
    stock_op: ops.Op,
    fundamentals_report_op: ops.Op,
    market_report_op: ops.Op,
    news_report_op: ops.Op,
    social_media_report_op: ops.Op,
    trader_decision_op: ops.Op,
    generation_config: GenerationConfig | None,
    set_cacheable: bool,
) -> ops.Op:
    messages = [
        ops.OpMessage(role="system", content=system_prompt),
        ops.OpMessage(
            role="user",
            content=ops.format_op(
                first_round_fmt,
                stock=stock_op,
                fundamentals_report=fundamentals_report_op,
                market_report=market_report_op,
                news_report=news_report_op,
                social_media_report=social_media_report_op,
                trader_decision=trader_decision_op,
            ),
        ),
    ]
    response_op = ops.llm_chat(
        messages, generation_config, return_history=True, cacheable=set_cacheable
    )
    return response_op


def risk_analyst_subsequent_round(
    subsequent_round_fmt: str,
    cur_history: ops.Op,
    other_histories: list[tuple[str, ops.Op]],
    generation_config: GenerationConfig | None,
) -> ops.Op:
    other_responses = {
        name: ops.get_last_message(other_history)
        for name, other_history in other_histories
    }
    messages = ops.append_message(
        cur_history,
        content=ops.format_op(subsequent_round_fmt, **other_responses),
        role="user",
    )
    response_op = ops.llm_chat(messages, generation_config, return_history=True)
    return response_op


def risk_judge(
    system_prompt: str,
    user_prompt_fmt: str,
    stock_op: ops.Op,
    debate: list[tuple[str, ops.Op]],
    generation_config: GenerationConfig | None,
) -> ops.Op:
    agent_names = [name for name, _ in debate]

    def _convert_debate_to_text(args: tuple[ops.SingleDtype, ...]) -> str:
        history_lines: list[str] = []
        agent_messages: tuple[Message]
        for agent_messages in zip(*args):
            for agent_name, message in zip(agent_names, agent_messages):
                if message.role == "assistant":
                    history_lines.append(
                        f"**{agent_name.capitalize()} Risk Analyst:** {message.content}"
                    )
        return "\n".join(history_lines)

    debate_op = ops.lambda_op(
        [history for _, history in debate], _convert_debate_to_text
    )
    messages = [
        ops.OpMessage(role="system", content=system_prompt),
        ops.OpMessage(
            role="user",
            content=ops.format_op(user_prompt_fmt, stock=stock_op, debate=debate_op),
        ),
    ]
    investment_recommendation = ops.llm_chat(messages, generation_config)
    return investment_recommendation


def risk_analyst_debate(
    risky_analyst_system_prompt: str,
    safe_analyst_system_prompt: str,
    neutral_analyst_system_prompt: str,
    judge_system_prompt: str,
    first_round_fmt: str,
    subsequent_round_fmt: str,
    judge_user_prompt_fmt: str,
    stock_op: ops.Op,
    fundamentals_report_op: ops.Op,
    market_report_op: ops.Op,
    news_report_op: ops.Op,
    social_media_report_op: ops.Op,
    trader_decision_op: ops.Op,
    num_rounds: int,
    generation_config: GenerationConfig | None,
    set_cacheable: bool,
) -> ops.Op:
    assert num_rounds >= 1
    prompt_map = {
        "risky": risky_analyst_system_prompt,
        "safe": safe_analyst_system_prompt,
        "neutral": neutral_analyst_system_prompt,
    }
    # First round
    histories: list[tuple[str, ops.Op]] = []
    for agent_name, system_prompt in prompt_map.items():
        history = risk_analyst_first_round(
            system_prompt,
            first_round_fmt,
            stock_op,
            fundamentals_report_op,
            market_report_op,
            news_report_op,
            social_media_report_op,
            trader_decision_op,
            generation_config,
            set_cacheable,
        )
        histories.append((agent_name, history))
    # Subsequent rounds
    for _ in range(1, num_rounds):
        new_histories: list[tuple[str, ops.Op]] = []
        for agent_name, history in histories:
            new_history = risk_analyst_subsequent_round(
                subsequent_round_fmt, history, histories, generation_config
            )
            new_histories.append((agent_name, new_history))
        histories = new_histories
    # Return the investment recommendation
    investment_recommendation = risk_judge(
        judge_system_prompt,
        judge_user_prompt_fmt,
        stock_op,
        histories,
        generation_config,
    )
    return investment_recommendation


def trader_analyst_chain(
    trader_system_prompt: str,
    risky_analyst_system_prompt: str,
    safe_analyst_system_prompt: str,
    neutral_analyst_system_prompt: str,
    judge_system_prompt: str,
    trader_user_prompt_fmt: str,
    first_round_fmt: str,
    subsequent_round_fmt: str,
    judge_user_prompt_fmt: str,
    stock_op: ops.Op,
    investment_plan_op: ops.Op,
    fundamentals_report_op: ops.Op,
    market_report_op: ops.Op,
    news_report_op: ops.Op,
    social_media_report_op: ops.Op,
    num_rounds: int,
    generation_config: GenerationConfig | None,
    set_cacheable: bool,
) -> ops.Op:
    messages = [
        ops.OpMessage(role="system", content=trader_system_prompt),
        ops.OpMessage(
            role="user",
            content=ops.format_op(
                trader_user_prompt_fmt,
                stock=stock_op,
                investment_plan=investment_plan_op,
            ),
        ),
    ]
    trader_decision = ops.llm_chat(messages, generation_config)
    investment_recommendation = risk_analyst_debate(
        risky_analyst_system_prompt,
        safe_analyst_system_prompt,
        neutral_analyst_system_prompt,
        judge_system_prompt,
        first_round_fmt,
        subsequent_round_fmt,
        judge_user_prompt_fmt,
        stock_op,
        fundamentals_report_op,
        market_report_op,
        news_report_op,
        social_media_report_op,
        trader_decision,
        num_rounds,
        generation_config,
        set_cacheable,
    )
    return investment_recommendation


def fund_manager(
    system_prompt: str,
    user_prompt_fmt: str,
    stock_op: ops.Op,
    investment_recommendations: dict[str, ops.Op],
    generation_config: GenerationConfig | None,
) -> ops.Op:
    messages = [
        ops.OpMessage(role="system", content=system_prompt),
        ops.OpMessage(
            role="user",
            content=ops.format_op(
                user_prompt_fmt, stock=stock_op, **investment_recommendations
            ),
        ),
    ]
    final_recommendation = ops.llm_chat(messages, generation_config)
    return final_recommendation


class TradingAgent(Agent):
    def __init__(
        self,
        trading_input: TradingInput,
        stock_op: ops.Op,
        company_profile_op: ops.Op,
        balance_sheet_op: ops.Op,
        income_stmt_op: ops.Op,
        cashflow_op: ops.Op,
        insider_sentiment_op: ops.Op,
        insider_transactions_op: ops.Op | None,
        stock_price_op: ops.Op,
        stock_stats_op: ops.Op,
        news_chunk_ops: list[ops.Op],
        social_media_chunk_ops: list[ops.Op],
        num_debate_rounds: int,
        server_config: HeliumServerConfig | None,
        generation_config: GenerationConfig | None,
        set_cacheable: bool,
    ) -> None:
        super().__init__(
            server_config=server_config,
            trading_input=trading_input,
            stock_op=stock_op,
            company_profile_op=company_profile_op,
            balance_sheet_op=balance_sheet_op,
            income_stmt_op=income_stmt_op,
            cashflow_op=cashflow_op,
            insider_sentiment_op=insider_sentiment_op,
            insider_transactions_op=insider_transactions_op,
            stock_price_op=stock_price_op,
            stock_stats_op=stock_stats_op,
            news_chunk_ops=news_chunk_ops,
            social_media_chunk_ops=social_media_chunk_ops,
            num_debate_rounds=num_debate_rounds,
            generation_config=generation_config,
            set_cacheable=set_cacheable,
        )

    def build_ops(
        self,
        trading_input: TradingInput,
        stock_op: ops.Op,
        company_profile_op: ops.Op,
        balance_sheet_op: ops.Op,
        income_stmt_op: ops.Op,
        cashflow_op: ops.Op,
        insider_sentiment_op: ops.Op,
        insider_transactions_op: ops.Op | None,
        stock_price_op: ops.Op,
        stock_stats_op: ops.Op,
        news_chunk_ops: list[ops.Op],
        social_media_chunk_ops: list[ops.Op],
        num_debate_rounds: int,
        generation_config: GenerationConfig | None,
        set_cacheable: bool,
    ) -> list[ops.OutputOp]:
        # Stage 1: Analysts
        fundamentals_report = fundamentals_analyst(
            trading_input.fundamentals_system_prompt,
            trading_input.fundamentals_extraction_fmt,
            trading_input.fundamentals_summary_fmt,
            stock_op,
            company_profile_op,
            balance_sheet_op,
            income_stmt_op,
            cashflow_op,
            insider_sentiment_op,
            insider_transactions_op,
            generation_config,
            trading_input.doc_descriptions,
        )
        market_report = market_analyst(
            trading_input.market_system_prompt,
            trading_input.market_extraction_fmt,
            trading_input.market_summary_fmt,
            stock_op,
            stock_price_op,
            stock_stats_op,
            generation_config,
            trading_input.doc_descriptions,
        )
        news_report = news_analyst(
            trading_input.news_system_prompt,
            trading_input.news_extraction_fmt,
            trading_input.news_summary_fmt,
            stock_op,
            news_chunk_ops,
            generation_config,
            trading_input.doc_descriptions,
        )
        social_media_report = social_media_analyst(
            trading_input.social_media_system_prompt,
            trading_input.social_extraction_fmt,
            trading_input.social_summary_fmt,
            stock_op,
            social_media_chunk_ops,
            generation_config,
            trading_input.doc_descriptions,
        )
        # Stage 2: Researchers
        investment_plan = researcher_debate(
            trading_input.bull_system_prompt,
            trading_input.bear_system_prompt,
            trading_input.research_manager_system_prompt,
            trading_input.researcher_first_round_fmt,
            trading_input.researcher_subsequent_round_fmt,
            trading_input.researcher_manager_user_prompt_fmt,
            stock_op,
            fundamentals_report,
            market_report,
            news_report,
            social_media_report,
            num_debate_rounds,
            generation_config,
            set_cacheable,
        )
        # Stage 3: Trader + Risk Analysts
        investment_recommendations: dict[str, ops.Op] = {}
        for (
            trader_name,
            trader_system_prompt,
        ) in trading_input.trader_system_prompts.items():
            investment_recommendation = trader_analyst_chain(
                trader_system_prompt,
                trading_input.risky_analyst_system_prompt,
                trading_input.safe_analyst_system_prompt,
                trading_input.neutral_analyst_system_prompt,
                trading_input.judge_system_prompt,
                trading_input.trader_user_prompt_fmt,
                trading_input.risk_analyst_first_round_fmt,
                trading_input.risk_analyst_subsequent_round_fmt,
                trading_input.judge_user_prompt_fmt,
                stock_op,
                investment_plan,
                fundamentals_report,
                market_report,
                news_report,
                social_media_report,
                num_debate_rounds,
                generation_config,
                set_cacheable,
            )
            investment_recommendations[trader_name] = investment_recommendation
        # Stage 4: Fund Manager
        final_recommendation = fund_manager(
            trading_input.fund_manager_system_prompt,
            trading_input.fund_manager_user_prompt_fmt,
            stock_op,
            investment_recommendations,
            generation_config,
        )
        return [ops.as_output("investment_recommendation", final_recommendation)]


class HeliumTradingProgram(TradingProgram, HeliumProgram):
    def __init__(
        self,
        request_config: HeliumRequestConfig | None = None,
        server_config: HeliumServerConfig | None = None,
        trading_agent: TradingAgent | None = None,
    ) -> None:
        TradingProgram.__init__(self)
        HeliumProgram.__init__(self, server_config=server_config)
        self.trading_agent = trading_agent
        self.request_config = request_config

    def create_agent(
        self, trading_input: TradingInput, set_cacheable: bool, **_
    ) -> TradingAgent:
        unchunked_inputs = {
            "stock": trading_input.stock_symbols,
            "company_profile": trading_input.funda_company_profiles,
            "balance_sheet": trading_input.funda_balance_sheets,
            "income_stmt": trading_input.funda_income_stmts,
            "cashflow": trading_input.funda_cashflow_stmts,
            "insider_sentiment": trading_input.funda_insider_sentiments,
            "stock_price": trading_input.market_stock_prices,
            "stock_stats": trading_input.market_stock_stats,
        }
        news_chunk_inputs = {
            f"news_chunk_{i}": list(chunk)
            for i, chunk in enumerate(zip(*trading_input.news_combined_chunked))
        }
        social_media_chunk_inputs = {
            f"social_media_chunk_{i}": list(chunk)
            for i, chunk in enumerate(zip(*trading_input.social_reddit_post_chunked))
        }

        if self.trading_agent is None:
            trading_agent = TradingAgent(
                trading_input,
                insider_transactions_op=(
                    None
                    if trading_input.funda_insider_transactions is None
                    else ops.input_placeholder("insider_transactions")
                ),
                news_chunk_ops=[
                    ops.input_placeholder(input_name)
                    for input_name in news_chunk_inputs
                ],
                social_media_chunk_ops=[
                    ops.input_placeholder(input_name)
                    for input_name in social_media_chunk_inputs
                ],
                num_debate_rounds=trading_input.num_debate_rounds,
                server_config=self.server_config,
                generation_config=trading_input.generation_config,
                **{
                    f"{input_name}_op": (ops.input_placeholder(input_name))
                    for input_name in unchunked_inputs
                },
                set_cacheable=set_cacheable,
            )
        else:
            trading_agent = self.trading_agent
            # Replace LLM ops' generation and cacheability config
            for op in trading_agent.graph.iter_ops(ops.LLMOp):
                op.config = (
                    trading_input.generation_config or GenerationConfig.from_env()
                )
                # Disable caching dynamic prefixes
                op.cacheable = False
                # if not set_cacheable:
                #     op.cacheable = False

        trading_agent.compile(
            **unchunked_inputs, **news_chunk_inputs, **social_media_chunk_inputs
        )
        return trading_agent

    async def _run(
        self, trading_input: TradingInput
    ) -> tuple[OutputType, HeliumSystemProfile]:
        indices = trading_input.shuffle()

        trading_agent = self.create_agent(trading_input, False)

        self.start_timer("generate")
        response = await trading_agent.run_async(self.request_config)
        self.stop_timer()

        outputs = response.outputs["investment_recommendation"]
        investment_recommendations = (
            self.OutputBuilder().update(zip(indices, outputs)).build()
        )

        return investment_recommendations, response.system_profile

    async def _precompute(
        self,
        trading_input: TradingInput,
        precompute_mode: Literal["none", "only", "both"],
    ) -> HeliumResponse:
        trading_agent = self.create_agent(trading_input, True)

        request_config = (
            HeliumRequestConfig()
            if self.request_config is None
            else self.request_config.model_copy()
        )
        request_config.precompute_mode = precompute_mode
        return await trading_agent.run_async(request_config)
