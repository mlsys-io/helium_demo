import asyncio
from typing import Any

from bench_programs.trading.base import OutputType, TradingInput, TradingProgram
from bench_programs.utils.parrot import (
    ParrotMixin,
    SemanticFunction,
    SemanticVariable,
    parrot_sampling_config,
    parrot_semantic_function,
    parrot_semantic_variable,
    parrot_start_benchmark,
    parrot_stop_benchmark,
)
from parrot import P

from helium.common import GenerationConfig
from helium.runtime.protocol import HeliumSystemProfile


class ParrotTradingProgram(ParrotMixin, TradingProgram):
    async def _run(
        self, trading_input: TradingInput
    ) -> tuple[OutputType, HeliumSystemProfile]:
        with self.get_vm() as vm:
            generation_config = (
                GenerationConfig.from_env()
                if trading_input.generation_config is None
                else trading_input.generation_config
            )

            # Start benchmarking
            parrot_start_benchmark(vm)

            # Start generation
            self.start_timer("generate")

            # Prepare inputs
            indices = trading_input.shuffle()

            self.start_timer("prepare")
            sampling_config = parrot_sampling_config(generation_config)
            include_insider_transactions = (
                trading_input.funda_insider_transactions is not None
            )
            # Create semantic functions once for the entire workflow
            semantic_functions = self._create_semantic_functions(
                vm,
                trading_input,
                generation_config.model,
                sampling_config,
                include_insider_transactions,
            )

            # Create semantic variables for all inputs
            input_variables: list[dict[str, Any]] = []
            output_variables: list[SemanticVariable] = []
            for original_idx, data in zip(indices, trading_input.iter_data()):
                input_vars = {
                    "index": original_idx,
                    "stock_symbol": P.variable(content=data["stock_symbol"]),
                    "company_profile": P.variable(
                        content=data["funda_company_profile"]
                    ),
                    "balance_sheet": P.variable(content=data["funda_balance_sheet"]),
                    "income_stmt": P.variable(content=data["funda_income_stmt"]),
                    "cashflow_stmt": P.variable(content=data["funda_cashflow_stmt"]),
                    "insider_sentiment": P.variable(
                        content=data["funda_insider_sentiment"]
                    ),
                    "insider_transactions": (
                        P.variable(content=data["funda_insider_transaction"])
                        if include_insider_transactions
                        else None
                    ),
                    "stock_price": P.variable(content=data["market_stock_price"]),
                    "stock_stats": P.variable(content=data["market_stock_stat"]),
                    "news_chunks": [
                        P.variable(content=chunk)
                        for chunk in data["news_combined_chunks"]
                    ],
                    "social_chunks": [
                        P.variable(content=chunk)
                        for chunk in data["social_reddit_post_chunks"]
                    ],
                }
                input_variables.append(input_vars)

                # Create semantic variables for this specific input using shared functions
                output_var = self._create_semantic_variable(
                    semantic_functions,
                    input_vars,
                    trading_input,
                    include_insider_transactions,
                )
                output_variables.append(output_var)
            self.stop_timer()

            # Execute all workflows
            outputs = await asyncio.gather(*[var.aget() for var in output_variables])

            # Stop generation
            self.stop_timer()

            # Stop benchmarking
            system_profile = parrot_stop_benchmark(vm)

            output_builder = self.OutputBuilder()
            for input_vars, output in zip(input_variables, outputs):
                output_builder.add(input_vars["index"], output)

            return output_builder.build(), system_profile

    def _create_semantic_functions(
        self,
        vm: P.VirtualMachine,
        trading_input: TradingInput,
        model: str,
        sampling_config: P.SamplingConfig,
        include_insider_transactions: bool,
    ) -> dict[str, Any]:
        semantic_functions: dict[str, SemanticFunction] = {}

        # Stage 1: Analysts
        # Fundamentals extraction
        doc_types = [
            "balance_sheet",
            "income_statement",
            "cashflow_statement",
            "insider_sentiment",
        ]
        if include_insider_transactions:
            doc_types.append("insider_transactions")
        for doc_type in doc_types:
            formatted_template = trading_input.fundamentals_extraction_fmt.format(
                doc_name=doc_type.replace("_", " "),
                doc_desc=trading_input.doc_descriptions.get(
                    doc_type.replace("_", " "), ""
                ),
            )
            formatted_template = formatted_template.replace("{stock}", "{{stock}}")
            formatted_template = formatted_template.replace(
                "{company_profile}", "{{company_profile}}"
            )
            formatted_template = formatted_template.replace("{doc}", "{{doc}}")
            messages = [
                {"role": "system", "content": trading_input.fundamentals_system_prompt},
                {"role": "user", "content": formatted_template},
                {"role": "assistant", "content": "{{summary}}"},
            ]
            semantic_functions[f"fundamentals_extract_{doc_type}"] = (
                parrot_semantic_function(
                    vm,
                    f"fundamentals_extract_{doc_type}",
                    model,
                    messages,
                    stock=P.Input,
                    doc=P.Input,
                    company_profile=P.Input,
                    summary=P.Output(sampling_config),
                )
            )
        # Fundamentals summary
        parrot_summary_template = trading_input.fundamentals_summary_fmt
        for k in ["stock", "company_profile", *doc_types]:
            parrot_summary_template = parrot_summary_template.replace(
                f"{{{k}}}", f"{{{{{k}}}}}"
            )
        messages = [
            {"role": "system", "content": trading_input.fundamentals_system_prompt},
            {"role": "user", "content": parrot_summary_template},
            {"role": "assistant", "content": "{{fundamentals_report}}"},
        ]
        semantic_functions["fundamentals_aggregate"] = parrot_semantic_function(
            vm,
            "fundamentals_aggregate",
            model,
            messages,
            stock=P.Input,
            company_profile=P.Input,
            balance_sheet=P.Input,
            income_statement=P.Input,
            cashflow_statement=P.Input,
            insider_sentiment=P.Input,
            **(
                dict(insider_transactions=P.Input)
                if include_insider_transactions
                else {}
            ),
            fundamentals_report=P.Output(sampling_config),
        )

        # Market extraction
        market_doc_types = ["stock_price_data", "stock_stats_indicators"]
        for doc_type in market_doc_types:
            formatted_template = trading_input.market_extraction_fmt.format(
                doc_name=doc_type.replace("_", " "),
                doc_desc=trading_input.doc_descriptions.get(
                    doc_type.replace("_", " "), ""
                ),
            )
            formatted_template = formatted_template.replace("{stock}", "{{stock}}")
            formatted_template = formatted_template.replace("{doc}", "{{doc}}")
            messages = [
                {"role": "system", "content": trading_input.market_system_prompt},
                {"role": "user", "content": formatted_template},
                {"role": "assistant", "content": "{{summary}}"},
            ]
            semantic_functions[f"market_extract_{doc_type}"] = parrot_semantic_function(
                vm,
                f"market_extract_{doc_type}",
                model,
                messages,
                stock=P.Input,
                doc=P.Input,
                summary=P.Output(sampling_config),
            )
        # Market summary
        parrot_market_summary_template = trading_input.market_summary_fmt
        for k in ["stock", "stock_price_data", "stock_stats_indicators"]:
            parrot_market_summary_template = parrot_market_summary_template.replace(
                f"{{{k}}}", f"{{{{{k}}}}}"
            )
        messages = [
            {"role": "system", "content": trading_input.market_system_prompt},
            {"role": "user", "content": parrot_market_summary_template},
            {"role": "assistant", "content": "{{market_report}}"},
        ]
        semantic_functions["market_aggregate"] = parrot_semantic_function(
            vm,
            "market_aggregate",
            model,
            messages,
            stock=P.Input,
            stock_price_data=P.Input,
            stock_stats_indicators=P.Input,
            market_report=P.Output(sampling_config),
        )

        # News extraction
        formatted_template = trading_input.news_extraction_fmt.format(
            doc_name="news chunk",
            doc_desc=trading_input.doc_descriptions.get("news chunk", ""),
        )
        formatted_template = formatted_template.replace("{stock}", "{{stock}}")
        formatted_template = formatted_template.replace("{doc}", "{{doc}}")
        messages = [
            {"role": "system", "content": trading_input.news_system_prompt},
            {"role": "user", "content": formatted_template},
            {"role": "assistant", "content": "{{summary}}"},
        ]
        semantic_functions["news_extract_chunk"] = parrot_semantic_function(
            vm,
            "news_extract_chunk",
            model,
            messages,
            stock=P.Input,
            doc=P.Input,
            summary=P.Output(sampling_config),
        )
        # News summary
        news_summary_template = trading_input.news_summary_fmt.replace(
            "{stock}", "{{stock}}"
        )
        for i in range(trading_input.num_news_chunks):
            news_summary_template = news_summary_template.replace(
                f"{{chunk_{i}}}", f"{{{{chunk_{i}}}}}"
            )
        messages = [
            {"role": "system", "content": trading_input.news_system_prompt},
            {"role": "user", "content": news_summary_template},
            {"role": "assistant", "content": "{{news_report}}"},
        ]
        semantic_functions["news_aggregate"] = parrot_semantic_function(
            vm,
            "news_aggregate",
            model,
            messages,
            stock=P.Input,
            **{f"chunk_{i}": P.Input for i in range(trading_input.num_news_chunks)},
            news_report=P.Output(sampling_config),
        )

        # Social extraction
        formatted_template = trading_input.social_extraction_fmt.format(
            doc_name="post chunk",
            doc_desc=trading_input.doc_descriptions.get("post chunk", ""),
        )
        formatted_template = formatted_template.replace("{stock}", "{{stock}}")
        formatted_template = formatted_template.replace("{doc}", "{{doc}}")
        messages = [
            {"role": "system", "content": trading_input.social_media_system_prompt},
            {"role": "user", "content": formatted_template},
            {"role": "assistant", "content": "{{summary}}"},
        ]
        semantic_functions["social_extract_chunk"] = parrot_semantic_function(
            vm,
            "social_extract_chunk",
            model,
            messages,
            stock=P.Input,
            doc=P.Input,
            summary=P.Output(sampling_config),
        )
        # Social summary
        social_summary_template = trading_input.social_summary_fmt.replace(
            "{stock}", "{{stock}}"
        )
        for i in range(trading_input.num_social_chunks):
            social_summary_template = social_summary_template.replace(
                f"{{chunk_{i}}}", f"{{{{chunk_{i}}}}}"
            )
        messages = [
            {"role": "system", "content": trading_input.social_media_system_prompt},
            {"role": "user", "content": social_summary_template},
            {"role": "assistant", "content": "{{social_report}}"},
        ]
        semantic_functions["social_aggregate"] = parrot_semantic_function(
            vm,
            "social_aggregate",
            model,
            messages,
            stock=P.Input,
            **{f"chunk_{i}": P.Input for i in range(trading_input.num_social_chunks)},
            social_report=P.Output(sampling_config),
        )

        # Stage 2: Researcher Debate
        # First rounds
        base_first_template = trading_input.researcher_first_round_fmt
        for k in [
            "stock",
            "fundamentals_report",
            "market_report",
            "news_report",
            "social_media_report",
        ]:
            base_first_template = base_first_template.replace(
                f"{{{k}}}", f"{{{{{k}}}}}"
            )
        messages = [
            {"role": "system", "content": trading_input.bull_system_prompt},
            {"role": "user", "content": base_first_template},
            {"role": "assistant", "content": "{{bull_response}}"},
        ]
        semantic_functions["bull_first_round"] = parrot_semantic_function(
            vm,
            "bull_first_round",
            model,
            messages,
            stock=P.Input,
            fundamentals_report=P.Input,
            market_report=P.Input,
            news_report=P.Input,
            social_media_report=P.Input,
            bull_response=P.Output(sampling_config),
        )
        messages = [
            {"role": "system", "content": trading_input.bear_system_prompt},
            {"role": "user", "content": base_first_template},
            {"role": "assistant", "content": "{{bear_response}}"},
        ]
        semantic_functions["bear_first_round"] = parrot_semantic_function(
            vm,
            "bear_first_round",
            model,
            messages,
            stock=P.Input,
            fundamentals_report=P.Input,
            market_report=P.Input,
            news_report=P.Input,
            social_media_report=P.Input,
            bear_response=P.Output(sampling_config),
        )
        num_rounds = trading_input.num_debate_rounds
        # Subsequent rounds
        if num_rounds > 1:
            researcher_first_template = trading_input.researcher_first_round_fmt
            for k in [
                "stock",
                "fundamentals_report",
                "market_report",
                "news_report",
                "social_media_report",
            ]:
                researcher_first_template = researcher_first_template.replace(
                    f"{{{k}}}", f"{{{{{k}}}}}"
                )
            sub_template_base = trading_input.researcher_subsequent_round_fmt
            for r in range(1, num_rounds):
                for role, system_prompt in [
                    ("bull", trading_input.bull_system_prompt),
                    ("bear", trading_input.bear_system_prompt),
                ]:
                    opponent = "bear" if role == "bull" else "bull"
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": researcher_first_template},
                        {"role": "assistant", "content": f"{{{{{role}_round_0}}}}"},
                    ]
                    # Intermediate history
                    for j in range(1, r):
                        filled_prev = sub_template_base.replace(
                            "{other_response}", f"{{{{{opponent}_round_{j-1}}}}}"
                        )
                        messages.append({"role": "user", "content": filled_prev})
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"{{{{{role}_round_{j}}}}}",
                            }
                        )
                    # Final user prompt for round r
                    filled_final = sub_template_base.replace(
                        "{other_response}", f"{{{{{opponent}_round_{r-1}}}}}"
                    )
                    messages.append({"role": "user", "content": filled_final})
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"{{{{{role}_round_{r}}}}}",
                        }
                    )
                    semantic_functions[f"{role}_round_{r}_with_history"] = (
                        parrot_semantic_function(
                            vm,
                            f"{role}_round_{r}_with_history",
                            model,
                            messages,
                            stock=P.Input,
                            fundamentals_report=P.Input,
                            market_report=P.Input,
                            news_report=P.Input,
                            social_media_report=P.Input,
                            **{f"bull_round_{i}": P.Input for i in range(r)},
                            **{f"bear_round_{i}": P.Input for i in range(r)},
                            **{f"{role}_round_{r}": P.Output(sampling_config)},
                        )
                    )
        # Research manager
        debate_lines = []
        for i in range(num_rounds):
            debate_lines.append(f"**Bull Researcher:** {{{{bull_round_{i}}}}}")
            debate_lines.append(f"**Bear Researcher:** {{{{bear_round_{i}}}}}")
        debate_block = "\n".join(debate_lines)
        manager_user_template = (
            trading_input.researcher_manager_user_prompt_fmt.replace(
                "{debate}", debate_block
            ).replace("{stock}", "{{stock}}")
        )
        manager_messages = [
            {"role": "system", "content": trading_input.research_manager_system_prompt},
            {"role": "user", "content": manager_user_template},
            {"role": "assistant", "content": "{{investment_plan}}"},
        ]
        semantic_functions["research_manager"] = parrot_semantic_function(
            vm,
            "research_manager",
            model,
            manager_messages,
            stock=P.Input,
            **{f"bull_round_{i}": P.Input for i in range(num_rounds)},
            **{f"bear_round_{i}": P.Input for i in range(num_rounds)},
            investment_plan=P.Output(sampling_config),
        )

        # Stage 3: Traders & Analysts Chains
        # Traders
        for (
            trader_name,
            trader_system_prompt,
        ) in trading_input.trader_system_prompts.items():
            trader_template = trading_input.trader_user_prompt_fmt
            trader_template = trader_template.replace("{stock}", "{{stock}}")
            trader_template = trader_template.replace(
                "{investment_plan}", "{{investment_plan}}"
            )
            messages = [
                {"role": "system", "content": trader_system_prompt},
                {"role": "user", "content": trader_template},
                {"role": "assistant", "content": "{{trader_decision}}"},
            ]
            semantic_functions[f"trader_{trader_name}"] = parrot_semantic_function(
                vm,
                f"trader_{trader_name}",
                model,
                messages,
                stock=P.Input,
                investment_plan=P.Input,
                trader_decision=P.Output(sampling_config),
            )
        # Risk analyst
        risk_types = ["risky", "safe", "neutral"]
        # First rounds
        first_template = trading_input.risk_analyst_first_round_fmt
        for k in [
            "stock",
            "fundamentals_report",
            "market_report",
            "news_report",
            "social_media_report",
            "trader_decision",
        ]:
            first_template = first_template.replace(f"{{{k}}}", f"{{{{{k}}}}}")
        for risk_type, risk_prompt in [
            ("risky", trading_input.risky_analyst_system_prompt),
            ("safe", trading_input.safe_analyst_system_prompt),
            ("neutral", trading_input.neutral_analyst_system_prompt),
        ]:
            messages = [
                {"role": "system", "content": risk_prompt},
                {"role": "user", "content": first_template},
                {"role": "assistant", "content": f"{{{{{risk_type}_round_0}}}}"},
            ]
            semantic_functions[f"risk_{risk_type}_round_0"] = parrot_semantic_function(
                vm,
                f"risk_{risk_type}_round_0",
                model,
                messages,
                stock=P.Input,
                fundamentals_report=P.Input,
                market_report=P.Input,
                news_report=P.Input,
                social_media_report=P.Input,
                trader_decision=P.Input,
                **{f"{risk_type}_round_0": P.Output(sampling_config)},
            )
        # Subsequent rounds
        if num_rounds > 1:
            sub_template_base = trading_input.risk_analyst_subsequent_round_fmt
            for r in range(1, num_rounds):
                for risk_type, risk_prompt in [
                    ("risky", trading_input.risky_analyst_system_prompt),
                    ("safe", trading_input.safe_analyst_system_prompt),
                    ("neutral", trading_input.neutral_analyst_system_prompt),
                ]:
                    messages = [
                        {"role": "system", "content": risk_prompt},
                        {"role": "user", "content": first_template},
                        {
                            "role": "assistant",
                            "content": f"{{{{{risk_type}_round_0}}}}",
                        },
                    ]
                    # Intermediate history
                    for j in range(1, r):
                        filled_prev = sub_template_base
                        for rt in risk_types:
                            filled_prev = filled_prev.replace(
                                f"{{{rt}}}", f"{{{{{rt}_round_{j-1}}}}}"
                            )
                        messages.append({"role": "user", "content": filled_prev})
                        messages.append(
                            {
                                "role": "assistant",
                                "content": f"{{{{{risk_type}_round_{j}}}}}",
                            }
                        )
                    # Final user prompt to elicit round r
                    filled_final = sub_template_base
                    for rt in risk_types:
                        filled_final = filled_final.replace(
                            f"{{{rt}}}", f"{{{{{rt}_round_{r-1}}}}}"
                        )
                    messages.append({"role": "user", "content": filled_final})
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"{{{{{risk_type}_round_{r}}}}}",
                        }
                    )
                    semantic_functions[f"risk_{risk_type}_round_{r}_with_history"] = (
                        parrot_semantic_function(
                            vm,
                            f"risk_{risk_type}_round_{r}_with_history",
                            model,
                            messages,
                            stock=P.Input,
                            fundamentals_report=P.Input,
                            market_report=P.Input,
                            news_report=P.Input,
                            social_media_report=P.Input,
                            trader_decision=P.Input,
                            **{f"risky_round_{i}": P.Input for i in range(r)},
                            **{f"safe_round_{i}": P.Input for i in range(r)},
                            **{f"neutral_round_{i}": P.Input for i in range(r)},
                            **{f"{risk_type}_round_{r}": P.Output(sampling_config)},
                        )
                    )
        # Risk judge
        risk_debate_block = []
        for i in range(num_rounds):
            risk_debate_block.append(f"**Risky Risk Analyst:** {{{{risky_round_{i}}}}}")
            risk_debate_block.append(f"**Safe Risk Analyst:** {{{{safe_round_{i}}}}}")
            risk_debate_block.append(
                f"**Neutral Risk Analyst:** {{{{neutral_round_{i}}}}}"
            )
        risk_debate_block_str = "\n".join(risk_debate_block)
        for trader_name in trading_input.trader_system_prompts.keys():
            judge_user_template = trading_input.judge_user_prompt_fmt.replace(
                "{debate}", risk_debate_block_str
            ).replace("{stock}", "{{stock}}")
            judge_messages = [
                {"role": "system", "content": trading_input.judge_system_prompt},
                {"role": "user", "content": judge_user_template},
                {"role": "assistant", "content": "{{investment_recommendation}}"},
            ]
            semantic_functions[f"risk_judge_{trader_name}"] = parrot_semantic_function(
                vm,
                f"risk_judge_{trader_name}",
                model,
                judge_messages,
                stock=P.Input,
                **{f"risky_round_{i}": P.Input for i in range(num_rounds)},
                **{f"safe_round_{i}": P.Input for i in range(num_rounds)},
                **{f"neutral_round_{i}": P.Input for i in range(num_rounds)},
                investment_recommendation=P.Output(sampling_config),
            )

        # Stage 4: Fund Manager
        fund_manager_template = trading_input.fund_manager_user_prompt_fmt
        for k in ["stock", *trading_input.trader_system_prompts]:
            fund_manager_template = fund_manager_template.replace(
                f"{{{k}}}", f"{{{{{k}}}}}"
            )
        messages = [
            {"role": "system", "content": trading_input.fund_manager_system_prompt},
            {"role": "user", "content": fund_manager_template},
            {"role": "assistant", "content": "{{final_recommendation}}"},
        ]
        semantic_functions["fund_manager"] = parrot_semantic_function(
            vm,
            "fund_manager",
            model,
            messages,
            stock=P.Input,
            **{
                trader_name: P.Input
                for trader_name in trading_input.trader_system_prompts
            },
            final_recommendation=P.Output(sampling_config),
        )

        return semantic_functions

    def _create_semantic_variable(
        self,
        semantic_functions: dict[str, Any],
        input_vars: dict[str, Any],
        trading_input: TradingInput,
        include_insider_transactions: bool,
    ) -> SemanticVariable:
        # Stage 1: Analysts
        # Fundamentals
        fundamentals_extracts: dict[str, SemanticVariable] = {}
        doc_mapping = {
            "balance_sheet": input_vars["balance_sheet"],
            "income_statement": input_vars["income_stmt"],
            "cashflow_statement": input_vars["cashflow_stmt"],
            "insider_sentiment": input_vars["insider_sentiment"],
        }
        if include_insider_transactions:
            doc_mapping["insider_transactions"] = input_vars["insider_transactions"]
        for doc_type, doc_var in doc_mapping.items():
            fundamentals_extracts[doc_type] = parrot_semantic_variable(
                semantic_functions[f"fundamentals_extract_{doc_type}"],
                stock=input_vars["stock_symbol"],
                doc=doc_var,
                company_profile=input_vars["company_profile"],
            )
        fundamentals_report = parrot_semantic_variable(
            semantic_functions["fundamentals_aggregate"],
            stock=input_vars["stock_symbol"],
            company_profile=input_vars["company_profile"],
            **fundamentals_extracts,
        )
        # Market
        market_extracts: dict[str, SemanticVariable] = {}
        for doc_type, doc_var in {
            "stock_price_data": input_vars["stock_price"],
            "stock_stats_indicators": input_vars["stock_stats"],
        }.items():
            market_extracts[doc_type] = parrot_semantic_variable(
                semantic_functions[f"market_extract_{doc_type}"],
                stock=input_vars["stock_symbol"],
                doc=doc_var,
            )
        market_report = parrot_semantic_variable(
            semantic_functions["market_aggregate"],
            stock=input_vars["stock_symbol"],
            **market_extracts,
        )
        # News
        news_chunk_summaries = {}
        for i, chunk in enumerate(input_vars["news_chunks"]):
            news_chunk_summaries[f"chunk_{i}"] = parrot_semantic_variable(
                semantic_functions["news_extract_chunk"],
                stock=input_vars["stock_symbol"],
                doc=chunk,
            )
        news_report = parrot_semantic_variable(
            semantic_functions["news_aggregate"],
            stock=input_vars["stock_symbol"],
            **news_chunk_summaries,
        )
        # Social
        social_chunk_summaries = {}
        for i, chunk in enumerate(input_vars["social_chunks"]):
            social_chunk_summaries[f"chunk_{i}"] = parrot_semantic_variable(
                semantic_functions["social_extract_chunk"],
                stock=input_vars["stock_symbol"],
                doc=chunk,
            )
        social_report = parrot_semantic_variable(
            semantic_functions["social_aggregate"],
            stock=input_vars["stock_symbol"],
            **social_chunk_summaries,
        )

        # Stage 2: Researcher Debate
        num_rounds = trading_input.num_debate_rounds
        bull_rounds: list[SemanticVariable] = []
        bear_rounds: list[SemanticVariable] = []
        # First rounds
        bull_rounds.append(
            parrot_semantic_variable(
                semantic_functions["bull_first_round"],
                stock=input_vars["stock_symbol"],
                fundamentals_report=fundamentals_report,
                market_report=market_report,
                news_report=news_report,
                social_media_report=social_report,
            )
        )
        bear_rounds.append(
            parrot_semantic_variable(
                semantic_functions["bear_first_round"],
                stock=input_vars["stock_symbol"],
                fundamentals_report=fundamentals_report,
                market_report=market_report,
                news_report=news_report,
                social_media_report=social_report,
            )
        )
        # Subsequent rounds
        for r in range(1, num_rounds):
            bull_histories = {f"bull_round_{i}": vr for i, vr in enumerate(bull_rounds)}
            bear_histories = {f"bear_round_{i}": vr for i, vr in enumerate(bear_rounds)}
            new_bull_round = parrot_semantic_variable(
                semantic_functions[f"bull_round_{r}_with_history"],
                stock=input_vars["stock_symbol"],
                fundamentals_report=fundamentals_report,
                market_report=market_report,
                news_report=news_report,
                social_media_report=social_report,
                **bull_histories,
                **bear_histories,
            )
            new_bear_round = parrot_semantic_variable(
                semantic_functions[f"bear_round_{r}_with_history"],
                stock=input_vars["stock_symbol"],
                fundamentals_report=fundamentals_report,
                market_report=market_report,
                news_report=news_report,
                social_media_report=social_report,
                **bull_histories,
                **bear_histories,
            )
            bull_rounds.append(new_bull_round)
            bear_rounds.append(new_bear_round)

        investment_plan = parrot_semantic_variable(
            semantic_functions["research_manager"],
            stock=input_vars["stock_symbol"],
            **{f"bull_round_{i}": vr for i, vr in enumerate(bull_rounds)},
            **{f"bear_round_{i}": vr for i, vr in enumerate(bear_rounds)},
        )

        # Stage 3: Traders & Analysts Chains
        trader_recommendations: dict[str, SemanticVariable] = {}
        for trader_name in trading_input.trader_system_prompts.keys():
            trader_decision = parrot_semantic_variable(
                semantic_functions[f"trader_{trader_name}"],
                stock=input_vars["stock_symbol"],
                investment_plan=investment_plan,
            )
            risky_rounds: list[SemanticVariable] = []
            safe_rounds: list[SemanticVariable] = []
            neutral_rounds: list[SemanticVariable] = []
            # First rounds
            risky_rounds.append(
                parrot_semantic_variable(
                    semantic_functions["risk_risky_round_0"],
                    stock=input_vars["stock_symbol"],
                    fundamentals_report=fundamentals_report,
                    market_report=market_report,
                    news_report=news_report,
                    social_media_report=social_report,
                    trader_decision=trader_decision,
                )
            )
            safe_rounds.append(
                parrot_semantic_variable(
                    semantic_functions["risk_safe_round_0"],
                    stock=input_vars["stock_symbol"],
                    fundamentals_report=fundamentals_report,
                    market_report=market_report,
                    news_report=news_report,
                    social_media_report=social_report,
                    trader_decision=trader_decision,
                )
            )
            neutral_rounds.append(
                parrot_semantic_variable(
                    semantic_functions["risk_neutral_round_0"],
                    stock=input_vars["stock_symbol"],
                    fundamentals_report=fundamentals_report,
                    market_report=market_report,
                    news_report=news_report,
                    social_media_report=social_report,
                    trader_decision=trader_decision,
                )
            )
            for r in range(1, num_rounds):
                risky_histories = {
                    f"risky_round_{i}": vr for i, vr in enumerate(risky_rounds)
                }
                safe_histories = {
                    f"safe_round_{i}": vr for i, vr in enumerate(safe_rounds)
                }
                neutral_histories = {
                    f"neutral_round_{i}": vr for i, vr in enumerate(neutral_rounds)
                }
                new_risky_round = parrot_semantic_variable(
                    semantic_functions[f"risk_risky_round_{r}_with_history"],
                    stock=input_vars["stock_symbol"],
                    fundamentals_report=fundamentals_report,
                    market_report=market_report,
                    news_report=news_report,
                    social_media_report=social_report,
                    trader_decision=trader_decision,
                    **risky_histories,
                    **safe_histories,
                    **neutral_histories,
                )
                new_safe_round = parrot_semantic_variable(
                    semantic_functions[f"risk_safe_round_{r}_with_history"],
                    stock=input_vars["stock_symbol"],
                    fundamentals_report=fundamentals_report,
                    market_report=market_report,
                    news_report=news_report,
                    social_media_report=social_report,
                    trader_decision=trader_decision,
                    **risky_histories,
                    **safe_histories,
                    **neutral_histories,
                )
                new_neutral_round = parrot_semantic_variable(
                    semantic_functions[f"risk_neutral_round_{r}_with_history"],
                    stock=input_vars["stock_symbol"],
                    fundamentals_report=fundamentals_report,
                    market_report=market_report,
                    news_report=news_report,
                    social_media_report=social_report,
                    trader_decision=trader_decision,
                    **risky_histories,
                    **safe_histories,
                    **neutral_histories,
                )
                risky_rounds.append(new_risky_round)
                safe_rounds.append(new_safe_round)
                neutral_rounds.append(new_neutral_round)
            trader_recommendations[trader_name] = parrot_semantic_variable(
                semantic_functions[f"risk_judge_{trader_name}"],
                stock=input_vars["stock_symbol"],
                **{f"risky_round_{i}": vr for i, vr in enumerate(risky_rounds)},
                **{f"safe_round_{i}": vr for i, vr in enumerate(safe_rounds)},
                **{f"neutral_round_{i}": vr for i, vr in enumerate(neutral_rounds)},
            )
        # Stage 4: Fund Manager
        final_recommendation = parrot_semantic_variable(
            semantic_functions["fund_manager"],
            stock=input_vars["stock_symbol"],
            **trader_recommendations,
        )

        return final_recommendation
