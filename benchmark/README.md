# Benchmarks

## Overview

This folder contains benchmarks to measure and compare performance across multiple systems or configurations. Each benchmark:

- Defines tasks to run (e.g., Map-Reduce QA, Multi-Agent Debate QA).
- Uses a runner to collect results and generate summaries.
- May run local or remote LLM services.

## Components

### BenchmarkMixin

A mixin that manages:

- Timing (start/stop and tracking via keys).
- Profiling results (to record custom performance info).
  Classes needing performance metrics, like benchmark programs, can subclass this to easily measure execution times.

### BenchmarkRunner

Coordinates:

- Running benchmark tasks with various configurations.
- Collecting, printing, and summarizing results.
  Designed to be subclassed if custom setup/cleanup logic is needed (e.g., spinning up servers).

### BenchmarkTask

Represents a single benchmark scenario. It:

- Sets up a program that implements the core logic (e.g., MapReduce, Debate).
- Provides a list of run configurations (multiple trials, parameter variations).
  Subclasses must implement `create_program()` and `get_run_configurations()`.

### bench_programs

Houses concrete program implementations for each benchmark type. Examples:

- MapReduceProgram for Map-Reduce QA tasks.
- DebateProgram for Multi-Agent Debate QA tasks.

By splitting logic into “programs,” we keep the tasks and runner generic.

## How to Implement a New Benchmark

1. Create a new `FooBenchmarkConfig` in `bench_tasks/foo.py` with your custom parameters.
2. Create a `FooBenchmarkTask` extending `BenchmarkTask` with its own `create_program()` and `get_run_configurations()`.
3. Optionally create a new program in `bench_programs/foo.py` that extends `BenchmarkMixin` to define the logic.
4. Register or yield the new task in a generator (e.g., `generate_tasks`) so it is discoverable.
5. Run the benchmark using your runner to evaluate performance.

Use the existing benchmarks as examples. This modular design helps you adapt to the specifics of your scenario while reusing the shared runners and mixin tools.
