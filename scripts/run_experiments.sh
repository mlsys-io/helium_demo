#!/bin/bash

set -eou pipefail

python benchmark/main.py -v >| logs/kvflow_trading_5.log
mv benchmark/bench.py benchmark/bench_1.py
mv benchmark/bench_2.py benchmark/bench.py
python benchmark/main.py -v >| logs/kvflow_microbenchmarks_5.log
mv benchmark/bench.py benchmark/bench_2.py
mv benchmark/bench_1.py benchmark/bench.py
