#!/bin/bash

set -e

# See https://docs.vllm.ai/en/latest/serving/engine_args.html for more information
python -m helium.runtime.utils.vllm.server $@
