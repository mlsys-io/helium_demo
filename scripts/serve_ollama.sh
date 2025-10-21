#!/bin/bash

set -e

source .env

OLLAMA_HOST="$VLLM_HOST:$VLLM_PORT" ollama run $LLM_MODEL $@
osascript -e 'tell app "Ollama" to quit'  # Kill the server running in the background