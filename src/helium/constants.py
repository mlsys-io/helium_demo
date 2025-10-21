# Best config according to profiling
DEFAULT_MAX_NUM_SEQS: int = 1024
DEFAULT_MAX_NUM_BATCHED_TOKENS: int = 4096

VLLM_CACHE_CAPACITY: dict[str, dict[str, int]] = {
    "NVIDIA H100 NVL": {
        "meta-llama/Meta-Llama-3-8B-Instruct": 543440,
        "meta-llama/Llama-3.1-8B-Instruct": 542960,
        "Qwen/Qwen3-8B": 478560,
        "Qwen/Qwen3-14B": 350304,
        "Qwen/Qwen3-32B": 81648,
    },
    "NVIDIA RTX 6000 Ada Generation": {
        "meta-llama/Meta-Llama-3-8B-Instruct": 208128,
        "meta-llama/Llama-3.1-8B-Instruct": 207536,
        "Qwen/Qwen3-8B": 191568,
        "Qwen/Qwen3-14B": 92000,
    },
}
LLM_CONTEXT_LENGTH: dict[str, int] = {
    "meta-llama/Meta-Llama-3-8B-Instruct": 8192,
    "meta-llama/Llama-3.1-8B-Instruct": 131072,
    "Qwen/Qwen3-8B": 40960,
    "Qwen/Qwen3-14B": 40960,
    "Qwen/Qwen3-32B": 40960,
}
