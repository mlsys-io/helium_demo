#include <torch/extension.h>
#include <c10/util/Optional.h>

void single_query_cached_kv_attention(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& head_mapping,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes);

void single_query_cached_kv_prev_attention(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& head_mapping,    // [num_heads]
  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  torch::Tensor& qk_maxs,    // [num_seqs]
  torch::Tensor& exp_sums,   // [num_seqs]
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes);

void single_query_cached_kv_post_attention(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& head_mapping,    // [num_heads]
  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  torch::Tensor& prev_qk_maxs,    // [num_seqs]
  torch::Tensor& prev_exp_sums,   // [num_seqs]
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "single_query_cached_kv_attention",
    &single_query_cached_kv_attention,
    "Compute the attention between an input query and the cached key/value tensors");
  m.def(
    "single_query_cached_kv_prev_attention",
    &single_query_cached_kv_prev_attention,
    "Compute the attention between an input query and the cached key/value tensors and log middle results");
  m.def(
    "single_query_cached_kv_post_attention",
    &single_query_cached_kv_post_attention,
    "Compute the attention between an input query and the cached key/value tensors based on previous results");
}
