# Qwen 3.5 fairseq2 Implementation Plan

> Companion to `qwen35_architecture.md`. All code goes in `src/fairseq2/models/qwen/`.

## Directory Layout (final state)

```
src/fairseq2/models/qwen/
├── __init__.py              # extend exports
├── config.py                # + Qwen35Config, Qwen35MoeConfig
├── factory.py               # + Qwen35Factory, Qwen35MoeFactory
├── attention.py             # NEW: Qwen35Attention
├── gated_delta_net.py       # NEW: GatedDeltaNet, GatedDeltaNetState
├── decoder_layer.py         # NEW: Qwen35DecoderLayer
├── moe.py                   # NEW: Qwen35MoeBlock, TopKRouter, Experts
├── interop.py               # extend with qwen35 + moe key maps
├── tokenizer.py             # shared (no changes)
├── hub.py                   # + qwen35 hub accessors
```

---

## Phase 1: Core Modules

### 1.1 `gated_delta_net.py` — GatedDeltaNet + State

**New classes:**
- `GatedDeltaNetState(IncrementalState)` — stores `conv_state` and `recurrent_state`
- `RMSNormGated(Module)` — `RMSNorm(x) * silu(gate)`, weight=ones
- `GatedDeltaNet(Module)` — the full linear attention module

**GatedDeltaNet members:**
```python
in_proj_qkv: Linear(hidden, key_dim*2 + value_dim, bias=False)
in_proj_z:   Linear(hidden, value_dim, bias=False)
in_proj_b:   Linear(hidden, num_v_heads, bias=False)
in_proj_a:   Linear(hidden, num_v_heads, bias=False)
conv1d:      Conv1d(conv_dim, conv_dim, kernel=4, groups=conv_dim, bias=False)
dt_bias:     Parameter(ones(num_v_heads))
A_log:       Parameter(log(uniform(0,16)))
norm:        RMSNormGated(head_v_dim, eps=1e-6)
out_proj:    Linear(value_dim, hidden, bias=False)
```

**Forward signature:** `forward(seqs, *, state_bag=None) -> Tensor`

**Port PyTorch fallbacks from HF** (no external deps):
- `torch_causal_conv1d_update` (HF lines 299-314)
- `torch_chunk_gated_delta_rule` (HF lines 323-400)
- `torch_recurrent_gated_delta_rule` (HF lines 403-442)
- `l2norm` helper (HF lines 317-320)

**State management:**
```python
class GatedDeltaNetState(IncrementalState):
    conv_state: Tensor      # (B, conv_dim, kernel_size - 1)
    recurrent_state: Tensor  # (B, num_v_heads, head_k_dim, head_v_dim)

    def reorder(self, new_order):  # beam search support
        self.conv_state = self.conv_state.index_select(0, new_order)
        self.recurrent_state = self.recurrent_state.index_select(0, new_order)
```

### 1.2 `attention.py` — Qwen35Attention

**New class:** `Qwen35Attention(Module)` — inspired by `OLMOMultiheadAttention`

**Key differences from `StandardMultiheadAttention`:**
1. `q_proj` output = `num_heads * head_dim * 2` (doubled for gate)
2. Split Q output into `query_states` and `gate`
3. QK-Norm: `RMSNorm(head_dim)` applied per-head before reshape
4. Partial RoPE via `pos_encoder` with `encoding_dim = head_dim * 0.25`
5. Output gating: `attn_output * sigmoid(gate)`

**Forward signature:** Same as `MultiheadAttention`:
```python
forward(seqs, seqs_layout, attn_bias_cache, *, state_bag=None) -> Tensor
```

### 1.3 `decoder_layer.py` — Qwen35DecoderLayer

**New class:** `Qwen35DecoderLayer(TransformerLMDecoderLayer)`

```python
class Qwen35DecoderLayer(TransformerLMDecoderLayer):
    layer_type: str  # "linear_attention" or "full_attention"
    # Conditionally holds ONE of:
    linear_attn: GatedDeltaNet | None
    self_attn: Qwen35Attention | None
    # Always holds:
    ffn: FeedForwardNetwork
    self_attn_layer_norm: RMSNorm  # pre-attention norm
    ffn_layer_norm: RMSNorm        # pre-ffn norm
```

Pre-norm order: `norm → attn/gdn → add → norm → ffn → add`

Attribute names `linear_attn` / `self_attn` match HF for clean interop mapping.

---

## Phase 2: Dense Model Integration

### 2.1 Config (`config.py`)

```python
QWEN35_FAMILY: Final = "qwen3_5"

@dataclass(kw_only=True)
class Qwen35Config:
    model_dim: int = 4096
    max_seq_len: int = 32_768
    vocab_size: int = 248_320
    pad_idx: int | None = None
    tied_embeddings: bool = False
    num_layers: int = 32
    num_attn_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    ffn_inner_dim: int = 12_288
    dropout_p: float = 0.0
    partial_rotary_factor: float = 0.25
    rope_theta: float = 1_000_000.0
    layer_types: list[str] | None = None
    full_attention_interval: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
```

Register arch configs: `qwen35_27b`, etc.

### 2.2 Factory (`factory.py`)

```python
class Qwen35Factory:
    def create_model(self) -> TransformerLM
    def create_embedding(self) -> Embedding
    def create_decoder_frontend(self, embed) -> TransformerFrontend
    def create_decoder(self) -> TransformerLMDecoder
    def create_position_encoder(self) -> RotaryEncoder
        # encoding_dim = int(head_dim * partial_rotary_factor) = 64
    def create_decoder_layer(self, layer_idx, pos_encoder) -> Qwen35DecoderLayer
    def create_gated_attention(self, layer_idx, pos_encoder) -> Qwen35Attention
    def create_gated_delta_net(self, layer_idx) -> GatedDeltaNet
    def create_ffn(self, layer_idx) -> GLUFeedForwardNetwork
    def create_layer_norm(self) -> RMSNorm
    def create_final_projection(self, embed) -> Projection
```

### 2.3 Interop (`interop.py`)

**Key maps — full attention layers:**
```
model.layers.{i}.self_attn.q_proj      → decoder.layers.{i}.self_attn.q_proj
model.layers.{i}.self_attn.k_proj      → decoder.layers.{i}.self_attn.k_proj
model.layers.{i}.self_attn.v_proj      → decoder.layers.{i}.self_attn.v_proj
model.layers.{i}.self_attn.o_proj      → decoder.layers.{i}.self_attn.output_proj
model.layers.{i}.self_attn.q_norm      → decoder.layers.{i}.self_attn.q_norm
model.layers.{i}.self_attn.k_norm      → decoder.layers.{i}.self_attn.k_norm
```

**Key maps — linear attention layers:**
```
model.layers.{i}.linear_attn.in_proj_qkv → decoder.layers.{i}.linear_attn.in_proj_qkv
model.layers.{i}.linear_attn.in_proj_z   → decoder.layers.{i}.linear_attn.in_proj_z
model.layers.{i}.linear_attn.in_proj_b   → decoder.layers.{i}.linear_attn.in_proj_b
model.layers.{i}.linear_attn.in_proj_a   → decoder.layers.{i}.linear_attn.in_proj_a
model.layers.{i}.linear_attn.conv1d      → decoder.layers.{i}.linear_attn.conv1d
model.layers.{i}.linear_attn.dt_bias     → decoder.layers.{i}.linear_attn.dt_bias
model.layers.{i}.linear_attn.A_log       → decoder.layers.{i}.linear_attn.A_log
model.layers.{i}.linear_attn.norm        → decoder.layers.{i}.linear_attn.norm
model.layers.{i}.linear_attn.out_proj    → decoder.layers.{i}.linear_attn.out_proj
```

**Key maps — FFN + norms + embeddings:**
```
model.layers.{i}.mlp.gate_proj           → decoder.layers.{i}.ffn.gate_proj
model.layers.{i}.mlp.up_proj             → decoder.layers.{i}.ffn.inner_proj
model.layers.{i}.mlp.down_proj           → decoder.layers.{i}.ffn.output_proj
model.layers.{i}.input_layernorm         → decoder.layers.{i}.self_attn_layer_norm
model.layers.{i}.post_attention_layernorm → decoder.layers.{i}.ffn_layer_norm
model.embed_tokens                        → decoder_frontend.embed
model.norm                                → decoder.layer_norm
lm_head                                   → final_proj
```

**Weight conversion in `convert_qwen35_state_dict`:**
- `Qwen3_5RMSNorm` weights: `weight += 1.0` (they use `(1+w)` formula)
- `RMSNormGated` weights in GDN: **no conversion** (standard `w * norm(x)`)
- Identify norm keys by matching `layernorm` or `model.norm` patterns

### 2.4 Registration (`composition/models.py`)

```python
register_model_family(
    container, QWEN35_FAMILY,
    kls=TransformerLM, config_kls=Qwen35Config,
    factory=create_qwen35_model,
    state_dict_converter=convert_qwen35_state_dict,
    compiler=compile_transformer_lm,
    fsdp_applier=apply_fsdp_to_transformer_lm,
)
register_qwen35_configs(container)
```

---

## Phase 3: Component Tests

| Test | What it validates |
|------|-------------------|
| `test_gated_delta_net_forward` | GDN output matches HF for same input + weights |
| `test_gated_delta_net_incremental` | Step-by-step decode matches full forward |
| `test_qwen35_attention_forward` | Gated attention matches HF |
| `test_qwen35_attention_incremental` | KV cache decode matches |
| `test_partial_rope` | 64/256 dim rotation matches HF |
| `test_rmsnorm_conversion` | `weight+1` conversion is numerically exact |
| `test_qwen35_interop_keys` | State dict round-trip HF↔fs2 |
| `test_qwen35_e2e_logits` | Full model forward matches HF on real weights |

**Partial RoPE verification task:** Create `RotaryEncoder(encoding_dim=64, max_seq_len=32768)`, apply to tensor with `head_dim=256`, verify only first 64 dims are modified.

---

## Phase 4: MoE Support

### 4.1 `moe.py` — MoE Block

**New classes:**
- `Qwen35TopKRouter(Module)` — softmax + top-k + renormalize
- `Qwen35Experts(Module)` — 3D parameter experts
- `Qwen35MoeBlock(FeedForwardNetwork)` — drop-in FFN replacement

```python
class Qwen35TopKRouter(Module):
    weight: Parameter(num_experts, hidden_dim)  # init zeros
    def forward(self, x) -> (logits, scores, indices):
        logits = softmax(linear(x, self.weight), dim=-1)
        values, indices = topk(logits, top_k)
        values /= values.sum(dim=-1, keepdim=True)
        return logits, values, indices

class Qwen35Experts(Module):
    gate_up_proj: Parameter(num_experts, 2*inter, hidden)
    down_proj: Parameter(num_experts, hidden, inter)

class Qwen35MoeBlock(FeedForwardNetwork):
    router: Qwen35TopKRouter
    experts: Qwen35Experts
    shared_expert: GLUFeedForwardNetwork
    shared_expert_gate: Linear(hidden, 1, bias=False)
    def forward(self, seqs) -> Tensor:
        expert_out = self.experts(seqs, *self.router(seqs)[1:])
        shared_out = sigmoid(self.shared_expert_gate(seqs)) * self.shared_expert(seqs)
        return expert_out + shared_out
```

### 4.2 Config + Factory

```python
QWEN35_MOE_FAMILY: Final = "qwen3_5_moe"

@dataclass(kw_only=True)
class Qwen35MoeConfig(Qwen35Config):
    model_dim: int = 2048
    num_layers: int = 40
    num_key_value_heads: int = 2
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    router_aux_loss_coef: float = 0.001

class Qwen35MoeFactory(Qwen35Factory):
    def create_ffn(self, layer_idx) -> Qwen35MoeBlock
```

### 4.3 MoE Interop Keys (additional)

```
model.layers.{i}.mlp.gate               → decoder.layers.{i}.ffn.router
model.layers.{i}.mlp.experts            → decoder.layers.{i}.ffn.experts
model.layers.{i}.mlp.shared_expert      → decoder.layers.{i}.ffn.shared_expert
model.layers.{i}.mlp.shared_expert_gate → decoder.layers.{i}.ffn.shared_expert_gate
```

### 4.4 MoE Tests

| Test | What it validates |
|------|-------------------|
| `test_topk_router` | softmax+topk+renorm matches HF |
| `test_moe_block` | Full block output matches HF |
| `test_shared_expert_gating` | `sigmoid(gate) * shared` formula |
| `test_moe_interop` | MoE state dict round-trip |

---

## Phase 5: Integration & Polish

- **Asset cards** (`assets/cards/models/qwen35.yaml`) for published models
- **Hub accessors** in `hub.py`
- **Sharder/FSDP specs** for tensor parallelism
- **Documentation** updates
- **Load balancing loss** support (optional, for training)
