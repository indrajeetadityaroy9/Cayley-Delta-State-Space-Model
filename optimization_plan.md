KSSM Throughput Optimization Plan: SOTA-Competitive Block                                                                                                                                                      
   
     Context                                                                                                                                                                                                        
                                                                                                                                                                                                                  
     Profiling the CUDA-migrated KSSM block (B=4, L=1024, d_model=384, H100) revealed the fused CUDA kernels account for only 15% of CUDA time. The actual bottlenecks are:

     - 39%: Unfused elementwise PyTorch ops (~600 separate kernel launches for sigmoid, softplus, normalize, multiply, add, etc.)
     - 35%: GEMM from 12 nn.Linear projections per block (vs 3-4 in Mamba-2/GatedDeltaNet)

     Current throughput: ~246K tokens/sec. Target: >500K tokens/sec.

     Root cause: 8 of 12 projections consume the same input (x_conv_cast) but are separate nn.Linear modules, each launching a separate GEMM. Between the GEMMs, ~15 elementwise Python ops launch individual
     CUDA kernels with high overhead.

     ---
     Phase 1: Fuse Gate Projections (Python-only)

     Merge 7 projections that share input x_conv_cast into one nn.Linear. Cuts 7 GEMM launches to 1.

     Current (7 separate projections from x_conv_cast):
     ┌─────────────────┬───────────────┬──────┬──────────────────┐
     │     Module      │     Shape     │ Bias │      Output      │
     ├─────────────────┼───────────────┼──────┼──────────────────┤
     │ dynamics_proj   │ (d_inner, 2H) │ Yes  │ alpha_raw, omega │
     ├─────────────────┼───────────────┼──────┼──────────────────┤
     │ selection_B     │ (d_inner, H)  │ No   │ sel_B            │
     ├─────────────────┼───────────────┼──────┼──────────────────┤
     │ selection_C     │ (d_inner, H)  │ No   │ sel_C            │
     ├─────────────────┼───────────────┼──────┼──────────────────┤
     │ selection_dt    │ (d_inner, H)  │ No   │ sel_dt           │
     ├─────────────────┼───────────────┼──────┼──────────────────┤
     │ beta_proj       │ (d_inner, H)  │ Yes  │ beta             │
     ├─────────────────┼───────────────┼──────┼──────────────────┤
     │ recurrence_gate │ (d_inner, H)  │ Yes  │ r_gate           │
     ├─────────────────┼───────────────┼──────┼──────────────────┤
     │ utility_gate    │ (d_inner, H)  │ Yes  │ u_gate           │
     └─────────────────┴───────────────┴──────┴──────────────────┘
     After: single gate_proj with layout:

     self.gate_proj = nn.Linear(self.d_inner, self.n_heads * 7, bias=True)
     # Output layout (split along last dim):
     #   [0:H]    alpha_raw     (softplus → alpha)
     #   [H:2H]   omega_raw     (+ RoPE → omega)
     #   [2H:3H]  sel_B_raw     (raw scalar, sign used post-normalize)
     #   [3H:4H]  sel_C_raw     (raw scalar, sign used post-normalize)
     #   [4H:5H]  sel_dt_raw    (softplus → added to dt)
     #   [5H:6H]  beta_raw      (sigmoid → beta)
     #   [6H:7H]  r_gate_raw    (sigmoid → r_gate)

     Note: Slot 7 (utility gate) is removed in Phase 2. If Phase 2 is deferred, use 8*H output.

     Initialization

     # gate_proj.bias initialization:
     bias[:H]     = 0.0      # alpha: spectral init overwrites in apply_spectral_init
     bias[H:2H]   = 0.0      # omega: spectral init overwrites
     bias[2H:5H]  = 0.0      # sel_B, sel_C, sel_dt: was no-bias (zero = equivalent)
     bias[5H:6H]  = 0.0      # beta: sigmoid(0) = 0.5
     bias[6H:7H]  = 1.0      # recurrence_gate: sigmoid(1) ≈ 0.73

     # gate_proj.weight initialization:
     weight[:2H]   ~ N(0, stds["dynamics_proj"])
     weight[2H:7H] ~ N(0, 1/sqrt(d_inner))    # LeCun fan-in for selections+gates

     Files changed

     - src/kssm/models/kssm_block.py: Replace 7 nn.Linear with gate_proj; update _init_weights(); update forward() to split gate_proj output and apply activations
     - src/kssm/models/components.py: apply_spectral_init references change from block.dynamics_proj.bias → block.gate_proj.bias
     - src/kssm/training/optim.py: Update ssm_keywords list — replace individual names with "gate_proj", keep "log_dt_scale"

     ---
     Phase 2: Remove Utility Gating

     Eliminates 2 projections (utility_gate in gate_proj, state_gate_proj), EMA bookkeeping, and associated elementwise ops. gate_proj output reduces from 8H to 7H.

     Changes

     - kssm_block.py:
       - Remove: state_gate_proj, _metabolic_loss, _ema_energy, _ema_decay buffer
       - Remove from forward(): state_signal computation (line 237), u_logit (238), u_gate sigmoid (240), metabolic_loss store (243), EMA update (268-271)
       - Simplify: V_gated = V * vp_scale.unsqueeze(-1) (remove * u_gate)
       - Move log_dt_scale from AdaptiveTimestep module to direct nn.Parameter on block (since AdaptiveTimestep is absorbed in Phase 3)
       - Move safety constants (_omega_thresh, _delta, _smoothness, _eps) from AdaptiveTimestep to block buffers
     - src/kssm/models/backbone.py: get_metabolic_loss() → returns torch.tensor(0.0)
     - src/kssm/training/trainer.py: Remove metabolic_loss from print statement (line 53); loss computation (lines 32-33) becomes just loss = task_loss (metabolic returns 0)
     - src/kssm/training/optim.py: Remove "utility_gate", "state_gate_proj" from ssm_keywords

     ---
     Phase 3: Fused Dynamics Kernel (CUDA)

     Replace adaptive_dt.cu + cayley_vp.cu + ~15 elementwise PyTorch ops with a single dynamics_fused.cu kernel. This is the highest-impact optimization (eliminates the 39% elementwise overhead).

     New kernel: src/kssm/csrc/kernels/dynamics_fused.cu

     Grid: (B, cdiv(L, 256), H), Block: 256 threads. Each thread processes one (b, l, h) position — pure elementwise, no cross-thread dependency.

     Forward inputs:
     - gate_raw: (B, L, 7*H) BF16 — raw output of gate_proj
     - log_dt_scale: (H,) FP32 — learnable per-head parameter
     - rope_freqs: (H,) FP32 — position encoding frequencies
     - gating_c: float, omega_thresh, delta, smoothness, eps: float — constants
     - seq_len: int — for position computation

     Forward outputs:
     - A_bar: (B, L, H, 2, 2) BF16
     - vp_scale: (B, L, H) BF16
     - sel_B_sign: (B, L, H) BF16 — sign(sel_B_raw), {+1, -1}
     - sel_C_sign: (B, L, H) BF16 — sign(sel_C_raw), {+1, -1}
     - beta: (B, L, H) BF16

     Forward algorithm per thread (b, l, h):
     1. Parse gate_raw: alpha_raw, omega_raw, sel_B_raw, sel_C_raw, sel_dt_raw, beta_raw, r_gate_raw
     2. alpha = softplus(alpha_raw)
     3. omega = omega_raw + l * rope_freqs[h]                    // RoPE modulation
     4. dt_scale = softplus(log_dt_scale[h])                      // per-head scale
     5. char_freq = alpha + |omega| + eps
     6. dt_raw = dt_scale / char_freq                             // adaptive dt
     7. dt_max = (2 - delta) / (alpha + eps)                      // safety cap
     8. blend = sigmoid((omega_thresh - |omega|) / smoothness)
     9. dt_base = blend * min(dt_raw, dt_max) + (1-blend) * dt_raw
     10. dt = dt_base + softplus(sel_dt_raw)                       // selection dt
     11. r_gate = sigmoid(r_gate_raw)
     12. A_bar = cayley_discretize(alpha, omega, dt)               // reuse cayley_math.cuh
     13. eig_sq, scale = recurrence_gate_modulation(eig_sq, r_gate, gating_c)
     14. A_bar *= scale
     15. vp_scale = sqrt(1 - effective_eig_sq)
     16. beta = sigmoid(beta_raw)
     17. sel_B_sign = sign(sel_B_raw)
     18. sel_C_sign = sign(sel_C_raw)
     19. Store all outputs

     Backward kernel: Same grid. Recomputes forward from saved gate_raw. Chains gradients through steps 16→1 in reverse, combining the math from existing adaptive_dt_bwd_kernel and cayley_vp_bwd_kernel.
     Outputs:
     - grad_gate_raw: (B, L, 7*H) FP32
     - grad_log_dt_scale: (H,) FP32 — via atomicAdd

     sel_B_sign and sel_C_sign gradients are None (non-differentiable sign function; gradient flows through the normalize kernel in Phase 4).

     Save-for-backward: Only gate_raw (B, L, 7*H). Everything else is recomputed.

     Integration files

     - src/kssm/csrc/binding.cpp: Add dynamics_fused_fwd_cuda, dynamics_fused_bwd_cuda
     - src/kssm/ops/__init__.py: Add DynamicsFusedFn(Function) + dynamics_fused_cuda() convenience function
     - src/kssm/models/kssm_block.py: Replace the elementwise chain (lines 202-262) with single dynamics_fused_cuda() call
     - Keep adaptive_dt.cu and cayley_vp.cu for reference/standalone testing

     ---
     Phase 4: Fused K/Q Normalization (CUDA)

     Absorb F.normalize(K, dim=-1) and F.normalize(Q, dim=-1) plus sel_B/sel_C sign multiplication into a single kernel. Eliminates 2 normalize calls (each launching ~4 elementwise + 2 reduction kernels).

     Key insight

     normalize(K_raw * sel_B) = sign(sel_B) * K_raw / ||K_raw||

     Since sel_B is a per-head scalar and normalize divides by L2 norm, the scalar magnitude cancels. Only its sign survives.

     New kernel: src/kssm/csrc/kernels/normalize_kq.cu

     Grid: (B*L, H), Block: D threads (64). Requires cross-thread reduction (sum of squares over D dimensions) — same block_reduce_to_scalar pattern as intra_chunk_scan.cu.

     Forward: Each block normalizes one K vector and one Q vector:
     K_out[d] = sel_B_sign * K_raw[d] / sqrt(sum_d(K_raw[d]^2) + eps)
     Q_out[d] = sel_C_sign * Q_raw[d] / sqrt(sum_d(Q_raw[d]^2) + eps)

     Backward: Standard normalize gradient with cross-thread dot product:
     grad_K_raw[d] = sel_B_sign / ||K_raw|| * (grad_K[d] - K_normed[d] * dot(grad_K, K_normed))

     Integration

     - src/kssm/csrc/binding.cpp: Add normalize_kq_fwd_cuda, normalize_kq_bwd_cuda
     - src/kssm/ops/__init__.py: Add NormalizeKQFn(Function) + normalize_kq_cuda()
     - src/kssm/models/kssm_block.py: Replace K = F.normalize(K * sel_B, dim=-1) and Q = F.normalize(Q * sel_C, dim=-1) with K, Q = normalize_kq_cuda(K_raw, Q_raw, sel_B_sign, sel_C_sign)

     ---
     Optimized Forward Pass

     def forward(self, x):
         residual = x
         x = self.norm(x)
         proj = self.in_proj(x)                                              # GEMM #1
         z, x_gate, K_raw, V = proj.split([d_inner, d_inner, d_inner, n_heads*2], dim=-1)
         K_raw = K_raw.view(B, L, H, D)
         V = V.view(B, L, H, 2)

         x_conv = self.conv(x_gate)                                          # CUDA (conv1d_silu)
         gate_raw = self.gate_proj(x_conv.to(weight_dtype))                  # GEMM #2 (fused 7-in-1)

         A_bar, vp_scale, sel_B_sign, sel_C_sign, beta = dynamics_fused_cuda(  # CUDA (fused)
             gate_raw, self.log_dt_scale.float(), self.rope_freqs, ...)

         Q_raw = self.Q_proj(x_conv_cast).view(B, L, H, D)                  # GEMM #3
         K, Q = normalize_kq_cuda(K_raw, Q_raw, sel_B_sign, sel_C_sign)     # CUDA (fused)

         V_gated = V * vp_scale.unsqueeze(-1)                                # 1 elementwise
         Y, _ = self.ssd_scan(A_bar, K, V_gated, beta)                      # CUDA (intra+inter scan)

         retrieved = torch.einsum('blhsd,blhd->blhs', Y, Q)                 # cuBLAS
         y = self.readout_proj(retrieved.reshape(B, L, -1))                  # GEMM #4
         y = group_norm(y) * silu(z) + D * x_conv                           # ~3 elementwise
         output = self.out_proj(y)                                           # GEMM #5
         return residual + output

     Kernel launch comparison
     ┌───────────────────────────────┬──────────────────────────────────┬─────────────────────────────────────────────┐
     │                               │              Before              │                    After                    │
     ├───────────────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────┤
     │ GEMM launches (fwd)           │ 12                               │ 5                                           │
     ├───────────────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────┤
     │ CUDA kernel launches (fwd)    │ 3 (conv, adaptive_dt, cayley_vp) │ 4 (conv, dynamics_fused, normalize_kq, ssd) │
     ├───────────────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────┤
     │ Elementwise PyTorch ops (fwd) │ ~15                              │ ~4                                          │
     ├───────────────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────┤
     │ Total fwd kernel launches     │ ~30                              │ ~13                                         │
     ├───────────────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────┤
     │ Total fwd+bwd launches        │ ~60                              │ ~26                                         │
     └───────────────────────────────┴──────────────────────────────────┴─────────────────────────────────────────────┘
     ---
     Files Summary
     ┌────────┬─────────────────────────────────────────┬─────────┐
     │ Action │                  File                   │  Phase  │
     ├────────┼─────────────────────────────────────────┼─────────┤
     │ Edit   │ src/kssm/models/kssm_block.py           │ 1,2,3,4 │
     ├────────┼─────────────────────────────────────────┼─────────┤
     │ Edit   │ src/kssm/models/components.py           │ 1,2     │
     ├────────┼─────────────────────────────────────────┼─────────┤
     │ Edit   │ src/kssm/models/backbone.py             │ 2       │
     ├────────┼─────────────────────────────────────────┼─────────┤
     │ Edit   │ src/kssm/training/trainer.py            │ 2       │
     ├────────┼─────────────────────────────────────────┼─────────┤
     │ Edit   │ src/kssm/training/optim.py              │ 1,2     │
     ├────────┼─────────────────────────────────────────┼─────────┤
     │ Create │ src/kssm/csrc/kernels/dynamics_fused.cu │ 3       │
     ├────────┼─────────────────────────────────────────┼─────────┤
     │ Create │ src/kssm/csrc/kernels/normalize_kq.cu   │ 4       │
     ├────────┼─────────────────────────────────────────┼─────────┤
     │ Edit   │ src/kssm/csrc/binding.cpp               │ 3,4     │
     ├────────┼─────────────────────────────────────────┼─────────┤
     │ Edit   │ src/kssm/ops/__init__.py                │ 3,4     │
     └────────┴─────────────────────────────────────────┴─────────┘
     Not touched: adaptive_dt.cu, cayley_vp.cu, intra_chunk_scan.cu, inter_chunk_scan.cu, conv1d_silu.cu, cayley_math.cuh, common.cuh, reduction.cuh, setup.py

     ---
     Implementation Order
     ┌──────────┬────────────────────────────────┬─────────────────────────────────────────────────┬─────────────────┐
     │ Priority │             Phase              │                     Impact                      │     Effort      │
     ├──────────┼────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────┤
     │ P0       │ Phase 1: Fuse gate projections │ -6 GEMM launches                                │ ~100 LOC Python │
     ├──────────┼────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────┤
     │ P0       │ Phase 2: Remove utility gating │ -2 more GEMM, -EMA overhead                     │ ~50 LOC Python  │
     ├──────────┼────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────┤
     │ P1       │ Phase 3: Fused dynamics kernel │ Eliminates ~15 elementwise ops + 2 CUDA kernels │ ~500 LOC CUDA   │
     ├──────────┼────────────────────────────────┼─────────────────────────────────────────────────┼─────────────────┤
     │ P2       │ Phase 4: Fused K/Q normalize   │ Eliminates ~12 elementwise/reduction ops        │ ~300 LOC CUDA   │
     └──────────┴────────────────────────────────┴─────────────────────────────────────────────────┴─────────────────┘
     Phases 1-2 are Python-only and can be tested with existing test suite before writing any CUDA. Phase 3 is the highest-impact CUDA change. Phase 4 is incremental.

     ---
     Verification: Actual Evaluation Runs

     No unit tests. Verification uses real WikiText-103 training and CUDA benchmarks.

     Step 0: Baseline (before any changes)

     # Record baseline throughput
     python -c "
     import torch, time
     from kssm.config.defaults import KSSMConfig
     from kssm.models.kssm_block import KSSMBlock
     config = KSSMConfig(d_model=384, n_layers=12, context_length=8192)
     block = KSSMBlock(config, layer_idx=0).cuda().bfloat16()
     x = torch.randn(4, 1024, 384, device='cuda', dtype=torch.bfloat16)
     # warmup
     for _ in range(10): block(x)
     torch.cuda.synchronize()
     # bench
     start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
     start.record()
     for _ in range(50): block(x)
     end.record(); torch.cuda.synchronize()
     ms = start.elapsed_time(end) / 50
     print(f'Baseline: {ms:.2f} ms/iter, {4*1024/(ms/1000):.0f} tok/s')
     "

     # Record baseline WikiText val PPL (1 epoch, small config for speed)
     python scripts/train.py --config configs/default.yaml --seed 42
     # Note: val PPL after epoch 1

     Step 1: After Phase 1-2 (Python-only changes)

     # Rebuild (pip install -e .) then:

     # 1. Throughput benchmark — must show improvement from fewer GEMMs
     #    Same benchmark snippet as Step 0. Expect ~1.3-1.5x speedup.

     # 2. WikiText training — 1 epoch, same config
     python scripts/train.py --config configs/default.yaml --seed 42
     # Acceptance: val PPL within 5% of baseline (fusion + utility removal are
     # mathematically neutral for fusion, minor for utility gate removal)

     Step 2: After Phase 3 (fused dynamics kernel)

     # Rebuild CUDA kernels (pip install -e . --no-build-isolation)

     # 1. Throughput benchmark — major improvement expected
     #    Same snippet. Expect ~2x speedup from eliminating elementwise overhead.

     # 2. WikiText training — 1 epoch
     python scripts/train.py --config configs/default.yaml --seed 42
     # Acceptance: val PPL within 5% of Step 1 (mathematically equivalent)

     Step 3: After Phase 4 (fused normalize kernel)

     # Rebuild CUDA kernels

     # 1. Final throughput benchmark
     #    Target: >500K tokens/sec (>2x baseline ~246K)

     # 2. WikiText training — full run (20 epochs)
     python scripts/train.py --config configs/default.yaml --seed 42
     # Report final val PPL. Must be ≤ baseline val PPL (no regression).
     # This is the definitive evaluation.

     Acceptance Criteria
     ┌──────────────────────────────────┬────────────────────────────────────────────────┐
     │              Metric              │                  Requirement                   │
     ├──────────────────────────────────┼────────────────────────────────────────────────┤
     │ Throughput (B=4, L=1024)         │ >500K tokens/sec                               │
     ├──────────────────────────────────┼────────────────────────────────────────────────┤
     │ WikiText-103 val PPL (20 epochs) │ ≤ baseline                                     │
     ├──────────────────────────────────┼────────────────────────────────────────────────┤
     │ Training stability               │ No NaN/Inf across 20 epochs                    │
     ├──────────────────────────────────┼────────────────────────────────────────────────┤
     │ Compilation                      │ pip install -e . --no-build-isolation succeeds │

