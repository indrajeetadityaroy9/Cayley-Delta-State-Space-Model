// CDSSM CUDA Extension Bindings

#include <torch/extension.h>
#include <c10/util/Optional.h>

namespace cdssm {

// Conv1d + SiLU kernels (conv1d_silu.cu)
torch::Tensor conv1d_silu_fwd_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
conv1d_silu_bwd_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor grad_out
);

// Intra-chunk delta-rule scan (intra_chunk_scan.cu)
std::tuple<torch::Tensor, torch::Tensor>
intra_chunk_scan_fwd_cuda(
    torch::Tensor A_flat,
    torch::Tensor K_flat,
    torch::Tensor V_flat,
    torch::Tensor beta_flat
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
intra_chunk_scan_bwd_cuda(
    torch::Tensor grad_local_h,
    torch::Tensor grad_cum_A,
    torch::Tensor A_flat,
    torch::Tensor K_flat,
    torch::Tensor V_flat,
    torch::Tensor beta_flat,
    torch::Tensor local_h,
    torch::Tensor cum_A
);

// Inter-chunk sequential scan (inter_chunk_scan.cu)
torch::Tensor inter_chunk_scan_fwd_cuda(
    torch::Tensor total_A,
    torch::Tensor final_local_h
);

std::tuple<torch::Tensor, torch::Tensor>
inter_chunk_scan_bwd_cuda(
    torch::Tensor grad_chunk_states,
    torch::Tensor total_A,
    torch::Tensor chunk_states
);

// Fused Cayley discretization + VP scale (cayley_vp.cu)
std::tuple<torch::Tensor, torch::Tensor>
cayley_vp_fwd_cuda(
    torch::Tensor alpha,
    torch::Tensor omega,
    torch::Tensor dt,
    torch::Tensor r_gate,
    float gating_c
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
cayley_vp_bwd_cuda(
    torch::Tensor grad_A_bar,
    torch::Tensor grad_vp_scale,
    torch::Tensor alpha,
    torch::Tensor omega,
    torch::Tensor dt,
    torch::Tensor r_gate,
    float gating_c
);

// Adaptive timestep (adaptive_dt.cu)
torch::Tensor adaptive_dt_fwd_cuda(
    torch::Tensor alpha,
    torch::Tensor omega,
    torch::Tensor log_dt_scale,
    float omega_thresh,
    float delta,
    float smoothness,
    float eps
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
adaptive_dt_bwd_cuda(
    torch::Tensor grad_dt,
    torch::Tensor alpha,
    torch::Tensor omega,
    torch::Tensor log_dt_scale,
    float omega_thresh,
    float delta,
    float smoothness,
    float eps
);

}  // namespace cdssm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CDSSM CUDA kernels";

    // Conv1d + SiLU
    m.def("conv1d_silu_fwd_cuda", &cdssm::conv1d_silu_fwd_cuda,
          "Fused Conv1d + SiLU forward",
          py::arg("x"), py::arg("weight"), py::arg("bias"));

    m.def("conv1d_silu_bwd_cuda", &cdssm::conv1d_silu_bwd_cuda,
          "Fused Conv1d + SiLU backward",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("grad_out"));

    // Intra-chunk scan
    m.def("intra_chunk_scan_fwd_cuda", &cdssm::intra_chunk_scan_fwd_cuda,
          "Intra-chunk delta-rule scan forward",
          py::arg("A_flat"), py::arg("K_flat"), py::arg("V_flat"), py::arg("beta_flat"));

    m.def("intra_chunk_scan_bwd_cuda", &cdssm::intra_chunk_scan_bwd_cuda,
          "Intra-chunk delta-rule scan backward",
          py::arg("grad_local_h"), py::arg("grad_cum_A"),
          py::arg("A_flat"), py::arg("K_flat"), py::arg("V_flat"), py::arg("beta_flat"),
          py::arg("local_h"), py::arg("cum_A"));

    // Inter-chunk scan
    m.def("inter_chunk_scan_fwd_cuda", &cdssm::inter_chunk_scan_fwd_cuda,
          "Inter-chunk sequential scan forward",
          py::arg("total_A"), py::arg("final_local_h"));

    m.def("inter_chunk_scan_bwd_cuda", &cdssm::inter_chunk_scan_bwd_cuda,
          "Inter-chunk sequential scan backward",
          py::arg("grad_chunk_states"), py::arg("total_A"), py::arg("chunk_states"));

    // Cayley + VP scale
    m.def("cayley_vp_fwd_cuda", &cdssm::cayley_vp_fwd_cuda,
          "Fused Cayley discretization + VP scale forward",
          py::arg("alpha"), py::arg("omega"), py::arg("dt"),
          py::arg("r_gate"), py::arg("gating_c"));

    m.def("cayley_vp_bwd_cuda", &cdssm::cayley_vp_bwd_cuda,
          "Fused Cayley discretization + VP scale backward",
          py::arg("grad_A_bar"), py::arg("grad_vp_scale"),
          py::arg("alpha"), py::arg("omega"), py::arg("dt"),
          py::arg("r_gate"), py::arg("gating_c"));

    // Adaptive timestep
    m.def("adaptive_dt_fwd_cuda", &cdssm::adaptive_dt_fwd_cuda,
          "Adaptive timestep forward",
          py::arg("alpha"), py::arg("omega"), py::arg("log_dt_scale"),
          py::arg("omega_thresh"), py::arg("delta"),
          py::arg("smoothness"), py::arg("eps"));

    m.def("adaptive_dt_bwd_cuda", &cdssm::adaptive_dt_bwd_cuda,
          "Adaptive timestep backward",
          py::arg("grad_dt"), py::arg("alpha"), py::arg("omega"),
          py::arg("log_dt_scale"), py::arg("omega_thresh"),
          py::arg("delta"), py::arg("smoothness"), py::arg("eps"));
}
