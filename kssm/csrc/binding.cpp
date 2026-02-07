// KSSM CUDA Extension Bindings

#include <torch/extension.h>
#include <c10/util/Optional.h>

namespace kssm {

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

}  // namespace kssm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "KSSM CUDA kernels";

    m.def("conv1d_silu_fwd_cuda", &kssm::conv1d_silu_fwd_cuda,
          "Fused Conv1d + SiLU forward",
          py::arg("x"), py::arg("weight"), py::arg("bias"));

    m.def("conv1d_silu_bwd_cuda", &kssm::conv1d_silu_bwd_cuda,
          "Fused Conv1d + SiLU backward",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("grad_out"));

}
