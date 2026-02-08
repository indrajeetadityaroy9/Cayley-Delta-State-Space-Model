"""CUDA extension building for KSSM.

All project metadata and dependencies are in pyproject.toml.
This file only handles CUDA extension compilation.
"""

from pathlib import Path

import pybind11
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


_PROJECT_ROOT = Path(__file__).resolve().parent


def get_cuda_extensions():
    """Build CUDA extensions."""
    csrc_dir = _PROJECT_ROOT / "src" / "kssm" / "csrc"
    kernel_dir = csrc_dir / "kernels"
    cu_files = list(kernel_dir.glob("*.cu"))

    sources = [str(csrc_dir / "binding.cpp")] + [str(f) for f in cu_files]
    include_dirs = [str(csrc_dir / "include"), pybind11.get_include()]

    ext_modules = [
        CUDAExtension(
            name="kssm._C",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode", "arch=compute_90,code=sm_90",
                    "--threads", "4",
                ],
            },
        )
    ]
    return ext_modules, {"build_ext": BuildExtension}


ext_modules, cmdclass = get_cuda_extensions()
setup(ext_modules=ext_modules, cmdclass=cmdclass)
