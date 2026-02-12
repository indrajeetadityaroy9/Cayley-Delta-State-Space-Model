"""CUDA extension building for the CDSSM package.

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
    csrc_dir = Path("cdssm") / "csrc"
    kernel_dir = csrc_dir / "kernels"
    cu_files = list((_PROJECT_ROOT / kernel_dir).glob("*.cu"))

    sources = [str(csrc_dir / "binding.cpp")] + [
        str(csrc_dir / "kernels" / f.name) for f in cu_files
    ]
    include_dirs = [str(_PROJECT_ROOT / csrc_dir / "include"), pybind11.get_include()]

    ext_modules = [
        CUDAExtension(
            name="cdssm._C",
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
