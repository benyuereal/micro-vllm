from setuptools import setup, Extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch

# 检查CUDA可用性
if not torch.cuda.is_available():
    raise RuntimeError("CUDA不可用，无法编译CUDA内核")

setup(
    name='fused_gptq_gemm_cuda',
    ext_modules=[
        CUDAExtension(
            name='fused_gptq_gemm_cuda',
            sources=['core/layer/gptq_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-use_fast_math',
                    '-lineinfo',
                    '-Xptxas=-O3',
                    '--ptxas-options=-v',
                    '-maxrregcount=255',
                    '--gpu-architecture=compute_80',  # 针对A100优化
                    '--gpu-code=sm_80'
                ]
            },
            libraries=['cublas', 'curand']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
