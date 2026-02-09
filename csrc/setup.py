from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import torch
import os

# 获取当前目录（csrc）
this_dir = os.path.dirname(os.path.abspath(__file__))
# 父目录（项目根目录）
parent_dir = os.path.dirname(this_dir)

setup(
    name="cpp_mlp",
    ext_modules=[
        CppExtension(
            name="cpp_mlp",
            sources=[os.path.join(this_dir, "mlp.cpp")],  # 显式指定完整路径
            extra_compile_args=[
                "-O3",
                "-std=c++17",
                "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
            ],
            include_dirs=[
                os.path.join(parent_dir, "core"),  # 如果需要包含其他头文件
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)