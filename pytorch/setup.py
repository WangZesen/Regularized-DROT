from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='reg_drot',
    ext_modules=[
        CUDAExtension('reg_drot', [
            './src/drot_cuda.cpp',
            './src/qr_drot_cuda_kernel.cu',
            './src/glr_drot_cuda_kernel.cu'],
            include_dirs=["../core/"])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version="1.0.0"
)
