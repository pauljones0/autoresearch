import pytest
from gpu_kernels.pipeline import GPUKernelPipeline

def test_gpu_kernels_instantiation(tmp_data_dir):
    pipeline = GPUKernelPipeline(data_dir=tmp_data_dir)
    pipeline.initialize()
    assert pipeline._initialized
