import json
import os
import pytest
from gpu_kernels.pipeline import GPUKernelPipeline
from gpu_kernels.schemas import KernelTarget
from gpu_kernels.generation.elementwise_generator import TritonElementwiseGenerator
from gpu_kernels.config.manager import KernelConfigManager


def test_gpu_kernels_instantiation(tmp_data_dir):
    pipeline = GPUKernelPipeline(data_dir=tmp_data_dir)
    pipeline.initialize()
    assert pipeline._initialized


# ---------------------------------------------------------------------------
# Elementwise generator tests
# ---------------------------------------------------------------------------

def _make_target(ops):
    return KernelTarget(group_id="test_group", op_sequence=ops)


def test_elementwise_detects_relu_square():
    gen = TritonElementwiseGenerator()
    target = _make_target(["aten::relu", "aten::square"])
    assert gen._detect_op_chain(target) == "relu_square"


def test_elementwise_detects_rms_norm():
    gen = TritonElementwiseGenerator()
    target = _make_target(["rms_norm"])
    assert gen._detect_op_chain(target) == "rms_norm"


def test_elementwise_detects_softcap_tanh():
    gen = TritonElementwiseGenerator()
    target = _make_target(["mul", "tanh"])
    assert gen._detect_op_chain(target) == "softcap_tanh"


def test_elementwise_detects_silu():
    gen = TritonElementwiseGenerator()
    target = _make_target(["sigmoid", "mul"])
    assert gen._detect_op_chain(target) == "silu"


def test_elementwise_raises_on_unrecognized_ops():
    gen = TritonElementwiseGenerator()
    target = _make_target(["unknown_op", "weird_thing"])
    with pytest.raises(ValueError, match="Unrecognized elementwise op chain"):
        gen._detect_op_chain(target)


# ---------------------------------------------------------------------------
# Config manager round-trip tests
# ---------------------------------------------------------------------------

def test_config_manager_save_load_roundtrip(tmp_path):
    config_dir = str(tmp_path / "kernels")
    os.makedirs(config_dir)

    mgr = KernelConfigManager(config_dir=config_dir)
    mgr._config = {
        "test_kernel": {
            "group_id": "elementwise_relu",
            "enabled": True,
            "speedup": 1.5,
        }
    }
    mgr.save()

    mgr2 = KernelConfigManager(config_dir=config_dir)
    loaded = mgr2.load()
    assert loaded["test_kernel"]["group_id"] == "elementwise_relu"
    assert loaded["test_kernel"]["speedup"] == 1.5
