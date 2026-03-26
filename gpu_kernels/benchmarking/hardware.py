"""
HardwareCapabilityDetector: auto-detects GPU capabilities and builds
a HardwareProfile with peak bandwidth from a lookup table.
"""

import json
import os

import torch

from ..schemas import HardwareProfile, save_json


# Peak memory bandwidth lookup table (GB/s)
# Values are for the highest-bandwidth SKU of each GPU
_BANDWIDTH_TABLE = {
    "A100-SXM4": 2039,
    "A100-PCIE": 1555,
    "A100": 2039,
    "H100-SXM5": 3352,
    "H100-PCIE": 2039,
    "H100": 3352,
    "H200": 4800,
    "RTX 4090": 1008,
    "RTX 4080": 717,
    "RTX 4070": 504,
    "RTX 3090": 936,
    "RTX 3080": 760,
    "RTX 3070": 448,
    "L40": 864,
    "L40S": 864,
    "L4": 300,
    "T4": 300,
    "V100-SXM2": 900,
    "V100": 900,
    "A10": 600,
    "A6000": 768,
}


def _lookup_bandwidth(gpu_name: str) -> float:
    """Look up peak bandwidth for a GPU name, with fuzzy matching."""
    name_upper = gpu_name.upper()
    # Try exact matches first, then substring
    for key, bw in sorted(_BANDWIDTH_TABLE.items(), key=lambda x: -len(x[0])):
        if key.upper() in name_upper:
            return float(bw)
    # Fallback: assume modest bandwidth
    return 500.0


class HardwareCapabilityDetector:
    """Auto-detects GPU hardware capabilities using torch.cuda."""

    def detect(self, device_index: int = 0) -> HardwareProfile:
        """Detect GPU capabilities and return a HardwareProfile.

        Args:
            device_index: CUDA device index (default 0).

        Returns:
            HardwareProfile populated with detected values.
        """
        if not torch.cuda.is_available():
            return HardwareProfile(gpu_name="no_gpu")

        props = torch.cuda.get_device_properties(device_index)

        gpu_name = props.name
        compute_cap = (props.major, props.minor)
        peak_bw = _lookup_bandwidth(gpu_name)

        profile = HardwareProfile(
            gpu_name=gpu_name,
            compute_capability=compute_cap,
            peak_bandwidth_gbps=peak_bw,
            sm_count=props.multi_processor_count,
            max_shared_memory_per_block_bytes=props.max_shared_memory_per_block,
            max_threads_per_block=props.max_threads_per_block,
            warp_size=props.warp_size if hasattr(props, "warp_size") else 32,
            max_registers_per_block=props.max_registers_per_block if hasattr(props, "max_registers_per_block") else 65536,
            clock_rate_mhz=props.clock_rate / 1000.0 if hasattr(props, "clock_rate") else 0.0,
            total_memory_bytes=props.total_memory,
        )
        return profile

    def detect_and_save(
        self, output_path: str = "hardware_profile.json", device_index: int = 0
    ) -> HardwareProfile:
        """Detect hardware and save profile to JSON.

        Args:
            output_path: Path to save the JSON file.
            device_index: CUDA device index.

        Returns:
            The detected HardwareProfile.
        """
        profile = self.detect(device_index)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        save_json(profile, output_path)
        return profile
