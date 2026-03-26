"""
Kernel configuration manager with atomic writes.

Manages kernel_config.json — enable/disable kernels, emergency disable,
and maintains kernel_disable_log.jsonl audit trail.
"""

import argparse
import json
import os
import tempfile
import time


class KernelConfigManager:
    """Manage kernel_config.json with atomic writes and audit logging."""

    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir or os.path.join(
            os.path.dirname(__file__), "..", "data"
        )
        self.config_path = os.path.join(self.config_dir, "kernel_config.json")
        self.disable_log_path = os.path.join(
            self.config_dir, "kernel_disable_log.jsonl"
        )
        self._config = {}

    def load(self) -> dict:
        """Load kernel config from disk."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._config = {}
        else:
            self._config = {}
        return self._config

    def save(self):
        """Save config atomically — write to temp file, then rename."""
        os.makedirs(self.config_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=self.config_dir, suffix=".tmp", prefix="kernel_config_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._config, f, indent=2)
            # Atomic rename (on same filesystem)
            if os.path.exists(self.config_path):
                os.replace(tmp_path, self.config_path)
            else:
                os.rename(tmp_path, self.config_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def disable_kernel(self, kernel_id: str, reason: str = ""):
        """Disable a specific kernel and log the action."""
        if not self._config:
            self.load()

        if kernel_id not in self._config:
            self._config[kernel_id] = {"enabled": False, "backend": "pytorch"}

        entry = self._config[kernel_id]
        if isinstance(entry, dict):
            entry["enabled"] = False
            entry["backend"] = "pytorch"
        else:
            entry = {"enabled": False, "backend": "pytorch"}
            self._config[kernel_id] = entry

        self.save()
        self._log_disable(kernel_id, reason, "manual_disable")

    def enable_kernel(self, kernel_id: str):
        """Re-enable a specific kernel."""
        if not self._config:
            self.load()

        if kernel_id in self._config:
            entry = self._config[kernel_id]
            if isinstance(entry, dict):
                entry["enabled"] = True
                entry["backend"] = entry.get("original_backend", "triton")
            self.save()
            self._log_action(kernel_id, "enable", "Kernel re-enabled")

    def emergency_disable_all(self):
        """Disable all kernels immediately — emergency fallback."""
        if not self._config:
            self.load()

        for kernel_id, entry in self._config.items():
            if isinstance(entry, dict) and entry.get("enabled", False):
                entry["enabled"] = False
                entry["backend"] = "pytorch"
                self._log_disable(kernel_id, "Emergency disable all", "emergency")

        self.save()

    def get_active_kernels(self) -> dict:
        """Return dict of currently active (enabled) kernels."""
        if not self._config:
            self.load()

        active = {}
        for kernel_id, entry in self._config.items():
            if isinstance(entry, dict) and entry.get("enabled", False):
                active[kernel_id] = entry
        return active

    def _log_disable(self, kernel_id: str, reason: str, action: str):
        """Append a disable event to the audit log."""
        self._log_action(kernel_id, action, reason)

    def _log_action(self, kernel_id: str, action: str, reason: str):
        """Append an action to the disable log (JSONL)."""
        entry = {
            "kernel_id": kernel_id,
            "action": action,
            "reason": reason,
            "timestamp": time.time(),
        }
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            with open(self.disable_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="GPU Kernel Config Manager",
        prog="python -m gpu_kernels.config.manager",
    )
    parser.add_argument(
        "--config-dir", default=None, help="Config directory path"
    )
    parser.add_argument(
        "--disable", metavar="KERNEL_ID", help="Disable a kernel by ID"
    )
    parser.add_argument(
        "--enable", metavar="KERNEL_ID", help="Enable a kernel by ID"
    )
    parser.add_argument(
        "--reason", default="", help="Reason for disable (used with --disable)"
    )
    parser.add_argument(
        "--emergency-disable-all",
        action="store_true",
        help="Emergency: disable all kernels",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_active",
        help="List active kernels",
    )
    args = parser.parse_args()

    mgr = KernelConfigManager(config_dir=args.config_dir)
    mgr.load()

    if args.emergency_disable_all:
        mgr.emergency_disable_all()
        print("All kernels disabled (emergency).")
    elif args.disable:
        mgr.disable_kernel(args.disable, args.reason)
        print(f"Kernel '{args.disable}' disabled. Reason: {args.reason or '(none)'}")
    elif args.enable:
        mgr.enable_kernel(args.enable)
        print(f"Kernel '{args.enable}' enabled.")
    elif args.list_active:
        active = mgr.get_active_kernels()
        if active:
            print(f"Active kernels ({len(active)}):")
            for kid, entry in active.items():
                print(f"  {kid}: backend={entry.get('backend', '?')}, "
                      f"speedup={entry.get('speedup', 0):.2f}x")
        else:
            print("No active kernels.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
