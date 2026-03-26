"""
Reheat decay verification — ensures temperature decreases after reheat.
"""

from bandit.schemas import BanditState, ArmState, ReheatDecayReport, load_jsonl


class ReheatDecayVerifier:
    """Verifies that temperature is monotonically non-increasing after reheat events."""

    def verify_decay(
        self,
        state: BanditState,
        log_path: str,
    ) -> ReheatDecayReport:
        """Verify temperature decay after reheat for all reheated arms.

        Reads the acceptance/reheat log and checks that after each reheat event,
        the arm's temperature values are monotonically non-increasing.

        Args:
            state: Current bandit state.
            log_path: Path to JSONL log containing temperature records.

        Returns:
            ReheatDecayReport with per-arm verification results.
        """
        entries = load_jsonl(log_path)

        # Track per-arm: segments of temperatures after each reheat
        arm_reheat_indices = {}  # arm_id -> list of indices where reheat occurred
        arm_temperatures = {}   # arm_id -> list of (index, temperature) tuples

        for i, entry in enumerate(entries):
            arm_id = entry.get("arm_id", "")
            if not arm_id:
                continue

            # Detect reheat events
            if entry.get("temperature_after") is not None and entry.get("temperature_before") is not None:
                if entry["temperature_after"] > entry["temperature_before"]:
                    if arm_id not in arm_reheat_indices:
                        arm_reheat_indices[arm_id] = []
                    arm_reheat_indices[arm_id].append(i)

            # Track temperature observations
            T = entry.get("T_effective", entry.get("temperature", entry.get("temperature_after")))
            if T is not None:
                if arm_id not in arm_temperatures:
                    arm_temperatures[arm_id] = []
                arm_temperatures[arm_id].append((i, T))

        per_arm = {}
        all_decaying = True

        for arm_id, reheat_indices in arm_reheat_indices.items():
            temps = arm_temperatures.get(arm_id, [])
            arm_ok = True

            for reheat_idx in reheat_indices:
                # Get temperatures after this reheat
                post_reheat = [(idx, t) for idx, t in temps if idx > reheat_idx]
                # Check monotonic non-increasing
                for j in range(1, len(post_reheat)):
                    if post_reheat[j][1] > post_reheat[j - 1][1]:
                        # Temperature increased after reheat (not from another reheat)
                        # Check if this increase is itself a reheat
                        increase_idx = post_reheat[j][0]
                        is_another_reheat = increase_idx in reheat_indices
                        if not is_another_reheat:
                            arm_ok = False
                            break
                    if not arm_ok:
                        break

            per_arm[arm_id] = {
                "decaying": arm_ok,
                "reheat_count": len(reheat_indices),
            }
            if not arm_ok:
                all_decaying = False

        return ReheatDecayReport(
            per_arm=per_arm,
            all_decaying=all_decaying,
        )
