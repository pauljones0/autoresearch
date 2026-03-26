"""
Regime change detection for bandit arms.
"""

from bandit.schemas import BanditState, ArmState, RegimeChangeEvent, load_jsonl


class RegimeChangeDetector:
    """Detects regime changes by comparing all-time vs rolling acceptance rates."""

    def detect(
        self,
        state: BanditState,
        log_path: str,
        window_size: int = 10,
    ) -> list:
        """Detect regime changes for each arm.

        For each arm, compare all-time acceptance rate vs rolling window rate.
        If rolling < all_time * 0.5, emit a RegimeChangeEvent.

        Args:
            state: Current bandit state.
            log_path: Path to JSONL acceptance log.
            window_size: Size of rolling window.

        Returns:
            List of RegimeChangeEvent for arms with detected changes.
        """
        entries = load_jsonl(log_path)
        if not entries:
            return []

        # Group entries by arm
        arm_entries = {}
        for entry in entries:
            arm_id = entry.get("arm_id", "")
            if arm_id:
                if arm_id not in arm_entries:
                    arm_entries[arm_id] = []
                arm_entries[arm_id].append(entry)

        events = []

        for arm_id, arm_log in arm_entries.items():
            if len(arm_log) < window_size:
                continue

            # All-time acceptance rate
            total = len(arm_log)
            accepted_all = sum(1 for e in arm_log if e.get("accepted", False))
            rate_all = accepted_all / total if total > 0 else 0.0

            # Rolling window acceptance rate
            recent = arm_log[-window_size:]
            accepted_recent = sum(1 for e in recent if e.get("accepted", False))
            rate_rolling = accepted_recent / window_size

            # Detect significant drop
            if rate_all > 0 and rate_rolling < rate_all * 0.5:
                drop_magnitude = (rate_all - rate_rolling) / rate_all

                recommended = []
                if rate_rolling == 0:
                    recommended.append("consider_reheat")
                    recommended.append("check_arm_viability")
                else:
                    recommended.append("monitor")
                    recommended.append("consider_constraint_review")

                event = RegimeChangeEvent(
                    arm_id=arm_id,
                    rate_all=rate_all,
                    rate_rolling=rate_rolling,
                    rate_drop_magnitude=drop_magnitude,
                    diagnostics_snapshot_summary=f"all_time={rate_all:.3f}, rolling_{window_size}={rate_rolling:.3f}",
                    recommended_actions=recommended,
                )
                events.append(event)

        return events
