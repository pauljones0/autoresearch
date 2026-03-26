"""
Phase 5 — DiffQualityAnalyzer: compare diff variants for the same technique,
identify patterns that distinguish successful from failed diffs.
"""

import json
import os
import re


class DiffQualityAnalyzer:
    """Analyze diff variants to discover patterns correlated with success/failure.

    When multiple diff variants are generated for the same technique, some may
    succeed while others fail.  This class captures what distinguishes them.
    """

    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self._patterns_path = os.path.join(data_dir, "diff_patterns.json")
        self._good_patterns: list[str] = []
        self._bad_patterns: list[str] = []

        if os.path.exists(self._patterns_path):
            self._load_patterns()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def analyze_variants(
        self,
        technique_id: str,
        variant_results: list,
    ) -> dict:
        """Analyze variants for a single technique.

        Args:
            technique_id: Identifier of the technique.
            variant_results: list of dicts, each with at least:
                diff_text, verdict ("accepted"/"rejected"/"crashed"),
                actual_delta (float).

        Returns:
            dict with good_patterns, bad_patterns.
        """
        good_diffs = [
            v for v in variant_results if v.get("verdict") == "accepted"
        ]
        bad_diffs = [
            v for v in variant_results if v.get("verdict") in ("rejected", "crashed")
        ]

        good_feats = [self._extract_features(v.get("diff_text", "")) for v in good_diffs]
        bad_feats = [self._extract_features(v.get("diff_text", "")) for v in bad_diffs]

        new_good = self._find_distinguishing(good_feats, bad_feats)
        new_bad = self._find_distinguishing(bad_feats, good_feats)

        # Accumulate patterns
        for p in new_good:
            if p not in self._good_patterns:
                self._good_patterns.append(p)
        for p in new_bad:
            if p not in self._bad_patterns:
                self._bad_patterns.append(p)

        self._save_patterns()

        return {
            "technique_id": technique_id,
            "n_good": len(good_diffs),
            "n_bad": len(bad_diffs),
            "good_patterns": list(self._good_patterns),
            "bad_patterns": list(self._bad_patterns),
        }

    def compute_variant_stats(self, results: list) -> dict:
        """Compute per-variant statistics.

        Args:
            results: list of dicts with variant_index, verdict, actual_delta.

        Returns:
            dict mapping variant_index to {total, accepted, success_rate, avg_delta,
            features}.
        """
        by_variant: dict[int, list] = {}
        for r in results:
            idx = r.get("variant_index", 0)
            by_variant.setdefault(idx, []).append(r)

        stats = {}
        for idx, entries in by_variant.items():
            total = len(entries)
            accepted = sum(1 for e in entries if e.get("verdict") == "accepted")
            deltas = [
                e.get("actual_delta", 0.0)
                for e in entries
                if e.get("verdict") != "crashed"
            ]
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

            # Aggregate diff features across this variant's entries
            all_feats = set()
            for e in entries:
                all_feats.update(self._extract_features(e.get("diff_text", "")))

            stats[idx] = {
                "total": total,
                "accepted": accepted,
                "success_rate": accepted / total if total else 0.0,
                "avg_delta": avg_delta,
                "features": sorted(all_feats),
            }

        return stats

    def get_good_patterns(self) -> list:
        """Return accumulated patterns from successful diffs."""
        return list(self._good_patterns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(diff_text: str) -> set:
        """Extract structural features from a diff."""
        features = set()
        if not diff_text:
            return features

        lines = diff_text.splitlines()
        added = [l for l in lines if l.startswith("+") and not l.startswith("+++")]
        removed = [l for l in lines if l.startswith("-") and not l.startswith("---")]

        features.add(f"added_lines:{len(added)}")
        features.add(f"removed_lines:{len(removed)}")

        all_text = " ".join(added + removed)

        # Detect common code patterns
        if re.search(r"class\s+\w+", all_text):
            features.add("defines_class")
        if re.search(r"def\s+\w+", all_text):
            features.add("defines_function")
        if re.search(r"import\s+", all_text):
            features.add("modifies_imports")
        if re.search(r"nn\.\w+", all_text):
            features.add("uses_nn_module")
        if re.search(r"\.backward\(", all_text):
            features.add("modifies_backward")
        if re.search(r"lr|learning.rate", all_text, re.IGNORECASE):
            features.add("touches_learning_rate")
        if re.search(r"attention|attn", all_text, re.IGNORECASE):
            features.add("touches_attention")
        if re.search(r"norm|LayerNorm|RMSNorm", all_text):
            features.add("touches_normalization")

        # Size category
        total = len(added) + len(removed)
        if total <= 5:
            features.add("size:tiny")
        elif total <= 20:
            features.add("size:small")
        elif total <= 50:
            features.add("size:medium")
        else:
            features.add("size:large")

        return features

    @staticmethod
    def _find_distinguishing(present_feats: list, absent_feats: list) -> list:
        """Find features common in *present_feats* but rare in *absent_feats*."""
        if not present_feats:
            return []

        # Count how often each feature appears
        present_counts: dict[str, int] = {}
        for feat_set in present_feats:
            for f in feat_set:
                present_counts[f] = present_counts.get(f, 0) + 1

        absent_counts: dict[str, int] = {}
        for feat_set in absent_feats:
            for f in feat_set:
                absent_counts[f] = absent_counts.get(f, 0) + 1

        n_present = len(present_feats)
        n_absent = max(len(absent_feats), 1)

        distinguishing = []
        for feat, count in present_counts.items():
            present_rate = count / n_present
            absent_rate = absent_counts.get(feat, 0) / n_absent
            # Feature appears in majority of present but minority of absent
            if present_rate >= 0.6 and absent_rate < 0.3:
                distinguishing.append(feat)

        return sorted(distinguishing)

    def _save_patterns(self) -> None:
        with open(self._patterns_path, "w") as f:
            json.dump(
                {
                    "good_patterns": self._good_patterns,
                    "bad_patterns": self._bad_patterns,
                },
                f,
                indent=2,
            )

    def _load_patterns(self) -> None:
        try:
            with open(self._patterns_path) as f:
                data = json.load(f)
            self._good_patterns = data.get("good_patterns", [])
            self._bad_patterns = data.get("bad_patterns", [])
        except (json.JSONDecodeError, OSError):
            pass
