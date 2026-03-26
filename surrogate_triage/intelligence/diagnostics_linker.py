"""
DiagnosticsIngestionLinker: reads DiagnosticsReport data and extracts
the top-3 bottlenecks, then computes relevance boosts for papers whose
abstracts match the identified bottleneck search terms.
"""

import math
import statistics


class DiagnosticsIngestionLinker:
    """Links model diagnostics to paper ingestion relevance scoring."""

    # Bottleneck detection thresholds
    ATTENTION_ENTROPY_THRESHOLD = 1.0
    GRADIENT_VANISHING_THRESHOLD = 0.01
    GRADIENT_EXPLODING_THRESHOLD = 100.0
    DEAD_NEURON_FRACTION_THRESHOLD = 0.30
    LOSS_IMBALANCE_FACTOR = 3.0
    LAYER_REDUNDANCY_CKA_THRESHOLD = 0.9

    def detect_bottlenecks(self, diagnostics: dict) -> list:
        """Detect bottlenecks from a diagnostics report dict.

        Returns a list of dicts with keys: bottleneck_type, severity, search_terms, boost_weight.
        Sorted by severity descending, limited to top 3.
        """
        candidates = []

        # (a) Attention entropy collapse
        attention_stats = diagnostics.get("attention_stats", [])
        if attention_stats:
            entropies = [s.get("entropy", float("inf")) for s in attention_stats
                         if isinstance(s, dict) and "entropy" in s]
            if entropies:
                mean_entropy = statistics.mean(entropies)
                if mean_entropy < self.ATTENTION_ENTROPY_THRESHOLD:
                    severity = (self.ATTENTION_ENTROPY_THRESHOLD - mean_entropy) / self.ATTENTION_ENTROPY_THRESHOLD
                    candidates.append({
                        "bottleneck_type": "attention_entropy_collapse",
                        "severity": min(severity, 1.0),
                        "search_terms": [],
                        "boost_weight": 0.0,
                    })

        # (b) Gradient vanishing in first 3 layers
        gradient_stats = diagnostics.get("gradient_stats", [])
        if gradient_stats:
            early_layers = [s for s in gradient_stats
                           if isinstance(s, dict) and s.get("layer_idx", 999) < 3]
            if early_layers:
                early_norms = [s.get("norm", float("inf")) for s in early_layers]
                mean_early_norm = statistics.mean(early_norms)
                if mean_early_norm < self.GRADIENT_VANISHING_THRESHOLD:
                    severity = (self.GRADIENT_VANISHING_THRESHOLD - mean_early_norm) / self.GRADIENT_VANISHING_THRESHOLD
                    candidates.append({
                        "bottleneck_type": "gradient_vanishing_early",
                        "severity": min(severity, 1.0),
                        "search_terms": [],
                        "boost_weight": 0.0,
                    })

        # (c) Gradient exploding
        if gradient_stats:
            all_norms = [s.get("norm", 0.0) for s in gradient_stats if isinstance(s, dict)]
            if all_norms:
                max_norm = max(all_norms)
                if max_norm > self.GRADIENT_EXPLODING_THRESHOLD:
                    severity = min((max_norm - self.GRADIENT_EXPLODING_THRESHOLD) / self.GRADIENT_EXPLODING_THRESHOLD, 1.0)
                    candidates.append({
                        "bottleneck_type": "gradient_exploding",
                        "severity": severity,
                        "search_terms": [],
                        "boost_weight": 0.0,
                    })

        # (d) Dead neurons
        activation_stats = diagnostics.get("activation_stats", [])
        if activation_stats:
            dead_fractions = [s.get("dead_neuron_fraction", 0.0) for s in activation_stats
                             if isinstance(s, dict)]
            if dead_fractions:
                max_dead = max(dead_fractions)
                if max_dead > self.DEAD_NEURON_FRACTION_THRESHOLD:
                    severity = min((max_dead - self.DEAD_NEURON_FRACTION_THRESHOLD) / (1.0 - self.DEAD_NEURON_FRACTION_THRESHOLD), 1.0)
                    candidates.append({
                        "bottleneck_type": "dead_neurons",
                        "severity": severity,
                        "search_terms": [],
                        "boost_weight": 0.0,
                    })

        # (e) Loss imbalance
        loss_decomposition = diagnostics.get("loss_decomposition", [])
        if loss_decomposition:
            buckets = {s.get("bucket_name", ""): s.get("mean_loss", 0.0)
                       for s in loss_decomposition if isinstance(s, dict)}
            rare_loss = buckets.get("rare", 0.0)
            top1k_loss = buckets.get("top_1k", 0.0)
            if top1k_loss > 0 and rare_loss > self.LOSS_IMBALANCE_FACTOR * top1k_loss:
                ratio = rare_loss / top1k_loss
                severity = min((ratio - self.LOSS_IMBALANCE_FACTOR) / self.LOSS_IMBALANCE_FACTOR, 1.0)
                candidates.append({
                    "bottleneck_type": "loss_imbalance",
                    "severity": severity,
                    "search_terms": [],
                    "boost_weight": 0.0,
                })

        # (f) Layer redundancy
        layer_similarity = diagnostics.get("layer_similarity_matrix", [])
        if layer_similarity:
            adjacent_ckas = []
            for entry in layer_similarity:
                if not isinstance(entry, dict):
                    continue
                li = entry.get("layer_i", -1)
                lj = entry.get("layer_j", -1)
                if abs(li - lj) == 1:
                    adjacent_ckas.append(entry.get("cka_score", 0.0))
            if adjacent_ckas:
                max_cka = max(adjacent_ckas)
                if max_cka > self.LAYER_REDUNDANCY_CKA_THRESHOLD:
                    severity = min((max_cka - self.LAYER_REDUNDANCY_CKA_THRESHOLD) / (1.0 - self.LAYER_REDUNDANCY_CKA_THRESHOLD), 1.0)
                    candidates.append({
                        "bottleneck_type": "layer_redundancy",
                        "severity": severity,
                        "search_terms": [],
                        "boost_weight": 0.0,
                    })

        # Sort by severity descending, return top 3
        candidates.sort(key=lambda x: x["severity"], reverse=True)
        return candidates[:3]

    def compute_relevance_boost(self, paper_abstract: str, bottlenecks: list) -> float:
        """Compute a relevance boost for a paper based on detected bottlenecks.

        Checks how many bottleneck search terms appear in the paper abstract.
        Returns a float in [0.0, 1.0].
        """
        if not bottlenecks or not paper_abstract:
            return 0.0

        abstract_lower = paper_abstract.lower()
        total_boost = 0.0

        for bn in bottlenecks:
            terms = bn.get("search_terms", [])
            severity = bn.get("severity", 0.0)
            if not terms:
                continue
            matches = sum(1 for t in terms if t.lower() in abstract_lower)
            if matches > 0:
                term_fraction = matches / len(terms)
                total_boost += severity * term_fraction

        # Normalize to [0, 1]
        return min(total_boost, 1.0)
