#!/usr/bin/env python3
"""
AutoResearch Orchestration Entry Point

Composes all layers and runs the autonomous research loop.

Usage:
    python run.py --data-dir ./data --iterations 100
    python run.py --no-meta                  # bandit + model_scientist only
    python run.py --no-surrogate --no-kernels  # minimal stack
"""

import argparse
import logging
import os
import sys
import time

logger = logging.getLogger("autoresearch")


def parse_args():
    parser = argparse.ArgumentParser(description="AutoResearch orchestration loop")
    parser.add_argument("--data-dir", default="./data",
                        help="Path to data directory (default: ./data)")
    parser.add_argument("--train-path", default="train.py",
                        help="Path to train.py (default: ./train.py)")
    parser.add_argument("--no-meta", action="store_true",
                        help="Disable meta-optimization layer")
    parser.add_argument("--no-surrogate", action="store_true",
                        help="Disable surrogate triage (no paper ingestion)")
    parser.add_argument("--no-kernels", action="store_true",
                        help="Disable GPU kernel generation")
    parser.add_argument("--config", default=None,
                        help="Path to meta_config.json override")
    parser.add_argument("--iterations", type=int, default=0,
                        help="Max iterations (0 = unlimited)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    return parser.parse_args()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_layers(args):
    """Instantiate layers bottom-up. Returns (meta, bandit, ms, st, gk)."""
    data_dir = os.path.abspath(args.data_dir)
    train_path = os.path.abspath(args.train_path)
    os.makedirs(data_dir, exist_ok=True)

    # 1. Leaf layers
    ms = None
    try:
        from model_scientist.pipeline import ModelScientistPipeline
        ms = ModelScientistPipeline(train_path=train_path, data_dir=data_dir)
        logger.info("model_scientist layer created")
    except Exception as e:
        logger.warning("model_scientist failed to create: %s", e)

    st = None
    if not args.no_surrogate:
        try:
            from surrogate_triage.pipeline import SurrogateTriagePipeline
            st = SurrogateTriagePipeline(
                train_path=train_path, data_dir=data_dir,
                model_scientist_pipeline=ms,
            )
            logger.info("surrogate_triage layer created")
        except Exception as e:
            logger.warning("surrogate_triage failed to create: %s", e)

    gk = None
    if not args.no_kernels:
        try:
            from gpu_kernels.pipeline import GPUKernelPipeline
            gk = GPUKernelPipeline(data_dir=data_dir)
            logger.info("gpu_kernels layer created")
        except Exception as e:
            logger.warning("gpu_kernels failed to create: %s", e)

    # 2. Bandit (requires at least model_scientist)
    bandit = None
    try:
        from bandit.pipeline import AdaptiveBanditPipeline
        bandit = AdaptiveBanditPipeline(
            work_dir=data_dir,
            model_scientist=ms,
            surrogate_triage=st,
            gpu_kernels=gk,
        )
        logger.info("bandit layer created")
    except Exception as e:
        logger.warning("bandit failed to create: %s", e)

    # 3. Meta (wraps all sub-layers)
    meta = None
    if not args.no_meta:
        try:
            from meta.pipeline import MetaAutoresearchPipeline
            meta = MetaAutoresearchPipeline(
                work_dir=data_dir,
                bandit_pipeline=bandit,
                model_scientist_pipeline=ms,
                surrogate_triage_pipeline=st,
                gpu_kernel_pipeline=gk,
            )
            logger.info("meta layer created")
        except Exception as e:
            logger.warning("meta failed to create: %s", e)

    return meta, bandit, ms, st, gk


def initialize_layers(meta, bandit, ms, st, gk):
    """Initialize all layers. Layers that fail to init are set to None."""
    if ms is not None:
        try:
            ms.initialize(baseline_val_bpb=1.5)
            logger.info("model_scientist initialized")
        except Exception as e:
            logger.warning("model_scientist init failed: %s", e)

    if st is not None:
        try:
            st.initialize()
            logger.info("surrogate_triage initialized")
        except Exception as e:
            logger.warning("surrogate_triage init failed: %s", e)

    if gk is not None:
        try:
            gk.initialize()
            logger.info("gpu_kernels initialized")
        except Exception as e:
            logger.warning("gpu_kernels init failed: %s", e)

    if bandit is not None:
        try:
            bandit.initialize()
            logger.info("bandit initialized")
        except Exception as e:
            logger.warning("bandit init failed: %s", e)

    if meta is not None:
        try:
            meta.initialize()
            logger.info("meta initialized")
        except Exception as e:
            logger.warning("meta init failed: %s", e)


def read_train_source(train_path: str) -> str:
    try:
        with open(train_path) as f:
            return f.read()
    except FileNotFoundError:
        return ""


def run_loop(meta, bandit, args):
    """Run the main experiment loop."""
    train_source = read_train_source(args.train_path)
    iteration = 0
    max_iter = args.iterations if args.iterations > 0 else float("inf")

    logger.info("Starting experiment loop (max_iterations=%s)",
                args.iterations if args.iterations > 0 else "unlimited")

    while iteration < max_iter:
        iteration += 1
        t0 = time.time()

        try:
            if meta is not None:
                result = meta.run_meta_iteration()
                logger.info("Meta iteration %d completed", iteration)
            elif bandit is not None:
                result = bandit.run_iteration(base_source=train_source)
                logger.info(
                    "Bandit iteration %d: arm=%s verdict=%s delta=%s (%.1fs)",
                    iteration,
                    getattr(result, "arm_selected", "?"),
                    getattr(result, "verdict", "?"),
                    getattr(result, "delta", None),
                    time.time() - t0,
                )
            else:
                logger.error("No runnable pipeline available")
                return
        except KeyboardInterrupt:
            logger.info("Interrupted by user at iteration %d", iteration)
            return
        except Exception as e:
            logger.exception("Error in iteration %d: %s", iteration, e)
            continue

    logger.info("Completed %d iterations", iteration)


def main():
    args = parse_args()
    setup_logging(args.log_level)

    logger.info("AutoResearch starting — data_dir=%s", args.data_dir)

    meta, bandit, ms, st, gk = build_layers(args)

    if bandit is None and meta is None:
        logger.error("Neither bandit nor meta layer could be created. Exiting.")
        sys.exit(1)

    initialize_layers(meta, bandit, ms, st, gk)
    run_loop(meta, bandit, args)


if __name__ == "__main__":
    main()
