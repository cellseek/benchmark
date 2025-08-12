#!/usr/bin/env python3
"""
SAM + XMem Benchmark Runner

This script runs the SAM + XMem benchmark on Cell Tracking Challenge datasets.
"""

import logging
import sys
from pathlib import Path

from src.sam_xmem_benchmark import SAMXMemBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Set up paths with default values
    data_root = Path("data/cell_tracking_challenge")
    results_dir = Path("benchmark_results")
    sam_model = Path("../sam/checkpoints/sam_model.pth")
    xmem_model = Path("../xmem/checkpoints/XMem-s012.pth")

    # Check if data directory exists
    if not data_root.exists():
        logger.error(f"Data directory not found: {data_root}")
        logger.info(
            "Please ensure the Cell Tracking Challenge data is downloaded and extracted."
        )
        return 1

    # Initialize benchmark
    try:
        benchmark = SAMXMemBenchmark(
            data_root=data_root,
            results_dir=results_dir,
            sam_model_path=sam_model if sam_model.exists() else None,
            xmem_model_path=xmem_model if xmem_model.exists() else None,
        )
    except Exception as e:
        logger.error(f"Failed to initialize benchmark: {e}")
        return 1

    # Run benchmark on all available datasets
    datasets_to_run = benchmark.loader.list_datasets("train")
    if not datasets_to_run:
        logger.error("No datasets found")
        return 1

    logger.info(
        f"Running benchmark on all available datasets: {', '.join(datasets_to_run)}"
    )

    # Run benchmarks
    all_results = []

    for dataset_name in datasets_to_run:
        dataset_info = benchmark.loader.get_dataset_info(dataset_name)
        if not dataset_info:
            logger.warning(f"Dataset {dataset_name} not found")
            continue

        sequences_to_run = dataset_info.sequences

        for sequence in sequences_to_run:
            if sequence not in dataset_info.sequences:
                logger.warning(f"Sequence {sequence} not found in {dataset_name}")
                continue

            try:
                logger.info(f"Running benchmark: {dataset_name} sequence {sequence}")

                # Use default parameters (defined in SAMXMemBenchmark)
                result = benchmark.run_benchmark(dataset_name, sequence)
                all_results.append(result)

                logger.info(f"Completed {dataset_name} sequence {sequence}")
                if result.segmentation_metrics:
                    logger.info(
                        f"  Segmentation Dice: {result.segmentation_metrics.dice:.3f}"
                    )
                if result.tracking_metrics:
                    logger.info(
                        f"  Tracking Score: {result.tracking_metrics.tra_score:.3f}"
                    )
                logger.info(f"  FPS: {result.performance_metrics.fps:.2f}")

            except Exception as e:
                logger.error(
                    f"Failed to run benchmark on {dataset_name} sequence {sequence}: {e}"
                )
                continue

    if all_results:
        # Save results
        benchmark.save_results(all_results)

        # Generate report
        report = benchmark.generate_report(all_results)
        print("\n" + "=" * 60)
        print(report)

        # Save report to file
        report_file = results_dir / "benchmark_report.md"
        with open(report_file, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {report_file}")

        return 0
    else:
        logger.error("No benchmarks completed successfully")
        return 1


if __name__ == "__main__":
    sys.exit(main())
