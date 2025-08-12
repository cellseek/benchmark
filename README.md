# Cell Tracking Challenge Benchmark Framework

This project implements a comprehensive benchmark framework for cell segmentation and tracking algorithms using the Cell Tracking Challenge dataset.

## Overview

The benchmark evaluates both segmentation accuracy and tracking performance across multiple microscopy modalities and cell types. It includes baseline models, integration with state-of-the-art methods like Cellpose, and comprehensive evaluation metrics.

## Project Structure

```
benchmark/
├── src/                      # Source code
│   ├── benchmark_framework.py   # Core benchmark framework
│   ├── baseline_models.py       # Simple baseline models for comparison
│   ├── cellpose.py              # Cellpose model wrapper
│   └── sam_xmem_benchmark.py    # SAM + XMem benchmark implementation
├── run_benchmark.py          # Main benchmark runner script
├── run_sam_xmem_benchmark.py # SAM + XMem benchmark runner
├── test_setup.py            # Setup verification script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Cell Tracking Challenge data
│   └── cell_tracking_challenge/
│       ├── train/          # Training data with ground truth
│       └── test/           # Test data (no ground truth)
├── benchmark_results/      # Output directory for results
└── test_results/          # Test output directory
```

## Installation

1. Clone or download this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install Cellpose for state-of-the-art comparison:
   ```bash
   pip install cellpose
   ```

## Quick Start

The framework now uses default parameters and automatically benchmarks all available datasets. No configuration files are needed.

### Basic Segmentation Benchmark

```python

```

### Run All Datasets (Recommended)

```python
# Simply run the main script to benchmark all datasets with default parameters
python run_benchmark.py
```

This will automatically:

- Use all available datasets in the `data/cell_tracking_challenge/train` folder
- Apply default parameters optimized for cell segmentation
- Run all baseline models and Cellpose models (if available)
- Generate comprehensive results and reports

### SAM + XMem Benchmark

```python
# Run SAM + XMem benchmark on all datasets
python run_sam_xmem_benchmark.py
```

### Custom Model Integration

```python
from src.benchmark_framework import BenchmarkFramework
from src.baseline_models import create_baseline_models

# Initialize benchmark
benchmark = BenchmarkFramework(
    data_root="data/cell_tracking_challenge",
    results_dir="benchmark_results"
)

# Get models (uses default parameters)
models = create_baseline_models()

# Framework will automatically use all available datasets and sequences
# No configuration needed!
```

## Default Parameters

The framework now uses sensible defaults for all parameters:

**Segmentation Parameters:**

- Threshold methods: Otsu, Li, adaptive thresholding
- Cellpose: Standard cytoplasm and nuclei models
- Minimum cell size: 15 pixels
- Morphological post-processing enabled

**SAM + XMem Parameters:**

- Cell diameter: 30 pixels (adjustable per dataset)
- Device: Automatic GPU/CPU selection
- Memory frames interval: 5
- Mixed precision training: Enabled

**Evaluation:**

- All standard metrics included (IoU, Dice, F1, etc.)
- Performance metrics tracked
- Visualization and reporting enabled

No configuration files needed - just run the scripts!

## Datasets

The framework supports multiple Cell Tracking Challenge datasets:

| Dataset         | Modality       | Cell Type                | Sequences |
| --------------- | -------------- | ------------------------ | --------- |
| BF-C2DL-HSC     | Brightfield    | Hematopoietic Stem Cells | 01, 02    |
| BF-C2DL-MuSC    | Brightfield    | Muscle Stem Cells        | 01, 02    |
| DIC-C2DH-HeLa   | DIC            | HeLa cells               | 01, 02    |
| Fluo-C2DL-Huh7  | Fluorescence   | Huh7 cells               | 01, 02    |
| Fluo-C2DL-MSC   | Fluorescence   | Mesenchymal Stem Cells   | 01, 02    |
| Fluo-N2DH-GOWT1 | Fluorescence   | GOWT1 cells              | 01, 02    |
| Fluo-N2DH-SIM+  | Fluorescence   | Simulated cells          | 01, 02    |
| Fluo-N2DL-HeLa  | Fluorescence   | HeLa cells               | 01, 02    |
| PhC-C2DH-U373   | Phase Contrast | U373 cells               | 01, 02    |
| PhC-C2DL-PSC    | Phase Contrast | Pancreatic Stem Cells    | 01, 02    |

## Evaluation Metrics

### Segmentation Metrics

- **IoU (Intersection over Union)**: Overlap between predicted and ground truth masks
- **Dice Coefficient**: Harmonic mean of precision and recall
- **F1 Score**: Balanced measure of precision and recall
- **Detection Rate**: Fraction of ground truth objects detected
- **False Positive Rate**: Fraction of false detections

### Tracking Metrics

- **TRA Score**: Cell Tracking Challenge tracking accuracy metric
- **DET Score**: Cell Tracking Challenge detection accuracy metric
- **MOTA**: Multi-Object Tracking Accuracy
- **MOTP**: Multi-Object Tracking Precision
- **IDF1**: Identification F1 score

### Performance Metrics

- **FPS**: Frames processed per second
- **Memory Usage**: Peak memory consumption
- **Inference Time**: Time per image

## Baseline Models

The framework includes several baseline models for comparison:

1. **Otsu Thresholding**: Classical threshold-based segmentation
2. **Li Thresholding**: Alternative thresholding method
3. **Canny Edge Detection**: Edge-based segmentation
4. **Adaptive Thresholding**: Handles varying illumination
5. **Random Baseline**: Random segmentation for lower bound

## Adding Custom Models

To add your own model, implement the `SegmentationModel` interface:

```python
from benchmark_framework import SegmentationModel
import numpy as np

class MyCustomModel(SegmentationModel):
    def __init__(self, name="my_model", **params):
        self.name = name
        self.params = params

    def predict(self, image: np.ndarray) -> np.ndarray:
        # Your segmentation logic here
        # Return integer mask with object labels
        return predicted_mask

    def get_info(self) -> dict:
        return {
            "name": self.name,
            "type": "custom",
            "params": self.params
        }

# Add to benchmark
models = get_baseline_models()
models.append(MyCustomModel())
```

## Output

The benchmark generates:

1. **Numerical Results**: CSV files with all metrics
2. **Visualizations**: Performance comparison plots
3. **HTML Report**: Comprehensive benchmark report
4. **Logs**: Detailed execution logs

Results are saved in the `benchmark_results` directory with timestamps.

## Performance Tips

1. **GPU Usage**: Set `use_gpu: true` in config for compatible models
2. **Memory Management**: Adjust `max_memory_gb` based on available RAM
3. **Parallel Processing**: Increase `num_workers` for faster processing
4. **Dataset Selection**: Start with smaller datasets for quick testing

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install all requirements with `pip install -r requirements.txt`
2. **Memory Errors**: Reduce `max_memory_gb` or process fewer images at once
3. **Cellpose Import Error**: Install with `pip install cellpose` or disable in config
4. **Data Path Issues**: Ensure data is in the correct directory structure

### Getting Help

1. Check the example scripts in `run_benchmark.py`
2. Review configuration options in `config.yaml`
3. Enable debug logging: set `level: "DEBUG"` in config

## Citation

If you use this benchmark framework in your research, please cite:

```bibtex
@software{cell_tracking_benchmark,
    title={Cell Tracking Challenge Benchmark Framework},
    author={Your Name},
    year={2024},
    url={https://github.com/your-repo/cell-tracking-benchmark}
}
```

Also cite the Cell Tracking Challenge:

```bibtex
@article{ctc2017,
    title={An objective comparison of cell-tracking algorithms},
    author={Maška, Martin and others},
    journal={Nature Methods},
    volume={14},
    number={12},
    pages={1141--1152},
    year={2017},
    publisher={Nature Publishing Group}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
