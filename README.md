# PEFT-CL: A Unified Framework for Parameter-Efficient Fine-Tuning in Continual Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PEFT-CL is a unified framework for implementing and evaluating various Parameter-Efficient Fine-Tuning (PEFT) methods in Continual Learning (CL) scenarios. This framework provides a standardized interface to easily compare and benchmark different PEFT approaches, including Prompt-based, LoRA-based, and Adapter-based methods.

## TODO List
### Prompt Series
- [x] L2P
- [x] DualPrompt
- [x] CODA-Prompt
- [x] HiDe-Prompt
- [x] C-PT
- [x] LFPT5
- [ ] Episodic Memory Prompts
- [ ] ProgPrompt
- [ ] LGCL
- [ ] OneStagePCL
- [ ] MoE_PromptCL
- [ ] C-CLIP

### LoRA Variations
- [x] infLoRA
- [x] SD-LoRA
- ~~[ ] FM-LoRA~~
- ~~[ ] TreeLoRA~~
- [ ] BI-LORA
- [ ] O-LoRA
### Adapter Tuning
- [x] MoE-Adapters4CL
- [x] SEED
- [x] CoSCL
- [ ] CN-DPM
- [ ] CPT
- [ ] AdatperCL
- [ ] Adaptive Compositional Model
- [ ] ADA
- [ ] HAM
### Others
- [x] Full Params Fine-tuning


## Features

- **Unified Interface**: Standardized API for all PEFT methods in continual learning
- **Comprehensive Methods**: Support for multiple PEFT approaches including Prompts, LoRA, and Adapters
- **Easy Configuration**: JSON-based configuration system for flexible experiment setup
- **Reproducible Results**: Built-in seed management and logging for reproducible experiments
- **Extensible Design**: Easy to add new PEFT methods following the framework structure

## Setup

```bash
git clone https://github.com/yourusername/PEFT-CL.git && cd PEFT-CL && pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Run an experiment using a configuration file:

```bash
python main.py --config exps/l2p.json
```

### Example Configurations

The `exps/` directory contains example configuration files for different methods:

- `l2p.json` - Learn to Prompt configuration
- `dualprompt.json` - DualPrompt configuration
- `coda_prompt.json` - CODA-Prompt configuration
- `inflora_c100.json` - InfLoRA configuration
- `sdlora.json` - SD-LoRA configuration
- `moe_adapters_c100.json` - MoE-Adapters configuration
- `seed_c100.json` - SEED configuration
- `coscl_c100.json` - CoSCL configuration
- `cpt_c100.json` - C-PT configuration
- `lfpt5_c100.json` - LFPT5 configuration
- `finetune.json` - Full fine-tuning baseline

### Configuration File Structure

A typical configuration file includes:

```json
{
    "prefix": "experiment_name",
    "dataset": "cifar224",
    "model_name": "l2p",
    "backbone_type": "vit_base_patch16_224_l2p",
    "init_cls": 10,
    "increment": 10,
    "device": ["0"],
    "seed": [1993],
    "batch_size": 16,
    "tuned_epoch": 5,
    "init_lr": 0.001875,
    "optimizer": "adam",
    "scheduler": "constant",
    ...
}
```

### Running Multiple Seeds

To run experiments with multiple random seeds, specify them in the configuration:

```json
{
    "seed": [0, 1993, 2024]
}
```

## Project Structure

```
PEFT-CL/
├── main.py                 # Main entry point
├── trainer.py              # Training logic
├── models/                 # Model implementations for each method
│   ├── l2p.py
│   ├── dualprompt.py
│   ├── inflora.py
│   └── ...
├── backbone/               # Backbone network implementations
│   ├── base_vit.py
│   ├── resnet.py
│   ├── prompt.py
│   ├── lora.py
│   └── ...
├── utils/                  # Utility functions
│   ├── factory.py          # Model factory
│   ├── data_manager.py     # Data management
│   ├── inc_net.py          # Incremental network utilities
│   └── ...
├── exps/                   # Configuration files
│   ├── l2p.json
│   ├── dualprompt.json
│   └── ...
├── logs/                   # Experiment logs (auto-generated)
└── third_party/            # Third-party implementations
```

## Configuration Parameters

### Common Parameters

- `model_name`: Name of the PEFT method (e.g., "l2p", "inflora", "dualprompt")
- `backbone_type`: Backbone architecture (e.g., "vit_base_patch16_224")
- `dataset`: Dataset name (e.g., "cifar224", "cifar100")
- `init_cls`: Number of classes in the first task
- `increment`: Number of new classes per task
- `device`: GPU device IDs (e.g., ["0"] or ["0", "1"])
- `seed`: Random seed(s) for reproducibility
- `batch_size`: Training batch size
- `tuned_epoch` / `epochs`: Number of training epochs
- `init_lr`: Initial learning rate
- `optimizer`: Optimizer type ("adam", "sgd", etc.)
- `scheduler`: Learning rate scheduler ("constant", "cosine", etc.)

### Method-Specific Parameters

Each method may have additional parameters. Refer to the example configuration files in `exps/` for method-specific settings.

## Output and Logging

- **Logs**: Training logs are automatically saved to `logs/{model_name}/{dataset}/{init_cls}/{increment}/`
- **Metrics**: The framework tracks and logs:
  - Top-1 and Top-5 accuracy per task
  - Average accuracy across all tasks
  - Forgetting measure (if enabled)
  - Parameter counts (total and trainable)

## Adding New Methods

To add a new PEFT method:

1. Create a new model file in `models/` (e.g., `models/new_method.py`)
2. Implement the `Learner` class following the base interface
3. Add the method to `utils/factory.py`
4. Create a configuration file in `exps/`
5. Add any necessary backbone components in `backbone/` if needed

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{peft-cl,
  title={PEFT-CL: A Unified Framework for Parameter-Efficient Fine-Tuning in Continual Learning},
  author={Wang, Ziyi},
  year={2025},
  license={MIT}
}
```