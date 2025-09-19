# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LM Evaluation Harness is a unified framework for testing generative language models on academic benchmarks, developed by EleutherAI. It supports 60+ standard academic benchmarks with hundreds of subtasks and serves as the backend for Hugging Face's Open LLM Leaderboard.

## Essential Commands

### Development Setup
```bash
# Install for development
pip install -e ".[dev]"
pre-commit install
```

### Testing
```bash
# Run all tests (preferred method)
python -m pytest --showlocals -s -vv -n=auto --ignore=tests/models/test_openvino.py

# Install testing dependencies
pip install lm_eval[testing]
```

### Linting and Code Quality
```bash
# Run pre-commit hooks (includes Ruff linting)
pre-commit run --all-files
```

### Basic Evaluation Commands
```bash
# List available tasks
lm_eval --tasks list

# Basic evaluation
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8

# Check task integrity
lm_eval --model hf --model_args pretrained=gpt2 --tasks hellaswag --check_integrity
```

## Architecture Overview

### Core Package Structure (`lm_eval/`)
- **`models/`** - Model implementations for different backends (HuggingFace, vLLM, OpenAI, Anthropic, etc.)
- **`tasks/`** - 194+ task categories covering various academic benchmarks
- **`evaluator.py`** - Core evaluation engine that orchestrates model evaluation
- **`__main__.py`** - CLI entry point
- **`api/`** - API interfaces and caching system (SQLite-based)
- **`filters/`** - Data filtering and preprocessing utilities
- **`loggers/`** - Logging and result output (supports Weights & Biases)

### Key Entry Points
- **CLI**: Use `lm-eval` or `lm_eval` commands
- **Python API**: `lm_eval.evaluate()` and `lm_eval.simple_evaluate()`

### Task System
- **YAML-based configuration** for task definitions
- **Jinja2 templating** for prompt design and customization
- **Grouped tasks** for easy evaluation of benchmark suites
- Tasks are organized by benchmark type (e.g., `tasks/arc/`, `tasks/hellaswag/`)

### Model Support
The framework supports multiple model backends:
- **Local Models**: HuggingFace Transformers, GGUF, vLLM, SGLang, NVIDIA NeMo
- **APIs**: OpenAI, Anthropic, TextSynth, IBM Watsonx
- **Specialized**: Steered models, multimodal models, quantized models (GPTQ, AutoGPTQ)

### Optional Dependencies
Use modular installation with extras:
- `[vllm]` - For vLLM acceleration
- `[api]` - For API model support
- `[wandb]` - For experiment tracking
- `[tasks]` - All task-specific dependencies
- `[dev]` - Development and testing tools

### Development Workflow
- **Code Style**: Ruff linting enforced via pre-commit hooks
- **Testing**: pytest with parallel execution support (`-n=auto`)
- **Caching**: SQLite-based result caching to avoid re-evaluation
- **Multi-GPU**: Supports data and tensor parallelism for large-scale evaluation

### Key Files for Understanding the Codebase
- `lm_eval/evaluator.py` - Main evaluation logic
- `lm_eval/base.py` - Base classes for models and tasks
- `lm_eval/models/` - Model implementations
- `lm_eval/tasks/` - Task configurations and implementations
- `lm_eval/api/` - Core API and utilities