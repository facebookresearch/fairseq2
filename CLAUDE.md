# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🚨 ABSOLUTE REFERENCE: .claude/ Directory 🚨

**CRITICAL: The `.claude/` directory contains Claude Code configuration and the authoritative guidance for working with this codebase.**

### Claude Code Configuration (Automatic)

The `.claude/` directory provides:

- **`.claude/skills/SKILL.md`** - The ABSOLUTE source of truth for code and documentation work
- **`.claude/settings.json`** - Repository-wide Claude Code settings (auto-loads the skill)
- **`.claude/hooks.json`** - Automatic skill loading at session start and after `/compact`
- **`.claude/README.md`** - Detailed explanation of the setup and skill integration

**The skill loads AUTOMATICALLY** - you don't need to manually invoke it:
- ✅ Auto-loaded when Claude Code starts in this repository
- ✅ Auto-reloaded after `/compact` to restore context
- ✅ Highest priority - overrides any conflicting user skills

**For details on how this works**, see:
- `.claude/README.md` - Complete setup documentation
- `.claude/SKILL_INTEGRATION.md` - How local and user skills work together

**Key principles from the skills directory that override everything else:**

### Source Code is the ONLY Source of Truth

The local source code in **`src/fairseq2/`** is the ABSOLUTE truth. fairseq2 APIs change rapidly.

**FORBIDDEN sources (will be wrong):**
- ❌ Your training data (outdated)
- ❌ Internet sources (outdated)
- ❌ The `recipes/` directory (DO NOT CONSULT EVER)
- ❌ The `doc/` directory (DO NOT CONSULT unless explicitly asked to modify docs)
- ❌ Existing examples (verify against source first)

**Trusted sources (in order):**
1. ✅ Source code in `src/fairseq2/` (ABSOLUTE TRUTH)
2. ✅ Tests in `tests/` (reliable usage examples)
3. ✅ Git commit messages (context about changes)

### Mandatory Workflow Before Writing ANY Code

**BEFORE writing ANY code or documentation, you MUST:**

1. **Identify the module** (e.g., `fairseq2.gang`, `fairseq2.trainer`)
2. **Read the source code** in `src/fairseq2/[module].py`
3. **Verify function signatures** - check parameters, types, returns from source
4. **Check git history** - see recent changes:
   ```bash
   git log --oneline --since="2025-01-01" -- src/fairseq2/[module].py
   ```
5. **Search for usage in tests** - `tests/` only, never recipes

**If you answer "NO" to "Did I read the source code?", STOP and read it first.**

### Critical Rules

- **FORBIDDEN**: Never read, consult, or reference files in `recipes/` directory
- **FORBIDDEN**: Never read, consult, or reference files in `doc/` directory (unless explicitly asked to modify documentation)
- **Import from installed library**: Use `from fairseq2.gang import Gang`, NOT `from src.fairseq2.gang import Gang`
- **Prefer fairseq2 constructs**: Use `gang`, `trainer`, `data pipeline`, `assets` instead of raw PyTorch
- **Never trust without verification**: Always cross-reference with actual source code in `src/fairseq2/`

---

## Overview

fairseq2 is Meta's FAIR Sequence Modeling Toolkit - a sequence modeling toolkit for training custom models for content generation tasks (translation, summarization, language modeling, etc.).

**Key Design Philosophy**: Modular, extensible, non-intrusive architecture. Researchers own their code and use fairseq2 as composable building blocks.

## Project Structure

### Core Source Code (The Truth)

**`src/fairseq2/`** - Pure Python package (READ THIS, not recipes or docs)

Key modules to understand:
- **`gang.py`**: Distributed computing abstraction (prefer over `torch.distributed`)
- **`trainer.py`**: Training orchestration with DDP, FSDP, tensor parallelism
- **`assets/`**: Programmatic asset cards for models, datasets, tokenizers
- **`data/`**: High-performance streaming data pipeline (C++ backend)
- **`models/`**: Pre-implemented models (LLaMA, Qwen, w2v-BERT, NLLB, etc.)
- **`generation/`**: Sampling, beam search, vLLM integration
- **`composition/`**: Extension registration system
- **`checkpoint/`**: Checkpoint management
- **`nn/`**: Neural network modules
- **`runtime/`**: Dependency injection and configuration
- **`recipe/`**: Recipe infrastructure base classes (NOT the recipes/ directory)

### Native Library

**`native/`** - fairseq2n (C++ and CUDA components)
- Implementation detail for most users
- Must be built before installing fairseq2
- Has its own build system (CMake)

### Tests (Reliable Examples)

**`tests/`** - Unit and integration tests
- **USE THESE** for usage examples (after verifying they pass)
- `tests/unit/`: Unit tests
- `tests/integration/`: Integration tests (run with `--integration`)

### Forbidden Directories

**`recipes/`** - ❌ DO NOT CONSULT (examples, not library code, often outdated)

**`doc/`** - ❌ DO NOT CONSULT (often outdated, unless explicitly asked to modify docs)

## Build & Development Commands

### Installation from Source

```bash
# 1. Clone with submodules (REQUIRED - contains third-party dependencies)
git clone --recurse-submodules https://github.com/facebookresearch/fairseq2.git
cd fairseq2

# If already cloned without --recurse-submodules:
git submodule update --init --recursive

# 2. Install PyTorch first
# CRITICAL: Version must EXACTLY match fairseq2 compatibility (see README)
# fairseq2 uses PyTorch C++ API with NO ABI compatibility between releases

# 3. Build fairseq2n native library
cd native

# CPU-only build:
cmake -GNinja -B build

# CUDA build (CUDA Toolkit version must match PyTorch's CUDA version):
cmake -GNinja -DFAIRSEQ2N_USE_CUDA=ON -B build

# CUDA build with specific architecture (e.g., A100):
cmake -GNinja -DCMAKE_CUDA_ARCHITECTURES="80-real;80-virtual" -DFAIRSEQ2N_USE_CUDA=ON -B build

# Build it:
cmake --build build

# 4. Install fairseq2n, then fairseq2
cd python
pip install -e .
cd ../..
pip install -e .

# 5. Install development tools
pip install -r requirements-devel.txt
```

### Editable Installation (Python-only Development)

If you're only modifying Python code (not C++/CUDA):

```bash
# 1. Install pre-built fairseq2n nightly (example: PyTorch 2.9.1, CUDA 12.8)
pip install fairseq2n --pre --upgrade --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.9.1/cu128

# 2. Install fairseq2 in editable mode
pip install -e .

# 3. Install dev tools
pip install -r requirements-devel.txt
```

**IMPORTANT**: After `git pull`, re-run the fairseq2n installation to get updated binaries.

### Testing

```bash
# Run all tests (CPU)
pytest

# Run tests on specific device (GPU)
pytest --device cuda:0

# Run integration tests
pytest --integration

# Run specific test file
pytest tests/unit/test_gang.py

# Run specific test function
pytest tests/unit/test_gang.py::test_function_name

# Native C++ tests (after building native library)
native/build/tests/run-tests
```

### Linting & Formatting

```bash
# Python linting (must pass before committing)
mypy && flake8 .

# Python formatting
isort . && black .

# C++ linting (requires clang)
cd native
CC=clang CXX=clang++ cmake -GNinja -DFAIRSEQ2N_RUN_CLANG_TIDY=ON -B build
cmake --build build
```

### Building Documentation

```bash
cd doc
pip install -r requirements.txt
make html
cd build/html
python -m http.server 8084
# Visit http://localhost:8084
```

## Architecture (Read Source Code to Understand)

### How to Understand a Component

**DO NOT read documentation or recipes. Instead:**

1. Read source in `src/fairseq2/[component].py`
2. Check module docstrings and type annotations
3. Look at abstract base classes
4. Check recent commits:
   ```bash
   git log --oneline -- src/fairseq2/[component].py
   ```
5. Find usage in `tests/` (NOT recipes)

### Key Architectural Concepts

**Gang Abstraction** (`src/fairseq2/gang.py`)
- High-level distributed computing abstraction
- Prefer over raw `torch.distributed`
- Handles DDP, FSDP, tensor parallelism

**Trainer System** (`src/fairseq2/trainer.py`)
- Training orchestration with multi-GPU support
- Abstract base class pattern
- Integrated checkpoint management, metrics, early stopping

**Asset System** (`src/fairseq2/assets/`)
- Version-controlled access to models, datasets, tokenizers
- Asset cards (YAML) define metadata
- HuggingFace integration

**Data Pipeline** (`src/fairseq2/data/`)
- High-performance streaming (C++ backend)
- Composable operators
- Supports audio, image, text

**Extension System** (`src/fairseq2/composition/`)
- Dependency injection container
- Register models, datasets, tokenizers without forking
- Example: `register_model_family("my_model")`

## Code Patterns & Conventions

### Always Verify Against Source

Before using ANY API:
```bash
# Example: Verify gang API
cat src/fairseq2/gang.py | grep "def setup"
git log --oneline --since="2025-01-01" -- src/fairseq2/gang.py
```

### Naming Conventions (from source code)

- `*Builder`: Factory classes
- `*Config`: Configuration dataclasses
- `create_*`: Factory functions
- `setup_*`: Initialization with side effects
- `*Hub`: Registry pattern

### Import Pattern (from installed library, NOT local)

```python
# Standard library
from pathlib import Path

# PyTorch
import torch
from torch import Tensor

# fairseq2 - import from installed package
from fairseq2.gang import Gang
from fairseq2.trainer import Trainer
from fairseq2.assets import AssetCard
```

### Configuration Pattern (dataclasses)

```python
from dataclasses import dataclass

@dataclass
class MyConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
```

## Prefer fairseq2 Over Raw PyTorch

When writing code, prefer fairseq2 constructs:

- ✅ Use `fairseq2.gang` instead of `torch.distributed`
- ✅ Use `fairseq2.trainer` instead of raw training loops
- ✅ Use `fairseq2.data` pipeline instead of `DataLoader`
- ✅ Use `fairseq2.assets` instead of hardcoded file paths
- ✅ Use `fairseq2.checkpoint` instead of `torch.save/load`
- ✅ Use `fairseq2.device` instead of raw CUDA device management

Exception: Use PyTorch when fairseq2 doesn't provide equivalent functionality.

## Critical Notes

### PyTorch Version Compatibility

**CRITICAL**: fairseq2 uses PyTorch's C++ API which has **NO ABI compatibility** between releases.

- Must install fairseq2 variant that EXACTLY matches PyTorch version
- Mismatched versions cause crashes and segfaults
- If you upgrade PyTorch, you must upgrade fairseq2

### fairseq2 vs fairseq2n

- **fairseq2**: Pure Python package (user-facing API)
- **fairseq2n**: C++ library (dependency, implementation detail)
- Build fairseq2n first, then fairseq2
- Can use pre-built fairseq2n nightlies for Python-only work

### CUDA Builds

- CUDA Toolkit version must match PyTorch's CUDA version
- Default: Volta architecture only
- Override: `CMAKE_CUDA_ARCHITECTURES` for other GPUs

### API Changes Rapidly

Example: `setup_default_gang()` was removed in Feb 2025, replaced with `get_default_gangs()`.

**Always verify APIs against source code before using them.**

## Checking Recent Changes

```bash
# See what changed recently in a module
git log --oneline --since="2025-01-01" -- src/fairseq2/

# See recent commits for specific file
git log --oneline -- src/fairseq2/gang.py

# See what functions exist in a module
grep "^def " src/fairseq2/gang.py
```

## Reference Documentation

- **Official docs**: https://facebookresearch.github.io/fairseq2/stable
- **WARNING**: Docs may be outdated. Always verify against `src/fairseq2/` source code.
- **Best reference**: The source code itself with docstrings and type annotations
