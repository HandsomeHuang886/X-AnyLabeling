# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

X-AnyLabeling is a cross-platform (Windows/Linux/macOS) PyQt6 desktop application for AI-assisted annotation of text, image, video, and multimodal data. It combines manual annotation tools with a plugin-style framework that runs ONNX/TensorRT/OpenCV-DNN models (YOLO family, SAM family, PP-OCR, Florence2, Grounding DINO, etc.) for auto-labeling, plus format import/export (COCO, VOC, YOLO, DOTA, MOT, MASK, PPOCR, XLSX, etc.) and an optional Ultralytics training platform. GPLv3, Python ≥ 3.11.

## Commands

```bash
# Install for development (pick one runtime; see note below)
pip install -e ".[cpu,dev]"       # CPU (onnxruntime)
pip install -e ".[gpu,dev]"       # CUDA 12.x (onnxruntime-gpu)
pip install -e ".[gpu-cu11,dev]"  # CUDA 11.x
pip install -e ".[gpu-cu13,dev]"  # CUDA 13.x
```

> **Important:** `onnxruntime` and `onnxruntime-gpu` must never be installed simultaneously. The extras are mutually exclusive (enforced via `[tool.uv]` conflicts in `pyproject.toml`).

```bash
# Launch the GUI
xanylabeling                       # or: python anylabeling/app.py
xanylabeling --filename IMG       # open an image/folder

# Headless CLI (no GUI)
xanylabeling checks               # system/package info
xanylabeling version
xanylabeling config               # print config file path (~/.xanylabelingrc)
xanylabeling convert --task yolo2xlabel --mode detect \
    --images ./images --labels ./labels --output ./output --classes classes.txt

# Tests (Qt must run offscreen in headless envs)
QT_QPA_PLATFORM=offscreen pytest
QT_QPA_PLATFORM=offscreen pytest tests/test_utils/test_general.py::TestIsRectangle::test_normal_rectangle
pytest tests/test_models -v      # a whole directory

# Lint / format / pre-commit
flake8                            # configured in .flake8 + [tool.flake8]
black --check .                   # line-length 79, configured in [tool.black]
pre-commit run --all-files

# Build distributable executable (PyInstaller specs in packaging/pyinstaller/specs/)
scripts/build_executable.sh linux-cpu   # {win-cpu|win-gpu|linux-cpu|linux-gpu|macos}

# Translations
scripts/generate_languages.py     # regenerate .ts from source strings
scripts/compile_languages.py      # compile .ts -> .qm in anylabeling/resources/translations
```

Notes:
- `pytest` is configured in `[tool.pytest.ini_options]`: `--doctest-modules --durations=30`, `testpaths = ["tests"]`, and a `slow` marker. The release CI runs `pytest --ignore=tests/test_widgets/test_toolbar_layout.py` with `QT_QPA_PLATFORM=offscreen`.
- Model/config docs live in `docs/en/` (`get_started.md`, `user_guide.md`, `cli.md`, `custom_model.md`).

## Architecture

The GUI is a thin main-window stack that delegates everything to the labeling widget:

```
anylabeling/app.py:main()  →  MainWindow (views/mainwindow.py)  →  LabelingWrapper (views/labeling/label_wrapper.py)
                              →  LabelingWidget (views/labeling/label_widget.py, ~7k lines, the core UI logic)
```

### UI layer (`anylabeling/views/`)
- **`labeling/label_widget.py`** — `LabelingWidget` (subclass of `LabelDialog`). The heart of the app: file I/O, canvas wiring, undo/redo, shape editing, menus/toolbars, settings, and the `ModelManager` connection. Very large; logic is broken out into the modules below.
- **`labeling/widgets/canvas.py`** — `Canvas` widget (~5.5k lines): renders shapes, handles mouse interaction, shape editing (move/resize vertices), brush/magic-wand modes, cuboid/rotated-box drawing. Most drawing-state bugs live here.
- **`labeling/shape.py`** — `Shape` data model (rectangle, polygon, circle, line, cuboid, rotated box, quadrilateral, mask …). Serializes via `to_dict()` / `load_from_dict()`.
- **`labeling/label_file.py`** + **`labeling/schema.py`** — the XLABEL JSON format (`LabelFile` class + template builder). The internal interchange format all converters target.
- **`labeling/widgets/auto_labeling/auto_labeling.py`** — the "AI" sidebar panel: model dropdown, per-model control widgets (conf/IoU sliders, prompt box, etc.), and download-progress UI. Drives `ModelManager`.
- **`labeling/widgets/`** — one file per dialog/panel (label dialog, shape dialog, classifier, chatbot, ppocr, vqa, remote server dialog, …).
- **`labeling/utils/`** — `qt.py`, `image.py`, `shape.py`, `crop.py`, `export.py`, `batch.py`, `video.py`, `theme.py`, `update_checker.py`, etc.
- **`common/converter.py`** — the `xanylabeling convert` framework. `SUPPORTED_TASKS` dict maps task names (`yolo2xlabel`, `xlabel2yolo`, `coco2xlabel`, `voc2xlabel`, `dota2xlabel`, `mot2xlabel`, `ppocr2xlabel`, `mask2xlabel`, …) to their modes/args; add a new format conversion by extending this dict.
- **`common/checks.py`** (`xanylabeling checks`), **`common/device_manager.py`**, **`common/toaster.py`**.

### Auto-labeling framework (`anylabeling/services/auto_labeling/`)
The model plugin system. Adding a new model touches these pieces:

1. **`configs/models.yaml`** — registry of every built-in model: `model_name` + `config_file` (referenced as `:/name.yaml` for bundled configs).
2. **`configs/auto_labeling/<name>.yaml`** — per-model config: `type`, `name`, `provider`, `display_name`, `model_path` (local path or download URL), `iou_threshold`, `conf_threshold`, `classes`, plus `Meta.required_config_names`. `type` is the discriminator that selects the implementation.
3. **`model.py`** — base `Model` class (QObject). Handles config parsing, `predict_shapes(image, filename)` / `unload()` abstract interface, model validation (`safe_check_model`, run in a subprocess to avoid native crashes), and downloading from GitHub releases or ModelScope into `~/xanylabeling_data/models/<model_name>/`. Cancellation + retry are built in.
4. **`model_manager.py`** — `ModelManager` (QObject): loads configs, maps a model `type` → implementation via a big dispatch in `_load_model`, runs inference on a QThread through `GenericWorker`, and emits signals (`new_auto_labeling_result`, `model_loaded`, `prediction_started/finished`, download progress).
5. **`__init__.py`** — **the model registry lists.** `_CUSTOM_MODELS` must contain the new model's `type` name (dispatch in `model_manager._load_model` also needs a branch). The other `_AUTO_LABELING_*` lists are allowlists that toggle which controls the UI shows for a model (conf slider, IoU slider, marks, mask fineness, cropping, prompt, reset tracker, etc.) — add the type name to the relevant lists.
6. **`engines/`** — `OnnxBaseModel` (ONNX Runtime), `DnnBaseModel` (OpenCV DNN), `TrtBaseModel` (TensorRT). Most implementations wrap one of these.
7. **`__base__/`** — shared per-family base classes (`yolo.py`, `sam.py`, `grounding_dino.py`, …) that concrete model files (e.g. `yolo11.py`, `segment_anything_2.py`) subclass.
8. **`types.py`** — `AutoLabelingResult`, `AutoLabelingMode`, `DownloadCancelledError`. `worker.py` — `GenericWorker`. `trackers/` — bot-sort / byte-track / track-track MOT implementations.

Concrete implementations are one file per model family at the top level (e.g. `yolo11.py`, `rfdetr.py`, `dfine_seg.py`, `ppocr_v6.py`, `florence2.py`). See `docs/en/custom_model.md` for the full end-to-end workflow.

### Config system (`anylabeling/config.py`)
Three sources merged in `get_config()`: default `configs/xanylabeling_config.yaml` → user `~/.xanylabelingrc` (or `--work-dir/.xanylabelingrc`) → CLI args. Legacy keys are migrated (`epsilon` → `canvas.epsilon`, etc.) via `normalize_user_config`. Settings persistence uses Qt `QSettings` (org `anylabeling`).

### Training (`anylabeling/services/auto_training/`)
Optional one-click Ultralytics training launched via the hidden `xanylabeling train-worker --payload ...` subcommand; UI in `views/training/`.

## Key conventions

- **Model files are downloaded at runtime** into `~/xanylabeling_data/models/<model_name>/` (migrated from the legacy `anylabeling_data` dir). Set `XANYLABELING_MODEL_HUB=modelscope` to force ModelScope as the download source; otherwise ModelScope is used only when `model_hub: modelscope` is configured or the UI language is `zh_CN`.
- `anylabeling/resources/resources.py` is **generated** (from `resources.qrc`) and excluded from black/flake8 — don't hand-edit it.
- Keep `anylabeling.views.labeling.logger` usage for all app logging.
- Format: black (line-length 79), flake8 (`select = B,C,E,F,W,T4,B9`), enforced by pre-commit hooks; tests run with `--doctest-modules`, so docstring code samples must be correct.
