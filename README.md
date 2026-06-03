# hand-wire-demo

Real-time hand gesture recognition using MediaPipe and OpenCV. Detects hand landmarks from a webcam feed and classifies gestures with a wireframe skeleton overlay.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue) ![License MIT](https://img.shields.io/badge/license-MIT-green)

## Demo

The app renders a live wireframe of 21 hand landmarks and displays the recognized gesture in the top-left corner.

**Supported gestures:** Open Palm · Fist · Peace · Point · Thumbs Up · OK · Rock

## Requirements

- Python 3.11+
- A webcam
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
git clone git@github.com:sbnet/gesture.git
cd gesture
uv sync          # or: pip install -e .
```

On first run the app downloads the MediaPipe hand landmark model (~1 MB) to `~/.cache/hand-wire-demo/`.

## Usage

```bash
uv run hand-wire-demo
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--camera`, `-c` | `0` | Webcam device index |
| `--width` | `1280` | Frame width |
| `--height` | `720` | Frame height |
| `--max-hands` | `1` | Maximum hands to detect simultaneously |
| `--model-complexity` | `1` | `0` = fast, `1` = full accuracy |
| `--min-detection-confidence` | `0.6` | Detection confidence threshold |
| `--min-tracking-confidence` | `0.5` | Tracking confidence threshold |
| `--mirror` / `--no-mirror` | mirror on | Flip image horizontally |
| `--debug` | off | Show landmark indices and raw feature values |

```bash
# Examples
uv run hand-wire-demo --camera 1 --debug
uv run hand-wire-demo --no-mirror --max-hands 2
```

Press **Q** or **Esc** to quit.

## Development

```bash
uv sync --dev

# Lint & format
uv run ruff check src tests
uv run ruff format src tests

# Type check
uv run mypy src

# Tests
uv run pytest
uv run pytest --cov=hand_wire_demo --cov-report=term-missing
```

CI runs on Python 3.11 and 3.12 via GitHub Actions (lint → format → type check → tests).

## License

MIT — see [LICENSE](LICENSE).
