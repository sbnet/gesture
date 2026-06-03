"""Entry point: uv run hand-wire-demo."""

from __future__ import annotations

import typer

from app import run
from config import AppConfig

app = typer.Typer(pretty_exceptions_enable=False, add_completion=False)


@app.command()
def _command(
    camera: int = typer.Option(0, "--camera", "-c", help="Webcam device index."),
    width: int = typer.Option(1280, "--width", help="Requested frame width."),
    height: int = typer.Option(720, "--height", help="Requested frame height."),
    max_hands: int = typer.Option(1, "--max-hands", help="Maximum number of hands to detect."),
    model_complexity: int = typer.Option(
        1, "--model-complexity", min=0, max=1, help="MediaPipe model complexity (0=fast, 1=full)."
    ),
    min_detection_confidence: float = typer.Option(
        0.6, "--min-detection-confidence", help="Minimum detection confidence."
    ),
    min_tracking_confidence: float = typer.Option(
        0.5, "--min-tracking-confidence", help="Minimum tracking confidence."
    ),
    mirror: bool = typer.Option(True, "--mirror/--no-mirror", help="Flip image horizontally."),
    debug: bool = typer.Option(False, "--debug", help="Show landmark/feature debug overlay."),
) -> None:
    """Real-time hand wireframe and gesture recognition demo."""
    cfg = AppConfig(
        camera_index=camera,
        frame_width=width,
        frame_height=height,
        max_hands=max_hands,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        mirror=mirror,
        debug=debug,
    )
    run(cfg)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    app()
