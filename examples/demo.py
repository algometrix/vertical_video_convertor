"""Demo: convert the bundled sample video to vertical.

Run from anywhere; all paths are resolved relative to this file:

    python examples/demo.py            # GPU if available
    python examples/demo.py --cpu      # force CPU
    python examples/demo.py --show     # live preview window (q to quit)
    python examples/demo.py --compare  # original | converted side by side

Output goes to <repo>/output/demo/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_VIDEO = REPO_ROOT / "assets" / "videos" / "Conan Busts His Employees Eating Cake.mp4"
OUTPUT_DIR = REPO_ROOT / "output" / "demo"

# Allow running straight from a clone without installing the package
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    parser.add_argument("--show", action="store_true", help="show a live preview window (q to quit)")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="preview original and converted side by side (implies --show)",
    )
    parser.add_argument(
        "--refocus-band",
        type=float,
        default=None,
        help="no re-aim while the face stays within this fraction of frame width "
        "of the current aim; larger = calmer camera, 0 disables (default: 0.03)",
    )
    args = parser.parse_args()

    if not SAMPLE_VIDEO.exists():
        print(f"Sample video not found: {SAMPLE_VIDEO}", file=sys.stderr)
        return 1

    from vertical_video_converter import VerticalVideoConverter

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected_output = OUTPUT_DIR / f"{SAMPLE_VIDEO.stem}_vertical_9x16.mp4"
    if expected_output.exists():
        print(f"[*] Removing previous demo output: {expected_output.name}")
        expected_output.unlink()

    print(f"[*] Converting: {SAMPLE_VIDEO.name}")
    print(f"[*] Inference:  {'CPU' if args.cpu else 'GPU (CUDA) with CPU fallback'}")
    converter = VerticalVideoConverter(use_gpu=not args.cpu)
    output = converter.create_vertical_video(
        SAMPLE_VIDEO,
        output_dir=OUTPUT_DIR,
        refocus_band=args.refocus_band,
        show_preview=args.show,
        compare_preview=args.compare,
    )
    print(f"\n[*] Demo output: {output}")
    print("[*] Open it next to the original to compare the framing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
