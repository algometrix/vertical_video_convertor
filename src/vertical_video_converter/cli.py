"""Command-line interface: `vvc <input> [options]`."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vvc",
        description="Convert horizontal video to a face-tracked vertical crop.",
    )
    parser.add_argument("input", help="input video file")
    parser.add_argument("-o", "--output-dir", default=None, help="output directory (default: next to input)")
    parser.add_argument("-r", "--ratio", default="9/16", help='output aspect ratio, e.g. "9/16" (default)')
    parser.add_argument(
        "--height-ratio",
        type=float,
        default=1.0,
        help="crop height as a fraction of source height (default: 1.0)",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=0.42,
        help="face position in the crop, 0.5 = centered (default: 0.42)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=2.0,
        help="hold the last face position this long on detection dropouts (default: 2.0)",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=28.0,
        help="scene-cut sensitivity, lower = more sensitive (default: 28.0)",
    )
    parser.add_argument(
        "--refocus-band",
        type=float,
        default=None,
        help="no re-aim while the face stays within this fraction of frame width "
        "of the current aim; larger = calmer camera, 0 disables (default: 0.03)",
    )
    parser.add_argument("--det-size", type=int, default=640, help="face detector input size (default: 640)")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    parser.add_argument("--show", action="store_true", help="show a live preview window (q to quit)")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="preview original and converted side by side (implies --show)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Import here so `vvc --help` stays fast (insightface is slow to import).
    from .converter import VerticalVideoConverter

    try:
        converter = VerticalVideoConverter(det_size=args.det_size, use_gpu=not args.cpu)
        converter.create_vertical_video(
            args.input,
            output_dir=args.output_dir,
            aspect_ratio=args.ratio,
            height_ratio=args.height_ratio,
            headroom=args.headroom,
            hold_seconds=args.hold_seconds,
            scene_cut_threshold=args.scene_threshold,
            refocus_band=args.refocus_band,
            show_preview=args.show,
            compare_preview=args.compare,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
