"""Example script demonstrating video conversion to vertical format."""

import argparse
import sys
from vertical_video_converter import VerticalVideoConverter


def main():
    parser = argparse.ArgumentParser(description='Convert video to vertical format')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('-o', '--output', help='Output video path', default=None)
    parser.add_argument('--height-ratio', help='Height ratio', type=float, default=1.0)
    parser.add_argument('--vertical-video-ratio', help='Vertical video ratio', type=str, default="9/16")
    args = parser.parse_args()

    try:
        # Initialize the video converter
        converter = VerticalVideoConverter()

        # Convert the video
        print(f"Converting {args.input} to vertical format...")
        converter.create_vertical_video(args.input, args.output, args.vertical_video_ratio, args.height_ratio)
        print(f"Conversion complete! Output saved to {args.output}")

    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 