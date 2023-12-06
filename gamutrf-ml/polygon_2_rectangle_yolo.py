"""Converts YOLO polygon labels to YOLO bounding box labels.

Polygon: (https://docs.ultralytics.com/datasets/segment/#ultralytics-yolo-format)
BB: (https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)
"""
import os
import argparse
from statistics import mean


def parse_cli() -> argparse.Namespace:
    """Parse the CLI to get an input directory and a directory 

    Returns:
        argparse.Namespace: Argparse argument payload
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=str, help="relative path to label file[s] directory"
    )
    parser.add_argument(
        "--output-dir", type=str, help="relative path to save label file[s] directory"
    )
    return parser.parse_args()


def polygon_2_rectangle(label_in_dir: str, label_out_dir: str) -> None:
    """Converts all YOLO polygon label files in a directory to bounding boxes and
    converts them to  YOLO bounding box labels in the format x_center, y_center,
    width, height. Those labels are then saved to the output directory (note), if
    the input and output directory are the same, you will overwrite the polygon labels.

    Args:
        label_in_dir (str): relative path to directory that contains YOLO polygon labels.
        label_out_dir (str): relative path to directory to save YOLO bounding box labels.
    """
    label_files_ls = [x for x in os.listdir(label_in_dir) if x.endswith(".txt")]
    for label_file in label_files_ls:
        with open(os.path.join(label_in_dir, label_file), "r", encoding="utf-8") as f:
            instances = f.readlines()
            for instance in instances:
                parsed_instance = instance.rstrip().split(" ")
                class_idx = parsed_instance[0]
                x_points = [
                    float(parsed_instance[1:][x])
                    for x in range(0, len(parsed_instance[1:]) - 1, 2)
                ]
                y_points = [
                    float(parsed_instance[1:][x + 1])
                    for x in range(0, len(parsed_instance[1:]), 2)
                ]

                x_center, y_center = mean([min(x_points), max(x_points)]), mean(
                    [min(y_points), max(y_points)]
                )
                width, height = max(x_points) - min(x_points), max(y_points) - min(
                    y_points
                )

                with open(
                    os.path.join(label_out_dir, label_file), "a+", encoding="utf-8"
                ) as f_out:
                    f_out.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":
    args = parse_cli()
    polygon_2_rectangle(args.input_dir, args.output_dir)
