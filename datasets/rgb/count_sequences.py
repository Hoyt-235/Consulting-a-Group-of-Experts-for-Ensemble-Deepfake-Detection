#!/usr/bin/env python3
import os
import sys

def count_image_groups(root_dir, group_size=8):
    """
    Recursively count images in groups of `group_size` per directory.
    For each subdirectory, only complete groups of `group_size` images
    are counted; any leftover images (< group_size) are discarded.
    """
    total = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter for supported image extensions
        images = [
            fname for fname in filenames
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        # Only complete groups of `group_size` count
        num_images = len(images)
        groups = num_images // group_size
        total += groups
    return total

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/root_dir")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a directory.")
        sys.exit(1)

    count = count_image_groups(root)
    print(f"Counted {count} images (in complete groups of 8) under '{root}'")
