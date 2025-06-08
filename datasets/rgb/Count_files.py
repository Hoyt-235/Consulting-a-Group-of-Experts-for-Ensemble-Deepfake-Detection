#!/usr/bin/env python3
import os
import sys

def count_jpg_files(root_dir):
    total = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                total += 1
    return total

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/root_dir")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a directory.")
        sys.exit(1)

    count = count_jpg_files(root)
    print(f"Found {count} .jpg/.jpeg files under '{root}'")
