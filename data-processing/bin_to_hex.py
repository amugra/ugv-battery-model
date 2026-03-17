#!/usr/bin/env python3
import argparse
import csv
import os
import struct

def main():
    parser = argparse.ArgumentParser(description="Convert raw binary to CSV of 16-bit hex values")
    parser.add_argument("infile", help="Input binary file")
    parser.add_argument("outfile", help="Output CSV file")
    args = parser.parse_args()

    filesize = os.path.getsize(args.infile)
    if filesize % 2 != 0:
        print(f"Warning: file size {filesize} is odd, last byte will be ignored")

    with open(args.infile, "rb") as f:
        raw = f.read()

    usable_len = len(raw) & ~1
    raw = raw[:usable_len]

    with open(args.outfile, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "hex16", "uint16"])

        for i in range(0, len(raw), 2):
            value = struct.unpack_from("<H", raw, i)[0]
            index = i // 2
            writer.writerow([index, f"0x{value:04X}", value])

    print(f"Wrote {usable_len // 2} samples to {args.outfile}")

if __name__ == "__main__":
    main()
