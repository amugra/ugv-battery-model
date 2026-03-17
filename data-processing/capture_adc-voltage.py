#!/usr/bin/env python3
import argparse
import serial
import time
import sys

def main():
    parser = argparse.ArgumentParser(description="Capture RP2040 ADC binary stream")
    parser.add_argument("port", help="Serial port, e.g. /dev/cu.usbmodemPICO2ADC01")
    parser.add_argument("outfile", help="Output binary file")
    parser.add_argument("--chunk", type=int, default=16384, help="Read chunk size")
    parser.add_argument("--startup-delay", type=float, default=0.5,
                        help="Delay after opening serial before sending start command")
    args = parser.parse_args()

    ser = serial.Serial(args.port, baudrate=115200, timeout=1)
    time.sleep(args.startup_delay)

    # Drain any idle/status text already sitting in the buffer
    ser.reset_input_buffer()

    # Start the run
    ser.write(b"s")
    ser.flush()

    print(f"Capturing from {args.port} to {args.outfile}")
    print("Press Ctrl+C to stop.")

    total = 0
    start = time.time()

    try:
        with open(args.outfile, "wb") as f:
            while True:
                data = ser.read(args.chunk)
                if data:
                    f.write(data)
                    total += len(data)

                    now = time.time()
                    elapsed = now - start
                    if elapsed > 0:
                        rate = total / elapsed / (1024 * 1024)
                        sys.stdout.write(
                            f"\rCaptured {total} bytes ({rate:.2f} MiB/s)"
                        )
                        sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nStopping capture...")
        try:
            ser.write(b"x")
            ser.flush()
            time.sleep(0.1)
        except Exception:
            pass
    finally:
        ser.close()
        print("Done.")

if __name__ == "__main__":
    main()
