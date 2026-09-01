"""Enumerate RealSense devices without starting streams."""

from ur5e_real.hardware.realsense import list_serials


if __name__ == "__main__":
    for serial in list_serials():
        print(serial)
