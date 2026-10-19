"""Thermal Image Toolkit launcher and backwards-compatible imports.

Implementation lives in the thermal_toolkit package.
"""

from thermal_toolkit.api import *  # Preserve existing import entry points.
from thermal_toolkit.app import main


if __name__ == "__main__":
    main()
