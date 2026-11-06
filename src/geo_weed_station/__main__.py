"""Allow running package directly via `python -m geo_weed_station`."""

import sys
from geo_weed_station.main import main

if __name__ == "__main__":
    sys.exit(main())
