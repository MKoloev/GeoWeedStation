#!/usr/bin/env python3
"""Run full original GeosAero evaluation; see --help for the fixed protocol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.geosaero_evaluation import main

if __name__ == '__main__':
    main('c06_rgb_species')
