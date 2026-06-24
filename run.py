#!/usr/bin/env python
"""
Perfect-Frame Auto-Snapper — one-command launcher.

Usage
-----
Activate the virtual environment once, then every session is just:

    python run.py

Activation:
    Windows  :  venv\\Scripts\\activate
    macOS/Linux:  source venv/bin/activate

The GUI window will open automatically.  Press Q or the Quit button to exit.
For the headless CLI instead:  python -m src.main
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.gui import main

if __name__ == "__main__":
    main()
