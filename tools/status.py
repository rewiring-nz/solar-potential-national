"""One line on the run: done / building / queued / failed, and who is on what.

    python tools/status.py

The same numbers are the object counts under gs://rewiring-solar-data/build/
{done,claims,queue,failed}/ -- open the bucket in the console for the same
picture without a terminal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gcs_queue import status

if __name__ == "__main__":
    status()
