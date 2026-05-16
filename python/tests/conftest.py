"""Make sure pytest imports the installed `listofclusters` (with the compiled
_listofclusters.so) instead of the source `python/listofclusters/` directory,
which only has the Python-side __init__.py and would fail to find the
compiled module."""
import os
import sys

_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _PARENT]
