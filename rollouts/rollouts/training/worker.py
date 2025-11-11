"""Worker re-export from miniray.

MiniRay is our general multiprocessing primitive (Heinrich-inspired).
For backward compatibility, re-export Worker from miniray here.
"""

from miniray import Worker

__all__ = ["Worker"]
