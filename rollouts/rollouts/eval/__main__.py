"""Allow running as: python -m rollouts.eval"""

import sys

from .run import main

if __name__ == "__main__":
    sys.exit(main())
