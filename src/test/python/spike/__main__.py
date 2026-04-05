"""Allow ``python -m spike`` as a shortcut to ``python -m spike.sweep_discover``."""
from spike.sweep_discover import main
import sys

sys.exit(main())
