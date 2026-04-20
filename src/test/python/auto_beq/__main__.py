"""Allow ``python -m spike`` as a shortcut to ``python -m spike.sweep_discover``."""
import sys

from auto_beq.sweep_discover import main

sys.exit(main())
