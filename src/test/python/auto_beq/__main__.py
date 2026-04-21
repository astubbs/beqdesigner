"""Allow ``python -m auto_beq`` as a shortcut to ``python -m auto_beq.media_discover``."""
import sys

from auto_beq.media_discover import main

sys.exit(main())
