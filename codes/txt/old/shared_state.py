from __future__ import annotations

import threading

ALNS_DONE = threading.Event()


class ALNSCompletion(Exception):
    """Signals ALNS has completed normally and should unwind."""
