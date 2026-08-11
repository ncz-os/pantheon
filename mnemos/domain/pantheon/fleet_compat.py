"""Compatibility check for the supported core 6.0/GRAEAE 0.1 release lines.

The supported GRAEAE package owns its Redis clients and no longer imports
private lifecycle discovery helpers from core. Keep the import-order hook as a
no-op, but never synthesize private core APIs: doing so makes an incompatible
GRAEAE installation appear healthy.
"""

from __future__ import annotations

def install_core_graeae_compat() -> None:
    """Retained public hook; supported fleet versions need no patching."""


__all__ = ["install_core_graeae_compat"]
