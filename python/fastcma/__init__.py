"""Fast CMA-ES with a Rust core.

The optimizer lives in the native extension ``fastcma.fastcma``; its public
names are re-exported here, so ``import fastcma`` works exactly as it does for a
plain extension module. A name added to the extension is added here too.
"""

from .fastcma import CMAES, ff, fmin, fmin_constrained, fmin_restart, fmin_vec

__all__ = ["CMAES", "ff", "fmin", "fmin_constrained", "fmin_restart", "fmin_vec"]
