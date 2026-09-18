"""Puts the app modules on sys.path so tests can `import policy` etc.

pytest's default import mode would only add the tests/ directory itself, so
without this the app modules (which live one level up, not in a package) would
not be importable no matter which directory pytest is invoked from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
