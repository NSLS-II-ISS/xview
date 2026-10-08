===================
Supported versions
===================

XView supports Python 3.10 and newer. The minimum version is declared as
``requires-python`` in ``pyproject.toml``. CI tests Python 3.10, 3.11 and 3.12.

Runtime dependency ranges are maintained in ``requirements.txt``; optional
Dash and legacy Databroker dependencies are declared in ``pyproject.toml``.
The NumPy minimum is 1.26, and NumPy 2 is permitted. These ranges are not an
exact deployment lock. ISS-maintained dependencies must be validated in the
beamline environment before production upgrades.
