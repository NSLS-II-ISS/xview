=====
XView
=====

ISS data visualisation and X-ray absorption spectroscopy analysis, with a
PyQt5 desktop application and an experimental Dash interface.

Installation
------------

On Linux x86-64, the recommended setup uses `Pixi <https://pixi.sh>`_::

    pixi install --locked
    pixi run start

``pixi.toml`` installs this checkout in editable mode, Qt, and pinned Git
revisions of ``xas``, ``isstools`` and ``issfactortools``. ``pixi.lock`` records
the resolved Conda and PyPI packages. The Pixi environment uses Python 3.10
and ``setuptools<81`` for compatibility with those ISS packages; XView itself
supports Python 3.10 and newer. No separate beamline-package installation is
needed for this environment. Cloud uploads still require site configuration.

Useful tasks::

    pixi run test       # regression tests, without a display
    pixi run smoke      # briefly open the complete desktop GUI offscreen
    pixi run check      # check installed dependency consistency
    pixi run build      # build wheel and source distribution
    pixi run dash       # experimental web UI; requires ISS Tiled access

``pixi run start`` needs a graphical desktop or X forwarding. Set
``MP_API_KEY`` in your shell to enable Materials Project searches. To update
resolved packages deliberately, run ``pixi update`` and review the lockfile
changes. The Git revisions in the manifest must be changed explicitly to
update the ISS packages. ``.pixi/`` contains the local environment and is
ignored by Git.

This configuration is validated with Pixi 0.81.0 on Linux x86-64.

For a manually managed environment instead of Pixi:

Python 3.10 or newer is required. Install into a dedicated environment::

    python3.10 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e .

The desktop application also requires the ISS-maintained ``xas``, ``isstools``
and ``issfactortools`` packages from your beamline environment. These are not
substituted with similarly named PyPI projects or bundled private copies.
Install compatible revisions of those projects in the same environment before
launching XView. Optional cloud uploads require ``isscloudtools`` and its site
configuration; local use does not require cloud credentials.

Runtime dependencies and supported version ranges are declared in
``requirements.txt``. Build metadata, entry points and optional dependencies are
in ``pyproject.toml``. The ranges permit dependency resolution on Python 3.10;
they are not an exact deployment lock. Validate with representative beamline
data before updating a production environment.

Start the desktop application with ``xview`` or ``python -m xview.xview``.
For Materials Project searches, set ``MP_API_KEY`` in your environment to your
own API key (``PMG_MAPI_KEY`` is also accepted). Search failures appear in the
GUI and release the worker thread.

To install and start the experimental web interface::

    python -m pip install -e '.[dash]'
    xview-dash

The web interface additionally needs an ``xas`` revision providing
``xas.tiled_io`` and access to the configured ISS Tiled service. It runs the
Dash development server. Legacy Databroker widgets and Mongo spectrum helpers
have separate dependencies, available through ``pip install -e '.[legacy]'``.
They retain their existing APIs and have not been migrated to Tiled.

Performance and compatibility changes
-------------------------------------

* File previews reuse parsed scans across selection, metadata display and
  plotting. The cache retains at most 32 files and approximately 64 MiB of
  DataFrame/header data. File identity, size and nanosecond timestamps
  invalidate changed scans; refreshing the folder clears the cache. Project
  imports read fresh data so processing cannot mutate cached previews.
* Plotting avoids unnecessary column copies and formats axes after all scans
  have been added. Each selected channel receives its own correct legend label.
* Materials Project searches use the public ``mp_api`` client, deduplicate
  formulas and reuse one connection per search.
* Bundled Qt forms, JSON schemas and CSS use ``importlib.resources`` instead of
  ``pkg_resources``. Package versions use ``setuptools-scm`` at build time and
  installed metadata at runtime, avoiding a Git subprocess during import.

Development
-----------

Install development tools and run the tests::

    python -m pip install -r requirements-dev.txt
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest
    python -m build
    python -m pip check

Tests cover cache invalidation and bounds, numerical plotting transformations,
Qt form loading, and worker completion on network or calculation failures.
They do not require beamline services or API credentials. CI resolves runtime
and Dash dependencies and tests Python 3.10, 3.11 and 3.12. Live catalog access,
cloud uploads and end-to-end beamline processing require site integration tests.

Free software: 3-clause BSD license.
