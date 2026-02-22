from __future__ import annotations

import importlib.metadata

import tidaldisruptionlrd as m


def test_version():
    assert importlib.metadata.version("tidaldisruptionlrd") == m.__version__
