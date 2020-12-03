"""Load tango-specific pytest fixtures."""

from tango.test_utils import state, typed_values, server_green_mode

import pytest

import sys
import os
import json

__all__ = ('state', 'typed_values', 'server_green_mode')

@pytest.hookimpl()
def pytest_sessionfinish(session):
    """ collets all tests to be run and outputs to bat script """
    if '--collect-only' in sys.argv and '-q' in sys.argv and 'nt' in os.name:
        print("Generating windows test script...")
        script_path = os.path.join(os.path.dirname(__file__),'run_tests_win.bat')
        with open(script_path,"w") as f:
            f.write("REM this script will run all tests separately.")
            for item in session.items:
                f.write("\n")
                f.write("pytest -c ../pytest_empty_config.txt ")#this empty file is created by appveyor
                f.write(item.nodeid) 
