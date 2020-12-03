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

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport():
    """ produces summary.json file for quick windows test summary """
    summary_path = "summary.json"
    
    outcome = yield  # Run all other pytest_runtest_makereport non wrapped hooks
    result = outcome.get_result()
    if result.when == "call" and 'nt' in os.name and os.path.isfile(summary_path):
        with open(summary_path, "r+") as f:
            summary = f.read()
            try:
                summary = json.loads(summary)
            except:
                summary = []
            finally:
                outcome = str(result.outcome).capitalize()
                test = {
                    "testName": result.nodeid,
                    "outcome": outcome,
                    "durationMilliseconds": result.duration,
                    "StdOut": result.capstdout,
                    "StdErr": result.capstderr,
                }
                summary.append(test)
                f.seek(0)
                f.write(json.dumps(summary))
                f.truncate()