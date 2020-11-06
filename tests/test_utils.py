# -*- coding: utf-8 -*-

import os

import pytest

from tango.utils import (
    InvalidTangoHostUriError,
    clear_test_context_tango_host_fdqn,
    get_device_uri_with_test_fdqn_if_necessary,
    set_test_context_tango_host_fdqn,
)


@pytest.fixture()
def restore_environ():
    yield
    clear_test_context_tango_host_fdqn()


@pytest.mark.parametrize(
    "environ_var, input_uri, expected_uri",
    [
        (None, "a/b/c", "a/b/c"),
        (None, "a/b/c/d", "a/b/c/d"),
        (None, "tango://host:12/a/b/c", "tango://host:12/a/b/c"),
        (None, "tango://host:12/a/b/c#dbase=no", "tango://host:12/a/b/c#dbase=no"),
        (None, "no://uri/validation", "no://uri/validation"),
        ("tango://host:12", "a/b/c", "tango://host:12/a/b/c"),
        ("tango://host:12", "a/b/c/d", "tango://host:12/a/b/c/d"),
        ("tango://host:12", "tango://host:12/a/b/c", "tango://host:12/a/b/c"),
        ("tango://host:12#dbase=no", "a/b/c", "tango://host:12/a/b/c#dbase=no"),
        ("tango://host:12#dbase=yes", "a/b/c", "tango://host:12/a/b/c#dbase=yes"),
        ("tango://127.0.0.1:12", "a/b/c", "tango://127.0.0.1:12/a/b/c"),
        (
            "ignore-environ-if-input-uri-already-resolved",
            "tango://host:12/a/b/c",
            "tango://host:12/a/b/c",
        ),
    ],
)
def test_get_uri_with_test_fdqn_if_necessary_success(
    environ_var, input_uri, expected_uri, restore_environ
):
    set_test_context_tango_host_fdqn(environ_var)
    actual_uri = get_device_uri_with_test_fdqn_if_necessary(input_uri)
    assert actual_uri == expected_uri


@pytest.mark.parametrize(
    "environ_var, input_uri",
    [
        ("host:123", "a/b/c"),  # missing scheme
        ("tango://", "a/b/c"),  # missing hostname and port
        ("tango://:123", "a/b/c"),  # missing hostname
        ("tango://host", "a/b/c"),  # missing port
        ("tango://host:0", "a/b/c"),  # zero-value port
        ("tango://host:12/path", "a/b/c"),  # non-empty path
        ("tango://host:123?query=1", "a/b/c"),  # non-empty query
        ("tango://host:123#dbase=invalid", "a/b/c"),  # invalid fragment
    ],
)
def test_get_uri_with_test_fdqn_if_necessary_failure(
    environ_var, input_uri, restore_environ
):
    set_test_context_tango_host_fdqn(environ_var)
    with pytest.raises(InvalidTangoHostUriError):
        get_device_uri_with_test_fdqn_if_necessary(input_uri)


def test_set_and_clear_environ_vars():
    environ_at_start = dict(os.environ)
    set_test_context_tango_host_fdqn("tango://localhost:1234")
    environ_after_set = dict(os.environ)
    clear_test_context_tango_host_fdqn()
    environ_after_clear = dict(os.environ)

    new_environ_vars = set(environ_after_set) - set(environ_at_start)
    assert len(new_environ_vars) == 1
    assert environ_at_start == environ_after_clear


def test_clear_environ_var_without_set_does_not_raise():
    clear_test_context_tango_host_fdqn()