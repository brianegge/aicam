"""Tests for the remote syslog handler setup."""
import configparser
import logging

from logsetup import SyslogFilter, setup_syslog


def _record(name, level):
    return logging.LogRecord(name, level, "f.py", 1, "msg", None, None)


def test_filter_passes_warnings_from_anywhere():
    assert SyslogFilter().filter(_record("aicam", logging.WARNING))
    assert SyslogFilter().filter(_record("urllib3", logging.ERROR))


def test_filter_passes_diagnostic_info_loggers():
    for name in ("detect", "notify", "camera"):
        assert SyslogFilter().filter(_record(name, logging.INFO))


def test_filter_drops_scan_spam():
    assert not SyslogFilter().filter(_record("aicam", logging.INFO))
    assert not SyslogFilter().filter(_record("detect", logging.DEBUG))


def test_setup_noop_without_section():
    config = configparser.ConfigParser()
    assert setup_syslog(config) is None


def test_setup_attaches_handler(tmp_path):
    config = configparser.ConfigParser()
    config["syslog"] = {"host": "127.0.0.1", "port": "5140"}
    handler = setup_syslog(config)
    try:
        assert handler is not None
        assert handler in logging.getLogger().handlers
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)


def test_setup_survives_bad_config():
    config = configparser.ConfigParser()
    config["syslog"] = {"host": "127.0.0.1", "port": "not-a-number"}
    assert setup_syslog(config) is None
