"""Remote syslog logging for aicam.

rsyslogd-style UDP forwarding survives the egge-nano USB-SSD wedge (resident
process, network only, no disk, no fork), so shipping the important log lines
to the LibreNMS syslog sink keeps them readable while the machine itself is
unreachable over SSH. Python's SysLogHandler sends the datagrams directly, so
this works even though journald output is never forwarded.
"""
import logging
import logging.handlers

# Loggers whose INFO lines carry the diagnostic story (notifications,
# tracking arrivals/departures, camera capture problems). Everything else
# only forwards WARNING and above — the per-scan "completed in" spam from
# the "aicam" logger stays local.
INFO_LOGGERS = ("detect", "notify", "camera")


class SyslogFilter(logging.Filter):
    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        return record.levelno >= logging.INFO and record.name in INFO_LOGGERS


def setup_syslog(config):
    """Attach a UDP SysLogHandler from an optional [syslog] config section.

    Returns the handler, or None when the section is absent or setup fails —
    remote logging must never take down detection.
    """
    if not config.has_section("syslog"):
        return None
    try:
        host = config["syslog"]["host"]
        port = config["syslog"].getint("port", 514)
        handler = logging.handlers.SysLogHandler(address=(host, port))
        # syslog-ng derives the program name from the "tag:" prefix.
        handler.setFormatter(
            logging.Formatter("aicam: %(name)s %(levelname)s %(message)s")
        )
        handler.addFilter(SyslogFilter())
        logging.getLogger().addHandler(handler)
        return handler
    except Exception:
        logging.getLogger(__name__).exception("Failed to set up remote syslog")
        return None
