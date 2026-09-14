"""Run-mode flags, per the team convention (KB infrastructure/email-testing.md).

TEST_EMAIL and DRY_RUN default to True so a bare run can never email a real
list or write production state; the workflow sets them explicitly.
"""

import os


def env_flag(name, default):
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    v = val.strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ValueError(f"{name}={val!r} is not a boolean")


def mode():
    m = {
        "TEST_EMAIL": env_flag("TEST_EMAIL", True),
        "SIMULATE_TRIGGER": env_flag("SIMULATE_TRIGGER", False),
        "DRY_RUN": env_flag("DRY_RUN", True),
    }
    print("mode: " + " ".join(f"{k}={v}" for k, v in m.items()))
    return m
