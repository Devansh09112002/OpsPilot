"""Tests for the committed-credential scanner.

The scanner is only useful if it is trusted. One false positive on a
documentation placeholder and the next person disables the check, so the
placeholder rules are tested as carefully as the detection rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infra.scan_secrets import dsn_is_a_leak, is_local_host, is_placeholder_password, scan

# ---------------------------------------------------------------------------
# Must be caught
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    'DATABASE_URL=postgresql+psycopg://postgres.abcdef:Hunter2RealPass@aws-0-us-west-1.pooler.supabase.com:6543/postgres',
    'postgres://admin:s3cr3tvalue@db.production.example.com:5432/app',
    'postgresql://user:aVeryRealLookingPassword@10.2.3.4:5432/prod',
])
def test_remote_dsn_with_a_real_password_is_a_leak(line):
    assert dsn_is_a_leak(line)


@pytest.mark.parametrize("snippet,label", [
    ("AIzaSyA" + "b" * 32, "Google"),
    ("AQ.Ab8RN6" + "c" * 45, "Google new format"),
    ("sk-ant-api03-" + "d" * 40, "Anthropic"),
    ("ghp_" + "e" * 36, "GitHub"),
    ("sbp_" + "f" * 40, "Supabase"),
    ("rnd_" + "g" * 24, "Render"),
])
def test_api_key_shapes_are_detected(snippet, label):
    from infra.scan_secrets import API_KEY_PATTERNS

    assert any(p.search(snippet) for _, p in API_KEY_PATTERNS), f"{label} key not detected"


# ---------------------------------------------------------------------------
# Must NOT be caught - a noisy scanner gets switched off
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    # Documented local development setup.
    'postgresql+psycopg://opspilot:opspilot_dev_pw@127.0.0.1:5433/opspilot',
    'postgresql+psycopg://opspilot:opspilot_dev_pw@localhost:5432/opspilot_test',
    # Docker Compose service name with a variable.
    'postgresql+psycopg://opspilot:${POSTGRES_PASSWORD:-opspilot_dev_pw}@db:5432/opspilot',
    # Documentation examples.
    'postgresql://postgres.abcdefghijklm:[YOUR-PASSWORD]@aws-0-us-west-1.pooler.supabase.com:6543/postgres',
    'postgresql://user:<password>@host.example.com:5432/db',
    'postgresql://u:PASSWORD@remote.example.com:5432/db',
    # A test fixture.
    'pooled = "postgresql+psycopg://u:p@aws-0-us-west-1.pooler.supabase.com:6543/postgres"',
    # An f-string that builds a DSN at runtime - the secret is a variable.
    'dsn = f"postgresql+psycopg://postgres.{ref}:{password}@{host}:6543/postgres"',
    # Not a DSN at all.
    'see https://supabase.com/dashboard for the connection string',
])
def test_placeholders_and_local_setup_are_not_leaks(line):
    assert not dsn_is_a_leak(line)


def test_local_host_recognition():
    assert is_local_host("127.0.0.1")
    assert is_local_host("localhost")
    assert is_local_host("db")
    assert not is_local_host("aws-0-us-west-1.pooler.supabase.com")


def test_placeholder_password_recognition():
    assert is_placeholder_password("[YOUR-PASSWORD]")
    assert is_placeholder_password("${POSTGRES_PASSWORD}")
    assert is_placeholder_password("PASSWORD")
    assert is_placeholder_password("opspilot_dev_pw")
    assert is_placeholder_password("{password}")
    assert is_placeholder_password("<YOUR-PASSWORD>")
    assert not is_placeholder_password("Hunter2RealPassword")


# ---------------------------------------------------------------------------
# The real repository
# ---------------------------------------------------------------------------

def test_this_repository_is_clean():
    """The scan CI runs, run against the working tree."""
    findings = scan()
    assert findings == [], "\n".join(
        f"{f.path}:{f.line_no} {f.reason}" for f in findings
    )

def test_scan_is_independent_of_working_directory(tmp_path, monkeypatch):
    """The scanner must cover the whole repository wherever it is invoked from.

    `git ls-files` with no path argument lists only the tree below the current
    directory, so a scanner that trusted it would report a subdirectory clean
    and call the repository clean - a false negative in the one tool whose
    false negatives publish credentials.
    """
    from infra import scan_secrets

    root_files = set(scan_secrets._tracked_files())
    assert len(root_files) > 1

    for subdir in ("backend", "infra", "frontend"):
        monkeypatch.chdir(Path(scan_secrets._repo_root()) / subdir)
        assert set(scan_secrets._tracked_files()) == root_files
        assert scan_secrets.scan() == []
