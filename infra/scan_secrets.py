"""Fail the build if a real credential is committed.

Run by CI. Deliberately not a chain of greps: a scanner that reports
placeholders as leaks gets switched off, which is worse than having none, so
the placeholder rules need to be readable and unit-tested. They are, in
`backend/tests/test_secret_scan.py`.

    python -m infra.scan_secrets

Exit 0 when clean, 1 with a report when something real is found.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass

# Provider key shapes. Placeholders in .env.example are empty, so any hit is a
# real key.
API_KEY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("Google API key (new format)", re.compile(r"\bAQ\.[0-9A-Za-z_\-]{40,}")),
    ("Anthropic API key", re.compile(r"sk-ant-[0-9A-Za-z_\-]{20,}")),
    ("OpenAI-style API key", re.compile(r"\bsk-[0-9A-Za-z]{40,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[0-9A-Za-z]{30,}")),
    ("Supabase access token", re.compile(r"\bsbp_[0-9a-f]{40,}")),
    ("Render API key", re.compile(r"\brnd_[0-9A-Za-z]{20,}")),
]

DSN = re.compile(
    r"postgres(?:ql)?(?:\+\w+)?://"
    r"(?P<user>[^:@\s/]+):(?P<password>[^@\s/]+)@"
    r"(?P<host>[^:/\s]+)"
)

# Hosts that are a developer's own machine or a compose service name. The
# documented local password is setup instructions, not a secret.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "db", "postgres", "host.docker.internal"}

# A password that is obviously a stand-in. Matching is case-insensitive.
PLACEHOLDER_PASSWORDS = {
    "password", "your-password", "your_password", "yourpassword",
    "changeme", "secret", "pass", "pwd", "xxx", "xxxx", "p",
    "opspilot_dev_pw", "example", "placeholder", "redacted",
}
PLACEHOLDER_SHAPES = [
    re.compile(r"^\[.*\]$"),          # [YOUR-PASSWORD]
    re.compile(r"^<.*>$"),            # <password>
    re.compile(r"^\$\{.*\}$"),        # ${POSTGRES_PASSWORD}
    re.compile(r"^\{.*\}$"),          # {password} - an f-string or template slot
    re.compile(r"^\$[A-Z_]+$"),       # $PGPASSWORD
    re.compile(r"^\*+$"),             # ****
    re.compile(r"^x+$", re.I),        # xxxx
]

SKIP_PATHS = (".lock", "package-lock.json", "uv.lock", "poetry.lock")


def is_placeholder_password(password: str) -> bool:
    if password.lower() in PLACEHOLDER_PASSWORDS:
        return True
    return any(shape.match(password) for shape in PLACEHOLDER_SHAPES)


def is_local_host(host: str) -> bool:
    return host.lower() in LOCAL_HOSTS


def dsn_is_a_leak(line: str) -> bool:
    """True when a line contains a DSN carrying a real credential.

    A DSN is a leak when it names a remote host *and* its password is not an
    obvious placeholder. A localhost DSN is documented setup; a remote DSN with
    `[YOUR-PASSWORD]` is documentation.
    """
    for match in DSN.finditer(line):
        if is_local_host(match.group("host")):
            continue
        if is_placeholder_password(match.group("password")):
            continue
        return True
    return False


@dataclass
class Finding:
    path: str
    line_no: int
    reason: str
    excerpt: str


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [
        f for f in out.splitlines()
        if f and not any(f.endswith(s) or s in f for s in SKIP_PATHS)
    ]


def scan() -> list[Finding]:
    findings: list[Finding] = []
    for path in _tracked_files():
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                lines = fh.readlines()
        except (OSError, IsADirectoryError):
            continue

        for i, line in enumerate(lines, 1):
            if len(line) > 4000:  # minified or generated
                continue
            for label, pattern in API_KEY_PATTERNS:
                if pattern.search(line):
                    findings.append(Finding(path, i, label, line.strip()[:120]))
            if dsn_is_a_leak(line):
                findings.append(
                    Finding(path, i, "database URL with a real credential",
                            line.strip()[:120])
                )
    return findings


def main() -> int:
    print("Scanning tracked files for committed credentials...")
    findings = scan()
    if not findings:
        print("Clean: no API keys and no remote database credentials.")
        return 0

    print(f"\n{len(findings)} finding(s):\n", file=sys.stderr)
    for f in findings:
        print(f"::error file={f.path},line={f.line_no}::{f.reason}", file=sys.stderr)
        print(f"  {f.path}:{f.line_no}  {f.reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
