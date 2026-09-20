"""Automated provisioning of the free-tier deployment.

Creates and configures everything that can be created over an API:

    Supabase  -> free project, wait for it, derive the pooler DSN
    local     -> alembic migrate, ingest the validated data
    Render    -> Docker web service + static site, env vars, deploys
    verify    -> poll until both are live and the journey works

Human steps are limited to what genuinely cannot be automated: creating the
two accounts and issuing an API token for each, plus connecting Render to
GitHub if the repository is private.

Secrets are read from the environment or the gitignored `.env`. Nothing in
this module prints a token, and nothing writes one to a tracked file.

    python -m infra.provision check          # what is configured so far
    python -m infra.provision supabase       # create the database project
    python -m infra.provision seed           # migrate + ingest
    python -m infra.provision render         # create/update the services
    python -m infra.provision verify         # end-to-end checks on the live URLs
    python -m infra.provision all            # the whole sequence
"""

from __future__ import annotations

import argparse
import os
import secrets
import string
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

SUPABASE_API = "https://api.supabase.com/v1"
RENDER_API = "https://api.render.com/v1"

GITHUB_REPO = "https://github.com/Devansh09112002/OpsPilot"
API_SERVICE = "opspilot-api"
WEB_SERVICE = "opspilot-web"
REGION_RENDER = "oregon"
REGION_SUPABASE = "us-west-1"


class ProvisionError(RuntimeError):
    """Something a human needs to know about, phrased without secrets."""


# ---------------------------------------------------------------------------
# .env handling (the only place secrets live locally)
# ---------------------------------------------------------------------------

def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    # Real environment wins, so CI and one-off overrides work.
    for key in [
        *values,
        "SUPABASE_ACCESS_TOKEN", "RENDER_API_KEY", "GEMINI_API_KEY",
        "DATABASE_URL", "SUPABASE_DB_PASSWORD",
    ]:
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def write_env(updates: dict[str, str]) -> None:
    """Update .env in place without reordering or dropping comments."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    remaining = dict(updates)

    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                out.append(f"{key}={remaining.pop(key)}")
                continue
        out.append(line)

    if remaining:
        out.append("")
        out.append("# --- added by infra.provision ---")
        out += [f"{k}={v}" for k, v in remaining.items()]

    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  updated {ENV_PATH.name}: {', '.join(updates)} "
          "(values not shown; file is gitignored)")


def _mask(value: str | None) -> str:
    if not value:
        return "not set"
    return f"set ({len(value)} chars, ends …{value[-4:]})"


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------

def _supabase(token: str) -> httpx.Client:
    return httpx.Client(
        base_url=SUPABASE_API,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60.0,
    )


def _random_password(n: int = 28) -> str:
    # Avoid characters that need percent-encoding inside a DSN.
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


def provision_supabase(env: dict[str, str]) -> dict[str, str]:
    token = env.get("SUPABASE_ACCESS_TOKEN")
    if not token:
        raise ProvisionError(
            "SUPABASE_ACCESS_TOKEN is not set. Create one at "
            "https://supabase.com/dashboard/account/tokens"
        )

    with _supabase(token) as client:
        orgs = client.get("/organizations")
        if orgs.status_code == 401:
            raise ProvisionError("The Supabase access token was rejected.")
        orgs.raise_for_status()
        org_list = orgs.json()
        if not org_list:
            raise ProvisionError(
                "The Supabase account has no organization. Create one in the "
                "dashboard, then re-run."
            )
        org_id = org_list[0]["id"]
        print(f"  organization: {org_list[0].get('name', org_id)}")

        projects = client.get("/projects")
        projects.raise_for_status()
        existing = next(
            (p for p in projects.json() if p.get("name") == "opspilot"), None
        )

        if existing:
            ref = existing["id"]
            region = existing.get("region", REGION_SUPABASE)
            print(f"  reusing existing project '{ref}' ({existing.get('status')})")
            password = env.get("SUPABASE_DB_PASSWORD")
            if not password:
                raise ProvisionError(
                    "A Supabase project named 'opspilot' already exists but "
                    "SUPABASE_DB_PASSWORD is not in .env, so its connection "
                    "string cannot be rebuilt. Either delete the project in the "
                    "dashboard and re-run, or reset the database password and "
                    "put it in .env as SUPABASE_DB_PASSWORD."
                )
        else:
            password = env.get("SUPABASE_DB_PASSWORD") or _random_password()
            print("  creating a free project (this takes a couple of minutes)…")
            created = client.post("/projects", json={
                "name": "opspilot",
                "organization_id": org_id,
                "plan": "free",
                "region": REGION_SUPABASE,
                "db_pass": password,
            })
            if created.status_code >= 400:
                raise ProvisionError(
                    f"Supabase refused to create the project "
                    f"(HTTP {created.status_code}): {created.text[:300]}"
                )
            body = created.json()
            ref = body["id"]
            region = body.get("region", REGION_SUPABASE)
            print(f"  project ref: {ref}")

        # Wait for it to become usable.
        deadline = time.time() + 600
        status = ""
        while time.time() < deadline:
            resp = client.get(f"/projects/{ref}")
            resp.raise_for_status()
            status = resp.json().get("status", "")
            if status in {"ACTIVE_HEALTHY", "ACTIVE"}:
                break
            print(f"    status={status}; waiting…")
            time.sleep(15)
        else:
            raise ProvisionError(
                f"Supabase project {ref} did not become healthy (last status "
                f"{status}). Check the dashboard."
            )
        print(f"  project is {status}")

    # Session pooler on 6543: a free project allows few direct connections and
    # a Render restart would exhaust them.
    host = f"aws-0-{region}.pooler.supabase.com"
    dsn = (
        f"postgresql+psycopg://postgres.{ref}:{password}@{host}:6543/postgres"
    )
    write_env({
        "DATABASE_URL": dsn,
        "SUPABASE_PROJECT_REF": ref,
        "SUPABASE_DB_PASSWORD": password,
    })
    return {"ref": ref, "dsn": dsn, "region": region}


# ---------------------------------------------------------------------------
# Local seeding against whatever DATABASE_URL points at
# ---------------------------------------------------------------------------

def seed_database(env: dict[str, str]) -> None:
    import subprocess

    dsn = env.get("DATABASE_URL")
    if not dsn:
        raise ProvisionError("DATABASE_URL is not set; run the supabase step first.")
    if "127.0.0.1" in dsn or "localhost" in dsn:
        print("  note: DATABASE_URL points at a local database")

    child_env = {**os.environ, "DATABASE_URL": dsn,
                 "PYTHONPATH": f"{REPO_ROOT / 'backend'}{os.pathsep}{REPO_ROOT}"}

    print("  applying migrations…")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT / "backend", env=child_env,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ProvisionError(f"alembic failed:\n{result.stdout}\n{result.stderr}")
    print("  migrations applied")

    print("  ingesting validated data…")
    result = subprocess.run(
        [sys.executable, "-m", "data_pipeline.ingest"],
        cwd=REPO_ROOT, env=child_env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ProvisionError(f"ingest failed:\n{result.stdout}\n{result.stderr}")
    print("  " + "\n  ".join(result.stdout.strip().splitlines()[-6:]))


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _render(token: str) -> httpx.Client:
    return httpx.Client(
        base_url=RENDER_API,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/json"},
        timeout=90.0,
    )


def _owner_id(client: httpx.Client) -> str:
    resp = client.get("/owners", params={"limit": 20})
    if resp.status_code == 401:
        raise ProvisionError("The Render API key was rejected.")
    resp.raise_for_status()
    owners = resp.json()
    if not owners:
        raise ProvisionError("The Render account has no owner record.")
    # Render wraps list items as {"owner": {...}, "cursor": "..."}; tolerate a
    # bare object too rather than KeyError on a shape change.
    first = owners[0]
    owner = first.get("owner", first) if isinstance(first, dict) else first
    if not isinstance(owner, dict) or "id" not in owner:
        raise ProvisionError(f"Unexpected /owners response shape: {str(first)[:200]}")
    print(f"  owner: {owner.get('name') or owner['id']}")
    return owner["id"]


def _find_service(client: httpx.Client, name: str) -> dict[str, Any] | None:
    resp = client.get("/services", params={"name": name, "limit": 20})
    resp.raise_for_status()
    for row in resp.json():
        service = row.get("service", row)
        if service.get("name") == name:
            return service
    return None


def _env_var_list(pairs: dict[str, str]) -> list[dict[str, str]]:
    return [{"key": k, "value": v} for k, v in pairs.items()]


def provision_render(env: dict[str, str]) -> dict[str, str]:
    token = env.get("RENDER_API_KEY")
    if not token:
        raise ProvisionError(
            "RENDER_API_KEY is not set. Create one at "
            "https://dashboard.render.com/u/settings#api-keys"
        )
    dsn = env.get("DATABASE_URL")
    if not dsn or "127.0.0.1" in dsn:
        raise ProvisionError(
            "DATABASE_URL must point at Supabase before deploying; run the "
            "supabase step first."
        )
    gemini = env.get("GEMINI_API_KEY", "")
    if not gemini:
        print("  warning: GEMINI_API_KEY is not set; the deploy will run with "
              "AI investigation disabled until it is added.")

    api_env = {
        "ENVIRONMENT": "production",
        "LOG_LEVEL": "INFO",
        "DATABASE_URL": dsn,
        "GEMINI_API_KEY": gemini,
        "LLM_MODEL": env.get("LLM_MODEL", "gemini-3.5-flash-lite"),
        "LLM_MODEL_FALLBACKS": env.get(
            "LLM_MODEL_FALLBACKS",
            "gemini-3.5-flash,gemini-3.6-flash,gemini-3-flash-preview,"
            "gemini-3.1-flash-lite"),
        "LLM_THINKING_BUDGET": "0",
        "LLM_TIMEOUT_SECONDS": "45",
        "COOKIE_SECURE": "true",
        "COOKIE_SAMESITE": "none",
        "SESSION_TTL_HOURS": "72",
        "INVESTIGATIONS_PER_SESSION_PER_DAY": "5",
        "INVESTIGATIONS_GLOBAL_PER_HOUR": "25",
        "INVESTIGATIONS_GLOBAL_PER_DAY": "80",
        "API_REQUESTS_PER_MINUTE": "120",
        # Provisional; corrected once the static site URL is known.
        "CORS_ORIGINS": "https://opspilot-web.onrender.com",
    }

    with _render(token) as client:
        owner_id = _owner_id(client)

        api = _find_service(client, API_SERVICE)
        if api:
            print(f"  {API_SERVICE}: exists ({api['id']}), updating environment")
            resp = client.put(f"/services/{api['id']}/env-vars",
                              json=_env_var_list(api_env))
            resp.raise_for_status()
        else:
            print(f"  {API_SERVICE}: creating Docker web service (free plan)")
            resp = client.post("/services", json={
                "type": "web_service",
                "name": API_SERVICE,
                "ownerId": owner_id,
                "repo": GITHUB_REPO,
                "branch": "main",
                "autoDeploy": "yes",
                "serviceDetails": {
                    # Render's API field is "runtime", not "env".
                    "runtime": "docker",
                    "region": REGION_RENDER,
                    "plan": "free",
                    "numInstances": 1,
                    "healthCheckPath": "/api/v1/health",
                    "envSpecificDetails": {
                        "dockerfilePath": "./backend/Dockerfile",
                        "dockerContext": ".",
                    },
                },
                "envVars": _env_var_list(api_env),
            })
            if resp.status_code >= 400:
                raise ProvisionError(
                    f"Render refused to create {API_SERVICE} "
                    f"(HTTP {resp.status_code}): {resp.text[:400]}"
                )
            api = resp.json().get("service", resp.json())
            print(f"  created {api['id']}")

        api_url = api.get("serviceDetails", {}).get("url") or (
            f"https://{API_SERVICE}.onrender.com"
        )

        web = _find_service(client, WEB_SERVICE)
        web_env = {"VITE_API_BASE_URL": api_url}
        if web:
            print(f"  {WEB_SERVICE}: exists ({web['id']}), updating environment")
            resp = client.put(f"/services/{web['id']}/env-vars",
                              json=_env_var_list(web_env))
            resp.raise_for_status()
        else:
            print(f"  {WEB_SERVICE}: creating static site")
            resp = client.post("/services", json={
                "type": "static_site",
                "name": WEB_SERVICE,
                "ownerId": owner_id,
                "repo": GITHUB_REPO,
                "branch": "main",
                "autoDeploy": "yes",
                "serviceDetails": {
                    "buildCommand": "cd frontend && npm ci && npm run build",
                    "publishPath": "./frontend/dist",
                    "routes": [
                        {"type": "rewrite", "source": "/*", "destination": "/index.html"}
                    ],
                },
                "envVars": _env_var_list(web_env),
            })
            if resp.status_code >= 400:
                raise ProvisionError(
                    f"Render refused to create {WEB_SERVICE} "
                    f"(HTTP {resp.status_code}): {resp.text[:400]}"
                )
            web = resp.json().get("service", resp.json())
            print(f"  created {web['id']}")

        web_url = web.get("serviceDetails", {}).get("url") or (
            f"https://{WEB_SERVICE}.onrender.com"
        )

        # Now that the real static-site origin is known, fix CORS. A wildcard
        # would break the credentialed session cookie.
        if api_env["CORS_ORIGINS"] != web_url:
            api_env["CORS_ORIGINS"] = web_url
            client.put(f"/services/{api['id']}/env-vars",
                       json=_env_var_list(api_env)).raise_for_status()
            print(f"  CORS_ORIGINS set to {web_url}")

        # Env-var changes are not deployed automatically, so a deploy is
        # triggered explicitly. 201 (created) and 202 (queued) are both success.
        for service in (api, web):
            resp = client.post(
                f"/services/{service['id']}/deploys",
                json={"clearCache": "do_not_clear"},
            )
            if resp.status_code >= 400:
                raise ProvisionError(
                    f"Render refused to deploy {service.get('name', service['id'])} "
                    f"(HTTP {resp.status_code}): {resp.text[:300]}"
                )
        print("  deploys triggered for both services")

    write_env({"OPSPILOT_API_URL": api_url, "OPSPILOT_WEB_URL": web_url})
    return {"api_url": api_url, "web_url": web_url,
            "api_id": api["id"], "web_id": web["id"]}


def wait_for_live(url: str, *, label: str, timeout: int = 1500) -> bool:
    """Poll a URL until it answers. Render's first Docker build is slow."""
    print(f"  waiting for {label} at {url} (first build can take 10+ minutes)…")
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            if resp.status_code < 400:
                print(f"  {label} is live (HTTP {resp.status_code})")
                return True
            last = f"HTTP {resp.status_code}"
        except Exception as exc:
            last = type(exc).__name__
        print(f"    not ready ({last}); retrying…")
        time.sleep(20)
    print(f"  {label} did not come up within {timeout}s (last: {last})")
    return False


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_check(env: dict[str, str]) -> int:
    print("Credentials and targets")
    for key in ("SUPABASE_ACCESS_TOKEN", "RENDER_API_KEY", "GEMINI_API_KEY"):
        print(f"  {key:<24s} {_mask(env.get(key))}")
    dsn = env.get("DATABASE_URL", "")
    where = (
        "local" if ("127.0.0.1" in dsn or "localhost" in dsn)
        else ("supabase" if "supabase" in dsn else ("set" if dsn else "not set"))
    )
    print(f"  {'DATABASE_URL':<24s} {where}")
    for key in ("OPSPILOT_API_URL", "OPSPILOT_WEB_URL"):
        print(f"  {key:<24s} {env.get(key, 'not set')}")
    return 0


def cmd_verify(env: dict[str, str]) -> int:
    api_url = env.get("OPSPILOT_API_URL")
    web_url = env.get("OPSPILOT_WEB_URL")
    if not api_url:
        raise ProvisionError("OPSPILOT_API_URL is not set; run the render step.")

    ok = True
    if not wait_for_live(f"{api_url}/api/v1/health", label="API"):
        return 1

    checks: list[tuple[str, str]] = [
        ("health", f"{api_url}/api/v1/health"),
        ("readiness", f"{api_url}/api/v1/health/ready"),
        ("meta", f"{api_url}/api/v1/meta"),
        ("snapshots", f"{api_url}/api/v1/snapshots"),
        ("orders", f"{api_url}/api/v1/orders?snapshot_id=2018-08-15&limit=3"),
    ]
    for label, url in checks:
        try:
            resp = httpx.get(url, timeout=60.0)
            body = resp.text[:200]
            status = "ok" if resp.status_code == 200 else f"HTTP {resp.status_code}"
            print(f"  {label:<12s} {status}  {body[:120]}")
            ok &= resp.status_code == 200
        except Exception as exc:
            print(f"  {label:<12s} FAILED {type(exc).__name__}")
            ok = False

    if web_url:
        ok &= wait_for_live(web_url, label="frontend", timeout=900)

    print("\nVERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["check", "supabase", "seed", "render", "verify", "all"],
    )
    args = parser.parse_args(argv)
    env = read_env()

    try:
        if args.command == "check":
            return cmd_check(env)
        if args.command == "supabase":
            provision_supabase(env)
            return 0
        if args.command == "seed":
            seed_database(env)
            return 0
        if args.command == "render":
            provision_render(env)
            return 0
        if args.command == "verify":
            return cmd_verify(env)

        # all
        print("[1/4] Supabase")
        provision_supabase(env)
        env = read_env()
        print("[2/4] Seeding the database")
        seed_database(env)
        print("[3/4] Render")
        provision_render(env)
        env = read_env()
        print("[4/4] Verifying")
        return cmd_verify(env)
    except ProvisionError as exc:
        print(f"\nACTION NEEDED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
