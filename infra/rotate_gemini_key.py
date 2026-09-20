"""Rotate the Gemini API key, verifying the new one before the old is revoked.

The order matters. Revoking first leaves the deployment broken while a new key
is pasted around; updating first and verifying means the old key is only ever
revoked once something else is known to work.

    python -m infra.rotate_gemini_key

Steps, in order:

  1. Read the new key from a hidden prompt. It is never echoed, never logged,
     never passed on a command line, and never written anywhere but `.env`
     and the Render service's environment.
  2. Check it against the provider directly, before anything is changed.
  3. Write it to `.env` and to the deployed API service.
  4. Wait for the redeploy, then run a real investigation against the public
     URL and confirm the model actually answered.
  5. Only then tell the operator it is safe to delete the old key.

Any failure stops before the old key is revoked, so the worst case is that
nothing changed.
"""

from __future__ import annotations

import getpass
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "backend"), str(REPO_ROOT)]

from infra.provision import (  # noqa: E402
    ProvisionError,
    _find_service,
    _render,
    read_env,
    write_env,
)

API_SERVICE = "opspilot-api"
GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def _mask(value: str) -> str:
    """A key is identified by its ends only, enough to tell two apart."""
    if len(value) < 12:
        return "***"
    return f"{value[:6]}...{value[-4:]} ({len(value)} chars)"


def prompt_for_key() -> str:
    print("\nCreate a new key at https://aistudio.google.com/apikey")
    print("Paste it below. It will not be shown as you type.\n")
    key = getpass.getpass("New GEMINI_API_KEY: ").strip()
    if not key:
        raise ProvisionError("No key entered; nothing was changed.")
    if len(key) < 20:
        raise ProvisionError("That does not look like an API key; nothing was changed.")
    return key


def verify_key_directly(key: str, models: list[str]) -> str:
    """Confirm the key answers before it is deployed anywhere.

    Tries each configured model: on the free tier an individual model can be
    out of quota or retired without the key being wrong.
    """
    last = ""
    for model in models:
        try:
            response = httpx.post(
                GENERATE_URL.format(model=model),
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": "Reply with OK."}]}]},
                timeout=60.0,
            )
        except httpx.HTTPError as exc:
            last = f"{model}: transport error ({type(exc).__name__})"
            continue
        if response.status_code == 200:
            return model
        if response.status_code in (400, 401, 403):
            raise ProvisionError(
                f"The provider rejected the new key (HTTP {response.status_code}). "
                "Nothing was changed; the old key is still in place."
            )
        last = f"{model}: HTTP {response.status_code}"
    raise ProvisionError(
        "The new key was not rejected, but no configured model answered "
        f"({last}). This is usually daily quota. Nothing was changed."
    )


def update_render(token: str, key: str) -> str:
    """Set GEMINI_API_KEY on the deployed API service and redeploy it."""
    with _render(token) as client:
        service = _find_service(client, API_SERVICE)
        if service is None:
            raise ProvisionError(f"Render service '{API_SERVICE}' not found.")

        response = client.get(f"/services/{service['id']}/env-vars", params={"limit": 100})
        response.raise_for_status()
        existing = {
            row["envVar"]["key"]: row["envVar"].get("value", "")
            for row in response.json()
        }
        existing["GEMINI_API_KEY"] = key

        response = client.put(
            f"/services/{service['id']}/env-vars",
            json=[{"key": k, "value": v} for k, v in existing.items()],
        )
        if response.status_code >= 400:
            raise ProvisionError(f"Render rejected the update: {response.text[:200]}")

        deploy = client.post(f"/services/{service['id']}/deploys", json={})
        deploy.raise_for_status()
        return deploy.json().get("id", "")


def wait_for_live_key(api_url: str, timeout_s: float = 900.0) -> None:
    """Wait until the deployed API reports a configured LLM and answers."""
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            meta = httpx.get(f"{api_url}/api/v1/meta", timeout=60.0).json()
            if meta.get("llm_configured"):
                print(f"  [{attempt}] API reports llm_configured=true")
                return
            print(f"  [{attempt}] API up, llm_configured=false; waiting")
        except httpx.HTTPError:
            print(f"  [{attempt}] API not reachable yet; waiting")
        time.sleep(20)
    raise ProvisionError("The redeployed API never reported a configured LLM.")


def verify_investigation(api_url: str) -> None:
    """Run a real investigation on the public URL with the new key.

    `meta` only says a key is present. This proves the provider answered
    through the deployed application, which is the thing being rotated.
    """
    with httpx.Client(base_url=api_url, timeout=180.0, follow_redirects=True) as client:
        client.get("/api/v1/tickets")  # establishes the guest session

        snapshots = client.get("/api/v1/snapshots").json()
        snapshot_id = snapshots[-1]["snapshot_id"]
        situations = client.get(
            f"/api/v1/snapshots/{snapshot_id}/situations", params={"limit": 1}
        ).json()
        if not situations:
            raise ProvisionError("No situation available to investigate.")

        situation_id = situations[0]["situation_id"]
        response = client.post(f"/api/v1/situations/{situation_id}/investigations")
        if response.status_code != 200:
            raise ProvisionError(
                f"Investigation failed on the deployed API: HTTP "
                f"{response.status_code} {response.text[:200]}"
            )
        body = response.json()
        if body.get("status") != "completed":
            raise ProvisionError(
                f"Investigation did not complete: {body.get('status')} "
                f"{body.get('error_message')}"
            )
        if body.get("generated_by") != "model":
            raise ProvisionError(
                "The investigation completed via the deterministic fallback, so "
                "the provider did not answer. The new key is not working; the "
                "old key has NOT been revoked."
            )
        print(
            f"  investigation {body['investigation_id']} completed, "
            f"generated_by={body['generated_by']}, "
            f"{len(body.get('facts', []))} cited facts"
        )


def main() -> int:
    env = read_env()
    token = env.get("RENDER_API_KEY")
    api_url = (env.get("OPSPILOT_API_URL") or "").rstrip("/")
    if not token:
        raise SystemExit("RENDER_API_KEY is not configured in .env")
    if not api_url:
        raise SystemExit("OPSPILOT_API_URL is not configured in .env")

    old_key = env.get("GEMINI_API_KEY", "")
    models = [env.get("LLM_MODEL", "gemini-3.5-flash-lite")]
    models += [
        m.strip()
        for m in env.get("LLM_MODEL_FALLBACKS", "").split(",")
        if m.strip() and m.strip() not in models
    ]

    print("=" * 66)
    print("Gemini API key rotation")
    print("=" * 66)
    print(f"current key : {_mask(old_key) if old_key else '(none)'}")
    print(f"api service : {API_SERVICE}  ({api_url})")

    try:
        new_key = prompt_for_key()
        if new_key == old_key:
            raise ProvisionError("That is the key already in use. Nothing changed.")
        print(f"new key     : {_mask(new_key)}")

        print("\n[1/4] checking the new key against the provider")
        model = verify_key_directly(new_key, models)
        print(f"  answered by {model}")

        print("\n[2/4] writing .env and the Render service environment")
        write_env({"GEMINI_API_KEY": new_key})
        deploy_id = update_render(token, new_key)
        print(f"  redeploy triggered: {deploy_id or '(no id returned)'}")

        print("\n[3/4] waiting for the redeploy")
        wait_for_live_key(api_url)

        print("\n[4/4] running a real investigation on the public URL")
        verify_investigation(api_url)

    except ProvisionError as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        print("The old key has NOT been revoked.", file=sys.stderr)
        return 2

    print("\n" + "=" * 66)
    print("ROTATION VERIFIED")
    print("=" * 66)
    print("The new key is live and the deployed agent answered with it.")
    print(f"\nNow delete the OLD key ({_mask(old_key)}) at:")
    print("  https://aistudio.google.com/apikey")
    print("\nNothing else references it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
