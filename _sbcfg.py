import pathlib, httpx, json
env = dict(l.split("=",1) for l in pathlib.Path(".env").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#") and "=" in l)
tok = env["SUPABASE_ACCESS_TOKEN"].strip(); ref = env["SUPABASE_PROJECT_REF"].strip()
h = {"Authorization": f"Bearer {tok}"}
for path in [f"/v1/projects/{ref}", f"/v1/projects/{ref}/config/database/pooler"]:
    try:
        r = httpx.get("https://api.supabase.com"+path, headers=h, timeout=60)
        print(f"--- {path} -> {r.status_code}")
        print(json.dumps(r.json(), indent=2)[:900])
    except Exception as e:
        print(f"--- {path} FAILED {e}")
