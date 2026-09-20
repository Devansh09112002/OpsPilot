import pathlib, httpx, json
env = dict(l.split("=",1) for l in pathlib.Path(".env").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#") and "=" in l)
tok = env["SUPABASE_ACCESS_TOKEN"].strip(); ref = env["SUPABASE_PROJECT_REF"].strip()
h = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
url = f"https://api.supabase.com/v1/projects/{ref}/database/query"
r = httpx.post(url, headers=h, json={"query": "select version(), current_database();"}, timeout=90)
print("POST database/query ->", r.status_code)
print(r.text[:400])
