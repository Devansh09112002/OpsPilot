import pathlib, httpx
env = dict(l.split("=",1) for l in pathlib.Path(".env").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#") and "=" in l)
tok=env["SUPABASE_ACCESS_TOKEN"].strip(); ref=env["SUPABASE_PROJECT_REF"].strip()
h={"Authorization":f"Bearer {tok}","Content-Type":"application/json"}
q="select (select count(*) from order_features) f,(select count(*) from order_outcomes) o,(select count(*) from snapshot_orders) s;"
r=httpx.post(f"https://api.supabase.com/v1/projects/{ref}/database/query",headers=h,json={"query":q},timeout=90)
print("progress:", r.json())
