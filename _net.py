import socket, pathlib
env = dict(l.split("=",1) for l in pathlib.Path(".env").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#") and "=" in l)
ref = env["SUPABASE_PROJECT_REF"].strip()
targets = [
    ("aws-0-us-west-1.pooler.supabase.com", 6543, "session/transaction pooler"),
    ("aws-0-us-west-1.pooler.supabase.com", 5432, "pooler on 5432 (session mode)"),
    (f"db.{ref}.supabase.co", 5432, "direct database"),
    ("api.supabase.com", 443, "control plane (known good)"),
]
for host, port, label in targets:
    try:
        ip = socket.gethostbyname(host)
    except Exception as e:
        print(f"  {label:<34s} DNS FAILED {e}"); continue
    s = socket.socket(); s.settimeout(12)
    try:
        s.connect((ip, port))
        print(f"  {label:<34s} OK      {host}:{port} ({ip})")
    except Exception as e:
        print(f"  {label:<34s} BLOCKED {host}:{port} ({ip}) -> {type(e).__name__}: {e}")
    finally:
        s.close()
