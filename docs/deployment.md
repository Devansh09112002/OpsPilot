# OpsPilot — Deployment

Target: a publicly reachable HTTPS demo at **zero cost**.

| Piece | Service | Plan | Why this one |
|---|---|---|---|
| API | Render Web Service (Docker) | Free | Runs the FastAPI process, the model artifact and the agent in one container. |
| Frontend | Render Static Site | Free | Serves the built Vite bundle over HTTPS on a CDN. |
| Database | Supabase PostgreSQL | Free | Render's free Postgres is **deleted after 30 days**; a Supabase free project persists. |
| LLM | Google Gemini API | Free tier | `gemini-3.5-flash`, no card required. |
| CI/CD | GitHub Actions | Free | Tests on every push; a scheduled ping keeps the free services warm. |

Nothing here requires a card, and no step enables billing.

---

## Known free-tier limits, and what the app does about them

| Limit | Effect | Mitigation in the code |
|---|---|---|
| Render free service sleeps after ~15 min idle | First request after a quiet period takes up to ~60 s | `frontend/src/lib/wakeup.ts` polls health and shows "Waking the server" instead of a broken page. The keepalive workflow pings every 10 min during waking hours. |
| Render free tier: 512 MB RAM, 0.1 CPU | OOM if the process is fat | Measured ~340 MB RSS. Image pins one worker and single-threaded BLAS/OpenMP. |
| Supabase free project pauses after ~7 days with no activity | Database unreachable until manually resumed | Keepalive touches the DB through `/health/ready`. `/health/ready` reports `unavailable`, and the UI says the database is paused rather than showing an empty queue. |
| Supabase free: 500 MB storage | Ingest must fit | Ingest stores feature documents only for the 3,912 scorable orders, so the database is **53 MB**. |
| Gemini free tier: per-minute and per-day request quotas | 429s under load | Per-session (10/day), global hourly (60) and global daily (180) caps are enforced *before* the call, returning a clear 429. |

---

## Step 1 — Gemini API key

1. Go to <https://aistudio.google.com/apikey>.
2. Sign in with a Google account and choose **Create API key**.
3. Pick **Create API key in new project** if prompted.
4. Copy the key (it starts with `AIza...`).

No billing account is required. The free tier is rate-limited, not charged.

---

## Step 2 — Supabase PostgreSQL

1. Go to <https://supabase.com/dashboard> and sign up (GitHub sign-in is fine).
2. **New project**:
   - *Name*: `opspilot`
   - *Database Password*: generate one and **save it** — it is shown once.
   - *Region*: pick the one nearest the Render region you will use
     (`us-west-1` pairs well with Render's Oregon).
   - *Plan*: **Free**.
3. Wait for provisioning (about two minutes).
4. Open **Project Settings → Database → Connection string → URI**.
5. Select the **Session pooler** entry, port **6543**. Copy it.

   It looks like:
   ```
   postgresql://postgres.abcdefghijklm:[YOUR-PASSWORD]@aws-0-us-west-1.pooler.supabase.com:6543/postgres
   ```

6. Two edits are required before OpsPilot can use it:
   - Replace `[YOUR-PASSWORD]` with the password from step 2.
   - Change the scheme to `postgresql+psycopg://` so SQLAlchemy uses psycopg 3.

   Final form:
   ```
   postgresql+psycopg://postgres.abcdefghijklm:THEPASSWORD@aws-0-us-west-1.pooler.supabase.com:6543/postgres
   ```

> **Use the session pooler (6543), not the direct connection (5432).** A free
> Supabase project allows few direct connections, and a Render restart can
> exhaust them, leaving the API unable to reach its own database.

> If the password contains `@`, `:`, `/` or `#`, percent-encode it
> (`@` → `%40`, and so on) or the URL will not parse.

---

## Step 3 — Load the data into Supabase

Run this **from the development machine**, once. It applies migrations and
loads the validated data.

```bash
# 1. Point at Supabase instead of the local database
#    (edit .env, or set the variable for one command)
export DATABASE_URL='postgresql+psycopg://postgres.xxx:PASSWORD@...pooler.supabase.com:6543/postgres'

# 2. Produce the artifacts if this is a fresh checkout
python -m data_pipeline.fetch     # verified download of the Olist CSVs
python -m data_pipeline.audit     # data gate -> data/processed/*.parquet
python -m ml_pipeline.train       # -> artifacts/delivery_risk_model.joblib

# 3. Schema, then data
cd backend && alembic upgrade head && cd ..
python -m data_pipeline.ingest
```

Expected output:

```
order_features      95,952 rows
order_outcomes      95,952 rows
snapshots                3 rows
snapshot_orders      3,955 rows
```

This is reproducible and idempotent: every table is replaced inside one
transaction, so re-running it restores the database from scratch if the
Supabase project is ever reset.

---

## Step 4 — Deploy the API to Render

1. Go to <https://dashboard.render.com> and **Sign up with GitHub**.
2. Grant Render access to the `OpsPilot` repository
   (*Configure account* → select the repo; a private repo needs this explicitly).
3. **New → Blueprint**, select the `OpsPilot` repo. Render reads `render.yaml`
   and proposes two services: `opspilot-api` and `opspilot-web`.
4. Render will ask for the values marked `sync: false`. Set them on
   **opspilot-api**:

   | Key | Value |
   |---|---|
   | `DATABASE_URL` | the Supabase session-pooler URL from step 2 |
   | `GEMINI_API_KEY` | the key from step 1 |
   | `CORS_ORIGINS` | leave blank for now; fixed in step 6 |

5. On **opspilot-web**, set `VITE_API_BASE_URL` to the API URL Render shows,
   e.g. `https://opspilot-api.onrender.com` (no trailing slash).
6. Click **Apply**. The first Docker build takes roughly 5–10 minutes.

---

## Step 5 — Close the CORS loop

The static site's URL is only known after it deploys.

1. Copy the frontend URL, e.g. `https://opspilot-web.onrender.com`.
2. On **opspilot-api** → *Environment*, set:
   ```
   CORS_ORIGINS=https://opspilot-web.onrender.com
   ```
3. Save. Render redeploys the API automatically.

> `CORS_ORIGINS` must be the exact origin, never `*`. The guest session travels
> in a credentialed cookie, and browsers refuse credentialed requests against a
> wildcard origin — every visitor would silently lose their tickets.
>
> The cookie is already configured `Secure` + `SameSite=None` in `render.yaml`,
> which is required because the API and the frontend are different origins.

---

## Step 6 — Keep the services awake

1. In GitHub → **Settings → Secrets and variables → Actions → Variables**,
   add a repository **variable**:
   - Name: `OPSPILOT_API_URL`
   - Value: `https://opspilot-api.onrender.com`
2. The `Keepalive` workflow then pings `/health/ready` every 10 minutes during
   waking hours, which keeps the Render service warm *and* counts as Supabase
   activity so the project does not pause.

It is a repository variable, not a secret: the URL is public.

---

## Step 7 — Verify the deployment

Never trust a green build alone. Check the running site.

```bash
API=https://opspilot-api.onrender.com

curl -s $API/api/v1/health
# {"status":"ok",...}

curl -s $API/api/v1/health/ready
# every check ok; "status":"ok" once the Gemini key is set

curl -s $API/api/v1/meta
# model_version, policy_version, llm_configured: true

curl -s "$API/api/v1/orders?snapshot_id=2018-08-15&limit=3"
# three real orders with model risk scores
```

Then the browser journey, against the deployed URL:

```bash
cd frontend
BASE_URL=https://opspilot-web.onrender.com \
  API_BASE_URL=https://opspilot-api.onrender.com \
  npx playwright test
```

The same suite that runs locally runs against production. The investigation and
approval journeys un-skip themselves once `llm_configured` is true.

---

## Rolling back

Render keeps previous deploys. **Dashboard → service → Events → Rollback**
restores the last working image; no repository change is needed.

The database is separate from the deploy, so a rollback never loses tickets.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Site loads, queue is empty, no error | API reachable but database empty | Re-run step 3. |
| "Waking the server" for over a minute | Render cold start, or the build is failing | Check *Events* and *Logs* on the API service. |
| Every reload loses tickets | Cookie rejected | `CORS_ORIGINS` must be the exact frontend origin; `COOKIE_SECURE=true`, `COOKIE_SAMESITE=none`. |
| `/health/ready` says database not ok | Supabase project paused | Resume it in the Supabase dashboard; confirm the keepalive variable from step 6 is set. |
| API restarts repeatedly, logs mention memory | 512 MB exceeded | Confirm one worker and the `*_NUM_THREADS=1` variables from the Dockerfile are in effect. |
| Investigations return 429 | Free-tier quota reached, working as designed | Wait, or raise the caps only if the provider quota genuinely allows it. |
| `psycopg.OperationalError: too many connections` | Using the direct connection (5432) | Switch to the session pooler (6543). |

---

## Secrets

- `GEMINI_API_KEY` and `DATABASE_URL` exist only in Render's environment
  configuration and in the local, gitignored `.env`.
- No key is ever sent to the browser: an E2E test asserts the built bundle
  contains no key material, and CI fails on a committed credential.
- To rotate the Gemini key: create a new one in AI Studio, update it on the
  Render service, delete the old one.
