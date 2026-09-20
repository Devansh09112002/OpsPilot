"""Seed a Supabase database over HTTPS instead of the PostgreSQL protocol.

Why this exists: on the development network, outbound TCP to 5432 and 6543 is
blocked (443 works, both PostgreSQL ports time out), so neither Alembic nor the
normal ingest can reach Supabase directly. Supabase exposes
`POST /v1/projects/{ref}/database/query`, which runs SQL over HTTPS, and that
path is open.

The deployed backend does **not** use this. It connects normally over the
PostgreSQL protocol from Render, which has no such restriction. This module is
only for seeding from a restricted network.

    python -m infra.seed_over_https schema    # create tables + alembic stamp
    python -m infra.seed_over_https data      # load the validated parquet files
    python -m infra.seed_over_https verify    # row counts and a sanity query
    python -m infra.seed_over_https all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "backend"), str(REPO_ROOT)]

from app.db.models import Base  # noqa: E402
from data_pipeline import spec  # noqa: E402
from data_pipeline.features import MODEL_FEATURES  # noqa: E402
from infra.provision import ProvisionError, read_env  # noqa: E402

# Rows per statement. Large enough to keep the request count sane (~200 for
# the whole seed), small enough that one statement stays well inside the API's
# payload limit.
CHUNK = 1000


def _client(env: dict[str, str]) -> tuple[httpx.Client, str]:
    token = env.get("SUPABASE_ACCESS_TOKEN")
    ref = env.get("SUPABASE_PROJECT_REF")
    if not token or not ref:
        raise ProvisionError(
            "SUPABASE_ACCESS_TOKEN and SUPABASE_PROJECT_REF must be set; run "
            "`python -m infra.provision supabase` first."
        )
    client = httpx.Client(
        base_url="https://api.supabase.com",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=180.0,
    )
    return client, ref


def run_sql(client: httpx.Client, ref: str, sql: str) -> list[dict]:
    resp = client.post(f"/v1/projects/{ref}/database/query", json={"query": sql})
    if resp.status_code >= 400:
        raise ProvisionError(
            f"SQL failed (HTTP {resp.status_code}): {resp.text[:400]}\n"
            f"  statement began: {sql[:160]}"
        )
    try:
        return resp.json()
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def create_schema(client: httpx.Client, ref: str) -> None:
    """Emit the SQLAlchemy metadata as DDL and apply it.

    The schema is generated from the same models Alembic migrates, so it cannot
    drift from the application. The alembic_version table is then stamped to
    head so a future migration run from an unrestricted network continues
    correctly rather than trying to re-create everything.
    """

    from sqlalchemy import create_mock_engine

    statements: list[str] = []

    def collect(sql, *args, **kwargs):
        statements.append(str(sql.compile(dialect=engine.dialect)).strip())

    engine = create_mock_engine("postgresql+psycopg://", collect)
    Base.metadata.create_all(engine, checkfirst=False)

    print(f"  applying {len(statements)} DDL statements")
    run_sql(client, ref, "drop schema public cascade; create schema public;")
    run_sql(client, ref, "grant all on schema public to postgres, anon, "
                         "authenticated, service_role;")

    for stmt in statements:
        if stmt:
            run_sql(client, ref, stmt + ";")

    head = _alembic_head()
    run_sql(client, ref,
            "create table if not exists alembic_version "
            "(version_num varchar(32) not null constraint alembic_version_pkc "
            "primary key);")
    run_sql(client, ref, "delete from alembic_version;")
    run_sql(client, ref, f"insert into alembic_version (version_num) values ('{head}');")
    print(f"  schema created and stamped at alembic revision {head}")


def _alembic_head() -> str:
    versions = REPO_ROOT / "backend" / "app" / "db" / "migrations" / "versions"
    revisions: dict[str, str | None] = {}
    for f in versions.glob("*.py"):
        text = f.read_text(encoding="utf-8")
        rev = down = None
        for line in text.splitlines():
            if line.startswith("revision ="):
                rev = line.split("=", 1)[1].strip().strip("'\"")
            elif line.startswith("down_revision ="):
                raw = line.split("=", 1)[1].strip()
                down = None if raw == "None" else raw.strip("'\"")
        if rev:
            revisions[rev] = down
    downs = {d for d in revisions.values() if d}
    heads = [r for r in revisions if r not in downs]
    if len(heads) != 1:
        raise ProvisionError(f"expected exactly one alembic head, found {heads}")
    return heads[0]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _sql_literal(value: Any) -> str:
    """Render one Python value as a SQL literal.

    The null check comes first and covers every missing-value type pandas
    produces. `pd.NaT` is NOT a `Timestamp` instance, so a type-first branch
    lets it reach the string fallback and emit the literal text 'NaT', which
    PostgreSQL rejects as an invalid timestamp.
    """
    if value is None:
        return "NULL"
    if isinstance(value, dict):
        return "'" + json.dumps(value).replace("'", "''") + "'::jsonb"
    try:
        if pd.isna(value):
            return "NULL"
    except (TypeError, ValueError):
        pass  # arrays and other non-scalars are not null-checkable
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, pd.Timestamp):
        return "'" + value.isoformat(sep=" ") + "'"
    return "'" + str(value).replace("'", "''") + "'"


def _insert_rows(client: httpx.Client, ref: str, table: str,
                 columns: list[str], rows: list[list[Any]]) -> None:
    cols = ", ".join(f'"{c}"' for c in columns)
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        values = ",".join(
            "(" + ",".join(_sql_literal(v) for v in row) + ")" for row in chunk
        )
        run_sql(client, ref, f'insert into "{table}" ({cols}) values {values};')
        done = min(i + CHUNK, len(rows))
        print(f"    {table}: {done:,}/{len(rows):,}", end="\r", flush=True)
    print(f"    {table}: {len(rows):,} rows inserted      ")


def load_scores_only(client: httpx.Client, ref: str) -> None:
    """Rewrite just the snapshot tables after a model change.

    The 96k feature and outcome rows are derived from the frozen dataset and do
    not change when the model does, so re-sending them over HTTPS would cost
    twenty minutes for nothing.
    """
    from ml_pipeline.model import band_for, calibrate, load_artifact, predict_risk

    features = pd.read_parquet(spec.PROCESSED_DIR / "order_features.parquet")
    members = pd.read_parquet(spec.PROCESSED_DIR / "snapshot_members.parquet")
    model, meta = load_artifact(spec.ARTIFACT_DIR)
    calibrator = meta.get("_calibrator")
    bands = meta.get("band_thresholds") or {"high": 0.60, "medium": 0.30}
    model_version = meta["model_version"]
    print(f"  rescoring with {model_version} (calibrated: {bool(calibrator)})")

    scored = members.merge(features, on="order_id", how="inner", validate="many_to_one")
    scored["ranking_score"] = predict_risk(model, scored)
    scored["risk_probability"] = calibrate(calibrator, scored["ranking_score"])
    scored["risk_band"] = [band_for(p, bands) for p in scored["risk_probability"]]

    run_sql(client, ref, 'delete from "snapshot_orders";')
    run_sql(client, ref, 'delete from "snapshots";')

    snapshot_rows = []
    for s in spec.SNAPSHOTS:
        sub = scored[scored["snapshot_id"] == s["snapshot_id"]]
        if sub.empty:
            continue
        snapshot_rows.append([
            s["snapshot_id"], s["label"], pd.Timestamp(s["snapshot_id"]),
            int(len(sub)), int((~sub["is_overdue"]).sum()), int(sub["is_overdue"].sum()),
        ])
    _insert_rows(client, ref, "snapshots",
                 ["snapshot_id", "label", "snapshot_at", "orders_in_transit",
                  "orders_pre_deadline", "orders_overdue"], snapshot_rows)

    member_rows = [
        [r.snapshot_id, r.order_id, bool(r.is_overdue), float(r.days_in_transit),
         float(r.risk_probability), float(r.ranking_score), r.risk_band, model_version]
        for r in scored.itertuples(index=False)
    ]
    _insert_rows(client, ref, "snapshot_orders",
                 ["snapshot_id", "order_id", "is_overdue", "days_in_transit",
                  "risk_probability", "ranking_score", "risk_band", "model_version"],
                 member_rows)
    run_sql(client, ref, "analyze;")


def load_data(client: httpx.Client, ref: str) -> None:
    features = pd.read_parquet(spec.PROCESSED_DIR / "order_features.parquet")
    outcomes = pd.read_parquet(spec.PROCESSED_DIR / "order_outcomes.parquet")
    members = pd.read_parquet(spec.PROCESSED_DIR / "snapshot_members.parquet")

    from data_pipeline.ingest import risk_band
    from ml_pipeline.model import load_artifact, predict_risk

    model, meta = load_artifact(spec.ARTIFACT_DIR)
    model_version = meta["model_version"]
    print(f"  scoring with {model_version}")

    scored = members.merge(features, on="order_id", how="inner", validate="many_to_one")
    scored["risk_probability"] = predict_risk(model, scored)
    scored["risk_band"] = [risk_band(p) for p in scored["risk_probability"]]
    scorable = set(members["order_id"])

    # Children first, so foreign keys stay satisfied.
    for table in ("snapshot_orders", "snapshots", "order_outcomes", "order_features"):
        run_sql(client, ref, f'delete from "{table}";')

    feature_cols = [
        "order_id", "split", "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_estimated_delivery_date",
        "features", "customer_state", "seller_state", "product_category",
        "n_items", "total_price", "total_freight",
    ]
    feature_rows = [
        [
            r.order_id, r.split, r.order_purchase_timestamp, r.order_approved_at,
            r.order_delivered_carrier_date, r.order_estimated_delivery_date,
            (
                {
                    f: (None if pd.isna(v := getattr(r, f))
                        else (v.item() if hasattr(v, "item") else v))
                    for f in MODEL_FEATURES
                } if r.order_id in scorable else None
            ),
            None if pd.isna(r.customer_state) else str(r.customer_state),
            None if pd.isna(r.seller_state) else str(r.seller_state),
            None if pd.isna(r.product_category) else str(r.product_category),
            int(r.n_items), float(r.total_price), float(r.total_freight),
        ]
        for r in features.itertuples(index=False)
    ]
    _insert_rows(client, ref, "order_features", feature_cols, feature_rows)

    outcome_rows = [
        [r.order_id, r.order_delivered_customer_date, bool(r.is_late)]
        for r in outcomes.itertuples(index=False)
    ]
    _insert_rows(client, ref, "order_outcomes",
                 ["order_id", "order_delivered_customer_date", "is_late"],
                 outcome_rows)

    snapshot_rows = []
    for s in spec.SNAPSHOTS:
        sub = scored[scored["snapshot_id"] == s["snapshot_id"]]
        if sub.empty:
            continue
        snapshot_rows.append([
            s["snapshot_id"], s["label"], pd.Timestamp(s["snapshot_id"]),
            int(len(sub)), int((~sub["is_overdue"]).sum()), int(sub["is_overdue"].sum()),
        ])
    _insert_rows(client, ref, "snapshots",
                 ["snapshot_id", "label", "snapshot_at", "orders_in_transit",
                  "orders_pre_deadline", "orders_overdue"], snapshot_rows)

    member_rows = [
        [r.snapshot_id, r.order_id, bool(r.is_overdue), float(r.days_in_transit),
         float(r.risk_probability), float(r.ranking_score), r.risk_band, model_version]
        for r in scored.itertuples(index=False)
    ]
    _insert_rows(client, ref, "snapshot_orders",
                 ["snapshot_id", "order_id", "is_overdue", "days_in_transit",
                  "risk_probability", "ranking_score", "risk_band", "model_version"],
                 member_rows)

    run_sql(client, ref, "analyze;")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify(client: httpx.Client, ref: str) -> bool:
    expected = {
        "order_features": 95952,
        "order_outcomes": 95952,
        "snapshots": 3,
        "snapshot_orders": 3955,
    }
    ok = True
    for table, want in expected.items():
        got = run_sql(client, ref, f'select count(*) as n from "{table}";')[0]["n"]
        mark = "ok" if int(got) == want else "MISMATCH"
        if int(got) != want:
            ok = False
        print(f"  {table:<18s} {int(got):>7,} (expected {want:,})  {mark}")

    rows = run_sql(client, ref, """
        select s.snapshot_id, count(*) filter (where not so.is_overdue) as pre_deadline,
               round(avg(so.risk_probability)::numeric, 4) as avg_risk,
               min(so.model_version) as model_version
        from snapshots s join snapshot_orders so using (snapshot_id)
        group by 1 order by 1;
    """)
    for r in rows:
        print(f"  snapshot {r['snapshot_id']}: {r['pre_deadline']} pre-deadline, "
              f"avg risk {r['avg_risk']}, model {r['model_version']}")

    leak = run_sql(client, ref, """
        select count(*) as n from information_schema.columns
        where table_name = 'order_features'
          and column_name in ('is_late', 'order_delivered_customer_date');
    """)[0]["n"]
    print(f"  outcome columns on order_features: {int(leak)} (must be 0)")
    ok &= int(leak) == 0
    return ok


def apply_migrations(client: httpx.Client, ref: str) -> None:
    """Bring the deployed database to Alembic head, over HTTPS.

    The development network blocks the Postgres ports, so Alembic cannot open
    its own connection to Supabase from here. It can still *generate* the
    migration SQL offline (`alembic upgrade <current>:head --sql`), which is
    the same SQL it would execute, and that is what gets sent.

    The current revision is read from the deployed `alembic_version` table
    rather than assumed, so running this twice is a no-op instead of an error.
    """
    rows = run_sql(client, ref, "select version_num from alembic_version;")
    if not rows:
        raise ProvisionError(
            "The deployed database has no alembic_version row. Run the "
            "'schema' command first."
        )
    current = rows[0]["version_num"]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "backend"), str(REPO_ROOT), env.get("PYTHONPATH", "")]
    )
    # Alembic still needs a URL to construct its config; it is never connected
    # to in --sql mode, so a local placeholder is correct here.
    env.setdefault("DATABASE_URL", "postgresql+psycopg://offline:offline@127.0.0.1:1/offline")

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", f"{current}:head", "--sql"],
        cwd=str(REPO_ROOT / "backend"),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProvisionError(f"Could not generate migration SQL:\n{result.stderr[-800:]}")

    sql = "\n".join(
        line for line in result.stdout.splitlines() if not line.startswith("INFO")
    ).strip()
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    statements = [
        s for s in statements if s.upper() not in {"BEGIN", "COMMIT", "BEGIN TRANSACTION"}
    ]
    if not statements:
        print(f"  already at head ({current}); nothing to apply")
        return

    print(f"  at {current}; applying {len(statements)} statements")
    for statement in statements:
        run_sql(client, ref, statement + ";")
    head = run_sql(client, ref, "select version_num from alembic_version;")[0]["version_num"]
    print(f"  now at {head}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["schema", "migrate", "data", "scores", "verify", "all"])
    args = parser.parse_args(argv)

    env = read_env()
    client, ref = _client(env)
    try:
        if args.command in ("schema", "all"):
            print("[schema]")
            create_schema(client, ref)
        if args.command in ("migrate", "all"):
            print("[migrate]")
            apply_migrations(client, ref)
        if args.command in ("data", "all"):
            print("[data]")
            load_data(client, ref)
        if args.command == "scores":
            print("[scores]")
            load_scores_only(client, ref)
        if args.command in ("verify", "all"):
            print("[verify]")
            if not verify(client, ref):
                print("VERIFY FAILED", file=sys.stderr)
                return 1
            print("VERIFY PASSED")
    except ProvisionError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
