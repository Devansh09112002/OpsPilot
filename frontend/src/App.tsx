/** App shell: header, routes, and a footer that always states what is real,
 *  what is historical and what is simulated.
 *
 *  The shell also owns cold-start handling. On a free-tier host the API sleeps
 *  after inactivity, so the app waits for it explicitly and explains the wait
 *  instead of rendering screens that would all fail to load.
 */

import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { api } from "./lib/api";
import { type WakeState, wakeBackend } from "./lib/wakeup";
import OrderInvestigation from "./pages/OrderInvestigation";
import RiskQueue from "./pages/RiskQueue";
import Tickets from "./pages/Tickets";
import type { Meta } from "./types";

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [wake, setWake] = useState<WakeState>("checking");
  const [dbDown, setDbDown] = useState(false);

  useEffect(() => {
    let cancelled = false;
    wakeBackend((s) => !cancelled && setWake(s)).then((state) => {
      if (cancelled || state !== "awake") return;
      api
        .meta()
        .then(setMeta)
        .catch(() => setMeta(null));
      // Readiness distinguishes "the API is up but its database is paused"
      // from a total outage, which matters on a free-tier database.
      api
        .readiness()
        .then((r) => setDbDown(r.checks?.database?.ok === false))
        .catch(() => undefined);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="app">
      <header className="site-header">
        <div className="site-header__inner">
          <NavLink to="/" className="brand">
            OpsPilot <small>Delivery risk &amp; AI investigation</small>
          </NavLink>
          <nav className="nav">
            <NavLink to="/" end>
              Risk queue
            </NavLink>
            <NavLink to="/tickets">Tickets</NavLink>
          </nav>
        </div>
      </header>

      <main className="main">
        {wake === "cold" && (
          <div className="banner banner--warn" data-testid="cold-start">
            <strong>Waking the server.</strong> OpsPilot runs on a free hosting
            tier that sleeps when idle, so the first visit after a quiet period
            takes up to a minute. Everything loads normally once it is up.
          </div>
        )}

        {wake === "unreachable" && (
          <div className="banner banner--error" data-testid="api-unreachable">
            <strong>The OpsPilot API is not responding.</strong> The free-tier
            server or its database may be paused. Please reload in a minute.
          </div>
        )}

        {dbDown && (
          <div className="banner banner--error">
            <strong>The database is unavailable.</strong> Free-tier databases
            pause after a period of inactivity. Order data cannot be loaded
            until it resumes.
          </div>
        )}

        {meta && !meta.llm_configured && (
          <div className="banner banner--warn">
            <strong>AI investigation is currently unavailable</strong> because no
            LLM provider key is configured on this deployment. The risk queue,
            order details and model predictions all work normally.
          </div>
        )}

        {meta && !meta.model_available && (
          <div className="banner banner--error">
            <strong>The delivery-risk model is not loaded.</strong> Risk scores
            cannot be served. No placeholder values are shown in their place.
          </div>
        )}

        {wake === "awake" || wake === "checking" ? (
          <Routes>
            <Route path="/" element={<RiskQueue />} />
            <Route path="/orders/:orderId" element={<OrderInvestigation />} />
            <Route path="/tickets" element={<Tickets />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        ) : (
          <div className="state">
            <h3>{wake === "cold" ? "Starting up…" : "Unavailable"}</h3>
            <p className="muted">
              {wake === "cold"
                ? "Connecting to the OpsPilot API."
                : "Could not reach the OpsPilot API."}
            </p>
          </div>
        )}
      </main>

      <footer className="site-footer">
        <div className="site-footer__inner">
          <div style={{ maxWidth: "58ch" }}>
            Order records are real, anonymised Brazilian marketplace orders from
            2016&ndash;2018. Risk scoring, AI investigation and ticketing execute
            live when you use them. All escalations are simulated inside this
            demo environment &mdash; no courier, seller or customer is contacted.
          </div>
          <div>
            {meta && (
              <>
                <div>
                  Model <code>{meta.model_version ?? "unavailable"}</code>
                  {meta.training_cutoff && <> &middot; trained to {meta.training_cutoff}</>}
                </div>
                <div>
                  Policy <code>{meta.policy_version}</code>
                </div>
                <div style={{ marginTop: "0.35rem" }}>
                  Data:{" "}
                  <a href={meta.dataset.url} target="_blank" rel="noreferrer noopener">
                    {meta.dataset.name}
                  </a>{" "}
                  &middot; {meta.dataset.license}
                </div>
              </>
            )}
          </div>
        </div>
      </footer>
    </div>
  );
}
