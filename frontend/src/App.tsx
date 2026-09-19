/** App shell: header, routes, and a footer that always states what is real,
 *  what is historical and what is simulated. */

import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { api } from "./lib/api";
import OrderInvestigation from "./pages/OrderInvestigation";
import RiskQueue from "./pages/RiskQueue";
import Tickets from "./pages/Tickets";
import type { Meta } from "./types";

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);

  useEffect(() => {
    api.meta().then(setMeta).catch(() => setMeta(null));
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

        <Routes>
          <Route path="/" element={<RiskQueue />} />
          <Route path="/orders/:orderId" element={<OrderInvestigation />} />
          <Route path="/tickets" element={<Tickets />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
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
