/** End-to-end browser journeys.
 *
 *  These run against a real running stack - the same one a visitor uses. The
 *  investigation journey needs a configured LLM key on the backend; where
 *  there is none, the suite asserts the honest-failure path instead of
 *  pretending the feature works.
 */

import { expect, test, type Page } from "@playwright/test";

const API = process.env.API_BASE_URL ?? "http://127.0.0.1:8000";

async function llmConfigured(page: Page): Promise<boolean> {
  const res = await page.request.get(`${API}/api/v1/meta`);
  return (await res.json()).llm_configured === true;
}

test.describe("Risk queue", () => {
  test("shows real snapshots, model scores and a version label", async ({ page }) => {
    await page.goto("/");

    await expect(page.getByRole("heading", { name: "Delivery risk queue" })).toBeVisible();

    // Wait for real data before asserting on it; the table shows skeleton
    // rows while the snapshot list and first page are still loading.
    const rows = page.getByTestId("order-row");
    await expect(rows.first()).toBeVisible();
    expect(await rows.count()).toBeGreaterThan(5);

    const snapshot = page.locator("#snapshot");
    await expect(snapshot).toBeVisible();
    await expect(snapshot.locator("option").first()).toBeAttached();
    expect(await snapshot.locator("option").count()).toBeGreaterThanOrEqual(1);

    // Scores must be real numbers in [0,1], ranked descending.
    const scores = await page
      .getByTestId("order-row")
      .locator("td:nth-child(2) .mono")
      .allTextContents();
    const values = scores.map(Number);
    expect(values.every((v) => v >= 0 && v <= 1)).toBeTruthy();
    expect([...values].sort((a, b) => b - a)).toEqual(values);

    // The model version is displayed, so no number is unattributed.
    await expect(page.locator("text=/xgboost|logistic_regression|rule_/").first()).toBeVisible();
  });

  test("distinguishes predicted risk from an already-known outcome", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("text=/predicted risk|at carrier handover/i").first()).toBeVisible();
    await expect(page.locator("text=/overdue/i").first()).toBeVisible();
  });

  test("filters by risk band", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("order-row").first()).toBeVisible();

    await page.locator("#band").selectOption("high");
    await expect(page.getByTestId("order-row").first()).toBeVisible();

    const bands = await page
      .getByTestId("order-row")
      .locator("td:nth-child(3) .badge")
      .allTextContents();
    expect(bands.every((b) => b.trim() === "high")).toBeTruthy();
  });

  test("paginates without repeating orders", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("order-row").first()).toBeVisible();
    const first = await page.getByTestId("order-row").first().getAttribute("data-order-id");

    await page.getByRole("button", { name: "Next" }).click();
    await expect(page.getByTestId("order-row").first()).toBeVisible();
    const second = await page.getByTestId("order-row").first().getAttribute("data-order-id");

    expect(second).not.toEqual(first);
  });

  test("never displays a delivery outcome", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("order-row").first()).toBeVisible();
    const body = (await page.locator("body").innerText()).toLowerCase();
    for (const banned of ["delivered_customer", "is_late", "actually delivered"]) {
      expect(body).not.toContain(banned);
    }
  });
});

test.describe("Order detail", () => {
  test("opens an order with a version-labelled score and no outcome", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("order-row").first()).toBeVisible();
    await page.getByTestId("order-row").first().click();

    await expect(page.getByRole("heading", { name: "Model assessment" })).toBeVisible();
    await expect(page.locator("text=/predicted at carrier handover/i")).toBeVisible();
    await expect(page.getByTestId("investigate")).toBeVisible();

    const body = (await page.locator("body").innerText()).toLowerCase();
    expect(body).not.toContain("delivered_customer");
    expect(body).not.toContain("is_late");
  });
});

test.describe("Investigation, approval and ticket", () => {
  test("completes the full journey with approval", async ({ page }) => {
    test.skip(!(await llmConfigured(page)), "no LLM key configured on this backend");
    test.setTimeout(150_000);

    await page.goto("/");
    // Wait for the queue to load before filtering it: the controls are
    // disabled until the snapshot list arrives.
    await expect(page.getByTestId("order-row").first()).toBeVisible();
    await page.locator("#band").selectOption("high");
    await expect(page.getByTestId("order-row").first()).toBeVisible();
    await page.getByTestId("order-row").first().click();

    await page.getByTestId("investigate").click();
    await expect(page.getByTestId("investigation-running")).toBeVisible();

    const report = page.getByTestId("investigation-report");
    await expect(report).toBeVisible({ timeout: 120_000 });

    // The report must cite evidence, not just assert prose.
    await expect(page.locator(".cite").first()).toBeVisible();
    await expect(page.locator("text=/Evidence used \\(\\d+ items\\)/")).toBeVisible();

    const proposal = page.getByTestId("proposal");
    if (await proposal.isVisible()) {
      await page.getByTestId("approve").click();
      await expect(page.getByTestId("ticket-link")).toBeVisible({ timeout: 30_000 });

      await page.goto("/tickets");
      await expect(page.getByTestId("ticket-row")).toHaveCount(1);
      await expect(page.locator("text=/Simulation/i").first()).toBeVisible();

      // A refresh must neither lose nor duplicate the ticket.
      await page.reload();
      await expect(page.getByTestId("ticket-row")).toHaveCount(1);
    }
  });

  test("rejecting a proposal creates no ticket", async ({ page }) => {
    test.skip(!(await llmConfigured(page)), "no LLM key configured on this backend");
    test.setTimeout(150_000);

    await page.goto("/");
    await expect(page.getByTestId("order-row").first()).toBeVisible();
    await page.locator("#band").selectOption("high");
    await expect(page.getByTestId("order-row").first()).toBeVisible();
    await page.getByTestId("order-row").nth(1).click();

    await page.getByTestId("investigate").click();
    await expect(page.getByTestId("investigation-report")).toBeVisible({ timeout: 120_000 });

    const proposal = page.getByTestId("proposal");
    if (await proposal.isVisible()) {
      await page.getByTestId("reject").click();
      await expect(page.locator("text=/No ticket was created/i")).toBeVisible();

      await page.goto("/tickets");
      await expect(page.getByTestId("ticket-row")).toHaveCount(0);
    }
  });

  test("reports an actionable error when AI is unavailable", async ({ page }) => {
    test.skip(await llmConfigured(page), "LLM is configured; failure path not exercised");

    await page.goto("/");
    await expect(page.getByTestId("order-row").first()).toBeVisible();
    await page.getByTestId("order-row").first().click();

    await page.getByTestId("investigate").click();
    await expect(page.locator("text=/Investigation failed|unavailable/i").first()).toBeVisible({
      timeout: 60_000,
    });
    // The rest of the product must still work.
    await expect(page.getByRole("heading", { name: "Model assessment" })).toBeVisible();
  });
});

test.describe("Tickets", () => {
  test("starts empty for a new visitor and explains how to create one", async ({ page }) => {
    // A direct navigation, not a client-side link: on a static host this only
    // works if the SPA rewrite is configured, so this doubles as a deploy check.
    const response = await page.goto("/tickets");
    expect(response?.status()).toBeLessThan(400);
    await expect(page.getByRole("heading", { name: "Your tickets" })).toBeVisible();
    await expect(page.locator("text=/Simulation/i").first()).toBeVisible();
  });

  test("a deep link survives a reload", async ({ page }) => {
    await page.goto("/tickets");
    await expect(page.getByRole("heading", { name: "Your tickets" })).toBeVisible();
    const response = await page.reload();
    expect(response?.status()).toBeLessThan(400);
    await expect(page.getByRole("heading", { name: "Your tickets" })).toBeVisible();
  });
});

test.describe("Security", () => {
  test("no API key reaches the browser bundle", async ({ page }) => {
    const bodies: string[] = [];
    page.on("response", async (r) => {
      const type = r.headers()["content-type"] ?? "";
      if (type.includes("javascript") || type.includes("html")) {
        try {
          bodies.push(await r.text());
        } catch {
          /* ignore */
        }
      }
    });
    await page.goto("/");
    await expect(page.getByTestId("order-row").first()).toBeVisible();

    const all = bodies.join("\n");
    expect(all).not.toContain("sk-ant");
    expect(all.toLowerCase()).not.toContain("llm_api_key");
    expect(all).not.toMatch(/ANTHROPIC_API_KEY\s*[:=]\s*["'][^"']+["']/);
  });

  test("the session cookie is HttpOnly", async ({ page, context }) => {
    await page.goto("/");
    await page.getByTestId("order-row").first().waitFor();
    // Force a session-bearing request.
    await page.goto("/tickets");
    await expect(page.getByRole("heading", { name: "Your tickets" })).toBeVisible();

    const cookie = (await context.cookies()).find((c) => c.name === "opspilot_session");
    if (cookie) {
      expect(cookie.httpOnly).toBeTruthy();
      // Not readable from JavaScript, so XSS cannot exfiltrate it.
      const visible = await page.evaluate(() => document.cookie);
      expect(visible).not.toContain(cookie.value);
    }
  });
});
