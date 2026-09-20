/** End-to-end browser journeys for lane situations.
 *
 *  The deterministic brief is exercised on every run because it needs no
 *  provider quota, which makes the full decision path - investigate, propose,
 *  approve, ticket - verifiable on the deployed site without spending the free
 *  tier. The LLM path is exercised separately, and skipped honestly where no
 *  key is configured.
 */

import { expect, test, type Page } from "@playwright/test";

const API = process.env.API_BASE_URL ?? "http://127.0.0.1:8000";

async function llmConfigured(page: Page): Promise<boolean> {
  const res = await page.request.get(`${API}/api/v1/meta`);
  return (await res.json()).llm_configured === true;
}

/** Open the first lane that the backend says is escalatable, if any. */
async function openEscalatableLane(page: Page): Promise<boolean> {
  await page.goto("/situations");
  await expect(page.getByRole("heading", { name: "Lane situations" })).toBeVisible();
  await expect(page.getByTestId("situation-row").first()).toBeVisible();

  const rows = page.getByTestId("situation-row");
  for (let i = 0; i < (await rows.count()); i += 1) {
    const badge = rows.nth(i).locator(".badge--high");
    if (await badge.count()) {
      await rows.nth(i).click();
      await expect(page.getByTestId("situation-detail-stats")).toBeVisible();
      return true;
    }
  }
  return false;
}

test.describe("Lane situations", () => {
  test("groups the flagged queue into lanes with real aggregates", async ({ page }) => {
    await page.goto("/situations");
    await expect(page.getByRole("heading", { name: "Lane situations" })).toBeVisible();

    const rows = page.getByTestId("situation-row");
    await expect(rows.first()).toBeVisible();
    expect(await rows.count()).toBeGreaterThan(0);

    // Ranked by expected late, descending. The page must not re-sort or
    // recompute; it renders what the backend ordered.
    const expected: number[] = [];
    for (let i = 0; i < Math.min(5, await rows.count()); i += 1) {
      const text = await rows.nth(i).locator("td").nth(2).innerText();
      expected.push(Number(text.trim()));
    }
    for (const value of expected) expect(Number.isNaN(value)).toBe(false);
    const sorted = [...expected].sort((a, b) => b - a);
    expect(expected).toEqual(sorted);

    await expect(page.getByTestId("situation-stats")).toBeVisible();
  });

  test("never displays a delivery outcome", async ({ page }) => {
    await page.goto("/situations");
    await expect(page.getByTestId("situation-row").first()).toBeVisible();
    const body = (await page.locator("body").innerText()).toLowerCase();
    expect(body).not.toContain("delivered on");
    expect(body).not.toContain("was late");
    expect(body).not.toContain("actually late");
  });

  test("a lane opens with its members and an as-of history caveat", async ({ page }) => {
    await page.goto("/situations");
    await page.getByTestId("situation-row").first().click();

    await expect(page.getByTestId("situation-detail-stats")).toBeVisible();
    await expect(page.getByTestId("situation-member").first()).toBeVisible();
    // The lane history must always carry its as-of caveat.
    await expect(page.locator("text=/delivered before/i").first()).toBeVisible();
  });

  test("a deep link to a lane survives a reload", async ({ page }) => {
    await page.goto("/situations");
    await page.getByTestId("situation-row").first().click();
    await expect(page.getByTestId("situation-detail-stats")).toBeVisible();

    const url = page.url();
    const response = await page.goto(url);
    expect(response?.status()).toBeLessThan(400);
    await expect(page.getByTestId("situation-detail-stats")).toBeVisible();
  });
});

test.describe("Deterministic brief", () => {
  test("produces a cited brief with no AI and labels itself as such", async ({ page }) => {
    await page.goto("/situations");
    await page.getByTestId("situation-row").first().click();
    await expect(page.getByTestId("situation-detail-stats")).toBeVisible();

    await page.getByTestId("investigate-deterministic").click();
    await expect(page.getByTestId("situation-report")).toBeVisible({ timeout: 60_000 });

    const label = page.getByTestId("generated-by");
    await expect(label).toContainText(/Deterministic brief/i);
    await expect(label).toContainText(/no language model/i);

    // Every finding must show the evidence ids it rests on.
    const cites = page.locator(".fact__cites .cite");
    expect(await cites.count()).toBeGreaterThan(0);
  });

  test("approving a lane creates one ticket covering every member order", async ({ page }) => {
    const found = await openEscalatableLane(page);
    test.skip(!found, "no escalatable lane in this snapshot");

    const memberCount = await page.getByTestId("situation-member").count();
    expect(memberCount).toBeGreaterThan(0);

    await page.getByTestId("investigate-deterministic").click();
    await expect(page.getByTestId("situation-report")).toBeVisible({ timeout: 60_000 });

    const proposal = page.getByTestId("situation-proposal");
    await expect(proposal).toBeVisible();
    // The approval control must state the scope rather than leave it implied.
    await expect(proposal).toContainText(new RegExp(`${memberCount}\\s+orders`));

    await page.getByTestId("approve-situation").click();
    await expect(page.getByTestId("situation-ticket-link")).toBeVisible({ timeout: 30_000 });

    await page.goto("/tickets");
    await expect(page.locator("text=/lane_escalation|Lane/i").first()).toBeVisible();
  });

  test("a lane below the escalation minimum offers no approval", async ({ page }) => {
    await page.goto("/situations");
    const rows = page.getByTestId("situation-row");
    await expect(rows.first()).toBeVisible();

    let opened = false;
    for (let i = 0; i < (await rows.count()); i += 1) {
      if ((await rows.nth(i).locator(".badge--high").count()) === 0) {
        await rows.nth(i).click();
        opened = true;
        break;
      }
    }
    test.skip(!opened, "every lane in this snapshot is escalatable");

    await expect(page.getByTestId("situation-detail-stats")).toBeVisible();
    await page.getByTestId("investigate-deterministic").click();
    await expect(page.getByTestId("situation-report")).toBeVisible({ timeout: 60_000 });

    await expect(page.getByTestId("situation-report")).toContainText(/Monitor/i);
    await expect(page.getByTestId("situation-proposal")).toHaveCount(0);
  });
});

test.describe("AI lane investigation", () => {
  test("produces a verified, cited assessment", async ({ page }) => {
    test.skip(!(await llmConfigured(page)), "no LLM key configured on this backend");

    await page.goto("/situations");
    await page.getByTestId("situation-row").first().click();
    await expect(page.getByTestId("situation-detail-stats")).toBeVisible();

    await page.getByTestId("investigate-situation").click();
    await expect(page.getByTestId("situation-report")).toBeVisible({ timeout: 90_000 });

    // Either the model wrote it, or the provider was unavailable and the
    // deterministic brief stood in. Both are correct outcomes; a broken page
    // is not.
    await expect(page.getByTestId("generated-by")).toBeVisible();
    const cites = page.locator(".fact__cites .cite");
    expect(await cites.count()).toBeGreaterThan(0);
  });
});

test.describe("Accessibility and empty states", () => {
  test("the queue and the lane list are reachable by keyboard", async ({ page }) => {
    // A row click is a mouse convenience. Without a real link in the row the
    // application's primary navigation is unusable without a pointer.
    await page.goto("/situations");
    await expect(page.getByTestId("situation-row").first()).toBeVisible();
    const laneLink = page.getByTestId("situation-row").first().getByRole("link");
    await expect(laneLink).toBeVisible();
    await laneLink.focus();
    await page.keyboard.press("Enter");
    await expect(page.getByTestId("situation-detail-stats")).toBeVisible();

    await page.goto("/");
    await expect(page.getByTestId("order-row").first()).toBeVisible();
    const orderLink = page.getByTestId("order-row").first().getByRole("link");
    await orderLink.focus();
    await page.keyboard.press("Enter");
    await expect(page.locator("text=/Predicted at carrier handover/i").first()).toBeVisible();
  });

  test("the risk queue points at the lane view", async ({ page }) => {
    await page.goto("/");
    const hint = page.getByTestId("situations-hint");
    await expect(hint).toBeVisible();
    await hint.getByRole("link").click();
    await expect(page.getByRole("heading", { name: "Lane situations" })).toBeVisible();
  });

  test("a snapshot with no lane clusters says so rather than showing nothing", async ({
    page,
  }) => {
    // Drive the empty state through the API contract rather than hoping a
    // snapshot happens to be empty.
    await page.route("**/situations?*", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
    );
    await page.goto("/situations");
    await expect(page.locator("text=/No lane situations/i")).toBeVisible();
  });
});

test.describe("Honest wording", () => {
  test("no screen presents the risk load as a forecast of a count", async ({ page }) => {
    // The sum of member probabilities overstates the number actually late by
    // about 1.5x on held-out data (docs/snapshot_calibration.md). The backend
    // surfaces are covered by a unit test; this covers what a visitor reads.
    // A stat card slipped through once and was caught by a screenshot.
    for (const path of ["/situations", "/"]) {
      await page.goto(path);
      await page.waitForTimeout(1500);
      const body = (await page.locator("body").innerText()).toLowerCase();
      for (const phrase of [
        "expected late deliveries",
        "expects to arrive late",
        "expects to be delivered late",
      ]) {
        expect(body, `${path} claims: ${phrase}`).not.toContain(phrase);
      }
    }

    await page.goto("/situations");
    await expect(page.locator("text=/overstates/i").first()).toBeVisible();
  });
});
