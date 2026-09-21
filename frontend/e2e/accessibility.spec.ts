/** Automated accessibility checks on every screen.
 *
 *  A previous pass fixed the blocking defect - table rows were clickable but
 *  not reachable by keyboard - and stopped there, leaving contrast and
 *  labelling unexamined. This runs axe-core over each screen against the
 *  WCAG 2.1 A and AA rule sets, which catches the categories a human reading
 *  the markup reliably misses: contrast ratios, missing form labels, landmark
 *  structure and heading order.
 *
 *  Serious violations fail the suite. Anything knowingly accepted is listed
 *  in ACCEPTED below with the reason, so an exception is a decision on the
 *  record rather than a silent skip.
 */

import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"];

/** Rule ids deliberately not enforced, each with a reason. */
const ACCEPTED: Record<string, string> = {};

type Violation = {
  id: string;
  impact?: string | null;
  help: string;
  nodes: { target: unknown[] }[];
};

async function scan(page: Page, label: string) {
  const results = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  const violations = (results.violations as Violation[]).filter(
    (v) => !(v.id in ACCEPTED),
  );

  if (violations.length) {
    const detail = violations
      .map(
        (v) =>
          `  [${v.impact ?? "unknown"}] ${v.id}: ${v.help}\n` +
          v.nodes
            .slice(0, 3)
            .map((n) => `      ${JSON.stringify(n.target)}`)
            .join("\n"),
      )
      .join("\n");
    throw new Error(`${label} has ${violations.length} accessibility violation(s):\n${detail}`);
  }
}

test.describe("Accessibility (axe-core, WCAG 2.1 AA)", () => {
  test("risk queue", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("order-row").first()).toBeVisible();
    await scan(page, "risk queue");
  });

  test("lane situations", async ({ page }) => {
    await page.goto("/situations");
    await expect(page.getByTestId("situation-row").first()).toBeVisible();
    await scan(page, "lane situations");
  });

  test("lane detail", async ({ page }) => {
    await page.goto("/situations");
    await page.getByTestId("situation-row").first().click();
    await expect(page.getByTestId("situation-detail-stats")).toBeVisible();
    await scan(page, "lane detail");
  });

  test("order detail", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("order-row").first().getByRole("link").click();
    await expect(page.locator("text=/Predicted at carrier handover/i").first()).toBeVisible();
    await scan(page, "order detail");
  });

  test("tickets", async ({ page }) => {
    await page.goto("/tickets");
    await expect(page.getByRole("heading", { name: "Your tickets" })).toBeVisible();
    await scan(page, "tickets");
  });

  test("a rendered lane brief", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto("/situations");
    await page.getByTestId("situation-row").first().click();
    await expect(page.getByTestId("situation-detail-stats")).toBeVisible();
    await page.getByTestId("investigate-deterministic").click();
    await expect(page.getByTestId("situation-report")).toBeVisible({ timeout: 90_000 });
    await scan(page, "lane brief");
  });

  test("every page has one main landmark and a page heading", async ({ page }) => {
    for (const path of ["/", "/situations", "/tickets"]) {
      await page.goto(path);
      await page.waitForTimeout(1200);
      expect(await page.locator("main").count(), `${path} main landmark`).toBe(1);
      // Exactly one h1 per page: three screens had none at all, so a screen
      // reader had no page title to announce.
      expect(await page.getByRole("heading", { level: 1 }).count(), `${path} h1`).toBe(1);
    }
  });
});
