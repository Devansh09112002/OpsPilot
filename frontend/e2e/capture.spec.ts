/** Capture README screenshots from the deployed site. Not part of the suite:
 *  run explicitly with `npx playwright test e2e/capture.spec.ts`. */
import { expect, test } from "@playwright/test";

const DIR = "../docs/screenshots";

test.describe("screenshots", () => {
  test.use({ viewport: { width: 1360, height: 900 } });

  test("situations", async ({ page }) => {
    await page.goto("/situations");
    await expect(page.getByTestId("situation-row").first()).toBeVisible();
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `${DIR}/live-situations.png`, fullPage: false });
  });

  test("lane detail with a deterministic brief", async ({ page }) => {
    test.setTimeout(180000);
    await page.goto("/situations");
    const rows = page.getByTestId("situation-row");
    await expect(rows.first()).toBeVisible();
    for (let i = 0; i < (await rows.count()); i += 1) {
      if (await rows.nth(i).locator(".badge--high").count()) { await rows.nth(i).click(); break; }
    }
    await expect(page.getByTestId("situation-detail-stats")).toBeVisible();
    await page.getByTestId("investigate-deterministic").click();
    await expect(page.getByTestId("situation-report")).toBeVisible({ timeout: 90_000 });
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${DIR}/live-lane-brief.png`, fullPage: false });
  });
});
