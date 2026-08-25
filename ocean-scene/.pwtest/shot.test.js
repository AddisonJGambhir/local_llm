const { test } = require('@playwright/test');
test('ocean screenshots', async ({ page }) => {
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('http://localhost:8931/');
  await page.waitForTimeout(6000);
  await page.screenshot({ path: '/tmp/ocean-golden.png' });
  await page.keyboard.press('KeyC');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: '/tmp/ocean-midday.png' });
  console.log('errors:', JSON.stringify(errors));
});
