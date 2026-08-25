const { test } = require('@playwright/test');
test('debug', async ({ page }) => {
  page.on('console', m => console.log('CONSOLE:', m.text()));
  page.on('requestfailed', r => console.log('FAILED:', r.url(), r.failure()?.errorText));
  const resp = [];
  page.on('response', r => { if (r.url().includes('waternormals')) resp.push(r.status() + ' ' + r.url()); });
  await page.goto('http://localhost:8931/');
  await page.waitForTimeout(5000);
  console.log('normals resp:', JSON.stringify(resp));
  const u = await page.evaluate(() => {
    const w = window.__water; return w ? 'exposed' : 'not exposed';
  });
  console.log(u);
});
