const { test } = require('@playwright/test');
test('probe', async ({ page }) => {
  page.on('console', m => console.log('CONSOLE:', m.text()));
  await page.goto('http://localhost:8931/');
  await page.waitForTimeout(4000);
  const info = await page.evaluate(() => {
    const { water, renderer } = window.__dbg;
    const u = water.material.uniforms;
    return {
      hasMirror: !!u.mirror,
      mirrorTex: u.mirror ? !!u.mirror.value : null,
      time: u.time.value,
      sunDirection: u.sunDirection.value.toArray(),
      waterColor: u.waterColor.value.getHex(),
      dist: u.distortionScale.value,
      programs: renderer.info.programs?.length,
      renderCalls: renderer.info.render.calls,
      gl: renderer.getContext().getParameter(renderer.getContext().VERSION)
    };
  });
  console.log(JSON.stringify(info, null, 1));
});
