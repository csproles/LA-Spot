const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto('http://localhost:7001/onboarding', { waitUntil: 'load', timeout: 20000 });
  await page.waitForTimeout(4000);
  await page.screenshot({ path: `retest-onboarding.png`, fullPage: true });
  const html = await page.content();
  require('fs').writeFileSync('onboarding.html', html);
  await browser.close();
})();
