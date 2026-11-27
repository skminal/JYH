// node build_pdf.js -> ../results/零基础读懂_OpenFOAM复刻等离子体气泡.pdf
const path = require('path');
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path.join(__dirname, 'guide.html'), { waitUntil: 'load' });
  await page.waitForTimeout(500);
  // print every equation at the same type size: images were rendered at 220 dpi
  // from 20-22 pt mathtext; scale them to ~12.5 pt on paper
  await page.evaluate(() => {
    document.querySelectorAll('.eq img').forEach((im) => {
      const fs = /eq_mom/.test(im.src) ? 20 : 22;
      im.style.width = (im.naturalWidth / 220 * 96 * 15 / fs) + 'px';
      im.style.maxWidth = '100%';
    });
  });
  const out = path.join(__dirname, '..', 'results', '零基础读懂_OpenFOAM复刻等离子体气泡.pdf');
  await page.pdf({
    path: out, format: 'A4', printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate: '<div style="width:100%;font-size:8px;color:#888;text-align:center;font-family:sans-serif"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
  });
  await browser.close();
  console.log('wrote', out);
})();
