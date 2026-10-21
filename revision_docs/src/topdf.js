const { chromium } = require('playwright');
(async () => {
  const [src, out] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + src, { waitUntil: 'load' });
  await p.emulateMedia({ media: 'print' });
  await p.pdf({ path: out, format: 'A4', printBackground: true,
    margin: { top: '14mm', bottom: '16mm', left: '13mm', right: '13mm' },
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate: '<div style="width:100%;font:9px -apple-system,Segoe UI,Roboto,sans-serif;color:#8a94a6;padding:0 13mm;display:flex;justify-content:space-between;"><span class="title"></span><span>Page <span class="pageNumber"></span> / <span class="totalPages"></span></span></div>' });
  await b.close();
  console.log('ok ->', out);
})();
