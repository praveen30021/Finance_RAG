const { chromium } = require('playwright');
(async () => {
  const [src, out, hl, hr] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + src, { waitUntil: 'load' });
  await p.waitForTimeout(1200);
  await p.emulateMedia({ media: 'print' });
  const hdr = `<div style="width:100%;font-family:Georgia,serif;font-size:8.5pt;color:#111;padding:0 20mm;">
    <div style="display:flex;justify-content:space-between;border-bottom:0.6pt solid #111;padding-bottom:2px;">
      <span style="font-style:italic">${hl}</span><span style="font-style:italic">${hr}</span></div></div>`;
  await p.pdf({ path: out, format: 'A4', printBackground: true,
    margin: { top: '22mm', bottom: '18mm', left: '20mm', right: '20mm' },
    displayHeaderFooter: true,
    headerTemplate: hdr,
    footerTemplate: '<div style="width:100%;text-align:center;font-family:Georgia,serif;font-size:9pt;color:#111;"><span class="pageNumber"></span></div>' });
  await b.close();
  console.log('ok ->', out);
})();
