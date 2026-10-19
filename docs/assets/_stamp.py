import io, re, os, sys
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import sys
SRC, DST = sys.argv[1], sys.argv[2]
pdfmetrics.registerFont(TTFont("DJV",  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DJVB", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))

reader = PdfReader(SRC); N = len(reader.pages)
PART_RE = re.compile(r"PART\s+(\d+)\s*[—–-]\s*([^\n]{0,60})")
part_of, starts, cur = [None]*N, [], None
for i, pg in enumerate(reader.pages):
    try: txt = pg.extract_text() or ""
    except Exception: txt = ""
    m = PART_RE.search(txt[:400])
    if m:
        lab = re.sub(r"\s+", " ", f"Part {m.group(1)} — {m.group(2).strip()}").strip(" .")
        if lab != cur: cur = lab; starts.append((i, lab))
    part_of[i] = cur

W, H = A4
writer = PdfWriter()
for i, page in enumerate(reader.pages):
    if i > 0:
        buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=(W, H)); y = 26
        c.setStrokeColorRGB(.84,.87,.90); c.setLineWidth(.6); c.line(34, y+12, W-34, y+12)
        c.setFont("DJV", 7.4); c.setFillColorRGB(.55,.58,.62)
        c.drawString(34, y, "Ensemble Learning — Interview Revision")
        if part_of[i]: c.drawCentredString(W/2, y, part_of[i][:58])
        c.setFont("DJVB", 7.8); c.setFillColorRGB(.15,.30,.72)
        c.drawRightString(W-34, y, f"{i}  /  {N-1}")
        c.save(); buf.seek(0)
        page.merge_page(PdfReader(buf).pages[0])
    writer.add_page(page)

for p in writer.pages:
    try: p.compress_content_streams(level=9)
    except Exception: pass
try: writer.compress_identical_objects(remove_identicals=True, remove_orphans=True)
except Exception as e: print("  (dedupe skipped:", e, ")")

writer.add_outline_item("Cover", 0)
writer.add_outline_item("Title & Contents", 1)
for idx, lab in starts: writer.add_outline_item(lab, idx)
writer.add_metadata({
    "/Title": "Ensemble Learning — Complete Revision Notes for Placement Interviews",
    "/Subject": "Voting, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost, Stacking",
    "/Keywords": "ensemble learning, bagging, boosting, random forest, adaboost, gradient boosting, "
                 "xgboost, lightgbm, catboost, stacking, machine learning interview",
    "/Creator": "pandoc + Chromium + pypdf",
})
writer.page_mode = "/UseOutlines"
with open(DST, "wb") as f: writer.write(f)
print(f"{N} pages · {len(starts)} bookmarks · {os.path.getsize(DST)/1e6:.2f} MB")
