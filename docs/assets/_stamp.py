"""Add running footers, page numbers, bookmarks and metadata to a rendered book PDF.
usage: python3 _stamp.py <raw.pdf> <out.pdf> "<Book title>" "<Series label>" """
import io, re, os, sys
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

SRC, DST = sys.argv[1], sys.argv[2]
TITLE  = sys.argv[3] if len(sys.argv) > 3 else "Ensemble Learning"
SERIES = sys.argv[4] if len(sys.argv) > 4 else ""
pdfmetrics.registerFont(TTFont("DJV",  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DJVB", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))

reader = PdfReader(SRC); N = len(reader.pages)
CH = re.compile(r"(?:CHAPTER|PART|APPENDIX)\s+([0-9A-Z]+)\s*[—–-]\s*([^\n]{0,58})")
chap, starts, cur = [None]*N, [], None
for i, pg in enumerate(reader.pages):
    try: txt = pg.extract_text() or ""
    except Exception: txt = ""
    m = CH.search(txt[:420])
    if m:
        lab = re.sub(r"\s+", " ", f"{m.group(1)} — {m.group(2).strip()}").strip(" .")
        if lab != cur: cur = lab; starts.append((i, lab))
    chap[i] = cur

W, H = A4
w = PdfWriter()
for i, page in enumerate(reader.pages):
    if i > 0:
        buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=(W, H)); y = 26
        c.setStrokeColorRGB(.84,.87,.90); c.setLineWidth(.6); c.line(36, y+12, W-36, y+12)
        c.setFont("DJV", 7.3); c.setFillColorRGB(.55,.58,.62)
        c.drawString(36, y, TITLE[:56])
        if chap[i]: c.drawCentredString(W/2, y, chap[i][:56])
        c.setFont("DJVB", 7.7); c.setFillColorRGB(.15,.30,.72)
        c.drawRightString(W-36, y, f"{i}  /  {N-1}")
        c.save(); buf.seek(0)
        page.merge_page(PdfReader(buf).pages[0])
    w.add_page(page)

for p in w.pages:
    try: p.compress_content_streams(level=9)
    except Exception: pass
try: w.compress_identical_objects()
except Exception: pass

w.add_outline_item("Cover", 0)
if N > 1: w.add_outline_item("Contents", 1)
for idx, lab in starts: w.add_outline_item(lab, idx)
w.add_metadata({"/Title": TITLE, "/Subject": SERIES,
    "/Keywords": "ensemble learning, bagging, boosting, random forest, adaboost, "
                 "gradient boosting, xgboost, lightgbm, catboost, stacking, ML interview",
    "/Creator": "pandoc + Chromium + pypdf"})
w.page_mode = "/UseOutlines"
with open(DST, "wb") as f: w.write(f)
print(f"   {os.path.basename(DST)}  ·  {N} pages  ·  {len(starts)} bookmarks  ·  {os.path.getsize(DST)/1e6:.2f} MB")
