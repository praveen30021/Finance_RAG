#!/usr/bin/env bash
# Rebuild Ensemble_Learning_Interview_Revision.pdf from the markdown source.
#   requires: pandoc, a Chromium binary, python3 with pypdf + reportlab
# usage: bash docs/build-pdf.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MD=docs/Ensemble_Learning_Interview_Revision.md
PDF=docs/Ensemble_Learning_Interview_Revision.pdf
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

CHROME=${CHROME:-$(command -v chromium || command -v chromium-browser || command -v google-chrome \
        || find /opt/pw-browsers -maxdepth 3 -name chrome -type f 2>/dev/null | head -1)}
[ -x "$CHROME" ] || { echo "No Chromium found. Set CHROME=/path/to/chrome"; exit 1; }

echo "1/4  markdown -> html"
pandoc "$MD" --from=gfm --to=html5 --wrap=preserve -o "$TMP/body.html"

echo "2/4  wrapping with print stylesheet + cover"
python3 - "$TMP" <<'PY'
import io, sys, datetime
tmp = sys.argv[1]
body = io.open(tmp + "/body.html", encoding="utf-8").read()
css  = io.open("docs/assets/pdf.css", encoding="utf-8").read()
body = body.replace('<h1 id=', '<h1 class="notop" id=', 1)
cover = """
<div class="cover">
  <div class="kicker">PLACEMENT INTERVIEW REVISION</div>
  <h1>Ensemble Learning</h1>
  <div class="rule"></div>
  <div class="sub">Theory · Architecture · Derivations · Diagrams · Formulas · Interview Q&amp;A<br>
  Everything from Voting to CatBoost, in one place.</div>
  <div class="chips">
    <span class="chip">Voting</span><span class="chip">Bagging</span><span class="chip">Random Forest</span>
    <span class="chip">AdaBoost</span><br>
    <span class="chip">Gradient Boosting</span><span class="chip">XGBoost</span>
    <span class="chip">LightGBM</span><span class="chip">CatBoost</span><span class="chip">Stacking</span>
  </div>
  <div class="foot">13 parts &nbsp;·&nbsp; 36 diagrams &nbsp;·&nbsp; 67 rapid-fire questions &nbsp;·&nbsp; %s</div>
</div>
""" % datetime.date.today().strftime("%d %B %Y")
io.open("docs/_print.html", "w", encoding="utf-8").write(
    f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
    f'<title>Ensemble Learning — Interview Revision</title><style>{css}</style></head>'
    f'<body>{cover}{body}</body></html>')
PY

echo "3/4  html -> pdf (Chromium)"
"$CHROME" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf="$TMP/raw.pdf" --virtual-time-budget=25000 \
  "file://$PWD/docs/_print.html" 2>/dev/null
rm -f docs/_print.html

echo "4/4  page numbers, footers, bookmarks"
python3 docs/assets/_stamp.py "$TMP/raw.pdf" "$PDF"
echo "done -> $PDF"
