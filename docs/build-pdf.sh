#!/usr/bin/env bash
# Build the Ensemble Learning revision books (markdown -> PDF).
#   requires: pandoc, a Chromium binary, python3 with pypdf + reportlab
# usage: bash docs/build-pdf.sh [book1|book2|book3|all]
set -euo pipefail
cd "$(dirname "$0")/.."
WHICH=${1:-all}

CHROME=${CHROME:-$(command -v chromium || command -v chromium-browser || command -v google-chrome \
        || find /opt/pw-browsers -maxdepth 3 -name chrome -type f 2>/dev/null | head -1)}
[ -x "$CHROME" ] || { echo "No Chromium found. Set CHROME=/path/to/chrome"; exit 1; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP" docs/_print.html' EXIT

build () {                       # $1 slug  $2 title  $3 subtitle  $4 book-number  $5 chips
  local md="docs/$1.md" pdf="docs/$1.pdf"
  [ -f "$md" ] || { echo "skip $md (missing)"; return; }
  echo "building $1 ..."
  pandoc "$md" --from=gfm+tex_math_dollars+fenced_divs --to=html5 --mathml \
         --highlight-style=tango --wrap=preserve -o "$TMP/body.html"
  BOOKNO="$4" TITLE="$2" SUBTITLE="$3" CHIPS="$5" python3 - "$TMP" <<'PY'
import io, os, re, sys, datetime
tmp = sys.argv[1]
body = io.open(tmp + "/body.html", encoding="utf-8").read()
css  = io.open("docs/assets/pdf.css", encoding="utf-8").read()
body = body.replace('<h1 id=', '<h1 class="notop" id=', 1)

# --- stop pandoc's blanket stretchy="true" from inflating ordinary parentheses.
#     Keep big delimiters only where the LaTeX actually asked for them.
MATH = re.compile(r"<math\b.*?</math>", re.S)
SIZED = re.compile(r"\\(left|right|middle|bigg?|Bigg?)\b")
def tame_delims(html):
    def fix(m):
        blk = m.group(0)
        ann = re.search(r"<annotation[^>]*>(.*?)</annotation>", blk, re.S)
        if ann and SIZED.search(ann.group(1)):
            return blk                      # author asked for sized delimiters: leave alone
        return blk.replace('stretchy="true"', 'stretchy="false"')
    return MATH.sub(fix, html)
body = tame_delims(body)

chips = "".join(f'<span class="chip">{c.strip()}</span>'
                for c in os.environ["CHIPS"].split(",") if c.strip())
cover = f"""<div class="cover">
  <div class="kicker">PLACEMENT INTERVIEW REVISION</div>
  <div class="bookno">{os.environ['BOOKNO']}</div>
  <h1>{os.environ['TITLE']}</h1>
  <div class="rule"></div>
  <div class="sub">{os.environ['SUBTITLE']}</div>
  <div class="chips">{chips}</div>
  <div class="series">A three-book series &nbsp;·&nbsp;
    <b>Book 1</b> Foundations &amp; Bagging &nbsp;·&nbsp;
    <b>Book 2</b> Boosting &nbsp;·&nbsp;
    <b>Book 3</b> LightGBM, CatBoost &amp; Stacking</div>
  <div class="foot">{datetime.date.today().strftime('%d %B %Y')}</div>
</div>"""
io.open("docs/_print.html", "w", encoding="utf-8").write(
  f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{os.environ["TITLE"]}</title>'
  f'<style>{css}</style></head><body>{cover}{body}</body></html>')
PY
  "$CHROME" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
    --print-to-pdf="$TMP/raw.pdf" --virtual-time-budget=30000 \
    "file://$PWD/docs/_print.html" 2>/dev/null
  python3 docs/assets/_stamp.py "$TMP/raw.pdf" "$pdf" "$2" "Ensemble Learning revision series"
}

B1=(Ensemble_1_Foundations_and_Bagging "Foundations & Bagging" \
    "Why ensembles work · Bias–Variance · Voting · Bagging · Random Forest" 1 \
    "Ensemble basics,Bias–Variance,Voting,Bagging,Random Forest,Extra Trees")
B2=(Ensemble_2_Boosting "Boosting" \
    "AdaBoost · Gradient Boosting · XGBoost — every derivation worked out" 2 \
    "AdaBoost,Gradient Boosting,XGBoost,Taylor derivation")
B3=(Ensemble_3_LightGBM_CatBoost_Stacking "LightGBM, CatBoost & Stacking" \
    "The modern boosters in depth · Stacking · Tuning · Cheat sheet · Q&amp;A" 3 \
    "LightGBM,CatBoost,Stacking,Blending,Tuning,Cheat sheet,Q&amp;A")

case "$WHICH" in
  book1) build "${B1[@]}" ;;
  book2) build "${B2[@]}" ;;
  book3) build "${B3[@]}" ;;
  all)   build "${B1[@]}"; build "${B2[@]}"; build "${B3[@]}" ;;
  *) echo "usage: bash docs/build-pdf.sh [book1|book2|book3|all]"; exit 1 ;;
esac
echo "done."
