# Revision Docs

Placement-interview revision guides for Python data visualization.

| File | Pages | Covers |
|---|---|---|
| `CatBoost_LightGBM_Revision_Guide.pdf` | 17 | CatBoost (ordered target statistics, ordered boosting, oblivious trees), LightGBM (histogram split finding, GOSS, EFB, leaf-wise growth), and a three-way comparison with XGBoost — theory, derivations, worked examples, mind maps, 16 Q&A |
| `Python_Data_Master_Revision_Guide.pdf` | 42 | **All four topics in one doc** — mindmaps, NumPy, Pandas, Matplotlib, Seaborn as function→use→code tables, 45 Q&A, master cheat sheet |
| `Matplotlib_Complete_Revision_Guide.pdf` | 26 | Architecture, Figure/Axes/Axis, pyplot vs OO, all chart types, subplots & GridSpec, colormaps, annotations, 3D, saving, animation, cheat sheet, 31 interview Q&A |
| `Seaborn_Complete_Revision_Guide.pdf` | 22 | Figure-level vs axes-level, long/wide data, relational, categorical, distribution, joint/pair, heatmap/clustermap, regression, FacetGrid, styling, cheat sheet, 30 interview Q&A |

## Rebuilding the PDFs

The PDFs are generated from the HTML sources in `src/` using headless Chromium.

```bash
cd revision_docs/src
node topdf.js "$PWD/matplotlib.html" ../Matplotlib_Complete_Revision_Guide.pdf
node topdf.js "$PWD/seaborn.html"    ../Seaborn_Complete_Revision_Guide.pdf
node topdf.js "$PWD/master.html"     ../Python_Data_Master_Revision_Guide.pdf
```

Edit the `.html` files to change content; `style.css` controls the shared look.

## Boosting guide (LaTeX-style)

`CatBoost_LightGBM_Revision_Guide.pdf` is built from `src/boost.html` + `src/tex.css` and
needs KaTeX for the maths. Fetch KaTeX once into `src/katex/`:

```bash
cd revision_docs/src
npm pack katex@0.16.11 && tar -xzf katex-0.16.11.tgz
mkdir -p katex && cp package/dist/katex.min.{css,js} katex/ \
  && cp -r package/dist/fonts katex/ && cp package/dist/contrib/auto-render.min.js katex/
node topdf_tex.js "$PWD/boost.html" ../CatBoost_LightGBM_Revision_Guide.pdf \
  "CatBoost, LightGBM &amp; XGBoost" "Revision Guide"
```
