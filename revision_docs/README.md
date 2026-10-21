# Revision Docs

Placement-interview revision guides for Python data visualization.

| File | Pages | Covers |
|---|---|---|
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
