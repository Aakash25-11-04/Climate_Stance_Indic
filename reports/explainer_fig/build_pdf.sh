#!/usr/bin/env bash
# Rebuild PROJECT_EXPLAINED.pdf from PROJECT_EXPLAINED.md (needs pandoc and Microsoft Edge). Run from the repo root.
set -e
python reports/explainer_fig/make_ranking.py
tail -n +2 PROJECT_EXPLAINED.md | pandoc -f markdown-implicit_figures -s --toc --toc-depth=2 --embed-resources --resource-path=. \
  --css reports/explainer_fig/style.css \
  --metadata title="The Indic Climate Stance project, explained simply" \
  --metadata subtitle="What we built, why, how, and how every model performed" \
  --metadata date="November 2026" -o reports/explainer_fig/project_explained.html
"/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$(cygpath -w "$PWD/PROJECT_EXPLAINED.pdf")" "file:///$(cygpath -m "$PWD/reports/explainer_fig/project_explained.html")"
