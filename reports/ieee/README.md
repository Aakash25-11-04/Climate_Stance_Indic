# IEEE-format technical report: build files

Produces `reports/technical_report_ieee.docx` and `.pdf` (two columns, US Letter, Times New Roman 10 pt).

```bash
python reports/ieee/figures.py . reports/ieee/fig        # figures from results/
cd reports/ieee && npm install docx@9 && node build.js report_raw.docx
powershell -File finalize.ps1 -In <abs>\report_raw.docx -OutDocx <abs>\technical_report_ieee.docx -OutPdf <abs>\technical_report_ieee.pdf
```

`finalize.ps1` uses Microsoft Word to fill in the table of contents and export the PDF.
All numbers in `build.js` are copied from `results/` (see `reports/technical_report.md`).
