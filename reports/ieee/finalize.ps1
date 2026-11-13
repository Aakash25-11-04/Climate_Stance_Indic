param([string]$In, [string]$OutDocx, [string]$OutPdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($In, $false, $false)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    $doc.SaveAs2($OutDocx, 16)          # wdFormatDocumentDefault (.docx)
    $doc.ExportAsFixedFormat($OutPdf, 17) # wdExportFormatPDF
    "pages: " + $doc.ComputeStatistics(2)
    $doc.Close($false)
} finally {
    $word.Quit()
}
