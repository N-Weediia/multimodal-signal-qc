$ErrorActionPreference = 'Stop'
$docx = (Resolve-Path 'D:\Python_File\EEF_analyze\outputs\teacher_report_20260908_plain\teacher_report_plain_20260908.docx').Path
$outDir = 'D:\Python_File\EEF_analyze\outputs\teacher_report_20260908_plain\rendered_pages'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$tmpPdf = 'D:\Python_File\EEF_analyze\plain_tmp\teacher_report_plain_20260908_word.pdf'
$pdf = Join-Path $outDir 'teacher_report_plain_20260908_word.pdf'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $word.Documents.Open($docx, $false, $true)
$doc.ExportAsFixedFormat($tmpPdf, 17)
$doc.Close($false)
$word.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
Copy-Item -LiteralPath $tmpPdf -Destination $pdf -Force
Write-Output $pdf
