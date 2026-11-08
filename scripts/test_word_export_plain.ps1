$ErrorActionPreference = 'Stop'
$docx = 'D:\Python_File\EEF_analyze\outputs\teacher_report_20260908_plain\teacher_report_plain_20260908.docx'
$pdf = 'D:\Python_File\EEF_analyze\plain_tmp\teacher_report_plain_test.pdf'
Write-Output 'BEFORE'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
Write-Output 'APP'
$doc = $word.Documents.Open($docx, $false, $true)
Write-Output 'OPEN'
$doc.ExportAsFixedFormat($pdf, 17)
Write-Output 'EXPORTED'
$doc.Close($false)
$word.Quit()
Write-Output 'QUIT'
Write-Output (Test-Path $pdf)
