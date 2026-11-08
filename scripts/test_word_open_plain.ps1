$ErrorActionPreference = 'Stop'
Write-Output 'BEFORE'
$word = New-Object -ComObject Word.Application
Write-Output 'APP'
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $word.Documents.Open('D:\Python_File\EEF_analyze\outputs\teacher_report_20260908_plain\teacher_report_plain_20260908.docx', $false, $true)
Write-Output 'OPEN'
$doc.Close($false)
Write-Output 'CLOSE'
$word.Quit()
Write-Output 'QUIT'
