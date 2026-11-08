$ErrorActionPreference = 'Stop'
$target = [System.IO.Path]::GetFullPath('D:\Python_File\EEF_analyze\outputs\teacher_report_20260908_plain\teacher_report_plain_20260908.docx')
$word = [Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
$closed = $false
foreach ($d in @($word.Documents)) {
    if ([System.IO.Path]::GetFullPath($d.FullName) -eq $target -and $d.ReadOnly) {
        $d.Close($false)
        $closed = $true
    }
}
if ($closed -and $word.Documents.Count -eq 0) { $word.Quit() }
Write-Output ('ClosedTarget=' + $closed)
