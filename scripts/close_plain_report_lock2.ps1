$ErrorActionPreference = 'Stop'
$word = [Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
Write-Output ('Before=' + $word.Documents.Count)
if ($word.Documents.Count -gt 0) {
    $d = $word.Documents.Item(1)
    Write-Output ('Path=' + $d.FullName)
    $d.Close(0)
    Write-Output ('AfterClose=' + $word.Documents.Count)
    [Runtime.InteropServices.Marshal]::ReleaseComObject($d) | Out-Null
}
$word.Quit(0)
[Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
[GC]::Collect()
[GC]::WaitForPendingFinalizers()
Write-Output 'Done'
