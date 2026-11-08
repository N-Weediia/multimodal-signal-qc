$ErrorActionPreference = 'Stop'
try {
    $word = [Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application')
    Write-Output ('Word version: ' + $word.Version)
    Write-Output ('Documents: ' + $word.Documents.Count)
    foreach ($d in $word.Documents) {
        Write-Output ('DOC: ' + $d.FullName + ' | ReadOnly=' + $d.ReadOnly)
    }
} catch {
    Write-Output ('NO_ACTIVE_WORD: ' + $_.Exception.Message)
}
