param(
    [string]$Source,
    [string]$Destination,
    [int]$FilesPerArchive = 500
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
if ($FilesPerArchive -lt 1) { throw 'FilesPerArchive must be positive.' }
$files = @(Get-ChildItem -LiteralPath $Source -Filter '*.xlsx' -File | Sort-Object Name)
if ($files.Count -eq 0) { throw 'No hourly workbooks found.' }
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$records = @()
for ($offset = 0; $offset -lt $files.Count; $offset += $FilesPerArchive) {
    $number = [int]($offset / $FilesPerArchive) + 1
    $name = 'hourly-data-{0:D2}.zip' -f $number
    $target = Join-Path $Destination $name
    if (Test-Path -LiteralPath $target) { throw "Archive already exists: $name" }
    $zip = [System.IO.Compression.ZipFile]::Open($target, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        $end = [Math]::Min($offset + $FilesPerArchive, $files.Count)
        for ($i = $offset; $i -lt $end; $i++) {
            $f = $files[$i]
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $f.FullName, $f.Name, [System.IO.Compression.CompressionLevel]::NoCompression) | Out-Null
            $records += [pscustomobject]@{building_id=$f.BaseName;filename=$f.Name;archive=$name;size_bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
        }
    } finally { $zip.Dispose() }
    Write-Output "$name complete ($($end - $offset) workbooks)"
}
$records | Export-Csv -LiteralPath (Join-Path $Destination 'hourly_archive_manifest.csv') -NoTypeInformation -Encoding utf8
Get-ChildItem -LiteralPath $Destination -Filter '*.zip' -File | ForEach-Object {
    [pscustomobject]@{archive=$_.Name;size_bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
} | Export-Csv -LiteralPath (Join-Path $Destination 'archive_checksums.csv') -NoTypeInformation -Encoding utf8
