param(
    [string]$EvkHost = "iq9075-evk",
    [string]$EvkUser = "ubuntu",
    [string]$GenieRoot = "/home/ubuntu/qai-conveyor/runtime/geniex-v0317-qairt245",
    [string]$Output = "",
    [string]$IdentityFile = ""
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $project "Build\ReleasePayloadSource\EVK\evk-geniex-runtime-v0317-qairt245.tar.gz"
}
$outputPath = [IO.Path]::GetFullPath($Output)
$outputParent = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputParent | Out-Null

foreach ($remotePath in @($GenieRoot)) {
    if ($remotePath -notmatch '^/home/ubuntu/[A-Za-z0-9._/-]+$' -or $remotePath.Contains('..')) {
        throw "Remote paths must be normalized children of /home/ubuntu: $remotePath"
    }
}
if ($EvkHost -notmatch '^[A-Za-z0-9.:-]+$' -or $EvkUser -notmatch '^[a-z_][a-z0-9_-]*$') {
    throw "Unsupported EVK host or user syntax."
}

$remote = "${EvkUser}@${EvkHost}"
$sshArgs = @()
$scpArgs = @()
if (-not [string]::IsNullOrWhiteSpace($IdentityFile)) {
    $identity = (Resolve-Path -LiteralPath $IdentityFile).Path
    $sshArgs += @('-i', $identity)
    $scpArgs += @('-i', $identity)
}
$identifier = [guid]::NewGuid().ToString('N')
$remoteArchive = "/home/ubuntu/qai-conveyor-release-${identifier}.tar.gz"
$rootName = Split-Path -Leaf $GenieRoot
$rootParent = $GenieRoot.Substring(0, $GenieRoot.Length - $rootName.Length).TrimEnd('/')

try {
    & ssh @sshArgs $remote "set -e; test -x '$GenieRoot/geniex-grammar'; curl -sf --max-time 5 http://127.0.0.1:18183/v1/models | grep -q 'local/cosmos-reason2-2b'"
    if ($LASTEXITCODE -ne 0) { throw "EVK runtime/model preflight failed." }

    $tarCommand = "set -e; tar -czf '$remoteArchive' --transform='s,^$rootName/,geniex/,' -C '$rootParent' '$rootName'; sha256sum '$remoteArchive'; du -h '$remoteArchive'"
    & ssh @sshArgs $remote $tarCommand
    if ($LASTEXITCODE -ne 0) { throw "EVK bundle creation failed." }

    $temporary = "${outputPath}.downloading"
    & scp @scpArgs "${remote}:${remoteArchive}" $temporary
    if ($LASTEXITCODE -ne 0) { throw "EVK bundle download failed." }
    Move-Item -Force -LiteralPath $temporary -Destination $outputPath

    $entries = @(& tar -tzf $outputPath)
    if ($LASTEXITCODE -ne 0 -or -not ($entries -contains 'geniex/geniex-grammar')) {
        throw "Collected EVK archive does not have the required geniex/ runtime layout."
    }
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath).Hash.ToLowerInvariant()
    Write-Host "EVK release bundle: $outputPath"
    Write-Host "bytes=$((Get-Item -LiteralPath $outputPath).Length) sha256=$hash"
}
finally {
    & ssh @sshArgs $remote "rm -f -- '$remoteArchive'" 2>$null
}
