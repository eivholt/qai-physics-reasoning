[CmdletBinding()]
param(
    [string]$IsaacSimRoot = 'C:\NVIDIA\isaac-sim-standalone-6.0.1',
    [switch]$Refresh,
    [switch]$VerifyExisting
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$collector = Join-Path $PSScriptRoot 'collect_omniverse_assets.py'
$isaacPython = Join-Path $IsaacSimRoot 'python.bat'

if (-not (Test-Path -LiteralPath $isaacPython -PathType Leaf)) {
    throw "Isaac Sim Python was not found at $isaacPython"
}

$arguments = @($collector, '--headless')
if ($Refresh) {
    $arguments += '--refresh'
}
if ($VerifyExisting) {
    $arguments += '--verify-existing'
}

& $isaacPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Omniverse asset collection failed with exit code $LASTEXITCODE"
}

$manifest = Join-Path $projectRoot 'Content\OmniversePayload\asset_manifest.json'
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "Asset collection did not create $manifest"
}
Write-Host "Omniverse payload is ready: $manifest"
