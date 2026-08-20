[CmdletBinding()]
param(
    [string]$ClientExecutable = "",
    [string[]]$ClientArguments = @(),
    [int]$HostReadyTimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$ExpectedModel = "Cosmos-Reason2-2B-Parcel-Speed-v1"
$HostUrl = "http://127.0.0.1:18084"
$ModelDirectory = Join-Path $ProjectDirectory "Build\ReleasePayloadSource\Models\ParcelSpeedV1"
$Model = Join-Path $ModelDirectory "Cosmos-Reason2-2B-Parcel-Speed-v1-Q8_0.gguf"
$Projector = Join-Path $ModelDirectory "mmproj-Cosmos-Reason2-2B-Parcel-Speed-v1-F16.gguf"
$Runtime = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) `
    "QaiConveyorDemo\Runtime\Windows\cuda"
$Server = Join-Path $Runtime "llama-server.exe"
$LogDirectory = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) `
    "QaiConveyorDemo\Logs\Servers"

function Test-HostReady {
    try {
        $Models = Invoke-RestMethod -Uri "$HostUrl/v1/models" -TimeoutSec 3
        return ($Models | ConvertTo-Json -Depth 8 -Compress) -match [regex]::Escape($ExpectedModel)
    } catch {
        return $false
    }
}

function Resolve-ClientExecutable {
    if (-not [string]::IsNullOrWhiteSpace($ClientExecutable)) {
        return [System.IO.Path]::GetFullPath($ClientExecutable)
    }
    $PackagedRoot = Join-Path $ProjectDirectory "Saved\Packaged"
    $Candidate = Get-ChildItem -LiteralPath $PackagedRoot -Filter "QaiConveyor.exe" `
            -File -Recurse -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if (-not $Candidate) {
        throw "No packaged QaiConveyor.exe found below $PackagedRoot."
    }
    return $Candidate.FullName
}

if (-not (Test-HostReady)) {
    foreach ($RequiredPath in @($Server, $Model, $Projector)) {
        if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
            throw "Missing GPU deployment artifact: $RequiredPath. Run the release setup/provisioner first."
        }
    }

    $ExpectedHashes = @{
        $Model = "106fe2e4a8ef60677a168dc44a45bd69fccc4a28f549925ba1c72d9e1561e85c"
        $Projector = "2fbf583701911b1ac67a03ee1ed03c066ae8f83aba1b3df31c7386ac96cd260a"
    }
    foreach ($Artifact in @($Model, $Projector)) {
        $ActualHash = (Get-FileHash -LiteralPath $Artifact -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne $ExpectedHashes[$Artifact]) {
            throw "GPU deployment artifact hash mismatch: $Artifact ($ActualHash)"
        }
    }

    New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
    $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $Stdout = Join-Path $LogDirectory "host-speed-v1-$Stamp.out.log"
    $Stderr = Join-Path $LogDirectory "host-speed-v1-$Stamp.err.log"
    $Arguments = @(
        "-m", "`"$Model`"",
        "--mmproj", "`"$Projector`"",
        "-ngl", "all",
        "-c", "512",
        "--host", "127.0.0.1",
        "--port", "18084",
        "--alias", $ExpectedModel,
        "--image-min-tokens", "112",
        "--image-max-tokens", "112",
        "--flash-attn", "on",
        "--no-cache-prompt",
        "--slot-prompt-similarity", "0"
    )
    $HostProcess = Start-Process -FilePath $Server -ArgumentList $Arguments `
        -WorkingDirectory $Runtime -WindowStyle Hidden `
        -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru

    $Deadline = [DateTime]::UtcNow.AddSeconds($HostReadyTimeoutSeconds)
    while (-not (Test-HostReady) -and [DateTime]::UtcNow -lt $Deadline) {
        $HostProcess.Refresh()
        if ($HostProcess.HasExited) {
            $Tail = (Get-Content -LiteralPath $Stderr -Tail 120 -ErrorAction SilentlyContinue) -join "`n"
            throw "GPU model server exited with code $($HostProcess.ExitCode):`n$Tail"
        }
        Start-Sleep -Seconds 1
    }
    if (-not (Test-HostReady)) {
        throw "GPU model server did not become ready within $HostReadyTimeoutSeconds seconds."
    }
    Write-Host "GPU model ready: $ExpectedModel ($HostUrl), PID $($HostProcess.Id)"
} else {
    Write-Host "GPU model already ready: $ExpectedModel ($HostUrl)"
}

$ResolvedClient = Resolve-ClientExecutable
if (-not (Test-Path -LiteralPath $ResolvedClient -PathType Leaf)) {
    throw "Packaged client not found: $ResolvedClient"
}
$ExistingClient = Get-Process -Name "QaiConveyor", "QaiConveyor-Win64-Shipping" `
    -ErrorAction SilentlyContinue
if ($ExistingClient) {
    Write-Host "QaiConveyor is already running (PID(s): $($ExistingClient.Id -join ', '))."
    return
}

$Client = Start-Process -FilePath $ResolvedClient -ArgumentList $ClientArguments `
    -WorkingDirectory (Split-Path -Parent $ResolvedClient) -PassThru
Write-Host "Demo launched: $ResolvedClient (PID $($Client.Id))"
