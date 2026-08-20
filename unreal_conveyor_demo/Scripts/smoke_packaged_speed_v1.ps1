[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ClientExecutable,
    [ValidateSet("evk", "host")]
    [string]$Backend = "evk",
    [string]$EvkServer = "http://192.168.1.158:18183",
    [string]$EvkModel = "local/cosmos-reason2-2b",
    [string]$HostServer = "http://127.0.0.1:18084",
    [string]$HostModel = "Cosmos-Reason2-2B-Parcel-Speed-v1",
    [string]$Output = "",
    [int]$TimeoutSeconds = 240,
    [switch]$UseRuntimeDefaults
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ClientExecutable = [System.IO.Path]::GetFullPath($ClientExecutable)
if (-not (Test-Path -LiteralPath $ClientExecutable -PathType Leaf)) {
    throw "Packaged client not found: $ClientExecutable"
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path (Split-Path -Parent $PSScriptRoot) `
        "..\docs\evidence\results\packaged_speed_v1_evk_smoke.json"
}
$Output = [System.IO.Path]::GetFullPath($Output)
$OutputDirectory = Split-Path -Parent $Output
[System.IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null

$SettingsRoot = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "QaiConveyorDemo"
$LogDirectory = Join-Path $SettingsRoot "Logs"
$SavedDirectory = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "QaiConveyor\Saved"
$CrashDirectory = Join-Path $SavedDirectory "Crashes"
$UserSettingsPath = Join-Path $SavedDirectory "Config\Windows\GameUserSettings.ini"
$UserSettingsExisted = Test-Path -LiteralPath $UserSettingsPath -PathType Leaf
[byte[]]$UserSettingsBytes = if ($UserSettingsExisted) {
    [System.IO.File]::ReadAllBytes($UserSettingsPath)
} else {
    @()
}
$CrashCountBefore = @(Get-ChildItem -LiteralPath $CrashDirectory -Directory -ErrorAction SilentlyContinue).Count
$StartedAt = [DateTime]::UtcNow
$Process = $null

try {
    $Arguments = @(
        "-QaiParcelEvaluation",
        "-QaiParcelEvaluationAutoExit",
        "-QaiParcelEvaluationSoloParcel",
        "-QaiParcelEvaluationMatchedBoundarySweeps",
        "-QaiParcelEvaluationCasesPerSignal=3",
        "-QaiParcelEvaluationCameraPolicy=runtime-exact",
        "-windowed",
        "-ResX=1280",
        "-ResY=720",
        "-NoSaveConfig",
        "-Unattended"
    )
    if (-not $UseRuntimeDefaults) {
        $Arguments += @("-Backend=$Backend", "-QaiParcelPrompt=speed-v1")
        if ($Backend -eq "evk") {
            $Arguments += @(
                "-EvkServer=$EvkServer",
                "-EvkModel=$EvkModel",
                "-QaiEvkCaptureWidth=448",
                "-QaiEvkCaptureHeight=256"
            )
        } else {
            $Arguments += @(
                "-HostServer=$HostServer",
                "-HostModel=$HostModel",
                "-QaiHostCaptureWidth=448",
                "-QaiHostCaptureHeight=256"
            )
        }
    }
    $Process = Start-Process -FilePath $ClientExecutable -ArgumentList $Arguments -PassThru
    $Deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while (-not $Process.HasExited -and [DateTime]::UtcNow -lt $Deadline) {
        Start-Sleep -Milliseconds 500
        $Process.Refresh()
    }
    if (-not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force
        throw "Packaged speed-v1 smoke exceeded $TimeoutSeconds seconds."
    }
    if ($Process.ExitCode -ne 0) {
        throw "Packaged speed-v1 smoke exited with code $($Process.ExitCode)."
    }

    $CrashCountAfter = @(Get-ChildItem -LiteralPath $CrashDirectory -Directory -ErrorAction SilentlyContinue).Count
    if ($CrashCountAfter -ne $CrashCountBefore) {
        throw "Packaged speed-v1 smoke created a crash report."
    }
    $Log = Get-ChildItem -LiteralPath $LogDirectory -Filter "simulator-*.log" -File -ErrorAction Stop |
        Where-Object { $_.LastWriteTimeUtc -ge $StartedAt } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if (-not $Log) {
        throw "Packaged client did not create a fresh simulator log."
    }
    $Lines = Get-Content -LiteralPath $Log.FullName
    $Configured = @($Lines | Where-Object {
        $_ -match "runtime_configured backend=$Backend" -and
        $_ -match "$($Backend)_capture=448x256"
    })
    $Results = @($Lines | Where-Object { $_ -match "parcel_evaluation_result " })
    $Complete = @($Lines | Where-Object { $_ -match "parcel_evaluation_complete .* cases=9 " })
    $Failures = @($Lines | Where-Object {
        $_ -match "inference_(failed|invalid_response|request_start_failed)"
    })
    $Incorrect = @($Results | Where-Object { $_ -notmatch " correct=true " })
    if ($Configured.Count -lt 1) {
        throw "Client log does not confirm $Backend speed-v1 at 448x256."
    }
    if ($Results.Count -ne 9 -or $Incorrect.Count -ne 0 -or $Complete.Count -ne 1) {
        throw "Client inference gate failed: results=$($Results.Count), incorrect=$($Incorrect.Count), complete=$($Complete.Count)."
    }
    if ($Failures.Count -ne 0) {
        throw "Client inference gate logged $($Failures.Count) inference failures."
    }

    $Hash = (Get-FileHash -LiteralPath $ClientExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
    $Evidence = [ordered]@{
        schema_version = 1
        passed = $true
        timestamp_utc = [DateTime]::UtcNow.ToString("o")
        client = $ClientExecutable
        package_sha256 = $Hash
        backend = $Backend
        server = if ($Backend -eq "evk") { $EvkServer } else { $HostServer }
        model = if ($Backend -eq "evk") { $EvkModel } else { $HostModel }
        prompt_profile = "speed-v1"
        image_width = 448
        image_height = 256
        runtime_defaults = [bool]$UseRuntimeDefaults
        cases = $Results.Count
        correct = $Results.Count - $Incorrect.Count
        simulator_log = $Log.FullName
        result_lines = $Results
    }
    $Evidence | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $Output -Encoding utf8
    Write-Host "Packaged $Backend speed-v1 smoke passed: $Output"
} finally {
    if ($Process -and -not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    }
    if ($UserSettingsExisted) {
        [System.IO.File]::WriteAllBytes($UserSettingsPath, $UserSettingsBytes)
    } elseif (Test-Path -LiteralPath $UserSettingsPath -PathType Leaf) {
        Remove-Item -LiteralPath $UserSettingsPath -Force
    }
}
