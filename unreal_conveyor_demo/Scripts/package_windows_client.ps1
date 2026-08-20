[CmdletBinding()]
param(
    [string]$UnrealRoot = "C:\Program Files\Epic Games\UE_5.8",
    [string]$ArchiveDirectory = "",
    [switch]$SkipBuild,
    [switch]$SkipCook,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectDirectory = Split-Path -Parent $PSScriptRoot
$ProjectFile = Join-Path $ProjectDirectory "QaiConveyor.uproject"
$BuildScript = Join-Path $UnrealRoot "Engine\Build\BatchFiles\Build.bat"
$EditorCommandlet = Join-Path $UnrealRoot "Engine\Binaries\Win64\UnrealEditor-Cmd.exe"
$AutomationTool = Join-Path $UnrealRoot "Engine\Build\BatchFiles\RunUAT.bat"

if ([string]::IsNullOrWhiteSpace($ArchiveDirectory)) {
    $ArchiveDirectory = Join-Path $ProjectDirectory "Saved\Packaged\Win64"
}
$ArchiveDirectory = [System.IO.Path]::GetFullPath($ArchiveDirectory)

foreach ($RequiredPath in @($ProjectFile, $BuildScript, $EditorCommandlet, $AutomationTool)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Required file not found: $RequiredPath"
    }
}

function Invoke-UnrealTool {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Unreal tool failed with exit code $LASTEXITCODE`: $Executable"
    }
}

if (-not $SkipBuild) {
    Write-Host "Building the Win64 Shipping client..."
    Invoke-UnrealTool -Executable $BuildScript -Arguments @(
        "QaiConveyor",
        "Win64",
        "Shipping",
        "-Project=$ProjectFile",
        "-WaitMutex"
    )
}

if (-not $SkipCook) {
    Write-Host "Cooking content with the UE 5.8.1 timing-assertion workaround..."
    # UE 5.8.1 can produce non-monotonic sub-microsecond DDC timestamps when
    # its commandlet worker pool is active on this machine. Restrict only the
    # headless cook to one engine thread; the packaged game remains normally
    # threaded and retains DX12, SM6, hardware ray tracing, and hardware Lumen.
    Invoke-UnrealTool -Executable $EditorCommandlet -Arguments @(
        $ProjectFile,
        "-run=Cook",
        "-TargetPlatform=Windows",
        "-unversioned",
        "-nothreading",
        "-unattended",
        "-NoLogTimes",
        "-stdout",
        "-CrashForUAT"
    )
}

Write-Host "Staging and archiving the Win64 Shipping client..."
Invoke-UnrealTool -Executable $AutomationTool -Arguments @(
    "BuildCookRun",
    "-project=$ProjectFile",
    "-noP4",
    "-platform=Win64",
    "-clientconfig=Shipping",
    "-skipbuild",
    "-skipcook",
    "-stage",
    "-pak",
    "-archive",
    "-archivedirectory=$ArchiveDirectory",
    "-unattended",
    "-utf8output"
)

$ClientCandidates = @(
    (Join-Path $ArchiveDirectory "Windows\QaiConveyor.exe"),
    (Join-Path $ArchiveDirectory "QaiConveyor.exe"),
    (Join-Path $ArchiveDirectory "QaiConveyor\Binaries\Win64\QaiConveyor-Win64-Shipping.exe")
)
$ClientExecutable = $ClientCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $ClientExecutable) {
    throw "Packaged client was not created in any expected archive layout: $($ClientCandidates -join ', ')"
}

if ($SmokeTest) {
    $SavedDirectory = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "QaiConveyor\Saved"
    $ScreenshotPath = Join-Path $SavedDirectory "Screenshots\runtime-motion.png"
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

    Write-Host "Running the packaged RTX client smoke test..."
    try {
        $Process = Start-Process -FilePath $ClientExecutable -ArgumentList @(
            "-QaiRuntimeMotionTest",
            "-windowed",
            "-ResX=1280",
            "-ResY=720",
            "-NoSaveConfig",
            "-Unattended"
        ) -PassThru -Wait

        $CrashCountAfter = @(Get-ChildItem -LiteralPath $CrashDirectory -Directory -ErrorAction SilentlyContinue).Count
        $Screenshot = Get-Item -LiteralPath $ScreenshotPath -ErrorAction SilentlyContinue
        if ($Process.ExitCode -ne 0) {
            throw "Packaged client smoke test exited with code $($Process.ExitCode)."
        }
        if ($CrashCountAfter -ne $CrashCountBefore) {
            throw "Packaged client smoke test created a crash report."
        }
        if (-not $Screenshot -or $Screenshot.LastWriteTimeUtc -lt $StartedAt) {
            throw "Packaged client smoke test did not write a fresh runtime screenshot."
        }
    } finally {
        # Unreal persists command-line ResX/ResY during shutdown even with
        # -NoSaveConfig on some Shipping builds. Diagnostics must never alter
        # the next interactive launch, so restore the exact prior bytes.
        if ($UserSettingsExisted) {
            [System.IO.File]::WriteAllBytes($UserSettingsPath, $UserSettingsBytes)
        } elseif (Test-Path -LiteralPath $UserSettingsPath -PathType Leaf) {
            Remove-Item -LiteralPath $UserSettingsPath -Force
        }
    }
    Write-Host "Smoke test passed: $ScreenshotPath"
}

Write-Host "Packaged client: $ClientExecutable"
