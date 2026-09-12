[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ClientExecutable,
    [int[]]$FrameCaps = @(30, 45, 60, 120),
    [ValidateSet("game_frame", "solver_step")][string[]]$Modes = @("game_frame", "solver_step"),
    [ValidateRange(5, 120)][int]$DurationSeconds = 15,
    [int]$Width = 2560,
    [int]$Height = 1440,
    [string]$Output = ""
)
$ErrorActionPreference = "Stop"
$ClientExecutable = (Resolve-Path -LiteralPath $ClientExecutable).Path
if (Get-Process -Name QaiConveyor,QaiConveyor-Win64-Shipping -ErrorAction SilentlyContinue) {
    throw "Close the interactive demo before running isolated physics tests."
}
if (-not $Output) {
    $Output = Join-Path $PSScriptRoot ("..\Saved\Diagnostics\physics-rates-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".json")
}
$Output = [IO.Path]::GetFullPath($Output)
[IO.Directory]::CreateDirectory((Split-Path -Parent $Output)) | Out-Null
$Settings = Join-Path $env:LOCALAPPDATA "QaiConveyor\Saved\Config\Windows\GameUserSettings.ini"
$HadSettings = Test-Path -LiteralPath $Settings
$SavedSettings = if ($HadSettings) { [IO.File]::ReadAllBytes($Settings) } else { $null }
$Results = @()
$Process = $null
try {
    foreach ($Cap in $FrameCaps) {
        foreach ($Mode in $Modes) {
            $Started = [DateTime]::UtcNow
            $Arguments = @("-windowed", "-noborder", "-ForceRes", "-ResX=$Width", "-ResY=$Height",
                "-RenderOffscreen", "-NoSaveConfig", "-Unattended", "-NoInference",
                "-QaiPhysicsRateCap=$Cap", "-QaiPhysicsRateTest=$DurationSeconds")
            if ($Mode -eq "solver_step") { $Arguments += "-QaiPhysicsStepForces" }
            # Run the Shipping binary directly so the PID is the tested process,
            # not the bootstrap executable that can exit before the game.
            $Process = Start-Process -FilePath $ClientExecutable -ArgumentList $Arguments -PassThru -WindowStyle Hidden
            $Deadline = [DateTime]::UtcNow.AddSeconds($DurationSeconds + 45)
            while (-not $Process.HasExited -and [DateTime]::UtcNow -lt $Deadline) {
                Start-Sleep -Milliseconds 250
                $Process.Refresh()
            }
            if (-not $Process.HasExited) { throw "Physics test timed out: $Mode/$Cap" }
            if ($Process.ExitCode -ne 0) { throw "Physics test exited $($Process.ExitCode): $Mode/$Cap" }
            $Log = Get-ChildItem -LiteralPath (Join-Path $env:LOCALAPPDATA "QaiConveyorDemo\Logs") -Filter "simulator-*.log" |
                Where-Object { $_.LastWriteTimeUtc -ge $Started -and $_.Name -like "*-$($Process.Id).log" } |
                Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
            if (-not $Log) { throw "Missing PID-matched test log." }
            $ResultLine = Get-Content -LiteralPath $Log.FullName |
                Where-Object { $_ -match "physics_rate_result " } | Select-Object -Last 1
            if (-not $ResultLine) { throw "No completed measurement in $($Log.FullName)" }
            $Metrics = @{}
            foreach ($Match in [regex]::Matches($ResultLine, '(\w+)=([^ ]+)')) {
                $Metrics[$Match.Groups[1].Value] = $Match.Groups[2].Value
            }
            $ParseNumber = { param($Text) [double]::Parse($Text, [Globalization.CultureInfo]::InvariantCulture) }
            $Elapsed = & $ParseNumber $Metrics.elapsed_s
            $Frames = [int]$Metrics.frames
            $Steps = [int]$Metrics.steps
            $MaxStep = & $ParseNumber $Metrics.max_step_ms
            $Row = [ordered]@{
                mode = $Mode; cap = $Cap; width = $Width; height = $Height
                actual_render_fps = [Math]::Round($Frames / $Elapsed, 2)
                force_steps = $Steps; render_frames = $Frames; max_step_ms = $MaxStep
                z_range_cm = & $ParseNumber $Metrics.z_range_cm
                vz_rms_cm_s = & $ParseNumber $Metrics.vz_rms_cm_s
                samples_s = & $ParseNumber $Metrics.samples_s
                log = $Log.FullName
            }
            $Results += $Row
            [ordered]@{ client=$ClientExecutable; results=$Results } | ConvertTo-Json -Depth 6 |
                Set-Content -LiteralPath $Output -Encoding utf8
            $Row | ConvertTo-Json -Compress
            if ($Metrics.mode -ne $Mode) { throw "Wrong force mode." }
            if ($Mode -eq "solver_step" -and ($Steps -le 0 -or $MaxStep -gt 8.5)) {
                throw "Solver-step timing gate failed: steps=$Steps maximum_ms=$MaxStep"
            }
        }
    }
    Write-Host "Physics comparison saved: $Output"
} finally {
    if ($Process -and -not $Process.HasExited) { Stop-Process -Id $Process.Id -Force }
    if ($HadSettings) { [IO.File]::WriteAllBytes($Settings, $SavedSettings) }
    elseif (Test-Path -LiteralPath $Settings) { Remove-Item -LiteralPath $Settings }
}
