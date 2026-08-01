[CmdletBinding()]
param(
    [string]$IsaacSimPath = "C:\NVIDIA\isaac-sim-standalone-6.0.1",
    [ValidateRange(1, 240)]
    [int]$FrameRateLimit = 60,
    [switch]$ResetUser
)

$resolvedIsaacPath = Resolve-Path -LiteralPath $IsaacSimPath -ErrorAction Stop
$launcher = Join-Path $resolvedIsaacPath "isaac-sim.bat"
$repositoryRoot = Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot "..\.."
) -ErrorAction Stop
$extensionFolder = Join-Path $repositoryRoot (
    "isaac_sim_supervisor_omniverse\exts"
)

if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Isaac Sim launcher not found: $launcher"
}
if (-not (Test-Path -LiteralPath $extensionFolder -PathType Container)) {
    throw "Tutorial extension folder not found: $extensionFolder"
}

$launchArguments = @(
    "--ext-folder",
    $extensionFolder,
    "--enable",
    "isaacsim.code_editor.python_server",
    "--enable",
    "qai.edge_ai_supervisor",
    "--enable",
    "qai.conveyor_safety",
    "--/app/runLoops/main/rateLimitEnabled=true",
    "--/app/runLoops/main/rateLimitFrequency=$FrameRateLimit",
    "--/app/runLoops/main/rateLimitUseBusyLoop=false",
    "--/app/runLoops/main/rateLimitUsePrecisionSleep=true",
    "--/app/runLoops/present/rateLimitEnabled=true",
    "--/app/runLoops/present/rateLimitFrequency=$FrameRateLimit",
    "--/app/runLoops/present/rateLimitUsePrecisionSleep=true",
    "--/app/runLoops/rendering_0/rateLimitEnabled=true",
    "--/app/runLoops/rendering_0/rateLimitFrequency=$FrameRateLimit",
    "--/app/runLoops/rendering_0/rateLimitUsePrecisionSleep=true",
    "--/app/runLoops/rendering_1/rateLimitEnabled=true",
    "--/app/runLoops/rendering_1/rateLimitFrequency=$FrameRateLimit",
    "--/app/runLoops/rendering_1/rateLimitUsePrecisionSleep=true"
)
if ($ResetUser) {
    $launchArguments += "--reset-user"
}

Write-Host "Starting Isaac Sim with its localhost Python server enabled..."
Write-Host "Launcher: $launcher"
Write-Host "Frame-rate cap: $FrameRateLimit FPS"
Write-Host "Supervisor panel: $extensionFolder"
& $launcher @launchArguments
