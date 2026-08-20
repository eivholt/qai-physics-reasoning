param(
    [ValidateSet('train', 'validation', 'test')]
    [string]$Split = 'train',
    [int]$CasesPerClass = 300,
    [int]$Seed = 31001,
    [string]$DatasetName = 'reason2-ft-parcel-v1',
    [string]$InferenceCamera = 'parcel-quarter-cell-occlusion-safe',
    [ValidateSet('runtime-neighborhood', 'runtime-exact', 'prompt-noise')]
    [string]$CameraPolicy = 'runtime-neighborhood',
    [string]$Executable = '',
    [ValidateSet('D3D11', 'D3D12')]
    [string]$CaptureRHI = 'D3D12',
    [switch]$DisableRayTracing,
    [switch]$SingleCore,
    [string]$CaptureDDC = '',
    [switch]$SoloParcel,
    [switch]$BlackBelt,
    [switch]$MatchedBoundarySweeps,
    [switch]$Bilateral
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$editor = 'C:\Program Files\Epic Games\UE_5.8\Engine\Binaries\Win64\UnrealEditor.exe'
$project = Join-Path $repo 'unreal_conveyor_demo\QaiConveyor.uproject'
$shipping = Join-Path $repo 'unreal_conveyor_demo\Binaries\Win64\QaiConveyor-Win64-Shipping.exe'
if (!$Executable) {
    $Executable = $editor
}
if (!(Test-Path $Executable)) { throw "Unreal executable not found: $Executable" }

$arguments = @(
    '-windowed', '-ResX=960', '-ResY=540', '-NoSound', '-NoSplash',
    '-Unattended',
    "-QaiInferenceCamera=$InferenceCamera",
    '-QaiParcelEvaluation',
    '-QaiParcelEvaluationCaptureOnly', '-QaiParcelEvaluationAutoExit',
    '-QaiParcelEvaluationRandomized',
    "-QaiParcelEvaluationCasesPerSignal=$CasesPerClass",
    "-QaiParcelEvaluationSeed=$Seed",
    "-QaiParcelEvaluationSplit=$Split",
    "-QaiParcelEvaluationCameraPolicy=$CameraPolicy",
    "-QaiParcelEvaluationName=$DatasetName",
    '-log'
)
$arguments += if ($CaptureRHI -eq 'D3D11') { '-d3d11' } else { '-d3d12' }
if ($DisableRayTracing -or $CaptureRHI -eq 'D3D11') {
    $arguments += @('-noraytracing', '-ExecCmds=r.RayTracing=0')
}
if ($CaptureDDC) { $arguments += "-DDC=$CaptureDDC" }
if ([IO.Path]::GetFileName($Executable) -eq 'UnrealEditor.exe') {
    $arguments = @($project, '-game') + $arguments
}
if ($SoloParcel) { $arguments += '-QaiParcelEvaluationSoloParcel' }
if ($BlackBelt) { $arguments += '-QaiBlackBeltVisual' }
if ($MatchedBoundarySweeps) {
    # Generate class triplets from the same parcel identity, camera, lighting,
    # and distractors while changing only its support geometry.  This targets
    # the production adapter's remaining AMBER/RED decision-boundary errors.
    $arguments += '-QaiParcelEvaluationMatchedBoundarySweeps'
}
if ($Bilateral) {
    # Alternate unsafe cases between the camera-near and divider-side floor
    # strips. Variant parity is shared by matched G/A/R triplets and recorded
    # in metadata, producing an exact 50/50 split without post-hoc sampling.
    $arguments += '-QaiParcelEvaluationBilateral'
}

if ($SingleCore) {
    # UE 5.8's startup DDC/GPU profiler can observe a non-monotonic timestamp
    # on this workstation when a process migrates between cores.  Setting
    # ProcessorAffinity after Start-Process is racy: UE can reach DDC startup
    # before PowerShell gets a chance to pin it.  cmd.exe's START /AFFINITY
    # creates the child with the mask already applied.
    $quotedArguments = @($arguments | ForEach-Object {
        '"' + ($_ -replace '"', '\"') + '"'
    }) -join ' '
    $command = "start `"`" /wait /b /affinity 1 `"$Executable`" $quotedArguments"
    Write-Host "Unreal dataset capture starting with affinity mask 0x1: executable=$Executable split=$Split cases=$CasesPerClass seed=$Seed camera=$InferenceCamera matchedBoundary=$MatchedBoundarySweeps bilateral=$Bilateral"
    & $env:ComSpec /d /s /c $command
    $captureExitCode = $LASTEXITCODE
} else {
    $process = Start-Process -FilePath $Executable -ArgumentList $arguments -PassThru
    Write-Host "Unreal dataset capture started: pid=$($process.Id) executable=$Executable split=$Split cases=$CasesPerClass seed=$Seed camera=$InferenceCamera matchedBoundary=$MatchedBoundarySweeps bilateral=$Bilateral"
    $process.WaitForExit()
    $captureExitCode = $process.ExitCode
}
if ($captureExitCode -ne 0) { throw "Unreal dataset capture failed with exit code $captureExitCode" }

$projectOutput = Join-Path $repo "unreal_conveyor_demo\Saved\Diagnostics\ParcelEvaluation\$DatasetName"
$packagedOutput = Join-Path $env:LOCALAPPDATA "QaiConveyor\Saved\Diagnostics\ParcelEvaluation\$DatasetName"
$output = if (Test-Path $projectOutput) {
    $projectOutput
} elseif (Test-Path $packagedOutput) {
    # A packaged build writes FPaths::ProjectSavedDir() below LocalAppData,
    # while editor/game captures write into the repository's Saved directory.
    # Accept both so the validated packaged executable can be used for large,
    # unattended dataset generation without silently reporting zero images.
    $packagedOutput
} else {
    $projectOutput
}
$images = @(Get-ChildItem $output -Recurse -Filter '*.png' -File -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "\\$Split\\[GAR]\\" })
$expected = [Math]::Max(3, $CasesPerClass) * 3
if ($images.Count -ne $expected) {
    throw "Expected $expected images for $Split, found $($images.Count) under $output"
}
Write-Host "Capture complete: $($images.Count) lossless PNGs under $output"
