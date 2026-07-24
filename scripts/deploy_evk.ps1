param(
    [Parameter(Mandatory = $true)]
    [string]$BundlePath,

    [Parameter(Mandatory = $true)]
    [string]$EvkIp,
    [string]$EvkUser = "ubuntu",
    [Parameter(Mandatory = $true)]
    [string]$IdentityFile,
    [string]$RemoteDirectory = "/home/ubuntu/cosmos_reason2_2b_qairt"
)

$ErrorActionPreference = "Stop"

$resolvedBundle = (Resolve-Path -LiteralPath $BundlePath).Path
if (-not (Test-Path -LiteralPath $IdentityFile -PathType Leaf)) {
    throw "SSH identity file not found: $IdentityFile"
}
if ($EvkUser -notmatch "^[a-z_][a-z0-9_-]*$") {
    throw "EvkUser contains unsupported characters."
}
if ($EvkIp -notmatch "^[A-Za-z0-9.:-]+$") {
    throw "EvkIp contains unsupported characters."
}
$RemoteDirectory = $RemoteDirectory.TrimEnd("/")
$remoteHome = "/home/$EvkUser"
if (
    [string]::IsNullOrWhiteSpace($RemoteDirectory) -or
    -not $RemoteDirectory.StartsWith("$remoteHome/") -or
    $RemoteDirectory -notmatch "^/[A-Za-z0-9._/-]+$" -or
    $RemoteDirectory -match "//" -or
    @($RemoteDirectory.Split("/", [StringSplitOptions]::RemoveEmptyEntries)) -contains "." -or
    @($RemoteDirectory.Split("/", [StringSplitOptions]::RemoveEmptyEntries)) -contains ".."
) {
    throw "RemoteDirectory must be a normalized child path below $remoteHome."
}

$remote = "${EvkUser}@${EvkIp}"
$temporaryRoot = $null
$stagingDirectory = $null
$stagingActivated = $false

try {
    $sourceRoot = $resolvedBundle
    if (Test-Path -LiteralPath $resolvedBundle -PathType Leaf) {
        if ([IO.Path]::GetExtension($resolvedBundle) -ne ".zip") {
            throw "BundlePath must be a directory or a .zip archive: $resolvedBundle"
        }
        $temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) (
            "qai-cosmos-deploy-" + [guid]::NewGuid().ToString("N")
        )
        New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
        Expand-Archive -LiteralPath $resolvedBundle -DestinationPath $temporaryRoot
        $sourceRoot = $temporaryRoot
    }

    $configs = @(
        Get-ChildItem -LiteralPath $sourceRoot -Recurse -File -Filter "genie_config.json"
    )
    if ($configs.Count -eq 0 -and (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
        $archives = @(Get-ChildItem -LiteralPath $sourceRoot -File -Filter "*.zip")
        if ($archives.Count -eq 1) {
            $temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) (
                "qai-cosmos-deploy-" + [guid]::NewGuid().ToString("N")
            )
            New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
            Expand-Archive -LiteralPath $archives[0].FullName -DestinationPath $temporaryRoot
            $sourceRoot = $temporaryRoot
            $configs = @(
                Get-ChildItem -LiteralPath $sourceRoot -Recurse -File -Filter "genie_config.json"
            )
        }
    }
    if ($configs.Count -ne 1) {
        throw "Expected exactly one genie_config.json below $sourceRoot; found $($configs.Count)."
    }

    $deployRoot = $configs[0].Directory.FullName
    $requiredFiles = @(
        "embedding_weights.raw",
        "genie_config.json",
        "genie-app-script.txt",
        "htp_backend_ext_config.json",
        "img-enc-htp.json",
        "metadata.json",
        "text-encoder.json",
        "text-generator.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json"
    )
    foreach ($name in $requiredFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $deployRoot $name) -PathType Leaf)) {
            throw "Incomplete VLM bundle: missing $name in $deployRoot"
        }
    }
    foreach ($name in @("genie_config.json", "text-generator.json", "img-enc-htp.json")) {
        $configText = Get-Content -LiteralPath (Join-Path $deployRoot $name) -Raw
        if ($configText -notmatch '"type"\s*:\s*"QnnHtp"') {
            throw "Bundle config $name does not select the QnnHtp backend."
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $deployRoot "sample_inputs") -PathType Container)) {
        throw "Incomplete VLM bundle: missing sample_inputs in $deployRoot"
    }
    $requiredSampleInputs = @(
        "pixel_values.raw",
        "position_ids_cos.raw",
        "position_ids_sin.raw",
        "window_attention_mask.raw",
        "full_attention_mask.raw",
        "prompt_prefix.txt",
        "prompt_suffix.txt"
    )
    foreach ($name in $requiredSampleInputs) {
        $samplePath = Join-Path (Join-Path $deployRoot "sample_inputs") $name
        if (-not (Test-Path -LiteralPath $samplePath -PathType Leaf)) {
            throw "Incomplete VLM bundle: missing sample_inputs/$name in $deployRoot"
        }
    }
    $requiredContextBins = @(
        "part1_of_4.bin",
        "part2_of_4.bin",
        "part3_of_4.bin",
        "part4_of_4.bin",
        "vision_encoder.bin"
    )
    foreach ($name in $requiredContextBins) {
        if (-not (Test-Path -LiteralPath (Join-Path $deployRoot $name) -PathType Leaf)) {
            throw "Incomplete VLM bundle: missing context binary $name in $deployRoot"
        }
    }

    $deploymentId = "{0}-{1}" -f (
        Get-Date -Format "yyyyMMdd-HHmmss"
    ), ([guid]::NewGuid().ToString("N").Substring(0, 8))
    $stagingDirectory = "${RemoteDirectory}.staging-${deploymentId}"
    $backupDirectory = "${RemoteDirectory}.previous"

    ssh -i $IdentityFile $remote "mkdir '$stagingDirectory'"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create remote staging directory."
    }

    scp -i $IdentityFile -r "$deployRoot\*" "${remote}:${stagingDirectory}/"
    if ($LASTEXITCODE -ne 0) {
        throw "Bundle copy failed; existing deployment was not changed."
    }

    $runner = Join-Path $PSScriptRoot "run_evk_qairt.sh"
    scp -i $IdentityFile $runner "${remote}:${stagingDirectory}/run_text_smoke.sh"
    if ($LASTEXITCODE -ne 0) {
        throw "Runner copy failed; existing deployment was not changed."
    }

    $evidenceCollector = Join-Path $PSScriptRoot "collect_evk_npu_evidence.py"
    scp -i $IdentityFile $evidenceCollector "${remote}:${stagingDirectory}/collect_npu_evidence.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Evidence collector copy failed; existing deployment was not changed."
    }

    ssh -i $IdentityFile $remote "set -e; chmod +x '$stagingDirectory/run_text_smoke.sh' '$stagingDirectory/collect_npu_evidence.py'; python3 '$stagingDirectory/collect_npu_evidence.py' --bundle '$stagingDirectory' --mode preflight --report '$stagingDirectory/npu_preflight.json'"
    if ($LASTEXITCODE -ne 0) {
        throw "Target bundle/runtime preflight failed; existing deployment was not changed."
    }

    ssh -i $IdentityFile $remote "set -e; if [ -e '$backupDirectory' ]; then rm -rf -- '$backupDirectory'; fi; if [ -e '$RemoteDirectory' ]; then mv '$RemoteDirectory' '$backupDirectory'; fi; if ! mv '$stagingDirectory' '$RemoteDirectory'; then if [ -e '$backupDirectory' ] && [ ! -e '$RemoteDirectory' ]; then mv '$backupDirectory' '$RemoteDirectory'; fi; exit 1; fi"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to activate staged deployment."
    }
    $stagingActivated = $true

    # The staged preflight report contains the staging path in its evidence.
    # Refresh it after activation so the retained report names paths that
    # continue to exist. A transient refresh failure must not misreport the
    # already-completed activation as a failed deployment.
    ssh -i $IdentityFile $remote "python3 '$RemoteDirectory/collect_npu_evidence.py' --bundle '$RemoteDirectory' --mode preflight --report '$RemoteDirectory/npu_preflight.json'"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning (
            "Deployment was activated, but final-path evidence refresh failed. " +
            "The staged preflight passed; rerun collect_npu_evidence.py manually."
        )
    }

    Write-Host "Deployed to ${remote}:${RemoteDirectory}"
    Write-Host "Rollback copy, if a deployment was replaced: ${remote}:${backupDirectory}"
}
finally {
    if ($null -ne $stagingDirectory -and -not $stagingActivated) {
        ssh -i $IdentityFile $remote "rm -rf -- '$stagingDirectory'" 2>$null
    }
    if ($null -ne $temporaryRoot -and (Test-Path -LiteralPath $temporaryRoot)) {
        $resolvedTemporaryRoot = (Resolve-Path -LiteralPath $temporaryRoot).Path
        $systemTemporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        if (-not $resolvedTemporaryRoot.StartsWith($systemTemporaryRoot)) {
            throw "Refusing to remove unexpected temporary path: $resolvedTemporaryRoot"
        }
        Remove-Item -LiteralPath $resolvedTemporaryRoot -Recurse -Force
    }
}
