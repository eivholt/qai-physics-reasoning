[CmdletBinding()]
param(
    [string]$RunDirectory = (
        "artifacts\isaac_sim_live_aisle\runs\20260728_072512_88034920"
    ),
    [string]$OutputDirectory = (
        "docs\media\isaac_sim_edge_supervisor_evk_r1"
    ),
    [ValidateRange(1, 30)]
    [int]$PlaybackFps = 6,
    [ValidateRange(0, 10000)]
    [int]$ContinueFrame = 10,
    [ValidateRange(1, 10000)]
    [int]$HazardFrame = 36,
    [ValidateRange(2, 10000)]
    [int]$RerouteFrame = 46,
    [ValidateRange(3, 10000)]
    [int]$LastFrame = 58,
    [string]$BackendLabel = "IQ9075 EVK | NPU Q4_0",
    [ValidateRange(0, 10000)]
    [int]$ContinueCount = 16,
    [ValidateRange(0, 10000)]
    [int]$NorthRerouteCount = 2,
    [ValidateRange(0, 10000)]
    [int]$FalseBaselineInterventions = 0,
    [ValidateRange(0, 10000)]
    [int]$LocalSafetyTakeovers = 0,
    [ValidateRange(0.0, 120.0)]
    [double]$RerouteLatencySeconds = 2.808984
)

$ErrorActionPreference = "Stop"

function Convert-ToWslPath {
    param([Parameter(Mandatory)][string]$Path)

    $absolute = [System.IO.Path]::GetFullPath($Path)
    if ($absolute -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Expected a Windows drive path, got: $absolute"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $tail = $Matches[2].Replace('\', '/')
    return "/mnt/$drive/$tail"
}

function Invoke-Ffmpeg {
    param([Parameter(Mandatory)][string[]]$Arguments)

    & wsl -e ffmpeg @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg exited with status $LASTEXITCODE"
    }
}

$resolvedRun = Resolve-Path -LiteralPath $RunDirectory
$framesDirectory = Join-Path $resolvedRun "frames"
if (-not (Test-Path -LiteralPath $framesDirectory -PathType Container)) {
    throw "Frame directory is missing: $framesDirectory"
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$resolvedOutput = Resolve-Path -LiteralPath $OutputDirectory

$clearStillFrame = [Math]::Min(30, $HazardFrame - 1)
$emergenceStillFrame = [Math]::Min(42, $RerouteFrame - 1)
$rerouteStillFrame = [Math]::Min(49, $LastFrame)

Copy-Item -LiteralPath (
    Join-Path $framesDirectory ("frame_{0:D4}.png" -f $clearStillFrame)
) -Destination (Join-Path $resolvedOutput "clear_route_continue.png") -Force
Copy-Item -LiteralPath (
    Join-Path $framesDirectory ("frame_{0:D4}.png" -f $emergenceStillFrame)
) -Destination (Join-Path $resolvedOutput "forklift_emerging.png") -Force
Copy-Item -LiteralPath (
    Join-Path $framesDirectory ("frame_{0:D4}.png" -f $rerouteStillFrame)
) -Destination (Join-Path $resolvedOutput "north_bypass_applied.png") -Force

$inputPattern = Convert-ToWslPath (
    Join-Path $framesDirectory "frame_%04d.png"
)
$fullMovie = Convert-ToWslPath (
    Join-Path $resolvedOutput "full_closed_loop.mp4"
)
$lastContinueFrame = $HazardFrame - 1
$lastInferenceFrame = $RerouteFrame - 1
$latencyLabel = $RerouteLatencySeconds.ToString(
    "0.000",
    [System.Globalization.CultureInfo]::InvariantCulture
)
$filter = @(
    "scale=1280:720:flags=lanczos"
    "pad=1280:900:0:0:color=0x111827"
    (
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/" +
        "DejaVuSans-Bold.ttf:text='Edge AI warehouse supervisor | " +
        "Cosmos-Reason2 2B | $BackendLabel':x=24:y=738:" +
        "fontsize=27:fontcolor=white"
    )
    (
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/" +
        "DejaVuSans-Bold.ttf:text='BUFFERING | clear baseline':" +
        "x=24:y=780:fontsize=25:fontcolor=0xFBBF24:" +
        "enable='between(n\,0\,$($ContinueFrame - 1))'"
    )
    (
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/" +
        "DejaVuSans-Bold.ttf:text='CONTINUE CURRENT_ROUTE | " +
        "corridor clear':x=24:y=780:fontsize=25:fontcolor=0x34D399:" +
        "enable='between(n\,$ContinueFrame\,$lastContinueFrame)'"
    )
    (
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/" +
        "DejaVuSans-Bold.ttf:text='FORKLIFT EMERGING | inference " +
        "in flight':x=24:y=780:fontsize=25:fontcolor=0xFBBF24:" +
        "enable='between(n\,$HazardFrame\,$lastInferenceFrame)'"
    )
    (
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/" +
        "DejaVuSans-Bold.ttf:text='REROUTE NORTH_BYPASS | " +
        "$latencyLabel s | " +
        "applied':x=24:y=780:fontsize=25:fontcolor=0x22D3EE:" +
        "enable='gte(n\,$RerouteFrame)'"
    )
    (
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/" +
        "DejaVuSans.ttf:text='$ContinueCount CONTINUE | " +
        "$NorthRerouteCount NORTH reroutes | " +
        "$FalseBaselineInterventions false baseline interventions | " +
        "$LocalSafetyTakeovers local-safety takeovers':" +
        "x=24:y=824:fontsize=19:fontcolor=0xD1D5DB"
    )
    (
        "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/" +
        "DejaVuSans.ttf:text='Clean roof frames go to Reason2; white, " +
        "cyan, and orange route lines are presentation-only.':" +
        "x=24:y=858:fontsize=18:fontcolor=0x9CA3AF"
    )
) -join ","

Invoke-Ffmpeg @(
    "-hide_banner", "-loglevel", "error", "-y",
    "-framerate", "$PlaybackFps",
    "-i", $inputPattern,
    "-vf", $filter,
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    $fullMovie
)

$clearStart = $ContinueFrame / $PlaybackFps
$clearDuration = ($HazardFrame - $ContinueFrame) / $PlaybackFps
$interventionStart = [Math]::Max(0, $HazardFrame - 4) / $PlaybackFps
$interventionDuration = (
    $LastFrame - [Math]::Max(0, $HazardFrame - 4) + 1
) / $PlaybackFps

$clearMovie = Convert-ToWslPath (
    Join-Path $resolvedOutput "clear_route_control.mp4"
)
$interventionMovie = Convert-ToWslPath (
    Join-Path $resolvedOutput "blind_corner_north_bypass.mp4"
)
Invoke-Ffmpeg @(
    "-hide_banner", "-loglevel", "error", "-y",
    "-ss", "$clearStart", "-i", $fullMovie,
    "-t", "$clearDuration",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
    $clearMovie
)
Invoke-Ffmpeg @(
    "-hide_banner", "-loglevel", "error", "-y",
    "-ss", "$interventionStart", "-i", $fullMovie,
    "-t", "$interventionDuration",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
    $interventionMovie
)

$gifFilter = (
    "[0:v]fps=$PlaybackFps,scale=960:-2:flags=lanczos," +
    "split[s0][s1];[s0]palettegen=max_colors=192[p];" +
    "[s1][p]paletteuse=dither=sierra2_4a"
)
foreach ($basename in @(
    "full_closed_loop",
    "clear_route_control",
    "blind_corner_north_bypass"
)) {
    $movie = Convert-ToWslPath (
        Join-Path $resolvedOutput "$basename.mp4"
    )
    $gif = Convert-ToWslPath (
        Join-Path $resolvedOutput "$basename.gif"
    )
    Invoke-Ffmpeg @(
        "-hide_banner", "-loglevel", "error", "-y",
        "-i", $movie,
        "-filter_complex", $gifFilter,
        "-loop", "0",
        $gif
    )
}

Get-ChildItem -LiteralPath $resolvedOutput |
    Sort-Object Name |
    Select-Object Name, Length
