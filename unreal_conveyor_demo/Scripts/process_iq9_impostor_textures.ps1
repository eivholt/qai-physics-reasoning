param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [switch]$UseLegacyBake
)

$ErrorActionPreference = 'Stop'

$referenceRoot = Join-Path $ProjectRoot 'ContentSource\IQ9EVK\Reference'
$referenceProcessor = Join-Path $PSScriptRoot 'process_iq9_reference_textures.py'
$referenceNames = @(
    'iq9_top_photo_primary.jpg',
    'iq9_top_photo_reference.jpg',
    'iq9_front_logo.png',
    'iq9_rear_power.png',
    'iq9_side_pcie_csi.png',
    'iq9_side_mode.png'
)
$hasReferenceSet = -not ($referenceNames | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $referenceRoot $_))
})
if ($hasReferenceSet -and -not $UseLegacyBake) {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        $python = Get-Command py -ErrorAction SilentlyContinue
    }
    if (-not $python) {
        throw 'The exact IQ9 product reference set is present. Run Scripts/process_iq9_reference_textures.py with a Python environment containing Pillow; use -UseLegacyBake only to intentionally restore the old CAD-bake textures.'
    }
    & $python.Source $referenceProcessor --project-root $ProjectRoot
    if ($LASTEXITCODE -ne 0) {
        throw "IQ9 reference texture processing failed with exit code $LASTEXITCODE"
    }
    exit 0
}

Add-Type -AssemblyName System.Drawing

$sourceRoot = Join-Path $ProjectRoot 'Saved\IQ9ImpostorBake'
$outputRoot = Join-Path $ProjectRoot 'ContentSource\IQ9EVK\Impostor'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

function Get-CroppedBitmap {
    param(
        [System.Drawing.Bitmap]$Source,
        [System.Drawing.Rectangle]$Crop,
        [int]$Width,
        [int]$Height
    )
    $bitmap = [System.Drawing.Bitmap]::new($Width, $Height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CompositingMode = [System.Drawing.Drawing2D.CompositingMode]::SourceCopy
        $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $graphics.DrawImage(
            $Source,
            [System.Drawing.Rectangle]::new(0, 0, $Width, $Height),
            $Crop,
            [System.Drawing.GraphicsUnit]::Pixel)
    }
    finally {
        $graphics.Dispose()
    }
    return $bitmap
}

function Set-ImpostorPalette {
    param(
        [System.Drawing.Bitmap]$Bitmap,
        [ValidateSet('Top', 'Side')][string]$Mode
    )
    $bounds = [System.Drawing.Rectangle]::new(0, 0, $Bitmap.Width, $Bitmap.Height)
    $data = $Bitmap.LockBits(
        $bounds,
        [System.Drawing.Imaging.ImageLockMode]::ReadWrite,
        [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    try {
        $bytes = [byte[]]::new([Math]::Abs($data.Stride) * $Bitmap.Height)
        [Runtime.InteropServices.Marshal]::Copy($data.Scan0, $bytes, 0, $bytes.Length)
        for ($y = 0; $y -lt $Bitmap.Height; $y++) {
            $ny = $y / [double]($Bitmap.Height - 1)
            for ($x = 0; $x -lt $Bitmap.Width; $x++) {
                $nx = $x / [double]($Bitmap.Width - 1)
                $index = $y * $data.Stride + $x * 4
                $blue = $bytes[$index]
                $green = $bytes[$index + 1]
                $red = $bytes[$index + 2]
                $luma = (0.0722 * $blue + 0.7152 * $green + 0.2126 * $red) / 255.0
                $detail = [Math]::Pow([Math]::Max(0.0, [Math]::Min(1.0, $luma * 1.65)), 0.78)

                if ($Mode -eq 'Top') {
                    $edge = $nx -lt 0.075 -or $nx -gt 0.925 -or $ny -lt 0.065 -or $ny -gt 0.925
                    $slot = (
                        ($nx -gt 0.16 -and $nx -lt 0.70 -and $ny -gt 0.11 -and $ny -lt 0.31) -or
                        ($nx -gt 0.16 -and $nx -lt 0.70 -and $ny -gt 0.56 -and $ny -lt 0.77)
                    ) -and $luma -gt 0.27
                    if ($edge) {
                        # Qualcomm blue enclosure lip, modulated by the captured bevel shading.
                        $outR = 28 + 35 * $detail
                        $outG = 28 + 36 * $detail
                        $outB = 150 + 80 * $detail
                    }
                    elseif ($slot) {
                        # Beige memory slots; keep the original grooves and end clips.
                        $outR = 135 + 82 * $detail
                        $outG = 119 + 80 * $detail
                        $outB = 87 + 70 * $detail
                    }
                    elseif ($luma -gt 0.48) {
                        # Exposed connectors and heat-spreader edges remain neutral metal.
                        $outR = 54 + 58 * $detail
                        $outG = 56 + 57 * $detail
                        $outB = 60 + 54 * $detail
                    }
                    else {
                        # The PCB itself is deliberately very dark satin black.
                        $outR = 10 + 47 * $detail
                        $outG = 11 + 49 * $detail
                        $outB = 14 + 52 * $detail
                    }
                }
                else {
                    if ($ny -lt 0.27) {
                        # Open-board profile and connectors above the enclosure wall.
                        $outR = 13 + 78 * $detail
                        $outG = 14 + 76 * $detail
                        $outB = 17 + 72 * $detail
                    }
                    else {
                        # Low-frequency photographed relief on the Qualcomm-blue enclosure.
                        $outR = 24 + 42 * $detail
                        $outG = 24 + 43 * $detail
                        $outB = 132 + 95 * $detail
                    }
                }
                $bytes[$index] = [byte][Math]::Clamp([int]$outB, 0, 255)
                $bytes[$index + 1] = [byte][Math]::Clamp([int]$outG, 0, 255)
                $bytes[$index + 2] = [byte][Math]::Clamp([int]$outR, 0, 255)
                $bytes[$index + 3] = 255
            }
        }
        [Runtime.InteropServices.Marshal]::Copy($bytes, 0, $data.Scan0, $bytes.Length)
    }
    finally {
        $Bitmap.UnlockBits($data)
    }
}

function Add-SideMarking {
    param([System.Drawing.Bitmap]$Bitmap)
    $graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
    $font = $null
    $brush = $null
    try {
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
        $font = [System.Drawing.Font]::new('Segoe UI Semibold', 24.0, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
        $brush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(205, 240, 238, 225))
        $graphics.DrawString('IQ9 EVK', $font, $brush, 34.0, $Bitmap.Height - 52.0)
    }
    finally {
        if ($brush) { $brush.Dispose() }
        if ($font) { $font.Dispose() }
        $graphics.Dispose()
    }
}

$specifications = @(
    @{ Name = 'top'; Crop = [System.Drawing.Rectangle]::new(86, 88, 852, 850); Width = 1024; Height = 1024; Mode = 'Top' },
    @{ Name = 'front'; Crop = [System.Drawing.Rectangle]::new(86, 282, 854, 458); Width = 512; Height = 256; Mode = 'Side' },
    @{ Name = 'back'; Crop = [System.Drawing.Rectangle]::new(86, 282, 854, 458); Width = 512; Height = 256; Mode = 'Side' },
    @{ Name = 'right'; Crop = [System.Drawing.Rectangle]::new(86, 280, 854, 462); Width = 512; Height = 256; Mode = 'Side' },
    @{ Name = 'left'; Crop = [System.Drawing.Rectangle]::new(86, 280, 854, 462); Width = 512; Height = 256; Mode = 'Side' }
)

foreach ($specification in $specifications) {
    $sourcePath = Join-Path $sourceRoot ("iq9_{0}.png" -f $specification.Name)
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Missing IQ9 bake source: $sourcePath"
    }
    $source = [System.Drawing.Bitmap]::FromFile($sourcePath)
    try {
        $output = Get-CroppedBitmap -Source $source -Crop $specification.Crop -Width $specification.Width -Height $specification.Height
        try {
            Set-ImpostorPalette -Bitmap $output -Mode $specification.Mode
            if ($specification.Mode -eq 'Side') {
                Add-SideMarking -Bitmap $output
            }
            $destination = Join-Path $outputRoot ("iq9_{0}.png" -f $specification.Name)
            $output.Save($destination, [System.Drawing.Imaging.ImageFormat]::Png)
            Write-Host "IQ9_IMPOSTOR_TEXTURE $destination $($output.Width)x$($output.Height)"
        }
        finally {
            $output.Dispose()
        }
    }
    finally {
        $source.Dispose()
    }
}
