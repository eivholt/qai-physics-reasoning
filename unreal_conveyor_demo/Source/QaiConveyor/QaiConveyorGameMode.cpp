#include "QaiConveyorGameMode.h"

#include "HAL/IConsoleManager.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "RHI.h"
#include "RHIGlobals.h"
#include "RenderUtils.h"
#include "HardwareInfo.h"
#include "Engine/DirectionalLight.h"
#include "Components/LightComponent.h"
#include "EngineUtils.h"
#include "QaiConveyorHUD.h"
#include "QaiConveyorPawn.h"
#include "QaiConveyorWorld.h"

namespace
{
    enum class EQaiRenderTier
    {
        Balanced,
        High,
        Ultra
    };

    FString GRenderProfileSummary = TEXT("render_profile unavailable");
    FString GRenderProfileDisplayName = TEXT("Render profile unavailable");
    FString GCompatibilityStatus = TEXT("render_profile_not_initialized");
    bool GIsUltraRenderTier = false;
    bool GUsesHardwareLumen = false;

    FString SanitizeLogField(FString Value)
    {
        return Value
            .Replace(TEXT("\r"), TEXT(" "))
            .Replace(TEXT("\n"), TEXT(" "))
            .Replace(TEXT("\""), TEXT("'"));
    }

    void SetRenderCVar(const TCHAR* Name, int32 Value)
    {
        if (IConsoleVariable* Variable = IConsoleManager::Get().FindConsoleVariable(Name))
        {
            Variable->Set(Value, ECVF_SetByGameOverride);
        }
    }

    void SetRenderCVar(const TCHAR* Name, float Value)
    {
        if (IConsoleVariable* Variable = IConsoleManager::Get().FindConsoleVariable(Name))
        {
            Variable->Set(Value, ECVF_SetByGameOverride);
        }
    }

    const TCHAR* TierName(EQaiRenderTier Tier)
    {
        switch (Tier)
        {
        case EQaiRenderTier::Ultra:
            return TEXT("Ultra");
        case EQaiRenderTier::High:
            return TEXT("High");
        default:
            return TEXT("Balanced");
        }
    }
}

AQaiConveyorGameMode::AQaiConveyorGameMode()
{
    DefaultPawnClass = AQaiConveyorPawn::StaticClass();
    HUDClass = AQaiConveyorHUD::StaticClass();
}

void AQaiConveyorGameMode::StartPlay()
{
    // Apply the quality tier before BeginPlay is dispatched to the world. This
    // makes the first interactive frame and its TSR history use one profile.
    ApplyHardwareRenderProfile();
    Super::StartPlay();

    if (!GetWorld())
    {
        return;
    }

    bool bHasRuntime = false;
    for (TActorIterator<AQaiConveyorWorld> It(GetWorld()); It; ++It)
    {
        bHasRuntime = true;
        break;
    }
    if (!bHasRuntime)
    {
        GetWorld()->SpawnActor<AQaiConveyorWorld>();
    }

    bool bHasDirectionalLight = false;
    for (TActorIterator<ADirectionalLight> It(GetWorld()); It; ++It)
    {
        bHasDirectionalLight = true;
        break;
    }
    if (!bHasDirectionalLight)
    {
        if (ADirectionalLight* Sun = GetWorld()->SpawnActor<ADirectionalLight>())
        {
            Sun->GetLightComponent()->SetMobility(EComponentMobility::Movable);
            Sun->SetActorRotation(FRotator(-48.0f, -32.0f, 0.0f));
            // The native map normally uses four authored warehouse panels.
            // This is only a restrained fallback for diagnostic/empty maps.
            Sun->GetLightComponent()->SetIntensity(1.5f);
            Sun->GetLightComponent()->SetLightColor(FLinearColor(1.0f, 0.97f, 0.92f));
        }
    }

}

const FString& AQaiConveyorGameMode::GetRenderProfileSummary()
{
    return GRenderProfileSummary;
}

const FString& AQaiConveyorGameMode::GetRenderProfileDisplayName()
{
    return GRenderProfileDisplayName;
}

const FString& AQaiConveyorGameMode::GetCompatibilityStatus()
{
    return GCompatibilityStatus;
}

bool AQaiConveyorGameMode::IsUltraRenderTier()
{
    return GIsUltraRenderTier;
}

bool AQaiConveyorGameMode::UsesHardwareLumen()
{
    return GUsesHardwareLumen;
}

void AQaiConveyorGameMode::ApplyHardwareRenderProfile()
{
    const FString AdapterName = GRHIAdapterName.IsEmpty() ? TEXT("Unknown GPU") : GRHIAdapterName;
    const FString AdapterLower = AdapterName.ToLower();
    const FString RhiName = FHardwareInfo::GetHardwareInfo(NAME_RHI);
    const uint64 DedicatedVideoMemory = GRHIGlobals.GpuInfo.DedicatedVideoMemory;
    const uint64 VideoMemoryMiB = DedicatedVideoMemory / (1024ull * 1024ull);
    const bool bIntegrated =
        AdapterLower.Contains(TEXT("intel")) ||
        AdapterLower.Contains(TEXT("integrated")) ||
        AdapterLower.Contains(TEXT("microsoft basic"));
    const bool bAppleSilicon = AdapterLower.Contains(TEXT("apple m"));

    EQaiRenderTier Tier = EQaiRenderTier::Balanced;
#if PLATFORM_MAC
    if (bAppleSilicon)
    {
        Tier = EQaiRenderTier::High;
    }
#else
    if (!bIntegrated && VideoMemoryMiB >= 12288)
    {
        Tier = EQaiRenderTier::Ultra;
    }
    else if (!bIntegrated && VideoMemoryMiB >= 6144)
    {
        Tier = EQaiRenderTier::High;
    }
#endif

    FString Override;
    if (FParse::Value(FCommandLine::Get(), TEXT("QaiRenderQuality="), Override))
    {
        if (Override.Equals(TEXT("ultra"), ESearchCase::IgnoreCase))
        {
            Tier = EQaiRenderTier::Ultra;
        }
        else if (Override.Equals(TEXT("high"), ESearchCase::IgnoreCase))
        {
            Tier = EQaiRenderTier::High;
        }
        else if (Override.Equals(TEXT("balanced"), ESearchCase::IgnoreCase))
        {
            Tier = EQaiRenderTier::Balanced;
        }
    }

    const bool bUltra = Tier == EQaiRenderTier::Ultra;
    const bool bHighOrBetter = Tier != EQaiRenderTier::Balanced;
    const int32 TsrHistoryPercentage = bUltra ? 200 : (bHighOrBetter ? 150 : 100);

    // The simulation remains fixed at 120 Hz, but presenting every frame on a
    // fast GPU needlessly drives an otherwise idle window to 99% utilization.
    // Sixty presented frames is the portable default; diagnostics and capture
    // runs can explicitly override it (zero means uncapped).
    int32 FrameRateCap = 60;
    FParse::Value(FCommandLine::Get(), TEXT("QaiMaxFPS="), FrameRateCap);
    FrameRateCap = FrameRateCap == 0 ? 0 : FMath::Clamp(FrameRateCap, 30, 240);
    SetRenderCVar(TEXT("t.MaxFPS"), static_cast<float>(FrameRateCap));

    FString RayTracingOverride = TEXT("auto");
    FParse::Value(FCommandLine::Get(), TEXT("QaiRayTracing="), RayTracingOverride);
    const bool bRayTracingForcedOff =
        RayTracingOverride.Equals(TEXT("off"), ESearchCase::IgnoreCase) ||
        RayTracingOverride.Equals(TEXT("false"), ESearchCase::IgnoreCase) ||
        RayTracingOverride.Equals(TEXT("0"), ESearchCase::IgnoreCase);
    const bool bRayTracingForcedOn =
        RayTracingOverride.Equals(TEXT("on"), ESearchCase::IgnoreCase) ||
        RayTracingOverride.Equals(TEXT("true"), ESearchCase::IgnoreCase) ||
        RayTracingOverride.Equals(TEXT("1"), ESearchCase::IgnoreCase);
    const bool bHardwareRayTracingAvailable = GRHISupportsRayTracing && IsRayTracingEnabled();
    const bool bUseHardwareLumen =
        bHardwareRayTracingAvailable && !bRayTracingForcedOff && (bUltra || bRayTracingForcedOn);
    const bool bUseLumen = bHighOrBetter;
    GIsUltraRenderTier = bUltra;
    GUsesHardwareLumen = bUseHardwareLumen;

    // This pool contains ray-tracing geometry acceleration structures, not
    // textures or the whole VRAM budget. Keep Unreal's default untouched when
    // the selected path does not use hardware ray tracing. macOS deliberately
    // remains on the portable software-Lumen path.
    int32 RayTracingGeometryPoolMiB = 0;
#if PLATFORM_WINDOWS
    if (bUseHardwareLumen)
    {
        RayTracingGeometryPoolMiB = bUltra || VideoMemoryMiB >= 12288
            ? 1024
            : (VideoMemoryMiB >= 8192 ? 768 : 512);
        SetRenderCVar(
            TEXT("r.RayTracing.ResidentGeometryMemoryPoolSizeInMB"),
            RayTracingGeometryPoolMiB);
    }
#endif

    TArray<FString> CompatibilityCodes;
    if (AdapterName == TEXT("Unknown GPU"))
    {
        CompatibilityCodes.Add(TEXT("gpu_unknown"));
    }
    if (DedicatedVideoMemory > 0 && VideoMemoryMiB < 4096)
    {
        CompatibilityCodes.Add(TEXT("low_vram"));
    }
    if (GRHIAdapterDriverOnDenyList)
    {
        CompatibilityCodes.Add(TEXT("gpu_driver_denylisted"));
    }
    if (bRayTracingForcedOn && !bHardwareRayTracingAvailable)
    {
        CompatibilityCodes.Add(TEXT("ray_tracing_requested_unavailable"));
    }
#if PLATFORM_WINDOWS
    if (!RhiName.Contains(TEXT("D3D12"), ESearchCase::IgnoreCase))
    {
        CompatibilityCodes.Add(TEXT("unexpected_windows_rhi"));
    }
#elif PLATFORM_MAC
    if (!RhiName.Contains(TEXT("Metal"), ESearchCase::IgnoreCase))
    {
        CompatibilityCodes.Add(TEXT("unexpected_macos_rhi"));
    }
#endif
    GCompatibilityStatus = CompatibilityCodes.IsEmpty()
        ? TEXT("ok")
        : FString::Join(CompatibilityCodes, TEXT(","));

    if (bRayTracingForcedOn && !bHardwareRayTracingAvailable)
    {
        UE_LOG(
            LogTemp,
            Warning,
            TEXT("Hardware ray tracing was requested but is unavailable; using the portable raster path"));
    }

    // TSR uses UE's cross-platform compute/pixel shader implementation on
    // DirectX 12 and Metal. A higher-resolution history is only enabled when
    // the detected GPU has enough local/unified memory for it.
    SetRenderCVar(TEXT("r.AntiAliasingMethod"), 4);
    SetRenderCVar(TEXT("r.PostProcessAAQuality"), bHighOrBetter ? 6 : 5);
    SetRenderCVar(TEXT("r.ScreenPercentage"), 100.0f);
    SetRenderCVar(TEXT("r.DynamicRes.OperationMode"), 0);
    SetRenderCVar(TEXT("r.TSR.History.R11G11B10"), 1);
    SetRenderCVar(TEXT("r.TSR.History.ScreenPercentage"), TsrHistoryPercentage);
    SetRenderCVar(TEXT("r.TSR.History.UpdateQuality"), bHighOrBetter ? 3 : 2);
    SetRenderCVar(TEXT("r.TSR.History.SampleCount"), bUltra ? 32 : (bHighOrBetter ? 24 : 16));
    SetRenderCVar(TEXT("r.TSR.ShadingRejection.Flickering"), 1);
    SetRenderCVar(TEXT("r.TSR.RejectionAntiAliasingQuality"), bUltra ? 3 : (bHighOrBetter ? 2 : 1));
    SetRenderCVar(TEXT("r.TSR.ReprojectionField"), 1);
    SetRenderCVar(TEXT("r.TSR.Resurrection"), 1);
    SetRenderCVar(TEXT("r.AnisotropicMaterials"), bHighOrBetter ? 1 : 0);

    const bool bPresentation4K = FParse::Param(FCommandLine::Get(), TEXT("QaiPresentation4K"));
    if (bPresentation4K)
    {
        // The camera remains still before capture, allowing the 200% TSR
        // history to converge over many frames at the final 3840x2160 output.
        SetRenderCVar(TEXT("r.TSR.History.ScreenPercentage"), 200.0f);
        SetRenderCVar(TEXT("r.TSR.History.SampleCount"), 64);
        SetRenderCVar(TEXT("r.TemporalAA.HistoryScreenPercentage"), 200.0f);
        SetRenderCVar(TEXT("r.TemporalAAFilterSize"), 0.7f);
    }

    // Ultra spends the available RT budget on hit lighting instead of the
    // lower-detail surface-cache approximation. High uses portable software
    // Lumen (including Apple Silicon); Balanced retains the SSR/SSAO path.
    SetRenderCVar(TEXT("r.DynamicGlobalIlluminationMethod"), bUseLumen ? 1 : 0);
    SetRenderCVar(TEXT("r.ReflectionMethod"), bUseLumen ? 1 : 2);
    SetRenderCVar(TEXT("r.Lumen.HardwareRayTracing"), bUseHardwareLumen ? 1 : 0);
    SetRenderCVar(TEXT("r.Lumen.HardwareRayTracing.LightingMode"), bUseHardwareLumen ? 1 : 0);
    SetRenderCVar(TEXT("r.Lumen.HardwareRayTracing.HitLighting.Allowed"), bUseHardwareLumen ? 1 : 0);
    SetRenderCVar(TEXT("r.Lumen.HardwareRayTracing.HitLighting.DirectLighting"), bUseHardwareLumen ? 1 : 0);
    SetRenderCVar(TEXT("r.Lumen.HardwareRayTracing.HitLighting.Skylight"), bUseHardwareLumen ? 1 : 0);
    SetRenderCVar(TEXT("r.Lumen.ScreenProbeGather.HardwareRayTracing"), bUseHardwareLumen ? 1 : 0);
    SetRenderCVar(TEXT("r.Lumen.Reflections.HardwareRayTracing"), bUseHardwareLumen ? 1 : 0);
    SetRenderCVar(TEXT("r.Lumen.Reflections.Allow"), bUseLumen ? 1 : 0);
    SetRenderCVar(TEXT("r.Lumen.Reflections.DownsampleFactor"), 1);
    SetRenderCVar(TEXT("r.Lumen.Reflections.MaxBounces"), bUltra ? 2 : 1);
    SetRenderCVar(TEXT("r.Lumen.Reflections.RadianceCache"), bUseLumen ? 1 : 0);
    SetRenderCVar(TEXT("r.Lumen.Reflections.Temporal"), 1);
    SetRenderCVar(TEXT("r.Lumen.ScreenProbeGather.DownsampleFactor"), bUltra ? 8 : 16);
    SetRenderCVar(TEXT("r.Lumen.ScreenProbeGather.TracingOctahedronResolution"), bUltra ? 16 : 8);
    SetRenderCVar(TEXT("r.Lumen.ScreenProbeGather.NumAdaptiveProbes"), bUltra ? 16 : 8);
    SetRenderCVar(TEXT("r.Lumen.ScreenProbeGather.ShortRangeAO"), bUseLumen ? 1 : 0);
    SetRenderCVar(
        TEXT("r.Lumen.ScreenProbeGather.ShortRangeAO.HardwareRayTracing"),
        bUseHardwareLumen ? 1 : 0);
    SetRenderCVar(TEXT("r.LumenScene.Radiosity.ProbeSpacing"), bUltra ? 2 : 4);
    SetRenderCVar(TEXT("r.LumenScene.Radiosity.HemisphereProbeResolution"), bUltra ? 8 : 4);
    // Lumen already supplies contact occlusion and diffuse bounce; running the
    // separate full-pipeline RTAO pass would duplicate work and hurt latency.
    SetRenderCVar(TEXT("r.RayTracing.AmbientOcclusion"), 0);

    SetRenderCVar(TEXT("r.MaxAnisotropy"), bHighOrBetter ? 16 : 8);
    SetRenderCVar(TEXT("r.SSR.Quality"), bHighOrBetter ? 3 : 2);
    SetRenderCVar(TEXT("r.SSR.HalfResSceneColor"), bHighOrBetter ? 0 : 1);
    SetRenderCVar(TEXT("r.SSR.Temporal"), 1);
    SetRenderCVar(TEXT("r.AmbientOcclusionMaxQuality"), bUltra ? 100.0f : (bHighOrBetter ? 80.0f : 60.0f));
    SetRenderCVar(TEXT("r.BloomQuality"), bHighOrBetter ? 5 : 4);
    SetRenderCVar(TEXT("r.VolumetricFog"), 1);
    SetRenderCVar(TEXT("r.VolumetricFog.GridPixelSize"), bUltra ? 4 : (bHighOrBetter ? 6 : 8));
    SetRenderCVar(TEXT("r.VolumetricFog.GridSizeZ"), bUltra ? 128 : (bHighOrBetter ? 96 : 64));
    SetRenderCVar(TEXT("r.VolumetricFog.HistoryWeight"), 0.88f);
    SetRenderCVar(TEXT("r.LocalFogVolume"), 1);
    SetRenderCVar(TEXT("r.LocalFogVolume.GlobalStartDistance"), 0.0f);
    // Voxelizing the local signal volumes allows the room-facing spotlights
    // to scatter through them; leaving them isolated produced flat colored
    // spheres with no actual light shaft.
    SetRenderCVar(TEXT("r.LocalFogVolume.RenderIntoVolumetricFog"), 1);
    SetRenderCVar(TEXT("r.LocalFogVolume.MaxDensityIntoVolumetricFog"), 0.03f);
    SetRenderCVar(TEXT("r.LocalFogVolume.UseHZB"), 0);
    SetRenderCVar(TEXT("r.LocalFogVolume.TilePixelSize"), bUltra ? 64 : (bHighOrBetter ? 96 : 128));
    SetRenderCVar(TEXT("r.Tonemapper.Quality"), 5);
    // Heavy output sharpening reconstructs stair steps after TSR. Keep only a
    // restrained clarity pass; Ultra spends its budget on the 200% history.
    const float TonemapperSharpen = bUltra ? 0.18f : (bHighOrBetter ? 0.14f : 0.10f);
    SetRenderCVar(TEXT("r.Tonemapper.Sharpen"), TonemapperSharpen);
    SetRenderCVar(TEXT("r.ShadowQuality"), 5);
    SetRenderCVar(TEXT("r.Shadow.MaxResolution"), bUltra ? 4096 : 2048);
    SetRenderCVar(TEXT("r.Shadow.CSM.MaxCascades"), bHighOrBetter ? 4 : 3);
    SetRenderCVar(TEXT("r.ContactShadows"), bHighOrBetter ? 1 : 0);
    SetRenderCVar(TEXT("r.AmbientOcclusionLevels"), bHighOrBetter ? 3 : 2);
    SetRenderCVar(TEXT("r.GTAO.SpatialFilter"), 1);
    SetRenderCVar(TEXT("r.Shadow.Virtual.Enable"), bUltra ? 1 : 0);
    SetRenderCVar(TEXT("r.Shadow.Virtual.MaxPhysicalPages"), bUltra ? 4096 : 2048);
    SetRenderCVar(TEXT("r.Shadow.Virtual.ResolutionLodBiasLocal"), bUltra ? -1.0f : 0.0f);
    SetRenderCVar(TEXT("r.Shadow.Virtual.ResolutionLodBiasLocalMoving"), bUltra ? 0.0f : 1.0f);
    SetRenderCVar(TEXT("r.Shadow.Virtual.SMRT.RayCountLocal"), bUltra ? 8 : 4);
    SetRenderCVar(TEXT("r.Shadow.Virtual.SMRT.SamplesPerRayLocal"), bUltra ? 8 : 4);
    SetRenderCVar(TEXT("r.Streaming.PoolSize"), bUltra ? 6144 : (bHighOrBetter ? 3072 : 1536));

    const TCHAR* LightingName = bUseHardwareLumen
        ? TEXT("hardware_lumen_hit_lighting")
        : (bUseLumen ? TEXT("software_lumen") : TEXT("raster_ssr"));
    GRenderProfileDisplayName = FString::Printf(
        TEXT("%s | TSR | %s"),
        TierName(Tier),
        bUseHardwareLumen ? TEXT("HW Lumen Hit RT") : (bUseLumen ? TEXT("SW Lumen") : TEXT("Raster SSR")));
    GRenderProfileSummary = FString::Printf(
        TEXT("render_profile tier=%s rhi=\"%s\" gpu=\"%s\" vendor_id=0x%04x device_id=0x%04x vram_mib=%llu driver_user=\"%s\" driver_internal=\"%s\" driver_date=\"%s\" driver_denylisted=%s aa=TSR history_pct=%d lighting=%s rt_available=%s rt_override=%s rt_geometry_pool_mib=%d tonemapper_sharpen=%.2f max_fps=%d compatibility=%s"),
        TierName(Tier),
        *SanitizeLogField(RhiName),
        *SanitizeLogField(AdapterName),
        GRHIVendorId,
        GRHIDeviceId,
        static_cast<unsigned long long>(VideoMemoryMiB),
        *SanitizeLogField(GRHIAdapterUserDriverVersion),
        *SanitizeLogField(GRHIAdapterInternalDriverVersion),
        *SanitizeLogField(GRHIAdapterDriverDate),
        GRHIAdapterDriverOnDenyList ? TEXT("true") : TEXT("false"),
        TsrHistoryPercentage,
        LightingName,
        bHardwareRayTracingAvailable ? TEXT("true") : TEXT("false"),
        *RayTracingOverride,
        RayTracingGeometryPoolMiB,
        TonemapperSharpen,
        FrameRateCap,
        *GCompatibilityStatus);
    UE_LOG(LogTemp, Display, TEXT("%s"), *GRenderProfileSummary);
}
