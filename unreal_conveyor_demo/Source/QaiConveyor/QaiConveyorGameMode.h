#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "QaiConveyorGameMode.generated.h"

UCLASS()
class QAICONVEYOR_API AQaiConveyorGameMode : public AGameModeBase
{
    GENERATED_BODY()

public:
    AQaiConveyorGameMode();
    virtual void StartPlay() override;

    /** Human-readable auto-selected GPU render tier, also written to the simulator log. */
    static const FString& GetRenderProfileSummary();

    /** Compact render path displayed by the standalone HUD. */
    static const FString& GetRenderProfileDisplayName();

    /** Machine-readable compatibility warning codes, or "ok". */
    static const FString& GetCompatibilityStatus();

    /** Ultra is the only tier that budgets a shadowed physical emitter per stack lens. */
    static bool IsUltraRenderTier();

    /** True when Ultra is using hardware ray traced Lumen for secondary bounce/reflections. */
    static bool UsesHardwareLumen();

    /** True when Lumen GI/reflections are enabled for the active session. */
    static bool IsRuntimeLumenEnabled();

    /** True when the global on-demand ray-tracing scene is currently enabled. */
    static bool IsRuntimeRayTracingEnabled();

    /** True when this RHI/project can change r.RayTracing.Enable without a restart. */
    static bool SupportsRuntimeRayTracingToggle();

    /** Toggle Lumen GI/reflections while retaining a raster/SSR fallback. */
    static bool ToggleRuntimeLumen();

    /** Toggle the global on-demand ray-tracing scene. Returns the applied state. */
    static bool ToggleRuntimeRayTracing();

private:
    static void ApplyHardwareRenderProfile();
    static void ApplyInteractiveWindowDefaults();
};
