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

private:
    static void ApplyHardwareRenderProfile();
};
