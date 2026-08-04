#pragma once

#include "CoreMinimal.h"
#include "GameFramework/HUD.h"
#include "QaiConveyorHUD.generated.h"

UCLASS()
class QAICONVEYOR_API AQaiConveyorHUD : public AHUD
{
    GENERATED_BODY()

public:
    virtual void DrawHUD() override;

private:
    double SmoothedFramesPerSecond = 0.0;
};
