#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "QaiConveyorPawn.generated.h"

class AQaiConveyorWorld;
class UCameraComponent;
class USceneComponent;
class USpringArmComponent;

UCLASS()
class QAICONVEYOR_API AQaiConveyorPawn : public APawn
{
    GENERATED_BODY()

public:
    AQaiConveyorPawn();
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaSeconds) override;
    virtual void SetupPlayerInputComponent(UInputComponent* PlayerInputComponent) override;

private:
    UPROPERTY()
    TObjectPtr<USceneComponent> Root;

    UPROPERTY()
    TObjectPtr<USpringArmComponent> SpringArm;

    UPROPERTY()
    TObjectPtr<UCameraComponent> Camera;

    UPROPERTY()
    TObjectPtr<AQaiConveyorWorld> Runtime;

    float ThrottleInput = 0.0f;
    float SteerInput = 0.0f;
    float LiftInput = 0.0f;
    float CameraYawRateInput = 0.0f;
    float CameraPitchRateInput = 0.0f;
    float OrbitYaw = 118.0f;
    float OrbitPitch = 68.0f;
    bool bBrake = false;
    bool bLoggedInputInitialization = false;
    bool bLoggedGamepadInput = false;
    bool bLoggedGamepadUnavailable = false;
    bool bDiagnosticCamera = false;
    bool bIntroCameraActive = true;
    bool bIntroCameraInitialized = false;
    bool bIntroCameraTransitionActive = false;
    float GamepadProbeSeconds = 0.0f;
    float IntroCameraTransitionElapsed = 0.0f;
    uint64 NativeGamepadButtons = 0;
    FVector DiagnosticCameraTarget = FVector::ZeroVector;
    FVector IntroTransitionStartTarget = FVector::ZeroVector;
    float IntroTransitionStartArmLength = 62.0f;
    float IntroTransitionStartYaw = -42.0f;
    float IntroTransitionStartPitch = 55.0f;
    float IntroTransitionStartFieldOfView = 32.0f;
    int32 ViewIndex = 2;

    void InputThrottle(float Value);
    void InputSteer(float Value);
    void InputLift(float Value);
    void InputCameraYawMouse(float Value);
    void InputCameraPitchMouse(float Value);
    void InputCameraYawGamepad(float Value);
    void InputCameraPitchGamepad(float Value);
    void InputCameraZoom(float Value);
    void BrakePressed();
    void BrakeReleased();
    void CycleForklift();
    void PreviousForklift();
    void NextForklift();
    void SelectForklift1();
    void SelectForklift2();
    void ResetScene();
    void CycleView();
    void ToggleBackend();
    void ToggleInference();
    void ToggleCollisionDebug();
    void QuitDemo();
    void AnyInputPressed();
    bool HandleIntroInput();
    void BeginIntroCameraTransition();
    void UpdateIntroCamera(float DeltaSeconds);
    void FindRuntime();
    void PollGamepadFallback(
        float DeltaSeconds,
        float& OutThrottle,
        float& OutSteer,
        float& OutLift,
        float& OutCameraYaw,
        float& OutCameraPitch,
        bool& OutBrake);
};
