#include "QaiConveyorPawn.h"

#include "Camera/CameraComponent.h"
#include "EngineUtils.h"
#include "GameFramework/PlayerController.h"
#include "GameFramework/SpringArmComponent.h"
#include "InputCoreTypes.h"
#include "Kismet/KismetSystemLibrary.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "QaiConveyorWorld.h"

#if PLATFORM_WINDOWS
#include "GameInputBaseModule.h"
#endif

namespace QaiCameraTuning
{
    constexpr float IntroArmLengthCm = 62.0f;
    constexpr float IntroYawDegrees = -42.0f;
    constexpr float IntroPitchDegrees = 55.0f;
    constexpr float IntroFieldOfView = 32.0f;
    constexpr float IntroFocusHeightCm = 1.5f;
    constexpr float TransitionSeconds = 1.55f;
    constexpr float GameplayArmLengthCm = 350.0f;
    constexpr float GameplayYawDegrees = -45.0f;
    constexpr float GameplayPitchDegrees = 32.0f;
    constexpr float GameplayFieldOfView = 75.0f;
    constexpr float GameplayFocusHeightCm = 125.0f;
}

AQaiConveyorPawn::AQaiConveyorPawn()
{
    PrimaryActorTick.bCanEverTick = true;

    Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
    SetRootComponent(Root);

    SpringArm = CreateDefaultSubobject<USpringArmComponent>(TEXT("SpringArm"));
    SpringArm->SetupAttachment(Root);
    SpringArm->TargetArmLength = 900.0f;
    SpringArm->bDoCollisionTest = false;
    SpringArm->bEnableCameraLag = true;
    SpringArm->CameraLagSpeed = 8.0f;

    Camera = CreateDefaultSubobject<UCameraComponent>(TEXT("Camera"));
    Camera->SetupAttachment(SpringArm, USpringArmComponent::SocketName);
    Camera->FieldOfView = 75.0f;

    // A locked EV is the portable manual-exposure baseline. It prevents signal
    // lights from pumping the camera while the gentler film toe and global
    // grade keep neutral walls bright without clipping parcel-label detail.
    Camera->PostProcessSettings.bOverride_AutoExposureMinBrightness = true;
    Camera->PostProcessSettings.AutoExposureMinBrightness = 7.65f;
    Camera->PostProcessSettings.bOverride_AutoExposureMaxBrightness = true;
    Camera->PostProcessSettings.AutoExposureMaxBrightness = 7.65f;
    Camera->PostProcessSettings.bOverride_LocalExposureMethod = true;
    Camera->PostProcessSettings.LocalExposureMethod = ELocalExposureMethod::Bilateral;
    Camera->PostProcessSettings.bOverride_LocalExposureHighlightContrastScale = true;
    Camera->PostProcessSettings.LocalExposureHighlightContrastScale = 0.66f;
    Camera->PostProcessSettings.bOverride_LocalExposureShadowContrastScale = true;
    Camera->PostProcessSettings.LocalExposureShadowContrastScale = 0.78f;
    Camera->PostProcessSettings.bOverride_LocalExposureDetailStrength = true;
    Camera->PostProcessSettings.LocalExposureDetailStrength = 1.04f;
    Camera->PostProcessSettings.bOverride_LocalExposureBlurredLuminanceBlend = true;
    Camera->PostProcessSettings.LocalExposureBlurredLuminanceBlend = 0.62f;
    Camera->PostProcessSettings.bOverride_WhiteTemp = true;
    Camera->PostProcessSettings.WhiteTemp = 6500.0f;
    Camera->PostProcessSettings.bOverride_WhiteTint = true;
    Camera->PostProcessSettings.WhiteTint = 0.0f;
    Camera->PostProcessSettings.bOverride_ColorSaturation = true;
    Camera->PostProcessSettings.ColorSaturation = FVector4(0.98f, 0.98f, 0.98f, 1.0f);
    Camera->PostProcessSettings.bOverride_ColorContrast = true;
    Camera->PostProcessSettings.ColorContrast = FVector4(0.94f, 0.94f, 0.94f, 1.0f);
    Camera->PostProcessSettings.bOverride_ColorGamma = true;
    Camera->PostProcessSettings.ColorGamma = FVector4(1.02f, 1.02f, 1.02f, 1.0f);
    Camera->PostProcessSettings.bOverride_ColorGain = true;
    Camera->PostProcessSettings.ColorGain = FVector4(1.025f, 1.025f, 1.025f, 1.0f);
    Camera->PostProcessSettings.bOverride_FilmSlope = true;
    Camera->PostProcessSettings.FilmSlope = 0.82f;
    Camera->PostProcessSettings.bOverride_FilmToe = true;
    Camera->PostProcessSettings.FilmToe = 0.42f;
    Camera->PostProcessSettings.bOverride_FilmShoulder = true;
    Camera->PostProcessSettings.FilmShoulder = 0.22f;
    Camera->PostProcessSettings.bOverride_FilmBlackClip = true;
    Camera->PostProcessSettings.FilmBlackClip = 0.0f;
    Camera->PostProcessSettings.bOverride_FilmWhiteClip = true;
    Camera->PostProcessSettings.FilmWhiteClip = 0.0f;
    Camera->PostProcessSettings.bOverride_BloomIntensity = true;
    Camera->PostProcessSettings.BloomIntensity = 0.50f;
    Camera->PostProcessSettings.bOverride_BloomThreshold = true;
    Camera->PostProcessSettings.BloomThreshold = 0.32f;
    Camera->PostProcessSettings.bOverride_VignetteIntensity = true;
    Camera->PostProcessSettings.VignetteIntensity = 0.12f;
}

void AQaiConveyorPawn::BeginPlay()
{
    Super::BeginPlay();
    FindRuntime();

    FString CameraPreset;
    if (FParse::Value(FCommandLine::Get(), TEXT("QaiCameraPreset="), CameraPreset))
    {
        bDiagnosticCamera = true;
        SpringArm->bEnableCameraLag = false;
        if (CameraPreset.Equals(TEXT("stack"), ESearchCase::IgnoreCase))
        {
            DiagnosticCameraTarget = FVector(385.0f, 35.0f, 155.0f);
            SpringArm->TargetArmLength = 660.0f;
            OrbitYaw = -62.0f;
            OrbitPitch = 29.0f;
            Camera->FieldOfView = 52.0f;
        }
        else if (CameraPreset.Equals(TEXT("signal"), ESearchCase::IgnoreCase))
        {
            DiagnosticCameraTarget = FVector(435.0f, -337.0f, 194.0f);
            SpringArm->TargetArmLength = 330.0f;
            OrbitYaw = -90.0f;
            OrbitPitch = 8.0f;
            Camera->FieldOfView = 45.0f;
        }
        else if (CameraPreset.Equals(TEXT("forklift"), ESearchCase::IgnoreCase))
        {
            DiagnosticCameraTarget = FVector(-70.0f, 85.0f, 125.0f);
            SpringArm->TargetArmLength = 780.0f;
            OrbitYaw = -62.0f;
            OrbitPitch = 31.0f;
            Camera->FieldOfView = 56.0f;
        }
        else if (CameraPreset.Equals(TEXT("reverse"), ESearchCase::IgnoreCase))
        {
            DiagnosticCameraTarget = FVector(105.0f, 50.0f, 110.0f);
            SpringArm->TargetArmLength = 1120.0f;
            OrbitYaw = -105.0f;
            OrbitPitch = 34.0f;
            Camera->FieldOfView = 58.0f;
        }
        else if (CameraPreset.Equals(TEXT("aisle"), ESearchCase::IgnoreCase))
        {
            DiagnosticCameraTarget = FVector(-65.0f, 80.0f, 115.0f);
            SpringArm->TargetArmLength = 735.0f;
            OrbitYaw = -96.0f;
            OrbitPitch = 19.0f;
            Camera->FieldOfView = 55.0f;
        }
        else if (CameraPreset.Equals(TEXT("conveyor"), ESearchCase::IgnoreCase))
        {
            DiagnosticCameraTarget = FVector(300.0f, 25.0f, 105.0f);
            SpringArm->TargetArmLength = 760.0f;
            OrbitYaw = -18.0f;
            OrbitPitch = 25.0f;
            Camera->FieldOfView = 54.0f;
        }
        else if (CameraPreset.Equals(TEXT("evktop"), ESearchCase::IgnoreCase)
            || CameraPreset.Equals(TEXT("iq9top"), ESearchCase::IgnoreCase))
        {
            DiagnosticCameraTarget = FVector(-95.6f, -312.3f, 112.2f);
            SpringArm->TargetArmLength = 55.0f;
            OrbitYaw = -42.0f;
            OrbitPitch = 58.0f;
            Camera->FieldOfView = 28.0f;
        }
        else if (CameraPreset.Equals(TEXT("evk"), ESearchCase::IgnoreCase))
        {
            DiagnosticCameraTarget = FVector(-95.6f, -312.3f, 112.0f);
            SpringArm->TargetArmLength = 145.0f;
            OrbitYaw = -90.0f;
            OrbitPitch = 42.0f;
            Camera->FieldOfView = 40.0f;
        }
        else if (CameraPreset.Equals(TEXT("evkfront"), ESearchCase::IgnoreCase))
        {
            DiagnosticCameraTarget = FVector(-95.6f, -312.3f, 112.2f);
            SpringArm->TargetArmLength = 42.0f;
            OrbitYaw = 168.0f;
            OrbitPitch = 32.0f;
            Camera->FieldOfView = 32.0f;
        }
        else if (CameraPreset.Equals(TEXT("evkback"), ESearchCase::IgnoreCase))
        {
            DiagnosticCameraTarget = FVector(-95.6f, -312.3f, 112.2f);
            SpringArm->TargetArmLength = 42.0f;
            OrbitYaw = -12.0f;
            OrbitPitch = 32.0f;
            Camera->FieldOfView = 32.0f;
        }
        else if (CameraPreset.Equals(TEXT("evkright"), ESearchCase::IgnoreCase))
        {
            DiagnosticCameraTarget = FVector(-95.6f, -312.3f, 112.2f);
            SpringArm->TargetArmLength = 42.0f;
            OrbitYaw = -102.0f;
            OrbitPitch = 32.0f;
            Camera->FieldOfView = 32.0f;
        }
        else if (CameraPreset.Equals(TEXT("evkleft"), ESearchCase::IgnoreCase))
        {
            DiagnosticCameraTarget = FVector(-95.6f, -312.3f, 112.2f);
            SpringArm->TargetArmLength = 42.0f;
            OrbitYaw = 78.0f;
            OrbitPitch = 32.0f;
            Camera->FieldOfView = 32.0f;
        }
        else
        {
            DiagnosticCameraTarget = FVector(110.0f, 55.0f, 115.0f);
            SpringArm->TargetArmLength = 1280.0f;
            OrbitYaw = -62.0f;
            OrbitPitch = 38.0f;
            Camera->FieldOfView = 58.0f;
        }
    }

    bIntroCameraActive = !bDiagnosticCamera
        && !FParse::Param(FCommandLine::Get(), TEXT("QaiSkipIntro"));
    if (bIntroCameraActive)
    {
        // The transition below is already temporally eased, so spring-arm lag
        // stays off until normal forklift control begins.
        SpringArm->bEnableCameraLag = false;
        UpdateIntroCamera(0.0f);
    }

    if (APlayerController* PC = Cast<APlayerController>(GetController()))
    {
        PC->SetShowMouseCursor(false);
        FInputModeGameOnly Mode;
        PC->SetInputMode(Mode);
    }
}

void AQaiConveyorPawn::FindRuntime()
{
    if (Runtime || !GetWorld())
    {
        return;
    }
    for (TActorIterator<AQaiConveyorWorld> It(GetWorld()); It; ++It)
    {
        Runtime = *It;
        return;
    }
}

void AQaiConveyorPawn::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);
    FindRuntime();
    if (!Runtime)
    {
        return;
    }
    if (!bLoggedInputInitialization)
    {
        bLoggedInputInitialization = true;
#if PLATFORM_WINDOWS
        Runtime->RecordControllerInput(TEXT("input_initialized keyboard_mouse=true gamepad=true backend=GameInputWindows+direct_poll"));
#elif PLATFORM_MAC
        Runtime->RecordControllerInput(TEXT("input_initialized keyboard_mouse=true gamepad=true backend=macos_generic+direct_poll"));
#else
        Runtime->RecordControllerInput(TEXT("input_initialized keyboard_mouse=true gamepad=true backend=generic+direct_poll"));
#endif
    }

    float AppliedThrottle = ThrottleInput;
    float AppliedSteer = SteerInput;
    float AppliedLift = LiftInput;
    float AppliedCameraYaw = CameraYawRateInput;
    float AppliedCameraPitch = CameraPitchRateInput;
    bool bAppliedBrake = bBrake;
    PollGamepadFallback(
        DeltaSeconds,
        AppliedThrottle,
        AppliedSteer,
        AppliedLift,
        AppliedCameraYaw,
        AppliedCameraPitch,
        bAppliedBrake);

    if (bIntroCameraActive || bIntroCameraTransitionActive)
    {
        Runtime->DriveActiveForklift(0.0f, 0.0f, 0.0f, false, DeltaSeconds);
        UpdateIntroCamera(DeltaSeconds);
        return;
    }

    Runtime->DriveActiveForklift(AppliedThrottle, AppliedSteer, AppliedLift, bAppliedBrake, DeltaSeconds);
    OrbitYaw = FRotator::NormalizeAxis(OrbitYaw + AppliedCameraYaw * 105.0f * DeltaSeconds);
    OrbitPitch = FMath::Clamp(OrbitPitch + AppliedCameraPitch * 75.0f * DeltaSeconds, 10.0f, 70.0f);
    const FVector Target = bDiagnosticCamera
        ? DiagnosticCameraTarget
        : (ViewIndex == 2
            ? FVector(120.0f, 50.0f, 100.0f)
            : Runtime->GetActiveForkliftLocation() + FVector(0.0f, 0.0f, 125.0f));
    SetActorLocation(Target);
    SpringArm->SetRelativeRotation(FRotator(-OrbitPitch, OrbitYaw, 0.0f));
}

void AQaiConveyorPawn::UpdateIntroCamera(float DeltaSeconds)
{
    if (!Runtime)
    {
        return;
    }

    if (bIntroCameraActive)
    {
        if (!Runtime->IsStageReady())
        {
            return;
        }

        const FVector EvkTarget = Runtime->GetEvkLocation()
            + FVector(0.0f, 0.0f, QaiCameraTuning::IntroFocusHeightCm);
        SetActorLocation(EvkTarget);
        SpringArm->TargetArmLength = QaiCameraTuning::IntroArmLengthCm;
        SpringArm->SetRelativeRotation(FRotator(
            -QaiCameraTuning::IntroPitchDegrees,
            QaiCameraTuning::IntroYawDegrees,
            0.0f));
        Camera->FieldOfView = QaiCameraTuning::IntroFieldOfView;

        if (!bIntroCameraInitialized)
        {
            bIntroCameraInitialized = true;
            Runtime->RecordControllerInput(FString::Printf(
                TEXT("intro_camera_ready subject=IQ9_EVK arm_cm=%.0f yaw=%.0f pitch=%.0f fov=%.0f dismiss=any_button"),
                QaiCameraTuning::IntroArmLengthCm,
                QaiCameraTuning::IntroYawDegrees,
                QaiCameraTuning::IntroPitchDegrees,
                QaiCameraTuning::IntroFieldOfView));
        }
        return;
    }

    if (!bIntroCameraTransitionActive)
    {
        return;
    }

    IntroCameraTransitionElapsed += DeltaSeconds;
    const float Alpha = FMath::Clamp(
        IntroCameraTransitionElapsed / QaiCameraTuning::TransitionSeconds,
        0.0f,
        1.0f);
    const float EasedAlpha = Alpha * Alpha * (3.0f - 2.0f * Alpha);
    const FVector GameplayTarget = Runtime->GetActiveForkliftLocation()
        + FVector(0.0f, 0.0f, QaiCameraTuning::GameplayFocusHeightCm);
    const float YawDelta = FMath::FindDeltaAngleDegrees(
        IntroTransitionStartYaw,
        QaiCameraTuning::GameplayYawDegrees);

    SetActorLocation(FMath::Lerp(IntroTransitionStartTarget, GameplayTarget, EasedAlpha));
    SpringArm->TargetArmLength = FMath::Lerp(
        IntroTransitionStartArmLength,
        QaiCameraTuning::GameplayArmLengthCm,
        EasedAlpha);
    SpringArm->SetRelativeRotation(FRotator(
        -FMath::Lerp(IntroTransitionStartPitch, QaiCameraTuning::GameplayPitchDegrees, EasedAlpha),
        IntroTransitionStartYaw + YawDelta * EasedAlpha,
        0.0f));
    Camera->FieldOfView = FMath::Lerp(
        IntroTransitionStartFieldOfView,
        QaiCameraTuning::GameplayFieldOfView,
        EasedAlpha);

    if (Alpha >= 1.0f)
    {
        bIntroCameraTransitionActive = false;
        SpringArm->bEnableCameraLag = true;
        Runtime->RecordControllerInput(FString::Printf(
            TEXT("intro_camera_transition_complete forklift=1 view=1 arm_cm=%.0f duration_s=%.2f"),
            QaiCameraTuning::GameplayArmLengthCm,
            QaiCameraTuning::TransitionSeconds));
    }
}

void AQaiConveyorPawn::BeginIntroCameraTransition()
{
    if (!bIntroCameraActive || bDiagnosticCamera || !Runtime)
    {
        return;
    }

    if (!bIntroCameraInitialized)
    {
        UpdateIntroCamera(0.0f);
        if (!bIntroCameraInitialized)
        {
            return;
        }
    }
    bIntroCameraActive = false;
    bIntroCameraTransitionActive = true;
    IntroCameraTransitionElapsed = 0.0f;
    IntroTransitionStartTarget = GetActorLocation();
    IntroTransitionStartArmLength = SpringArm->TargetArmLength;
    const FRotator StartRotation = SpringArm->GetRelativeRotation();
    IntroTransitionStartYaw = StartRotation.Yaw;
    IntroTransitionStartPitch = -StartRotation.Pitch;
    IntroTransitionStartFieldOfView = Camera->FieldOfView;

    Runtime->SelectForklift(0);
    ViewIndex = 0;
    OrbitYaw = QaiCameraTuning::GameplayYawDegrees;
    OrbitPitch = QaiCameraTuning::GameplayPitchDegrees;
    Runtime->RecordControllerInput(FString::Printf(
        TEXT("intro_camera_dismissed destination=forklift_1 view=1 transition_s=%.2f"),
        QaiCameraTuning::TransitionSeconds));
}

bool AQaiConveyorPawn::HandleIntroInput()
{
    if (bDiagnosticCamera)
    {
        return false;
    }
    if (bIntroCameraActive)
    {
        BeginIntroCameraTransition();
        return true;
    }
    return bIntroCameraTransitionActive;
}

void AQaiConveyorPawn::AnyInputPressed()
{
    HandleIntroInput();
}

void AQaiConveyorPawn::PollGamepadFallback(
    float DeltaSeconds,
    float& OutThrottle,
    float& OutSteer,
    float& OutLift,
    float& OutCameraYaw,
    float& OutCameraPitch,
    bool& OutBrake)
{
    GamepadProbeSeconds += DeltaSeconds;

#if PLATFORM_WINDOWS && GAME_INPUT_SUPPORT
    // Poll GameInput directly, independently of UE platform-user assignment.
    // Prefer the standardized gamepad state, then map generic HID controllers
    // (DualSense, 8BitDo and similar devices) onto the same canonical layout.
    if (IGameInput* GameInput = FGameInputBaseModule::GetGameInput())
    {
        const auto ApplyDeadZone = [](float Value, float DeadZone)
        {
            return FMath::Abs(Value) < DeadZone ? 0.0f : Value;
        };
        const auto ApplyNativeState = [this, &OutThrottle, &OutSteer, &OutLift,
            &OutCameraYaw, &OutCameraPitch, &OutBrake, &ApplyDeadZone](
            float LeftX,
            float LeftY,
            float RightX,
            float RightY,
            float LeftTrigger,
            float RightTrigger,
            uint64 Buttons,
            const TCHAR* Backend)
        {
            // Forklift controls deliberately mirror a vehicle: RT accelerates
            // forward, LT reverses, while the left stick only steers.
            const uint64 Pressed = Buttons & ~NativeGamepadButtons;
            NativeGamepadButtons = Buttons;
            const bool bMeaningfulAnalogInput = FMath::Abs(LeftX) > 0.25f
                || FMath::Abs(LeftY) > 0.25f
                || FMath::Abs(RightX) > 0.25f
                || FMath::Abs(RightY) > 0.25f
                || LeftTrigger > 0.12f
                || RightTrigger > 0.12f;
            if ((Pressed != 0 || bMeaningfulAnalogInput) && HandleIntroInput())
            {
                OutThrottle = 0.0f;
                OutSteer = 0.0f;
                OutLift = 0.0f;
                OutCameraYaw = 0.0f;
                OutCameraPitch = 0.0f;
                OutBrake = false;
                return;
            }

            OutThrottle = ApplyDeadZone(RightTrigger - LeftTrigger, 0.08f);
            OutSteer = ApplyDeadZone(LeftX, 0.12f);
            OutLift = ((Buttons & static_cast<uint64>(GameInputGamepadDPadUp)) != 0 ? 1.0f : 0.0f)
                - ((Buttons & static_cast<uint64>(GameInputGamepadDPadDown)) != 0 ? 1.0f : 0.0f);
            OutCameraYaw = ApplyDeadZone(RightX, 0.12f);
            OutCameraPitch = -ApplyDeadZone(RightY, 0.12f);
            OutBrake = OutBrake || (Buttons & static_cast<uint64>(GameInputGamepadA)) != 0;
            if (Runtime)
            {
                if ((Pressed & static_cast<uint64>(GameInputGamepadB)) != 0)
                {
                    Runtime->CycleForklift();
                }
                if ((Pressed & static_cast<uint64>(GameInputGamepadX)) != 0)
                {
                    ResetScene();
                }
                if ((Pressed & static_cast<uint64>(GameInputGamepadY)) != 0)
                {
                    CycleView();
                }
                if ((Pressed & static_cast<uint64>(GameInputGamepadLeftShoulder)) != 0)
                {
                    Runtime->CycleForklift(-1);
                }
                if ((Pressed & static_cast<uint64>(GameInputGamepadRightShoulder)) != 0)
                {
                    Runtime->CycleForklift(1);
                }
                if ((Pressed & static_cast<uint64>(GameInputGamepadMenu)) != 0)
                {
                    Runtime->ToggleBackend();
                }
                if ((Pressed & static_cast<uint64>(GameInputGamepadView)) != 0)
                {
                    CycleView();
                }
            }

            if (!bLoggedGamepadInput)
            {
                bLoggedGamepadInput = true;
                const FString Details = FString::Printf(
                    TEXT("controller_detected backend=%s lx=%.2f ly=%.2f rx=%.2f ry=%.2f lt=%.2f rt=%.2f"),
                    Backend,
                    LeftX,
                    LeftY,
                    RightX,
                    RightY,
                    LeftTrigger,
                    RightTrigger);
                UE_LOG(LogTemp, Display, TEXT("%s"), *Details);
                Runtime->RecordControllerInput(Details);
            }
        };

        IGameInputReading* Reading = nullptr;
        const HRESULT Result = GameInput->GetCurrentReading(GameInputKindGamepad, nullptr, &Reading);
        if (SUCCEEDED(Result) && Reading)
        {
            GameInputGamepadState State{};
            const bool bStateReady = Reading->GetGamepadState(&State);
            Reading->Release();
            if (bStateReady)
            {
                ApplyNativeState(
                    State.leftThumbstickX,
                    State.leftThumbstickY,
                    State.rightThumbstickX,
                    State.rightThumbstickY,
                    State.leftTrigger,
                    State.rightTrigger,
                    static_cast<uint64>(State.buttons),
                    TEXT("GameInputStandard"));
                return;
            }
        }

        Reading = nullptr;
        const HRESULT GenericResult = GameInput->GetCurrentReading(
            GameInputKindController, nullptr, &Reading);
        if (SUCCEEDED(GenericResult) && Reading)
        {
            const uint32 AxisCount = Reading->GetControllerAxisCount();
            const uint32 ButtonCount = Reading->GetControllerButtonCount();
            TArray<float> Axes;
            TArray<bool> ButtonStates;
            Axes.AddZeroed(AxisCount);
            ButtonStates.AddZeroed(ButtonCount);
            const bool bAxesReady = AxisCount > 0
                && Reading->GetControllerAxisState(AxisCount, Axes.GetData()) > 0;
            const bool bButtonsReady = ButtonCount > 0
                && Reading->GetControllerButtonState(ButtonCount, ButtonStates.GetData()) > 0;

            IGameInputDevice* Device = nullptr;
            IGameInputMapper* Mapper = nullptr;
            Reading->GetDevice(&Device);
            if (Device)
            {
                Device->CreateInputMapper(&Mapper);
            }

            const auto ReadAxis = [&Axes, Mapper](
                GameInputGamepadAxes CanonicalAxis,
                uint32 FallbackIndex,
                bool bSigned,
                bool bFallbackInverted)
            {
                uint32 SourceIndex = FallbackIndex;
                bool bInverted = bFallbackInverted;
                GameInputAxisMapping Mapping{};
                if (Mapper
                    && Mapper->GetGamepadAxisMappingInfo(CanonicalAxis, &Mapping)
                    && Mapping.controllerElementKind == GameInputElementKindAxis)
                {
                    SourceIndex = Mapping.controllerIndex;
                    bInverted = Mapping.isInverted;
                }
                if (!Axes.IsValidIndex(static_cast<int32>(SourceIndex)))
                {
                    return 0.0f;
                }
                float Value = FMath::Clamp(Axes[SourceIndex], 0.0f, 1.0f);
                if (bInverted)
                {
                    Value = 1.0f - Value;
                }
                return bSigned ? Value * 2.0f - 1.0f : Value;
            };
            const auto ReadButton = [&Axes, &ButtonStates, Mapper](
                GameInputGamepadButtons CanonicalButton,
                uint32 FallbackIndex)
            {
                GameInputButtonMapping Mapping{};
                if (Mapper && Mapper->GetGamepadButtonMappingInfo(CanonicalButton, &Mapping))
                {
                    if (Mapping.controllerElementKind == GameInputElementKindButton
                        && ButtonStates.IsValidIndex(static_cast<int32>(Mapping.controllerIndex)))
                    {
                        return ButtonStates[Mapping.controllerIndex];
                    }
                    if (Mapping.controllerElementKind == GameInputElementKindAxis
                        && Axes.IsValidIndex(static_cast<int32>(Mapping.controllerIndex)))
                    {
                        const float Value = Axes[Mapping.controllerIndex];
                        return Mapping.isInverted ? Value < 0.35f : Value > 0.65f;
                    }
                }
                return ButtonStates.IsValidIndex(static_cast<int32>(FallbackIndex))
                    && ButtonStates[FallbackIndex];
            };

            if (bAxesReady || bButtonsReady)
            {
                uint64 Buttons = 0;
                const auto AddButton = [&Buttons, &ReadButton](
                    GameInputGamepadButtons Button, uint32 FallbackIndex)
                {
                    if (ReadButton(Button, FallbackIndex))
                    {
                        Buttons |= static_cast<uint64>(Button);
                    }
                };
                AddButton(GameInputGamepadA, 0);
                AddButton(GameInputGamepadB, 1);
                AddButton(GameInputGamepadX, 2);
                AddButton(GameInputGamepadY, 3);
                AddButton(GameInputGamepadLeftShoulder, 4);
                AddButton(GameInputGamepadRightShoulder, 5);
                AddButton(GameInputGamepadView, 6);
                AddButton(GameInputGamepadMenu, 7);
                AddButton(GameInputGamepadDPadUp, 10);
                AddButton(GameInputGamepadDPadDown, 11);
                ApplyNativeState(
                    ReadAxis(GameInputGamepadLeftThumbstickX, 0, true, false),
                    ReadAxis(GameInputGamepadLeftThumbstickY, 1, true, true),
                    ReadAxis(GameInputGamepadRightThumbstickX, 2, true, false),
                    ReadAxis(GameInputGamepadRightThumbstickY, 3, true, true),
                    ReadAxis(GameInputGamepadLeftTrigger, 4, false, false),
                    ReadAxis(GameInputGamepadRightTrigger, 5, false, false),
                    Buttons,
                    TEXT("GameInputGenericMapped"));
            }
            if (Mapper)
            {
                Mapper->Release();
            }
            if (Device)
            {
                Device->Release();
            }
            Reading->Release();
            if (bAxesReady || bButtonsReady)
            {
                return;
            }
        }
    }
#endif

    APlayerController* PC = Cast<APlayerController>(GetController());
    if (!PC)
    {
        return;
    }

    const float LeftX = PC->GetInputAnalogKeyState(EKeys::Gamepad_LeftX);
    const float LeftY = PC->GetInputAnalogKeyState(EKeys::Gamepad_LeftY);
    const float RightX = PC->GetInputAnalogKeyState(EKeys::Gamepad_RightX);
    const float RightY = PC->GetInputAnalogKeyState(EKeys::Gamepad_RightY);
    const float LeftTrigger = PC->GetInputAnalogKeyState(EKeys::Gamepad_LeftTriggerAxis);
    const float RightTrigger = PC->GetInputAnalogKeyState(EKeys::Gamepad_RightTriggerAxis);
    const bool bLiftUp = PC->IsInputKeyDown(EKeys::Gamepad_DPad_Up);
    const bool bLiftDown = PC->IsInputKeyDown(EKeys::Gamepad_DPad_Down);
    const bool bHasAnalogInput = FMath::Abs(LeftX) > 0.12f
        || FMath::Abs(LeftY) > 0.12f
        || FMath::Abs(RightX) > 0.12f
        || FMath::Abs(RightY) > 0.12f
        || LeftTrigger > 0.08f
        || RightTrigger > 0.08f
        || bLiftUp
        || bLiftDown;
    if (!bHasAnalogInput)
    {
        if (!bLoggedGamepadUnavailable && GamepadProbeSeconds >= 5.0f)
        {
            bLoggedGamepadUnavailable = true;
#if PLATFORM_WINDOWS
            Runtime->RecordControllerInput(TEXT("controller_not_detected backend=GameInputDirect waited_seconds=5 hint=connect_usb_or_bluetooth_controller"));
#else
            Runtime->RecordControllerInput(TEXT("controller_not_detected backend=platform waited_seconds=5 hint=connect_usb_or_bluetooth_controller"));
#endif
        }
        return;
    }
    if (HandleIntroInput())
    {
        OutThrottle = 0.0f;
        OutSteer = 0.0f;
        OutLift = 0.0f;
        OutCameraYaw = 0.0f;
        OutCameraPitch = 0.0f;
        OutBrake = false;
        return;
    }

    // This direct poll is deliberate. It keeps standard controller axes live
    // even when a platform backend reports keys but skips legacy axis delegates.
    OutThrottle = FMath::Abs(RightTrigger - LeftTrigger) < 0.08f ? 0.0f : RightTrigger - LeftTrigger;
    OutSteer = FMath::Abs(LeftX) < 0.12f ? 0.0f : LeftX;
    OutLift = (bLiftUp ? 1.0f : 0.0f) - (bLiftDown ? 1.0f : 0.0f);
    OutCameraYaw = FMath::Abs(RightX) < 0.12f ? 0.0f : RightX;
    OutCameraPitch = FMath::Abs(RightY) < 0.12f ? 0.0f : -RightY;
    OutBrake = OutBrake || PC->IsInputKeyDown(EKeys::Gamepad_FaceButton_Bottom);

    if (!bLoggedGamepadInput)
    {
        bLoggedGamepadInput = true;
        const FString Details = FString::Printf(
            TEXT("controller_input_active lx=%.2f ly=%.2f rx=%.2f ry=%.2f lt=%.2f rt=%.2f"),
            LeftX, LeftY, RightX, RightY, LeftTrigger, RightTrigger);
        UE_LOG(LogTemp, Display, TEXT("%s"), *Details);
        Runtime->RecordControllerInput(Details);
    }
}

void AQaiConveyorPawn::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
    Super::SetupPlayerInputComponent(PlayerInputComponent);
    PlayerInputComponent->BindKey(
        EKeys::AnyKey,
        IE_Pressed,
        this,
        &AQaiConveyorPawn::AnyInputPressed).bConsumeInput = false;
    PlayerInputComponent->BindAxis(TEXT("Throttle"), this, &AQaiConveyorPawn::InputThrottle);
    PlayerInputComponent->BindAxis(TEXT("Steer"), this, &AQaiConveyorPawn::InputSteer);
    PlayerInputComponent->BindAxis(TEXT("Lift"), this, &AQaiConveyorPawn::InputLift);
    PlayerInputComponent->BindAxis(TEXT("CameraYawMouse"), this, &AQaiConveyorPawn::InputCameraYawMouse);
    PlayerInputComponent->BindAxis(TEXT("CameraPitchMouse"), this, &AQaiConveyorPawn::InputCameraPitchMouse);
    PlayerInputComponent->BindAxis(TEXT("CameraYawGamepad"), this, &AQaiConveyorPawn::InputCameraYawGamepad);
    PlayerInputComponent->BindAxis(TEXT("CameraPitchGamepad"), this, &AQaiConveyorPawn::InputCameraPitchGamepad);
    PlayerInputComponent->BindAxis(TEXT("CameraZoom"), this, &AQaiConveyorPawn::InputCameraZoom);
    PlayerInputComponent->BindAction(TEXT("Brake"), IE_Pressed, this, &AQaiConveyorPawn::BrakePressed);
    PlayerInputComponent->BindAction(TEXT("Brake"), IE_Released, this, &AQaiConveyorPawn::BrakeReleased);
    PlayerInputComponent->BindAction(TEXT("CycleForklift"), IE_Pressed, this, &AQaiConveyorPawn::CycleForklift);
    PlayerInputComponent->BindAction(TEXT("PreviousForklift"), IE_Pressed, this, &AQaiConveyorPawn::PreviousForklift);
    PlayerInputComponent->BindAction(TEXT("NextForklift"), IE_Pressed, this, &AQaiConveyorPawn::NextForklift);
    PlayerInputComponent->BindAction(TEXT("Forklift1"), IE_Pressed, this, &AQaiConveyorPawn::SelectForklift1);
    PlayerInputComponent->BindAction(TEXT("Forklift2"), IE_Pressed, this, &AQaiConveyorPawn::SelectForklift2);
    PlayerInputComponent->BindAction(TEXT("ResetScene"), IE_Pressed, this, &AQaiConveyorPawn::ResetScene);
    PlayerInputComponent->BindAction(TEXT("CycleView"), IE_Pressed, this, &AQaiConveyorPawn::CycleView);
    PlayerInputComponent->BindAction(TEXT("ToggleBackend"), IE_Pressed, this, &AQaiConveyorPawn::ToggleBackend);
    PlayerInputComponent->BindAction(TEXT("ToggleInference"), IE_Pressed, this, &AQaiConveyorPawn::ToggleInference);
    PlayerInputComponent->BindAction(TEXT("ToggleCollisionDebug"), IE_Pressed, this, &AQaiConveyorPawn::ToggleCollisionDebug);
    PlayerInputComponent->BindAction(TEXT("Quit"), IE_Pressed, this, &AQaiConveyorPawn::QuitDemo);
}

void AQaiConveyorPawn::InputThrottle(float Value)
{
    if (bIntroCameraTransitionActive || (FMath::Abs(Value) >= 0.12f && HandleIntroInput()))
    {
        ThrottleInput = 0.0f;
        return;
    }
    ThrottleInput = FMath::Abs(Value) < 0.12f ? 0.0f : Value;
}

void AQaiConveyorPawn::InputSteer(float Value)
{
    if (bIntroCameraTransitionActive || (FMath::Abs(Value) >= 0.12f && HandleIntroInput()))
    {
        SteerInput = 0.0f;
        return;
    }
    SteerInput = FMath::Abs(Value) < 0.12f ? 0.0f : Value;
}

void AQaiConveyorPawn::InputLift(float Value)
{
    if (bIntroCameraTransitionActive || (FMath::Abs(Value) >= 0.08f && HandleIntroInput()))
    {
        LiftInput = 0.0f;
        return;
    }
    LiftInput = FMath::Abs(Value) < 0.08f ? 0.0f : Value;
}

void AQaiConveyorPawn::InputCameraYawMouse(float Value)
{
    if (!bIntroCameraActive && !bIntroCameraTransitionActive)
    {
        OrbitYaw = FRotator::NormalizeAxis(OrbitYaw + Value * 2.0f);
    }
}

void AQaiConveyorPawn::InputCameraPitchMouse(float Value)
{
    if (!bIntroCameraActive && !bIntroCameraTransitionActive)
    {
        OrbitPitch = FMath::Clamp(OrbitPitch + Value * 1.6f, 10.0f, 70.0f);
    }
}

void AQaiConveyorPawn::InputCameraYawGamepad(float Value)
{
    if (bIntroCameraTransitionActive || (FMath::Abs(Value) >= 0.12f && HandleIntroInput()))
    {
        CameraYawRateInput = 0.0f;
        return;
    }
    CameraYawRateInput = FMath::Abs(Value) < 0.12f ? 0.0f : Value;
}

void AQaiConveyorPawn::InputCameraPitchGamepad(float Value)
{
    if (bIntroCameraTransitionActive || (FMath::Abs(Value) >= 0.12f && HandleIntroInput()))
    {
        CameraPitchRateInput = 0.0f;
        return;
    }
    CameraPitchRateInput = FMath::Abs(Value) < 0.12f ? 0.0f : Value;
}

void AQaiConveyorPawn::InputCameraZoom(float Value)
{
    if (FMath::Abs(Value) > KINDA_SMALL_NUMBER)
    {
        if (HandleIntroInput())
        {
            return;
        }
        SpringArm->TargetArmLength = FMath::Clamp(SpringArm->TargetArmLength - Value * 70.0f, 300.0f, 1600.0f);
    }
}
void AQaiConveyorPawn::BrakePressed() { if (!HandleIntroInput()) bBrake = true; }
void AQaiConveyorPawn::BrakeReleased() { bBrake = false; }
void AQaiConveyorPawn::CycleForklift() { if (!HandleIntroInput() && Runtime) Runtime->CycleForklift(); }
void AQaiConveyorPawn::PreviousForklift() { if (!HandleIntroInput() && Runtime) Runtime->CycleForklift(-1); }
void AQaiConveyorPawn::NextForklift() { if (!HandleIntroInput() && Runtime) Runtime->CycleForklift(1); }
void AQaiConveyorPawn::SelectForklift1() { if (!HandleIntroInput() && Runtime) Runtime->SelectForklift(0); }
void AQaiConveyorPawn::SelectForklift2() { if (!HandleIntroInput() && Runtime) Runtime->SelectForklift(1); }
void AQaiConveyorPawn::ResetScene()
{
    if (HandleIntroInput())
    {
        return;
    }
    if (Runtime)
    {
        Runtime->ResetScene();
    }
    // X is an emergency return to the authored demo state, including a
    // predictable presentation camera rather than only reattaching cargo.
    ViewIndex = 2;
    SpringArm->TargetArmLength = 900.0f;
    OrbitYaw = -45.0f;
    OrbitPitch = 38.0f;
}
void AQaiConveyorPawn::ToggleBackend() { if (!HandleIntroInput() && Runtime) Runtime->ToggleBackend(); }
void AQaiConveyorPawn::ToggleInference() { if (!HandleIntroInput() && Runtime) Runtime->ToggleInference(); }
void AQaiConveyorPawn::ToggleCollisionDebug() { if (!HandleIntroInput() && Runtime) Runtime->ToggleCollisionDebug(); }

void AQaiConveyorPawn::CycleView()
{
    if (HandleIntroInput())
    {
        return;
    }
    static const float Distances[] = {350.0f, 600.0f, 900.0f};
    ViewIndex = (ViewIndex + 1) % UE_ARRAY_COUNT(Distances);
    SpringArm->TargetArmLength = Distances[ViewIndex];
}

void AQaiConveyorPawn::QuitDemo()
{
    if (HandleIntroInput())
    {
        return;
    }
    UKismetSystemLibrary::QuitGame(this, Cast<APlayerController>(GetController()), EQuitPreference::Quit, false);
}
