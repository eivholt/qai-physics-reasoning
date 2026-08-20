#include "QaiConveyorPawn.h"

#include "Camera/CameraComponent.h"
#include "EngineUtils.h"
#include "Framework/Application/SlateApplication.h"
#include "Engine/Engine.h"
#include "GameFramework/GameUserSettings.h"
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
    // Pull the opening EVK product shot 50 cm farther out than the original
    // 184 cm framing without changing its viewing angle or lens.
    constexpr float IntroArmLengthCm = 234.0f;
    constexpr float IntroYawDegrees = -42.0f;
    constexpr float IntroPitchDegrees = 55.0f;
    constexpr float IntroFieldOfView = 36.0f;
    constexpr float IntroFocusHeightCm = 1.5f;
    constexpr float TransitionSeconds = 1.55f;
    constexpr float GameplayArmLengthCm = 350.0f;
    constexpr float GameplayYawDegrees = -45.0f;
    constexpr float GameplayPitchDegrees = 32.0f;
    constexpr float GameplayFieldOfView = 75.0f;
    constexpr float GameplayFocusHeightCm = 125.0f;
    constexpr float OrbitStickDeadZone = 0.18f;
    // Digital keys should turn the steering linkage progressively instead of
    // teleporting it from centre to full lock. Return is slightly quicker so
    // releasing a key still feels responsive.
    constexpr float KeyboardSteerRisePerSecond = 4.50f;
    constexpr float KeyboardSteerReturnPerSecond = 5.50f;

    bool IsDesktopWindowSwitchKey(const FKey& Key)
    {
        return Key == EKeys::LeftAlt
            || Key == EKeys::RightAlt
            || Key == EKeys::Tab;
    }

    FVector2D ApplyRadialStickDeadZone(float X, float Y)
    {
        const FVector2D RawInput(X, Y);
        const float RawMagnitude = RawInput.Size();
        if (RawMagnitude <= OrbitStickDeadZone)
        {
            return FVector2D::ZeroVector;
        }

        // Remove the inactive centre and remap the remaining travel to 0..1,
        // avoiding a response jump at the edge of the dead zone.
        const float ClampedMagnitude = FMath::Min(RawMagnitude, 1.0f);
        const float RemappedMagnitude =
            (ClampedMagnitude - OrbitStickDeadZone) / (1.0f - OrbitStickDeadZone);
        return RawInput * (RemappedMagnitude / RawMagnitude);
    }
}

AQaiConveyorPawn::AQaiConveyorPawn()
{
    PrimaryActorTick.bCanEverTick = true;
    AutoPossessPlayer = EAutoReceiveInput::Player0;

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
        else if (CameraPreset.Equals(TEXT("conveyorseam"), ESearchCase::IgnoreCase))
        {
            // QA view centred on the authored A08/A05 join.  Keep this close
            // preset available so roller/material regressions cannot be signed
            // off from a whole-room screenshot again.
            DiagnosticCameraTarget = FVector(292.0f, -120.0f, 76.0f);
            SpringArm->TargetArmLength = 350.0f;
            OrbitYaw = -18.0f;
            OrbitPitch = 35.0f;
            Camera->FieldOfView = 42.0f;
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

    AcquireGameInputFocus();
}

void AQaiConveyorPawn::PossessedBy(AController* NewController)
{
    Super::PossessedBy(NewController);
    bInputModeApplied = false;
    AcquireGameInputFocus();
}

void AQaiConveyorPawn::AcquireGameInputFocus()
{
    APlayerController* PC = Cast<APlayerController>(GetController());
    if (!PC)
    {
        bInputModeApplied = false;
        return;
    }
    PC->SetShowMouseCursor(false);
    PC->SetIgnoreMoveInput(false);
    PC->SetIgnoreLookInput(false);
    FInputModeGameOnly Mode;
    PC->SetInputMode(Mode);
    if (FSlateApplication::IsInitialized())
    {
        FSlateApplication::Get().SetAllUserFocusToGameViewport(EFocusCause::SetDirectly);
    }
    bInputModeApplied = true;
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
    UserInputIdleSeconds += DeltaSeconds;
    FindRuntime();
    if (!bInputModeApplied)
    {
        // BeginPlay may run before the default pawn is possessed. Apply the
        // game-only mode on the first possessed tick instead of relying on a
        // controller reconnect or a mouse click to repair viewport focus.
        AcquireGameInputFocus();
    }
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
    PollKeyboardFallback(
        DeltaSeconds,
        AppliedThrottle,
        AppliedSteer,
        AppliedLift,
        bAppliedBrake);

    // Use the already dead-zoned values so controller drift cannot keep the
    // contextual control drawer hidden while the demo is unattended.
    if (FMath::Abs(AppliedThrottle) >= 0.08f
        || FMath::Abs(AppliedSteer) >= 0.12f
        || FMath::Abs(AppliedLift) >= 0.08f
        || FMath::Abs(AppliedCameraYaw) >= QaiCameraTuning::OrbitStickDeadZone
        || FMath::Abs(AppliedCameraPitch) >= QaiCameraTuning::OrbitStickDeadZone
        || bAppliedBrake)
    {
        MarkUserActivity();
    }

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

void AQaiConveyorPawn::PollKeyboardFallback(
    float DeltaSeconds,
    float& OutThrottle,
    float& OutSteer,
    float& OutLift,
    bool& OutBrake)
{
    APlayerController* PC = Cast<APlayerController>(GetController());
    if (!PC)
    {
        return;
    }

    const bool bForward = PC->IsInputKeyDown(EKeys::W) || PC->IsInputKeyDown(EKeys::Up);
    const bool bReverse = PC->IsInputKeyDown(EKeys::S) || PC->IsInputKeyDown(EKeys::Down);
    const bool bLeft = PC->IsInputKeyDown(EKeys::A) || PC->IsInputKeyDown(EKeys::Left);
    const bool bRight = PC->IsInputKeyDown(EKeys::D) || PC->IsInputKeyDown(EKeys::Right);
    const bool bLiftUp = PC->IsInputKeyDown(EKeys::E);
    const bool bLiftDown = PC->IsInputKeyDown(EKeys::Q);
    const bool bSpace = PC->IsInputKeyDown(EKeys::SpaceBar);

    const bool bDirectionalInput = bForward || bReverse || bLeft || bRight
        || bLiftUp || bLiftDown || bSpace;
    if (bDirectionalInput && HandleIntroInput())
    {
        KeyboardSteerFiltered = 0.0f;
        OutThrottle = 0.0f;
        OutSteer = 0.0f;
        OutLift = 0.0f;
        OutBrake = false;
        return;
    }

    if (bForward || bReverse)
    {
        OutThrottle = (bForward ? 1.0f : 0.0f) - (bReverse ? 1.0f : 0.0f);
    }
    const bool bKeyboardSteer = bLeft || bRight;
    if (bKeyboardSteer)
    {
        const float TargetSteer = (bRight ? 1.0f : 0.0f) - (bLeft ? 1.0f : 0.0f);
        KeyboardSteerFiltered = FMath::FInterpConstantTo(
            KeyboardSteerFiltered,
            TargetSteer,
            DeltaSeconds,
            QaiCameraTuning::KeyboardSteerRisePerSecond);
        OutSteer = KeyboardSteerFiltered;
    }
    else
    {
        KeyboardSteerFiltered = FMath::FInterpConstantTo(
            KeyboardSteerFiltered,
            0.0f,
            DeltaSeconds,
            QaiCameraTuning::KeyboardSteerReturnPerSecond);
        // Do not replace a live analog stick value while centring the residual
        // keyboard steering after key release.
        if (FMath::Abs(OutSteer) < 0.12f && !FMath::IsNearlyZero(KeyboardSteerFiltered))
        {
            OutSteer = KeyboardSteerFiltered;
        }
        else if (FMath::Abs(OutSteer) >= 0.12f)
        {
            KeyboardSteerFiltered = 0.0f;
        }
    }
    if (bLiftUp || bLiftDown)
    {
        OutLift = (bLiftUp ? 1.0f : 0.0f) - (bLiftDown ? 1.0f : 0.0f);
    }
    OutBrake = OutBrake || bSpace;

    if (bDirectionalInput && !bLoggedKeyboardInput && Runtime)
    {
        bLoggedKeyboardInput = true;
        Runtime->RecordControllerInput(TEXT("keyboard_input_active backend=direct_player_controller_poll"));
    }
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
                TEXT("intro_camera_ready subject=IQ9_EVK arm_cm=%.0f yaw=%.0f pitch=%.0f fov=%.0f dismiss=any_button_except_alt_tab"),
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

void AQaiConveyorPawn::AnyInputPressed(FKey PressedKey)
{
    // Alt+Tab is part of the recording workflow, not an instruction to leave
    // the EVK product shot. AnyKey sees the initial Alt press before Windows
    // changes applications, so filter both modifier variants and Tab while
    // the intro is active. Outside the intro they remain ordinary activity,
    // and Alt+Enter still toggles fullscreen because Enter is not filtered.
    if (bIntroCameraActive
        && QaiCameraTuning::IsDesktopWindowSwitchKey(PressedKey))
    {
        if (Runtime)
        {
            Runtime->RecordControllerInput(FString::Printf(
                TEXT("intro_camera_input_ignored key=%s reason=desktop_window_switch"),
                *PressedKey.ToString()));
        }
        return;
    }
    MarkUserActivity();
    HandleIntroInput();
}

void AQaiConveyorPawn::MarkUserActivity()
{
    UserInputIdleSeconds = 0.0f;
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
            if (Pressed != 0 || bMeaningfulAnalogInput)
            {
                MarkUserActivity();
            }
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
            const FVector2D OrbitInput = QaiCameraTuning::ApplyRadialStickDeadZone(RightX, -RightY);
            OutCameraYaw = OrbitInput.X;
            OutCameraPitch = OrbitInput.Y;
            OutBrake = OutBrake || (Buttons & static_cast<uint64>(GameInputGamepadA)) != 0;
            if (Runtime)
            {
                if ((Pressed & static_cast<uint64>(GameInputGamepadB)) != 0)
                {
                    Runtime->ToggleBackend();
                }
                if ((Pressed & static_cast<uint64>(GameInputGamepadX)) != 0)
                {
                    ResetScene();
                }
                if ((Pressed & static_cast<uint64>(GameInputGamepadY)) != 0)
                {
                    CycleView();
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
    const FVector2D OrbitInput = QaiCameraTuning::ApplyRadialStickDeadZone(RightX, -RightY);
    OutCameraYaw = OrbitInput.X;
    OutCameraPitch = OrbitInput.Y;
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
    PlayerInputComponent->BindAction(TEXT("ResetScene"), IE_Pressed, this, &AQaiConveyorPawn::ResetScene);
    PlayerInputComponent->BindAction(TEXT("CycleView"), IE_Pressed, this, &AQaiConveyorPawn::CycleView);
    PlayerInputComponent->BindAction(TEXT("ToggleBackend"), IE_Pressed, this, &AQaiConveyorPawn::ToggleBackend);
    PlayerInputComponent->BindAction(TEXT("ToggleInference"), IE_Pressed, this, &AQaiConveyorPawn::ToggleInference);
    // Bind the render toggles directly so binary-only updates remain usable with an
    // already cooked build whose input configuration predates these controls.
    PlayerInputComponent->BindKey(EKeys::F6, IE_Pressed, this, &AQaiConveyorPawn::ToggleLumen);
    PlayerInputComponent->BindKey(EKeys::F7, IE_Pressed, this, &AQaiConveyorPawn::ToggleRayTracing);
    PlayerInputComponent->BindKey(
        EKeys::F9,
        IE_Pressed,
        this,
        &AQaiConveyorPawn::ToggleSensorViewOverlay);
    PlayerInputComponent->BindKey(
        EKeys::F11,
        IE_Pressed,
        this,
        &AQaiConveyorPawn::ToggleFullscreen);
    PlayerInputComponent->BindKey(
        FInputChord(EKeys::Enter, false, false, true, false),
        IE_Pressed,
        this,
        &AQaiConveyorPawn::ToggleFullscreen);
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
    if (FMath::Abs(Value) > KINDA_SMALL_NUMBER)
    {
        MarkUserActivity();
    }
    if (!bIntroCameraActive && !bIntroCameraTransitionActive)
    {
        OrbitYaw = FRotator::NormalizeAxis(OrbitYaw + Value * 2.0f);
    }
}

void AQaiConveyorPawn::InputCameraPitchMouse(float Value)
{
    if (FMath::Abs(Value) > KINDA_SMALL_NUMBER)
    {
        MarkUserActivity();
    }
    if (!bIntroCameraActive && !bIntroCameraTransitionActive)
    {
        OrbitPitch = FMath::Clamp(OrbitPitch + Value * 1.6f, 10.0f, 70.0f);
    }
}

void AQaiConveyorPawn::InputCameraYawGamepad(float Value)
{
    if (bIntroCameraTransitionActive
        || (FMath::Abs(Value) >= QaiCameraTuning::OrbitStickDeadZone && HandleIntroInput()))
    {
        CameraYawRateInput = 0.0f;
        return;
    }
    CameraYawRateInput = FMath::Abs(Value) < QaiCameraTuning::OrbitStickDeadZone ? 0.0f : Value;
}

void AQaiConveyorPawn::InputCameraPitchGamepad(float Value)
{
    if (bIntroCameraTransitionActive
        || (FMath::Abs(Value) >= QaiCameraTuning::OrbitStickDeadZone && HandleIntroInput()))
    {
        CameraPitchRateInput = 0.0f;
        return;
    }
    CameraPitchRateInput = FMath::Abs(Value) < QaiCameraTuning::OrbitStickDeadZone ? 0.0f : Value;
}

void AQaiConveyorPawn::InputCameraZoom(float Value)
{
    if (FMath::Abs(Value) > KINDA_SMALL_NUMBER)
    {
        MarkUserActivity();
        if (HandleIntroInput())
        {
            return;
        }
        SpringArm->TargetArmLength = FMath::Clamp(SpringArm->TargetArmLength - Value * 70.0f, 300.0f, 1600.0f);
    }
}
void AQaiConveyorPawn::BrakePressed() { if (!HandleIntroInput()) bBrake = true; }
void AQaiConveyorPawn::BrakeReleased() { bBrake = false; }
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
void AQaiConveyorPawn::ToggleLumen() { if (!HandleIntroInput() && Runtime) Runtime->ToggleLumen(); }
void AQaiConveyorPawn::ToggleRayTracing() { if (!HandleIntroInput() && Runtime) Runtime->ToggleRayTracing(); }
void AQaiConveyorPawn::ToggleCollisionDebug() { if (!HandleIntroInput() && Runtime) Runtime->ToggleCollisionDebug(); }
void AQaiConveyorPawn::ToggleSensorViewOverlay() { if (Runtime) Runtime->ToggleSensorViewOverlay(); }

void AQaiConveyorPawn::ToggleFullscreen()
{
    UGameUserSettings* Settings = GEngine ? GEngine->GetGameUserSettings() : nullptr;
    if (!Settings)
    {
        return;
    }

    const bool bEnterFullscreen =
        Settings->GetFullscreenMode() == EWindowMode::Windowed;
    Settings->SetFullscreenMode(
        bEnterFullscreen
            ? EWindowMode::WindowedFullscreen
            : EWindowMode::Windowed);
    Settings->ApplyResolutionSettings(false);
    Settings->ConfirmVideoMode();
    UE_LOG(
        LogTemp,
        Display,
        TEXT("window_mode_toggle mode=%s shortcuts=F11+AltEnter"),
        bEnterFullscreen ? TEXT("borderless_fullscreen") : TEXT("windowed"));
}

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
#if WITH_EDITOR
    // `UnrealEditor.exe -game` deliberately sets GIsEditor=false even though
    // this is still an editor build. Request a normal close so EndPlay can
    // cancel HTTP, drain capture workers, and release renderer resources.
    if (Runtime)
    {
        Runtime->RecordControllerInput(TEXT("exit_requested source=escape mode=editor_target_graceful"));
    }
    FPlatformMisc::RequestExit(false);
    return;
#else
    UKismetSystemLibrary::QuitGame(this, Cast<APlayerController>(GetController()), EQuitPreference::Quit, false);
#endif
}
