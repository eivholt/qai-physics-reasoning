#include "QaiConveyorHUD.h"

#include "Engine/Canvas.h"
#include "Engine/Engine.h"
#include "Engine/Font.h"
#include "Engine/Texture2D.h"
#include "EngineUtils.h"
#include "Misc/App.h"
#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "QaiConveyorGameMode.h"
#include "QaiConveyorWorld.h"

namespace
{
    FLinearColor SignalColor(const FString& Signal)
    {
        if (Signal == TEXT("R")) return FLinearColor(0.95f, 0.08f, 0.05f, 1.0f);
        if (Signal == TEXT("A")) return FLinearColor(1.0f, 0.55f, 0.03f, 1.0f);
        if (Signal == TEXT("G")) return FLinearColor(0.04f, 0.85f, 0.18f, 1.0f);
        return FLinearColor::Gray;
    }

    FString SignalLabel(const FString& Signal)
    {
        if (Signal == TEXT("R")) return TEXT("RED");
        if (Signal == TEXT("A")) return TEXT("AMBER");
        if (Signal == TEXT("G")) return TEXT("GREEN");
        return TEXT("WAITING");
    }
}

void AQaiConveyorHUD::DrawHUD()
{
    Super::DrawHUD();
    if (FParse::Param(FCommandLine::Get(), TEXT("QaiCleanScreenshot")))
    {
        return;
    }
    if (!Canvas || !GetWorld())
    {
        return;
    }

    AQaiConveyorWorld* Runtime = nullptr;
    for (TActorIterator<AQaiConveyorWorld> It(GetWorld()); It; ++It)
    {
        Runtime = *It;
        break;
    }
    if (!Runtime)
    {
        return;
    }

    const float Scale = FMath::Clamp(Canvas->SizeY / 1080.0f, 0.72f, 1.2f);
    const double FrameSeconds = FMath::Max(FApp::GetDeltaTime(), 1.0 / 500.0);
    const double InstantFramesPerSecond = 1.0 / FrameSeconds;
    const double Blend = 1.0 - FMath::Exp(-FrameSeconds * 3.0);
    SmoothedFramesPerSecond = SmoothedFramesPerSecond > 0.0
        ? FMath::Lerp(SmoothedFramesPerSecond, InstantFramesPerSecond, Blend)
        : InstantFramesPerSecond;
    const float X = 22.0f * Scale;
    const float Y = 22.0f * Scale;
    const float W = 590.0f * Scale;
    const float H = Runtime->IsStageReady() ? 248.0f * Scale : 158.0f * Scale;
    DrawRect(FLinearColor(0.015f, 0.02f, 0.025f, 0.88f), X, Y, W, H);
    // The bar and the physical stack lamps both expose raw Reason2 output.
    DrawRect(SignalColor(Runtime->GetRawModelSignal()), X, Y, 8.0f * Scale, H);

    UFont* Font = GEngine ? GEngine->GetSmallFont() : nullptr;
    float LineY = Y + 14.0f * Scale;
    auto Line = [&](const FString& Text, const FLinearColor& Color = FLinearColor::White)
    {
        DrawText(Text, Color, X + 20.0f * Scale, LineY, Font, Scale, false);
        LineY += 22.0f * Scale;
    };

    Line(
        FString::Printf(
            TEXT("REASON2 CONVEYOR SAFETY  |  %d FPS"),
            FMath::RoundToInt(SmoothedFramesPerSecond)),
        FLinearColor(0.35f, 0.78f, 1.0f));
    Line(AQaiConveyorGameMode::GetRenderProfileDisplayName(), FLinearColor(0.55f, 0.82f, 1.0f));
    if (!Runtime->GetStageError().IsEmpty())
    {
        Line(Runtime->GetStageError(), FLinearColor(1.0f, 0.25f, 0.2f));
        Line(TEXT("Re-run the native asset import, then rebuild the installer."), FLinearColor(1.0f, 0.75f, 0.3f));
        return;
    }

    Line(FString::Printf(TEXT("Forklift %d  |  speed %+.2f m/s  |  lift %.2f m"), Runtime->GetActiveForkliftIndex() + 1, Runtime->GetActiveSpeedMetersPerSecond(), Runtime->GetActiveLiftMeters()));
    const FString PhysicsText = FString::Printf(
        TEXT("Physics %s"),
        *SignalLabel(Runtime->GetGroundTruthSignal()));
    const FString SeparatorText = TEXT("  |  ");
    const FString ReasonText = FString::Printf(
        TEXT("Reason2 %s"),
        *SignalLabel(Runtime->GetRawModelSignal()));
    float PhysicsWidth = 0.0f;
    float PhysicsHeight = 0.0f;
    float SeparatorWidth = 0.0f;
    float SeparatorHeight = 0.0f;
    Canvas->StrLen(Font, PhysicsText, PhysicsWidth, PhysicsHeight);
    Canvas->StrLen(Font, SeparatorText, SeparatorWidth, SeparatorHeight);
    DrawText(
        PhysicsText,
        SignalColor(Runtime->GetGroundTruthSignal()),
        X + 20.0f * Scale,
        LineY,
        Font,
        Scale,
        false);
    DrawText(
        SeparatorText,
        FLinearColor(0.72f, 0.76f, 0.8f),
        X + 20.0f * Scale + PhysicsWidth * Scale,
        LineY,
        Font,
        Scale,
        false);
    DrawText(
        ReasonText,
        SignalColor(Runtime->GetRawModelSignal()),
        X + 20.0f * Scale + (PhysicsWidth + SeparatorWidth) * Scale,
        LineY,
        Font,
        Scale,
        false);
    LineY += 22.0f * Scale;
    Line(FString::Printf(TEXT("%s: %s  |  %s"), *Runtime->GetBackendName().ToUpper(), *Runtime->GetModelName(), *Runtime->GetBackendStatus()));
    Line(FString::Printf(TEXT("Inference %s%s  |  skipped AI captures %d"), Runtime->IsInferenceEnabled() ? TEXT("ON") : TEXT("OFF"), Runtime->IsInferenceBusy() ? TEXT(" (busy)") : TEXT(""), Runtime->GetDroppedInferenceFrames()));
    Line(TEXT("WASD/arrows or RT/LT: drive   left stick: steer   Q/E or D-pad: lift"), FLinearColor(0.72f, 0.76f, 0.8f));
    Line(TEXT("Space/A: brake   X: reset scene   LB/RB: previous/next forklift"), FLinearColor(0.72f, 0.76f, 0.8f));
    Line(TEXT("Mouse/right stick: orbit   View: zoom preset   B/Start: switch Host/EVK   I: AI"), FLinearColor(0.72f, 0.76f, 0.8f));
    Line(
        FString::Printf(TEXT("F8: collision bounds %s"), Runtime->IsCollisionDebugEnabled() ? TEXT("ON") : TEXT("OFF")),
        Runtime->IsCollisionDebugEnabled() ? FLinearColor(0.3f, 1.0f, 0.45f) : FLinearColor(0.72f, 0.76f, 0.8f));

    const float PreviewW = 246.0f * Scale;
    const float PreviewH = PreviewW * 9.0f / 16.0f;
    const float PreviewGap = 10.0f * Scale;
    const float PreviewPanelW = PreviewW * 2.0f + PreviewGap + 28.0f * Scale;
    const float PreviewPanelH = PreviewH + 58.0f * Scale;
    const float PreviewPanelX = Canvas->SizeX - PreviewPanelW - 22.0f * Scale;
    const float PreviewPanelY = 22.0f * Scale;
    DrawRect(
        FLinearColor(0.015f, 0.02f, 0.025f, 0.88f),
        PreviewPanelX,
        PreviewPanelY,
        PreviewPanelW,
        PreviewPanelH);
    DrawText(
        TEXT("REASON2 INPUT  |  LAST SUBMITTED PAIR"),
        FLinearColor(0.35f, 0.78f, 1.0f),
        PreviewPanelX + 14.0f * Scale,
        PreviewPanelY + 10.0f * Scale,
        Font,
        Scale,
        false);

    const int32 SubmittedCount = Runtime->GetSubmittedInferenceFrameCount();
    for (int32 Index = 0; Index < 2; ++Index)
    {
        const float FrameX = PreviewPanelX + 14.0f * Scale + Index * (PreviewW + PreviewGap);
        const float FrameY = PreviewPanelY + 34.0f * Scale;
        DrawRect(
            FLinearColor(0.001f, 0.002f, 0.003f, 1.0f),
            FrameX - 2.0f,
            FrameY - 2.0f,
            PreviewW + 4.0f,
            PreviewH + 4.0f);
        if (UTexture2D* Frame = Runtime->GetSubmittedInferenceFrame(Index))
        {
            DrawTexture(Frame, FrameX, FrameY, PreviewW, PreviewH, 0.0f, 0.0f, 1.0f, 1.0f);
        }
        const FString FrameRole = Index == 0 ? TEXT("OLDER") : TEXT("NEWEST");
        const FString FrameTime = Runtime->GetSubmittedInferenceFrameTime(Index);
        DrawText(
            SubmittedCount > Index
                ? FString::Printf(TEXT("%s  %s"), *FrameRole, FrameTime.IsEmpty() ? TEXT("--:--:--") : *FrameTime)
                : FString::Printf(TEXT("%s  waiting for inference"), *FrameRole),
            FLinearColor(0.72f, 0.76f, 0.8f),
            FrameX,
            FrameY + PreviewH + 5.0f * Scale,
            Font,
            Scale * 0.92f,
            false);
    }
}
