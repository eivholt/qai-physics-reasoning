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
#include "QaiConveyorPawn.h"
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

    TArray<FString> WrapCanvasText(
        UCanvas* Canvas,
        UFont* Font,
        const FString& Text,
        float MaximumWidth,
        float TextScale)
    {
        TArray<FString> Lines;
        TArray<FString> Paragraphs;
        Text.ParseIntoArrayLines(Paragraphs, false);
        if (Paragraphs.IsEmpty())
        {
            Paragraphs.Add(Text);
        }
        for (const FString& Paragraph : Paragraphs)
        {
            TArray<FString> Words;
            Paragraph.ParseIntoArrayWS(Words);
            if (Words.IsEmpty())
            {
                Lines.Add(TEXT("(empty)"));
                continue;
            }
            FString CurrentLine;
            for (const FString& Word : Words)
            {
                const FString Candidate = CurrentLine.IsEmpty()
                    ? Word
                    : CurrentLine + TEXT(" ") + Word;
                float Width = 0.0f;
                float Height = 0.0f;
                Canvas->StrLen(Font, Candidate, Width, Height);
                if (!CurrentLine.IsEmpty() && Width * TextScale > MaximumWidth)
                {
                    Lines.Add(CurrentLine);
                    CurrentLine = Word;
                }
                else
                {
                    CurrentLine = Candidate;
                }
            }
            if (!CurrentLine.IsEmpty())
            {
                Lines.Add(CurrentLine);
            }
        }
        return Lines;
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

    Runtime->DrawCollisionDebugOverlay(Canvas);

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
    // The always-visible block contains only live demo state. Controls live in
    // an independent drawer below it, leaving substantially more scene visible
    // whenever the operator is actively driving or moving the camera.
    const float H = 158.0f * Scale;
    DrawRect(FLinearColor(0.015f, 0.02f, 0.025f, 0.704f), X, Y, W, H);
    // The bar and physical lamps use the latest valid Reason2 presentation state;
    // the full answer panel continues to expose the newest raw response.
    DrawRect(SignalColor(Runtime->GetModelSignal()), X, Y, 8.0f * Scale, H);

    UFont* Font = GEngine ? GEngine->GetSmallFont() : nullptr;
    float LineY = Y + 14.0f * Scale;
    auto Line = [&](const FString& Text, const FLinearColor& Color = FLinearColor::White)
    {
        DrawText(Text, Color, X + 20.0f * Scale, LineY, Font, Scale, false);
        LineY += 22.0f * Scale;
    };

    const bool bEvkBackend = Runtime->GetBackendName().Equals(TEXT("evk"), ESearchCase::IgnoreCase);
    const FString InferenceDeviceLabel = bEvkBackend ? TEXT("EVK NPU") : TEXT("HOST GPU");
    const FLinearColor InferenceDeviceColor = bEvkBackend
        ? FLinearColor(0.78f, 0.48f, 1.0f)
        : FLinearColor(0.35f, 0.78f, 1.0f);
    Line(
        FString::Printf(
            TEXT("REASON2 PARCEL SAFETY  |  %s  |  %d FPS"),
            *InferenceDeviceLabel,
            FMath::RoundToInt(SmoothedFramesPerSecond)),
        InferenceDeviceColor);
    Line(AQaiConveyorGameMode::GetRenderProfileDisplayName(), FLinearColor(0.55f, 0.82f, 1.0f));
    if (!Runtime->GetStageError().IsEmpty())
    {
        Line(Runtime->GetStageError(), FLinearColor(1.0f, 0.25f, 0.2f));
        Line(TEXT("Re-run the native asset import, then rebuild the installer."), FLinearColor(1.0f, 0.75f, 0.3f));
        return;
    }

    const FString PhysicsText = FString::Printf(
        TEXT("Simulator state %s"),
        *SignalLabel(Runtime->GetSubmittedGroundTruthSignal()));
    const FString SeparatorText = TEXT("  |  ");
    const FString ReasonText = FString::Printf(
        TEXT("Reason2 %s"),
        *SignalLabel(Runtime->GetModelSignal()));
    float PhysicsWidth = 0.0f;
    float PhysicsHeight = 0.0f;
    float SeparatorWidth = 0.0f;
    float SeparatorHeight = 0.0f;
    Canvas->StrLen(Font, PhysicsText, PhysicsWidth, PhysicsHeight);
    Canvas->StrLen(Font, SeparatorText, SeparatorWidth, SeparatorHeight);
    DrawText(
        PhysicsText,
        SignalColor(Runtime->GetSubmittedGroundTruthSignal()),
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
        SignalColor(Runtime->GetModelSignal()),
        X + 20.0f * Scale + (PhysicsWidth + SeparatorWidth) * Scale,
        LineY,
        Font,
        Scale,
        false);
    LineY += 22.0f * Scale;
    DrawRect(
        FLinearColor(InferenceDeviceColor.R, InferenceDeviceColor.G, InferenceDeviceColor.B, 0.16f),
        X + 14.0f * Scale,
        LineY - 3.0f * Scale,
        W - 28.0f * Scale,
        21.0f * Scale);
    Line(
        FString::Printf(TEXT("ACTIVE INFERENCE DEVICE: %s"), *InferenceDeviceLabel),
        InferenceDeviceColor);
    Line(FString::Printf(TEXT("MODEL: %s"), *Runtime->GetModelName()));
    const double LastInferenceMilliseconds = Runtime->GetLastInferenceMilliseconds();
    const FString LatencyText = LastInferenceMilliseconds > 0.0
        ? FString::Printf(TEXT("%.0f ms"), LastInferenceMilliseconds)
        : TEXT("--");
    Line(FString::Printf(
        TEXT("Inference %s%s  |  latency %s"),
        Runtime->IsInferenceEnabled() ? TEXT("ON") : TEXT("OFF"),
        Runtime->IsInferenceBusy() ? TEXT(" (busy)") : TEXT(""),
        *LatencyText));

    constexpr float ControlHelpIdleSeconds = 10.0f;
    const AQaiConveyorPawn* DemoPawn = PlayerOwner
        ? Cast<AQaiConveyorPawn>(PlayerOwner->GetPawn())
        : nullptr;
    const bool bShowControlHelp = !DemoPawn
        || DemoPawn->GetUserInputIdleSeconds() >= ControlHelpIdleSeconds;
    const float DrawerTargetAlpha = bShowControlHelp ? 1.0f : 0.0f;
    const float DrawerSpeed = bShowControlHelp ? 3.6f : 7.0f;
    ControlHelpDrawerAlpha = FMath::FInterpTo(
        ControlHelpDrawerAlpha,
        DrawerTargetAlpha,
        static_cast<float>(FrameSeconds),
        DrawerSpeed);
    if (FMath::IsNearlyEqual(ControlHelpDrawerAlpha, DrawerTargetAlpha, 0.002f))
    {
        ControlHelpDrawerAlpha = DrawerTargetAlpha;
    }

    // Ease the drawer's position so its start and stop feel physical rather
    // than like a panel being teleported. It slides through the left screen
    // edge, keeping the status block completely stationary.
    const float DrawerEase = ControlHelpDrawerAlpha * ControlHelpDrawerAlpha
        * (3.0f - 2.0f * ControlHelpDrawerAlpha);
    const float DrawerX = FMath::Lerp(-W - 4.0f * Scale, X, DrawerEase);
    const float DrawerY = Y + H + 4.0f * Scale;
    const float DrawerH = 112.0f * Scale;
    if (ControlHelpDrawerAlpha > 0.001f)
    {
        DrawRect(FLinearColor(0.015f, 0.02f, 0.025f, 0.704f), DrawerX, DrawerY, W, DrawerH);
        DrawRect(InferenceDeviceColor, DrawerX, DrawerY, 8.0f * Scale, DrawerH);

        float DrawerLineY = DrawerY + 10.0f * Scale;
        auto DrawerLine = [&](const FString& Text)
        {
            DrawText(
                Text,
                FLinearColor(0.72f, 0.76f, 0.8f),
                DrawerX + 20.0f * Scale,
                DrawerLineY,
                Font,
                Scale,
                false);
            DrawerLineY += 20.5f * Scale;
        };
        DrawerLine(TEXT("WASD/arrows or RT/LT: drive   left stick: steer   Q/E or D-pad: lift"));
        DrawerLine(TEXT("Space/A: brake   X: reset scene"));
        DrawerLine(TEXT("Mouse/right stick: orbit   View: zoom preset   B/Start: switch GPU/EVK   I: AI"));
        DrawerLine(FString::Printf(
            TEXT("F6: Lumen %s   F7: ray tracing %s%s   F8: collision %s"),
            AQaiConveyorGameMode::IsRuntimeLumenEnabled() ? TEXT("ON") : TEXT("OFF"),
            AQaiConveyorGameMode::IsRuntimeRayTracingEnabled() ? TEXT("ON") : TEXT("OFF"),
            AQaiConveyorGameMode::SupportsRuntimeRayTracingToggle() ? TEXT("") : TEXT(" (fixed)"),
            Runtime->IsCollisionDebugEnabled() ? TEXT("ON") : TEXT("OFF")));
        DrawerLine(FString::Printf(
            TEXT("F9: sensor view %s   F11 or Alt+Enter: fullscreen"),
            Runtime->IsSensorViewOverlayEnabled() ? TEXT("ON") : TEXT("OFF")));
    }

    const int32 PreviewSlots = 1;
    const int32 PreviewColumns = 1;
    const int32 PreviewRows = 1;
    // Keep all three left-side surfaces on one common column. The 14 cm
    // padding on either side makes this panel exactly as wide as the status
    // panel above it.
    const float PreviewW = 562.0f * Scale;
    const float PreviewH = PreviewW * 9.0f / 16.0f;
    const float PreviewGapX = 8.0f * Scale;
    const float PreviewGapY = 23.0f * Scale;
    const float PreviewPanelW = PreviewW * PreviewColumns
        + PreviewGapX * (PreviewColumns - 1)
        + 28.0f * Scale;
    const float PreviewPanelH = PreviewH * PreviewRows
        + PreviewGapY * (PreviewRows - 1)
        + 58.0f * Scale;
    const float AnswerPanelW = PreviewPanelW;
    const float AnswerTextScale = Scale * 0.82f;
    const float AnswerNameW = 150.0f * Scale;
    const float AnswerValueW = AnswerPanelW - AnswerNameW - 36.0f * Scale;
    const float AnswerLineH = 18.0f * Scale;
    const int32 AnswerCount = Runtime->GetModelAnswerPropertyCount();
    TArray<TArray<FString>> WrappedValues;
    WrappedValues.Reserve(AnswerCount);
    float AnswerContentH = 0.0f;
    for (int32 Index = 0; Index < AnswerCount; ++Index)
    {
        TArray<FString> Wrapped = WrapCanvasText(
            Canvas,
            Font,
            Runtime->GetModelAnswerPropertyValue(Index),
            AnswerValueW,
            AnswerTextScale);
        AnswerContentH += FMath::Max(1, Wrapped.Num()) * AnswerLineH + 4.0f * Scale;
        WrappedValues.Add(MoveTemp(Wrapped));
    }
    if (AnswerCount == 0)
    {
        AnswerContentH = 26.0f * Scale;
    }
    const float AnswerPanelH = 50.0f * Scale + AnswerContentH;
    const float PanelMargin = X;
    const float PanelGap = 10.0f * Scale;
    const float AnswerPanelX = PanelMargin;
    const float AnswerPanelY = Canvas->SizeY - PanelMargin - AnswerPanelH;
    const float PreviewPanelX = PanelMargin;
    const float PreviewPanelY = AnswerPanelY - PanelGap - PreviewPanelH;
    DrawRect(
        FLinearColor(0.015f, 0.02f, 0.025f, 0.704f),
        PreviewPanelX,
        PreviewPanelY,
        PreviewPanelW,
        PreviewPanelH);
    DrawText(
        TEXT("REASON2 INPUT  |  CURRENT LOSSLESS OBSERVATION"),
        FLinearColor(0.35f, 0.78f, 1.0f),
        PreviewPanelX + 14.0f * Scale,
        PreviewPanelY + 10.0f * Scale,
        Font,
        Scale,
        false);

    const int32 SubmittedCount = Runtime->GetSubmittedInferenceFrameCount();
    for (int32 Index = 0; Index < PreviewSlots; ++Index)
    {
        const int32 Column = Index % PreviewColumns;
        const int32 Row = Index / PreviewColumns;
        const float FrameX = PreviewPanelX + 14.0f * Scale
            + Column * (PreviewW + PreviewGapX);
        const float FrameY = PreviewPanelY + 34.0f * Scale
            + Row * (PreviewH + PreviewGapY);
        DrawRect(
            FLinearColor(0.001f, 0.002f, 0.003f, 1.0f),
            FrameX - 2.0f,
            FrameY - 2.0f,
            PreviewW + 4.0f,
            PreviewH + 4.0f);
        if (UTexture2D* Frame = Runtime->GetSubmittedInferenceFrame(Index))
        {
            // Diagnostic source frames are opaque RGB. Explicit opaque
            // blending also makes the panel independent of scene-capture
            // alpha conventions across D3D12, Metal and render profiles.
            DrawTexture(
                Frame,
                FrameX,
                FrameY,
                PreviewW,
                PreviewH,
                0.0f,
                0.0f,
                1.0f,
                1.0f,
                FLinearColor::White,
                BLEND_Opaque);
        }
        const FString FrameTime = Runtime->GetSubmittedInferenceFrameTime(Index);
        DrawText(
            SubmittedCount > Index
                ? FString::Printf(
                    TEXT("CURRENT  %s"),
                    FrameTime.IsEmpty() ? TEXT("--:--:--") : *FrameTime)
                : TEXT("waiting for first submission"),
            FLinearColor(0.72f, 0.76f, 0.8f),
            FrameX,
            FrameY + PreviewH + 5.0f * Scale,
            Font,
            Scale * 0.72f,
            false);
    }

    DrawRect(
        FLinearColor(0.015f, 0.02f, 0.025f, 0.72f),
        AnswerPanelX,
        AnswerPanelY,
        AnswerPanelW,
        AnswerPanelH);
    DrawText(
        TEXT("FULL MODEL ANSWER  |  PROPERTY : VALUE"),
        FLinearColor(0.35f, 0.78f, 1.0f),
        AnswerPanelX + 14.0f * Scale,
        AnswerPanelY + 10.0f * Scale,
        Font,
        Scale,
        false);
    float AnswerY = AnswerPanelY + 38.0f * Scale;
    if (AnswerCount == 0)
    {
        DrawText(
            Runtime->IsInferenceBusy()
                ? TEXT("model is processing the current observation...")
                : TEXT("waiting for the first model answer..."),
            FLinearColor(0.62f, 0.67f, 0.72f),
            AnswerPanelX + 14.0f * Scale,
            AnswerY,
            Font,
            AnswerTextScale,
            false);
    }
    for (int32 Index = 0; Index < AnswerCount; ++Index)
    {
        const FString Name = Runtime->GetModelAnswerPropertyName(Index) + TEXT(":");
        DrawText(
            Name,
            FLinearColor(0.38f, 0.82f, 1.0f),
            AnswerPanelX + 14.0f * Scale,
            AnswerY,
            Font,
            AnswerTextScale,
            false);
        const TArray<FString>& ValueLines = WrappedValues[Index];
        for (int32 LineIndex = 0; LineIndex < ValueLines.Num(); ++LineIndex)
        {
            DrawText(
                ValueLines[LineIndex],
                FLinearColor::White,
                AnswerPanelX + 14.0f * Scale + AnswerNameW,
                AnswerY + LineIndex * AnswerLineH,
                Font,
                AnswerTextScale,
                false);
        }
        AnswerY += FMath::Max(1, ValueLines.Num()) * AnswerLineH + 4.0f * Scale;
    }
}
