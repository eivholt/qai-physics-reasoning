#include "QaiConveyorWorld.h"
#include "QaiConveyorGameMode.h"

#include "Async/Async.h"
#include "Components/PointLightComponent.h"
#include "Components/SpotLightComponent.h"
#include "Components/MaterialBillboardComponent.h"
#include "Components/PrimitiveComponent.h"
#include "Components/RectLightComponent.h"
#include "Components/LineBatchComponent.h"
#include "Components/LightComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Components/ExponentialHeightFogComponent.h"
#include "Components/LocalFogVolumeComponent.h"
#include "Dom/JsonObject.h"
#include "DrawDebugHelpers.h"
#include "Engine/ExponentialHeightFog.h"
#include "Engine/Engine.h"
#include "Engine/GameViewportClient.h"
#include "Engine/LocalFogVolume.h"
#include "Engine/SceneCapture2D.h"
#include "Engine/RectLight.h"
#include "Engine/SpotLight.h"
#include "Engine/StaticMesh.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformMemory.h"
#include "HAL/PlatformMisc.h"
#include "HAL/PlatformProperties.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "HttpModule.h"
#include "HighResScreenshot.h"
#include "ImageUtils.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Internationalization/Regex.h"
#include "JsonObjectConverter.h"
#include "Misc/Base64.h"
#include "Misc/CommandLine.h"
#include "Misc/DateTime.h"
#include "Misc/EngineVersion.h"
#include "Misc/FileHelper.h"
#include "Misc/App.h"
#include "Misc/Paths.h"
#include "Misc/Parse.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInterface.h"
#include "Math/PerspectiveMatrix.h"
#include "RHICommandList.h"
#include "RHIGPUReadback.h"
#include "RenderingThread.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "UnrealClient.h"

namespace ConveyorTuning
{
    constexpr float FixedStep = 1.0f / 120.0f;
    constexpr int32 MaxFixedStepsPerFrame = 8;
    constexpr float MaxSpeedCm = 100.0f;
    constexpr float AccelerationCm = 100.0f;
    constexpr float BrakingCm = 150.0f;
    constexpr float WheelbaseCm = 165.0f;
    constexpr float MaxSteerRadians = 0.52f;
    constexpr float SteeringRisePerSecond = 2.35f;
    constexpr float SteeringReturnPerSecond = 3.75f;
    constexpr float LiftSpeedCm = 90.0f;
    constexpr float MaxLiftCm = 200.0f;
    constexpr float BeltCenterX = 442.0f;
    // Native module bounds place the curve centres at Y=-361.9404 and
    // Y=108.3344. Their midpoint is negative; the old positive sign shifted
    // parcels and safety-distance tests 253.606 cm away from the rollers.
    constexpr float BeltCenterY = -126.803f;
    constexpr float BeltHalfWidth = 55.0f;
    constexpr float BeltHalfLength = 440.1374f;
    constexpr float ForkliftHalfLength = 187.0f;
    constexpr float ForkliftHalfWidth = 69.0f;
    constexpr float WorkerRadiusCm = 38.0f;
    constexpr float CollisionMarginCm = 5.0f;
    constexpr float MinimumCollisionLookaheadCm = 8.0f;
    constexpr float WorldMinimumX = -650.0f;
    constexpr float WorldMaximumX = 750.0f;
    constexpr float WorldMinimumY = -650.0f;
    constexpr float WorldMaximumY = 700.0f;
    constexpr float RedClearanceCm = 100.0f;
    constexpr float MovingThresholdCm = 5.0f;
    constexpr float ConveyorSpeedCm = 55.0f;
    constexpr float GravityCmPerSecondSquared = 980.0f;
    constexpr float ParcelRestitution = 0.14f;
    constexpr float ConveyorRollerTopFallbackCm = 76.9307f;
    // Moving rollers transfer momentum through a Coulomb-friction contact.
    // Heading follows the curved roller field through a bounded torque rather
    // than a direct transform interpolation.
    constexpr float ConveyorHeadingAngularAccelerationDegrees = 150.0f;
    constexpr float ConveyorMaximumYawSpeedDegrees = 65.0f;
    constexpr float ConveyorYawDampingPerSecond = 2.8f;
    constexpr float ParcelMaximumImpactYawSpeedDegrees = 240.0f;
    constexpr float ForkliftWheelRadiusCm = 31.0f;
    constexpr float ForkliftTireClearanceCm = 0.6f;
    constexpr float WorkerSpeedCm = 72.0f;
    constexpr float WorkerMaximumVisualTurnRateDegreesPerSecond = 220.0f;
    constexpr float WorkerHeadingResponsePerSecond = 7.0f;
    constexpr float WorkerHeadingCommitSeconds = 0.10f;
    constexpr float WorkerHeadingDeadbandDegrees = 12.0f;
    constexpr float WorkerRouteReversalCooldownSeconds = 1.15f;
    constexpr float CaptureInterval = 1.0f;
    // Render once at the host model's validated 16:9 input size. EVK frames
    // are downsampled before PNG encoding because its full-NPU vision path is
    // validated at 384x216 and larger pairs can abort GenieX's DSP queue.
    // A single render target is allocated at startup. Host-only diagnostic
    // overrides can raise it to 768 or 1024; EVK output remains 384x216.
    constexpr int32 DefaultHostCaptureWidth = 512;
    constexpr int32 DefaultHostCaptureHeight = 288;
    constexpr int32 MaximumHostCaptureWidth = 1024;
    constexpr int32 EvkCaptureWidth = 384;
    constexpr int32 EvkCaptureHeight = 216;
    const FVector IQ9EvkHalfExtentCm(5.30f, 5.30f, 2.15f);
    constexpr float IQ9EvkMassKg = 0.78f;
    // Keep the second authored vehicle in the project, but remove it from the
    // active demo until its physical behaviour and visual classification are
    // ready for another validation pass.
    constexpr int32 EnabledForkliftCount = 1;

    bool OverlapsOnAxis(
        const FVector2D& Delta,
        const FVector2D& Axis,
        const FVector2D& AForward,
        const FVector2D& ARight,
        const FVector2D& BForward,
        const FVector2D& BRight,
        float AHalfLength = ForkliftHalfLength,
        float AHalfWidth = ForkliftHalfWidth,
        float BHalfLength = ForkliftHalfLength,
        float BHalfWidth = ForkliftHalfWidth)
    {
        const float CenterDistance = FMath::Abs(FVector2D::DotProduct(Delta, Axis));
        const float ARadius = FMath::Abs(FVector2D::DotProduct(AForward, Axis)) * AHalfLength
            + FMath::Abs(FVector2D::DotProduct(ARight, Axis)) * AHalfWidth;
        const float BRadius = FMath::Abs(FVector2D::DotProduct(BForward, Axis)) * BHalfLength
            + FMath::Abs(FVector2D::DotProduct(BRight, Axis)) * BHalfWidth;
        return CenterDistance < ARadius + BRadius;
    }

    bool ObbOverlapsAabb(
        const FVector& VehicleLocation,
        const FRotator& VehicleRotation,
        const FVector2D& ObstacleCenter,
        const FVector2D& ObstacleHalfExtent,
        float MarginCm = CollisionMarginCm)
    {
        const FVector VehicleForward3 = VehicleRotation.Vector();
        const FVector VehicleRight3 = FRotationMatrix(VehicleRotation).GetScaledAxis(EAxis::Y);
        const FVector2D VehicleForward(VehicleForward3.X, VehicleForward3.Y);
        const FVector2D VehicleRight(VehicleRight3.X, VehicleRight3.Y);
        const FVector2D WorldX(1.0f, 0.0f);
        const FVector2D WorldY(0.0f, 1.0f);
        const FVector2D Delta = ObstacleCenter - FVector2D(VehicleLocation.X, VehicleLocation.Y);
        const float ObstacleHalfX = ObstacleHalfExtent.X + MarginCm;
        const float ObstacleHalfY = ObstacleHalfExtent.Y + MarginCm;
        return OverlapsOnAxis(
                   Delta, VehicleForward, VehicleForward, VehicleRight, WorldX, WorldY,
                   ForkliftHalfLength, ForkliftHalfWidth, ObstacleHalfX, ObstacleHalfY)
            && OverlapsOnAxis(
                   Delta, VehicleRight, VehicleForward, VehicleRight, WorldX, WorldY,
                   ForkliftHalfLength, ForkliftHalfWidth, ObstacleHalfX, ObstacleHalfY)
            && OverlapsOnAxis(
                   Delta, WorldX, VehicleForward, VehicleRight, WorldX, WorldY,
                   ForkliftHalfLength, ForkliftHalfWidth, ObstacleHalfX, ObstacleHalfY)
            && OverlapsOnAxis(
                   Delta, WorldY, VehicleForward, VehicleRight, WorldX, WorldY,
                   ForkliftHalfLength, ForkliftHalfWidth, ObstacleHalfX, ObstacleHalfY);
    }

    bool ObbOverlapsObb(
        const FVector& ACenter,
        const FQuat& ARotation,
        const FVector& AHalfExtent,
        const FVector& BCenter,
        const FQuat& BRotation,
        const FVector& BHalfExtent,
        float MarginCm = CollisionMarginCm)
    {
        if (ACenter.Z + AHalfExtent.Z + MarginCm <= BCenter.Z - BHalfExtent.Z
            || BCenter.Z + BHalfExtent.Z + MarginCm <= ACenter.Z - AHalfExtent.Z)
        {
            return false;
        }
        const FVector AForward3 = ARotation.GetForwardVector();
        const FVector ARight3 = ARotation.GetRightVector();
        const FVector BForward3 = BRotation.GetForwardVector();
        const FVector BRight3 = BRotation.GetRightVector();
        const FVector2D AForward(AForward3.X, AForward3.Y);
        const FVector2D ARight(ARight3.X, ARight3.Y);
        const FVector2D BForward(BForward3.X, BForward3.Y);
        const FVector2D BRight(BRight3.X, BRight3.Y);
        const FVector2D Delta(BCenter.X - ACenter.X, BCenter.Y - ACenter.Y);
        const float BHalfX = BHalfExtent.X + MarginCm;
        const float BHalfY = BHalfExtent.Y + MarginCm;
        return OverlapsOnAxis(
                   Delta, AForward, AForward, ARight, BForward, BRight,
                   AHalfExtent.X, AHalfExtent.Y, BHalfX, BHalfY)
            && OverlapsOnAxis(
                   Delta, ARight, AForward, ARight, BForward, BRight,
                   AHalfExtent.X, AHalfExtent.Y, BHalfX, BHalfY)
            && OverlapsOnAxis(
                   Delta, BForward, AForward, ARight, BForward, BRight,
                   AHalfExtent.X, AHalfExtent.Y, BHalfX, BHalfY)
            && OverlapsOnAxis(
                   Delta, BRight, AForward, ARight, BForward, BRight,
                   AHalfExtent.X, AHalfExtent.Y, BHalfX, BHalfY);
    }

    float ObbPlanarPenetrationCm(
        const FVector& ACenter,
        const FQuat& ARotation,
        const FVector& AHalfExtent,
        const FVector& BCenter,
        const FQuat& BRotation,
        const FVector& BHalfExtent,
        float MarginCm = CollisionMarginCm)
    {
        if (ACenter.Z + AHalfExtent.Z + MarginCm <= BCenter.Z - BHalfExtent.Z
            || BCenter.Z + BHalfExtent.Z + MarginCm <= ACenter.Z - AHalfExtent.Z)
        {
            return 0.0f;
        }

        const FVector AForward3 = ARotation.GetForwardVector();
        const FVector ARight3 = ARotation.GetRightVector();
        const FVector BForward3 = BRotation.GetForwardVector();
        const FVector BRight3 = BRotation.GetRightVector();
        const FVector2D AForward(AForward3.X, AForward3.Y);
        const FVector2D ARight(ARight3.X, ARight3.Y);
        const FVector2D BForward(BForward3.X, BForward3.Y);
        const FVector2D BRight(BRight3.X, BRight3.Y);
        const FVector2D Delta(BCenter.X - ACenter.X, BCenter.Y - ACenter.Y);
        const FVector2D Axes[] = {AForward, ARight, BForward, BRight};
        float MinimumPenetration = TNumericLimits<float>::Max();
        for (const FVector2D& RawAxis : Axes)
        {
            const FVector2D Axis = RawAxis.GetSafeNormal();
            if (Axis.IsNearlyZero())
            {
                continue;
            }
            const float CenterDistance = FMath::Abs(FVector2D::DotProduct(Delta, Axis));
            const float ARadius = FMath::Abs(FVector2D::DotProduct(AForward, Axis)) * AHalfExtent.X
                + FMath::Abs(FVector2D::DotProduct(ARight, Axis)) * AHalfExtent.Y;
            const float BRadius = FMath::Abs(FVector2D::DotProduct(BForward, Axis))
                    * (BHalfExtent.X + MarginCm)
                + FMath::Abs(FVector2D::DotProduct(BRight, Axis))
                    * (BHalfExtent.Y + MarginCm);
            const float Penetration = ARadius + BRadius - CenterDistance;
            if (Penetration <= 0.0f)
            {
                return 0.0f;
            }
            MinimumPenetration = FMath::Min(MinimumPenetration, Penetration);
        }
        return MinimumPenetration == TNumericLimits<float>::Max()
            ? 0.0f
            : MinimumPenetration;
    }

    bool SphereOverlapsObb(
        const FVector& SphereCenter,
        float SphereRadius,
        const FVector& BoxCenter,
        const FQuat& BoxRotation,
        const FVector& BoxHalfExtent,
        float MarginCm = CollisionMarginCm)
    {
        const FVector Local = BoxRotation.Inverse().RotateVector(SphereCenter - BoxCenter);
        const FVector Closest(
            FMath::Clamp(Local.X, -BoxHalfExtent.X, BoxHalfExtent.X),
            FMath::Clamp(Local.Y, -BoxHalfExtent.Y, BoxHalfExtent.Y),
            FMath::Clamp(Local.Z, -BoxHalfExtent.Z, BoxHalfExtent.Z));
        return FVector::DistSquared(Local, Closest)
            < FMath::Square(FMath::Max(0.0f, SphereRadius + MarginCm));
    }

    bool SpheresOverlap(
        const FVector& ACenter,
        float ARadius,
        const FVector& BCenter,
        float BRadius,
        float MarginCm = CollisionMarginCm)
    {
        return FVector::DistSquared(ACenter, BCenter)
            < FMath::Square(FMath::Max(0.0f, ARadius + BRadius + MarginCm));
    }

    float PointToObbDistance2D(
        const FVector& Point,
        const FVector& BoxCenter,
        const FQuat& BoxRotation,
        const FVector& BoxHalfExtent)
    {
        const FVector Local = BoxRotation.Inverse().RotateVector(Point - BoxCenter);
        const float DeltaX = FMath::Max(FMath::Abs(Local.X) - BoxHalfExtent.X, 0.0f);
        const float DeltaY = FMath::Max(FMath::Abs(Local.Y) - BoxHalfExtent.Y, 0.0f);
        return FMath::Sqrt(DeltaX * DeltaX + DeltaY * DeltaY);
    }

    bool WheelDiskOverlapsSupport2D(
        const FVector& WheelCenter,
        const FQuat& WheelRotation,
        float WheelRadius,
        float WheelHalfLength,
        const FVector& BoxCenter,
        const FQuat& BoxRotation,
        const FVector& BoxHalfExtent,
        float& OutLongitudinalGap)
    {
        const FVector2D WheelForward(
            WheelRotation.GetForwardVector().X,
            WheelRotation.GetForwardVector().Y);
        const FVector2D WheelRight(
            WheelRotation.GetRightVector().X,
            WheelRotation.GetRightVector().Y);
        const FVector2D BoxForward(
            BoxRotation.GetForwardVector().X,
            BoxRotation.GetForwardVector().Y);
        const FVector2D BoxRight(
            BoxRotation.GetRightVector().X,
            BoxRotation.GetRightVector().Y);
        const FVector2D Delta(BoxCenter.X - WheelCenter.X, BoxCenter.Y - WheelCenter.Y);
        const float BoxLongitudinalRadius =
            FMath::Abs(FVector2D::DotProduct(BoxForward, WheelForward)) * BoxHalfExtent.X
            + FMath::Abs(FVector2D::DotProduct(BoxRight, WheelForward)) * BoxHalfExtent.Y;
        const float BoxLateralRadius =
            FMath::Abs(FVector2D::DotProduct(BoxForward, WheelRight)) * BoxHalfExtent.X
            + FMath::Abs(FVector2D::DotProduct(BoxRight, WheelRight)) * BoxHalfExtent.Y;
        const float LongitudinalDistance = FMath::Abs(FVector2D::DotProduct(Delta, WheelForward));
        const float LateralDistance = FMath::Abs(FVector2D::DotProduct(Delta, WheelRight));
        if (LateralDistance > BoxLateralRadius + WheelHalfLength)
        {
            OutLongitudinalGap = TNumericLimits<float>::Max();
            return false;
        }
        OutLongitudinalGap = FMath::Max(0.0f, LongitudinalDistance - BoxLongitudinalRadius);
        return OutLongitudinalGap < WheelRadius;
    }

    float EvaluateSupportCoverage(
        const FVector& BodyCenter,
        const FQuat& BodyRotation,
        const FVector& BodyHalfExtent,
        const FVector& SupportCenter,
        const FQuat& SupportRotation,
        const FVector& SupportHalfExtent,
        FVector& OutOverhangDirection)
    {
        constexpr int32 SamplesPerAxis = 5;
        int32 SupportedSamples = 0;
        FVector UnsupportedCentroid = FVector::ZeroVector;
        const FVector BodyForward = BodyRotation.GetForwardVector();
        const FVector BodyRight = BodyRotation.GetRightVector();
        const FQuat InverseSupport = SupportRotation.Inverse();
        for (int32 X = 0; X < SamplesPerAxis; ++X)
        {
            const float LocalX = FMath::Lerp(
                -BodyHalfExtent.X,
                BodyHalfExtent.X,
                (static_cast<float>(X) + 0.5f) / SamplesPerAxis);
            for (int32 Y = 0; Y < SamplesPerAxis; ++Y)
            {
                const float LocalY = FMath::Lerp(
                    -BodyHalfExtent.Y,
                    BodyHalfExtent.Y,
                    (static_cast<float>(Y) + 0.5f) / SamplesPerAxis);
                const FVector Sample = BodyCenter + BodyForward * LocalX + BodyRight * LocalY;
                const FVector SupportLocal = InverseSupport.RotateVector(Sample - SupportCenter);
                if (FMath::Abs(SupportLocal.X) <= SupportHalfExtent.X
                    && FMath::Abs(SupportLocal.Y) <= SupportHalfExtent.Y)
                {
                    ++SupportedSamples;
                }
                else
                {
                    UnsupportedCentroid += Sample;
                }
            }
        }
        const int32 TotalSamples = SamplesPerAxis * SamplesPerAxis;
        const int32 UnsupportedSamples = TotalSamples - SupportedSamples;
        if (UnsupportedSamples > 0)
        {
            UnsupportedCentroid /= static_cast<float>(UnsupportedSamples);
            OutOverhangDirection = (UnsupportedCentroid - BodyCenter).GetSafeNormal2D();
        }
        else
        {
            OutOverhangDirection = FVector::ZeroVector;
        }
        return static_cast<float>(SupportedSamples) / static_cast<float>(TotalSamples);
    }

    bool IsCenterOfMassSupported(
        const FVector& CenterOfMass,
        const FVector& SupportCenter,
        const FQuat& SupportRotation,
        const FVector& SupportHalfExtent,
        float MarginCm = 0.0f)
    {
        const FVector Local = SupportRotation.Inverse().RotateVector(
            CenterOfMass - SupportCenter);
        return FMath::Abs(Local.X) <= SupportHalfExtent.X + MarginCm
            && FMath::Abs(Local.Y) <= SupportHalfExtent.Y + MarginCm;
    }

    void ApplyCoulombFriction(
        FVector& Velocity,
        const FVector& SupportVelocity,
        float StaticFriction,
        float DynamicFriction,
        float SleepSpeedCm,
        float StepSeconds)
    {
        FVector Relative = Velocity - SupportVelocity;
        Relative.Z = 0.0f;
        const float Speed = Relative.Size2D();
        const float StaticHoldSpeed = StaticFriction
            * GravityCmPerSecondSquared
            * StepSeconds;
        if (Speed <= FMath::Max(SleepSpeedCm, StaticHoldSpeed))
        {
            Velocity.X = SupportVelocity.X;
            Velocity.Y = SupportVelocity.Y;
            return;
        }
        const float NewSpeed = FMath::Max(
            0.0f,
            Speed - DynamicFriction * GravityCmPerSecondSquared * StepSeconds);
        Relative *= NewSpeed / FMath::Max(Speed, UE_SMALL_NUMBER);
        Velocity.X = SupportVelocity.X + Relative.X;
        Velocity.Y = SupportVelocity.Y + Relative.Y;
    }

    float YawImpulseDeltaDegrees(
        const FVector& LeverArmCm,
        const FVector& ImpulseKgCmPerSecond,
        float InertiaZKgCm2,
        float ResponseScale)
    {
        const float AngularImpulseZ = FVector::CrossProduct(
            LeverArmCm,
            ImpulseKgCmPerSecond).Z;
        return FMath::RadiansToDegrees(
            AngularImpulseZ / FMath::Max(InertiaZKgCm2, 1.0f))
            * ResponseScale;
    }

    float ProjectedVerticalHalfExtent(const FQuat& Rotation, const FVector& HalfExtent)
    {
        return FMath::Abs(Rotation.GetForwardVector().Z) * HalfExtent.X
            + FMath::Abs(Rotation.GetRightVector().Z) * HalfExtent.Y
            + FMath::Abs(Rotation.GetUpVector().Z) * HalfExtent.Z;
    }

    bool CircleOverlapsBox2D(
        const FVector2D& CircleCenter,
        float CircleRadius,
        const FVector& BoxCenter,
        const FQuat& BoxRotation,
        const FVector& BoxHalfExtent,
        float MarginCm = CollisionMarginCm)
    {
        const FVector Forward3 = BoxRotation.GetForwardVector();
        const FVector Right3 = BoxRotation.GetRightVector();
        const FVector2D Forward(Forward3.X, Forward3.Y);
        const FVector2D Right(Right3.X, Right3.Y);
        const FVector2D Relative = CircleCenter - FVector2D(BoxCenter.X, BoxCenter.Y);
        const float LocalX = FVector2D::DotProduct(Relative, Forward);
        const float LocalY = FVector2D::DotProduct(Relative, Right);
        const FVector2D Delta(
            LocalX - FMath::Clamp(LocalX, -BoxHalfExtent.X, BoxHalfExtent.X),
            LocalY - FMath::Clamp(LocalY, -BoxHalfExtent.Y, BoxHalfExtent.Y));
        return Delta.SizeSquared() < FMath::Square(CircleRadius + MarginCm);
    }

    bool FindPlanarCircleObbSeparation(
        const FVector& CircleCenter,
        float CircleRadius,
        const FVector& BoxCenter,
        const FQuat& BoxRotation,
        const FVector& BoxHalfExtent,
        FVector& OutBoxSeparation)
    {
        const FVector Forward3 = BoxRotation.GetForwardVector();
        const FVector Right3 = BoxRotation.GetRightVector();
        const FVector2D Forward(Forward3.X, Forward3.Y);
        const FVector2D Right(Right3.X, Right3.Y);
        const FVector2D Relative = FVector2D(CircleCenter.X, CircleCenter.Y)
            - FVector2D(BoxCenter.X, BoxCenter.Y);
        const FVector2D CircleLocal(
            FVector2D::DotProduct(Relative, Forward),
            FVector2D::DotProduct(Relative, Right));
        const FVector2D Closest(
            FMath::Clamp(CircleLocal.X, -BoxHalfExtent.X, BoxHalfExtent.X),
            FMath::Clamp(CircleLocal.Y, -BoxHalfExtent.Y, BoxHalfExtent.Y));
        const FVector2D Delta = CircleLocal - Closest;
        const float DistanceSquared = Delta.SizeSquared();
        FVector2D LocalSeparation = FVector2D::ZeroVector;
        if (DistanceSquared > UE_SMALL_NUMBER)
        {
            const float Distance = FMath::Sqrt(DistanceSquared);
            if (Distance >= CircleRadius)
            {
                return false;
            }
            // Delta points from the box toward the circle; move the box in
            // the opposite direction by the exact planar penetration.
            LocalSeparation = -Delta / Distance * (CircleRadius - Distance + 0.35f);
        }
        else
        {
            // Circle centre is inside the box footprint. Exit through the
            // nearest face, preserving a stable deterministic axis choice.
            const float ExitX = BoxHalfExtent.X - FMath::Abs(CircleLocal.X);
            const float ExitY = BoxHalfExtent.Y - FMath::Abs(CircleLocal.Y);
            if (ExitX <= ExitY)
            {
                const float Direction = CircleLocal.X >= 0.0f ? -1.0f : 1.0f;
                LocalSeparation.X = Direction * (ExitX + CircleRadius + 0.35f);
            }
            else
            {
                const float Direction = CircleLocal.Y >= 0.0f ? -1.0f : 1.0f;
                LocalSeparation.Y = Direction * (ExitY + CircleRadius + 0.35f);
            }
        }
        const FVector2D World = Forward * LocalSeparation.X + Right * LocalSeparation.Y;
        OutBoxSeparation = FVector(World.X, World.Y, 0.0f);
        return true;
    }

    bool CircleOverlapsObb(
        const FVector2D& CircleCenter,
        float CircleRadius,
        const FVector& VehicleLocation,
        const FRotator& VehicleRotation,
        float MarginCm = CollisionMarginCm)
    {
        const FVector VehicleForward3 = VehicleRotation.Vector();
        const FVector VehicleRight3 = FRotationMatrix(VehicleRotation).GetScaledAxis(EAxis::Y);
        const FVector2D VehicleForward(VehicleForward3.X, VehicleForward3.Y);
        const FVector2D VehicleRight(VehicleRight3.X, VehicleRight3.Y);
        const FVector2D Relative = CircleCenter - FVector2D(VehicleLocation.X, VehicleLocation.Y);
        const float LocalForward = FVector2D::DotProduct(Relative, VehicleForward);
        const float LocalRight = FVector2D::DotProduct(Relative, VehicleRight);
        const float NearestForward = FMath::Clamp(LocalForward, -ForkliftHalfLength, ForkliftHalfLength);
        const float NearestRight = FMath::Clamp(LocalRight, -ForkliftHalfWidth, ForkliftHalfWidth);
        const FVector2D Delta(LocalForward - NearestForward, LocalRight - NearestRight);
        return Delta.SizeSquared() < FMath::Square(CircleRadius + MarginCm);
    }

    void EvaluateConveyor(float Distance, FVector& Position, FVector& Tangent)
    {
        constexpr float StraightHalf = 235.1374f;
        constexpr float Radius = 150.0f;
        constexpr float StraightLength = 2.0f * StraightHalf;
        constexpr float ArcLength = PI * Radius;
        if (Distance < StraightLength)
        {
            Position = FVector(BeltCenterX - Radius, BeltCenterY - StraightHalf + Distance, 0.0f);
            Tangent = FVector(0.0f, 1.0f, 0.0f);
        }
        else if ((Distance -= StraightLength) < ArcLength)
        {
            // Top semicircle: left straight (+Y) to right straight (-Y).
            const float Angle = PI - Distance / Radius;
            Position = FVector(
                BeltCenterX + Radius * FMath::Cos(Angle),
                BeltCenterY + StraightHalf + Radius * FMath::Sin(Angle),
                0.0f);
            Tangent = FVector(FMath::Sin(Angle), -FMath::Cos(Angle), 0.0f);
        }
        else if ((Distance -= ArcLength) < StraightLength)
        {
            Position = FVector(BeltCenterX + Radius, BeltCenterY + StraightHalf - Distance, 0.0f);
            Tangent = FVector(0.0f, -1.0f, 0.0f);
        }
        else
        {
            Distance -= StraightLength;
            // Bottom semicircle closes the right straight back onto the left.
            const float Angle = -Distance / Radius;
            Position = FVector(
                BeltCenterX + Radius * FMath::Cos(Angle),
                BeltCenterY - StraightHalf + Radius * FMath::Sin(Angle),
                0.0f);
            Tangent = FVector(FMath::Sin(Angle), -FMath::Cos(Angle), 0.0f);
        }
    }

    void ProjectToConveyor(
        const FVector& WorldPosition,
        float& OutDistance,
        FVector& OutPosition,
        FVector& OutTangent,
        float& OutLateralDistance)
    {
        constexpr float StraightHalf = 235.1374f;
        constexpr float Radius = 150.0f;
        constexpr float StraightLength = 2.0f * StraightHalf;
        constexpr float ArcLength = PI * Radius;
        const float BottomY = BeltCenterY - StraightHalf;
        const float TopY = BeltCenterY + StraightHalf;
        float BestSquared = TNumericLimits<float>::Max();

        const auto Consider = [&](float Distance, const FVector& Position, const FVector& Tangent)
        {
            const float Squared = FVector::DistSquared2D(WorldPosition, Position);
            if (Squared < BestSquared)
            {
                BestSquared = Squared;
                OutDistance = Distance;
                OutPosition = Position;
                OutTangent = Tangent;
            }
        };

        const float LeftY = FMath::Clamp(WorldPosition.Y, BottomY, TopY);
        Consider(
            LeftY - BottomY,
            FVector(BeltCenterX - Radius, LeftY, 0.0f),
            FVector(0.0f, 1.0f, 0.0f));

        const FVector2D TopRelative(WorldPosition.X - BeltCenterX, WorldPosition.Y - TopY);
        const float TopAngle = FMath::Clamp(FMath::Atan2(TopRelative.Y, TopRelative.X), 0.0f, PI);
        Consider(
            StraightLength + (PI - TopAngle) * Radius,
            FVector(
                BeltCenterX + Radius * FMath::Cos(TopAngle),
                TopY + Radius * FMath::Sin(TopAngle),
                0.0f),
            FVector(FMath::Sin(TopAngle), -FMath::Cos(TopAngle), 0.0f));

        const float RightY = FMath::Clamp(WorldPosition.Y, BottomY, TopY);
        Consider(
            StraightLength + ArcLength + TopY - RightY,
            FVector(BeltCenterX + Radius, RightY, 0.0f),
            FVector(0.0f, -1.0f, 0.0f));

        const FVector2D BottomRelative(WorldPosition.X - BeltCenterX, WorldPosition.Y - BottomY);
        const float BottomAngle = FMath::Clamp(FMath::Atan2(BottomRelative.Y, BottomRelative.X), -PI, 0.0f);
        Consider(
            2.0f * StraightLength + ArcLength - BottomAngle * Radius,
            FVector(
                BeltCenterX + Radius * FMath::Cos(BottomAngle),
                BottomY + Radius * FMath::Sin(BottomAngle),
                0.0f),
            FVector(FMath::Sin(BottomAngle), -FMath::Cos(BottomAngle), 0.0f));

        OutLateralDistance = FMath::Sqrt(BestSquared);
    }

    bool FindPlanarObbSeparation(
        const FVector& ACenter,
        const FQuat& ARotation,
        const FVector& AHalfExtent,
        const FVector& BCenter,
        const FQuat& BRotation,
        const FVector& BHalfExtent,
        FVector& OutSeparation)
    {
        if (ACenter.Z + AHalfExtent.Z <= BCenter.Z - BHalfExtent.Z
            || BCenter.Z + BHalfExtent.Z <= ACenter.Z - AHalfExtent.Z)
        {
            return false;
        }
        const FVector2D AForward(ARotation.GetForwardVector().X, ARotation.GetForwardVector().Y);
        const FVector2D ARight(ARotation.GetRightVector().X, ARotation.GetRightVector().Y);
        const FVector2D BForward(BRotation.GetForwardVector().X, BRotation.GetForwardVector().Y);
        const FVector2D BRight(BRotation.GetRightVector().X, BRotation.GetRightVector().Y);
        const FVector2D Axes[] = {AForward, ARight, BForward, BRight};
        const FVector2D Delta(ACenter.X - BCenter.X, ACenter.Y - BCenter.Y);
        float MinimumOverlap = TNumericLimits<float>::Max();
        FVector2D MinimumAxis = FVector2D::ZeroVector;
        for (FVector2D Axis : Axes)
        {
            if (!Axis.Normalize())
            {
                continue;
            }
            const float ARadius = FMath::Abs(FVector2D::DotProduct(AForward, Axis)) * AHalfExtent.X
                + FMath::Abs(FVector2D::DotProduct(ARight, Axis)) * AHalfExtent.Y;
            const float BRadius = FMath::Abs(FVector2D::DotProduct(BForward, Axis)) * BHalfExtent.X
                + FMath::Abs(FVector2D::DotProduct(BRight, Axis)) * BHalfExtent.Y;
            const float SignedDistance = FVector2D::DotProduct(Delta, Axis);
            const float Overlap = ARadius + BRadius - FMath::Abs(SignedDistance);
            if (Overlap <= 0.0f)
            {
                return false;
            }
            if (Overlap < MinimumOverlap)
            {
                MinimumOverlap = Overlap;
                MinimumAxis = Axis * (SignedDistance >= 0.0f ? 1.0f : -1.0f);
            }
        }
        OutSeparation = FVector(MinimumAxis.X, MinimumAxis.Y, 0.0f) * (MinimumOverlap + 0.35f);
        return true;
    }
}

namespace
{
    constexpr int32 SimulatorRunsToRetain = 3;
    constexpr int64 MaxSimulatorLogBytes = 8ll * 1024ll * 1024ll;

    FString SanitizeLogField(FString Value)
    {
        return Value
            .Replace(TEXT("\r"), TEXT(" "))
            .Replace(TEXT("\n"), TEXT(" "))
            .Replace(TEXT("\""), TEXT("'"));
    }

    FString SimulatorLogPath()
    {
        static const FString Path = []
        {
#if PLATFORM_MAC
            const FString Root = FPaths::Combine(
                FPlatformProcess::UserHomeDir(),
                TEXT("Library/Logs/QaiConveyorDemo"));
#else
            const FString Root = FPaths::Combine(
                FPlatformProcess::UserSettingsDir(),
                TEXT("QaiConveyorDemo/Logs"));
#endif
            IFileManager& FileManager = IFileManager::Get();
            FileManager.MakeDirectory(*Root, true);

            // A new run gets one new file, so retain only the two newest old
            // runs before selecting this run's path. Failed/crashed runs are
            // ordinary files and therefore remain available for diagnosis.
            TArray<FString> ExistingRuns;
            FileManager.FindFiles(ExistingRuns, *FPaths::Combine(Root, TEXT("simulator-*.log")), true, false);
            ExistingRuns.Sort([&FileManager, &Root](const FString& Left, const FString& Right)
            {
                const FDateTime LeftTime = FileManager.GetTimeStamp(*FPaths::Combine(Root, Left));
                const FDateTime RightTime = FileManager.GetTimeStamp(*FPaths::Combine(Root, Right));
                return LeftTime == RightTime ? Left > Right : LeftTime > RightTime;
            });
            for (int32 Index = SimulatorRunsToRetain - 1; Index < ExistingRuns.Num(); ++Index)
            {
                const FString OldPath = FPaths::Combine(Root, ExistingRuns[Index]);
                if (!FileManager.Delete(*OldPath, false, true, true))
                {
                    UE_LOG(LogTemp, Warning, TEXT("Could not remove expired simulator log: %s"), *OldPath);
                }
            }
            return FPaths::Combine(
                Root,
                FString::Printf(
                    TEXT("simulator-%s-%u.log"),
                    *FDateTime::UtcNow().ToString(TEXT("%Y%m%dT%H%M%SZ")),
                    FPlatformProcess::GetCurrentProcessId()));
        }();
        return Path;
    }

    void CompactSimulatorLogIfNeeded(const FString& Path)
    {
        IFileManager& FileManager = IFileManager::Get();
        if (FileManager.FileSize(*Path) < MaxSimulatorLogBytes)
        {
            return;
        }

        FString Existing;
        if (!FFileHelper::LoadFileToString(Existing, *Path))
        {
            UE_LOG(LogTemp, Warning, TEXT("Could not compact simulator log: %s"), *Path);
            return;
        }

        // Preserve startup compatibility data plus the newest events. This is
        // preferable to dropping later failures during a multi-day session.
        constexpr int32 HeaderCharacters = 16 * 1024;
        constexpr int32 TailCharacters = 4 * 1024 * 1024;
        const FString Header = Existing.Left(HeaderCharacters);
        FString Tail = Existing.Right(TailCharacters);
        int32 FirstNewline = INDEX_NONE;
        if (Tail.FindChar(TEXT('\n'), FirstNewline) && FirstNewline + 1 < Tail.Len())
        {
            Tail.RightChopInline(FirstNewline + 1, EAllowShrinking::No);
        }
        const FString Marker = FString::Printf(
            TEXT("\n%s log_compacted max_bytes=%lld retained=startup_and_recent\n"),
            *FDateTime::UtcNow().ToIso8601(),
            MaxSimulatorLogBytes);
        if (!FFileHelper::SaveStringToFile(
                Header + Marker + Tail,
                *Path,
                FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
        {
            UE_LOG(LogTemp, Warning, TEXT("Failed writing compacted simulator log: %s"), *Path);
        }
    }

    void SimulatorLog(const FString& Message)
    {
        const FString& Path = SimulatorLogPath();
        CompactSimulatorLogIfNeeded(Path);
        const FString Line = FString::Printf(
            TEXT("%s %s\n"),
            *FDateTime::UtcNow().ToIso8601(),
            *Message.Replace(TEXT("\r"), TEXT(" ")).Replace(TEXT("\n"), TEXT(" ")));
        FFileHelper::SaveStringToFile(
            Line,
            *Path,
            FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM,
            &IFileManager::Get(),
            FILEWRITE_Append | FILEWRITE_AllowRead);
    }

    FString SystemProfileLogLine()
    {
        const FPlatformMemoryStats Memory = FPlatformMemory::GetStats();
        return FString::Printf(
            TEXT("system_profile engine=\"%s\" platform=%s build=%s os=\"%s\" cpu=\"%s\" logical_cores=%d memory_total_mib=%llu memory_available_mib=%llu process_id=%u"),
            *SanitizeLogField(FEngineVersion::Current().ToString()),
            ANSI_TO_TCHAR(FPlatformProperties::PlatformName()),
            LexToString(FApp::GetBuildConfiguration()),
            *SanitizeLogField(FPlatformMisc::GetOSVersion()),
            *SanitizeLogField(FPlatformMisc::GetCPUBrand()),
            FPlatformMisc::NumberOfCoresIncludingHyperthreads(),
            static_cast<unsigned long long>(Memory.TotalPhysical / (1024ull * 1024ull)),
            static_cast<unsigned long long>(Memory.AvailablePhysical / (1024ull * 1024ull)),
            FPlatformProcess::GetCurrentProcessId());
    }
}

AQaiConveyorWorld::AQaiConveyorWorld()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PrePhysics;
}

UTexture2D* AQaiConveyorWorld::GetSubmittedInferenceFrame(int32 Index) const
{
    return SubmittedFrameTextures.IsValidIndex(Index) ? SubmittedFrameTextures[Index].Get() : nullptr;
}

FString AQaiConveyorWorld::GetSubmittedInferenceFrameTime(int32 Index) const
{
    return SubmittedFrameTimes.IsValidIndex(Index) ? SubmittedFrameTimes[Index] : FString();
}

void AQaiConveyorWorld::BeginPlay()
{
    Super::BeginPlay();
    bIsShuttingDown = false;
    if (GEngine)
    {
        // Keep the standalone HUD clean (notably from RT residency warnings)
        // while preserving the same diagnostics in Unreal and rolling logs.
        GEngine->Exec(GetWorld(), TEXT("DisableAllScreenMessages"));
    }
    SimulatorLog(TEXT("session_start version=0.1.0"));
    SimulatorLog(SystemProfileLogLine());
    SimulatorLog(FString::Printf(
        TEXT("log_policy retain_runs=%d max_bytes_per_run=%lld path=\"%s\""),
        SimulatorRunsToRetain,
        MaxSimulatorLogBytes,
        *SanitizeLogField(SimulatorLogPath())));
    SimulatorLog(AQaiConveyorGameMode::GetRenderProfileSummary());
    SimulatorLog(FString::Printf(
        TEXT("compatibility_status status=%s"),
        *AQaiConveyorGameMode::GetCompatibilityStatus()));
    bRuntimeMotionTest = FParse::Param(FCommandLine::Get(), TEXT("QaiRuntimeMotionTest"));
    bPhysicsContactTest = FParse::Param(FCommandLine::Get(), TEXT("QaiPhysicsContactTest"));
    bShelfForkTest = FParse::Param(FCommandLine::Get(), TEXT("QaiShelfForkTest"));
    bEvkPhysicsTest = FParse::Param(FCommandLine::Get(), TEXT("QaiEvkPhysicsTest"));
    bForkliftClimbTest = FParse::Param(FCommandLine::Get(), TEXT("QaiForkliftClimbTest"));
    bForkEdgeBalanceTest = FParse::Param(FCommandLine::Get(), TEXT("QaiForkEdgeBalanceTest"));
    bForkWedgeTest = FParse::Param(FCommandLine::Get(), TEXT("QaiForkWedgeTest"));
    bRuntimeMotionTest = bRuntimeMotionTest || bPhysicsContactTest || bShelfForkTest
        || bEvkPhysicsTest || bForkliftClimbTest || bForkEdgeBalanceTest || bForkWedgeTest;
    bHudScreenshotTest = FParse::Param(FCommandLine::Get(), TEXT("QaiHudScreenshotTest"));
    bWorkerSoakTest = FParse::Param(FCommandLine::Get(), TEXT("QaiWorkerSoakTest"));
    bPresentation4K = FParse::Param(FCommandLine::Get(), TEXT("QaiPresentation4K"));
    bResolutionDataset = FParse::Param(FCommandLine::Get(), TEXT("QaiResolutionDataset"));
    bResolutionDatasetWhiteFloor = FParse::Param(FCommandLine::Get(), TEXT("QaiResolutionDatasetWhiteFloor"));
    bResolutionDatasetHideWorkers = FParse::Param(FCommandLine::Get(), TEXT("QaiResolutionDatasetHideWorkers"));
    bResolutionDatasetHideParcels = FParse::Param(FCommandLine::Get(), TEXT("QaiResolutionDatasetHideParcels"));
    FParse::Value(FCommandLine::Get(), TEXT("QaiResolutionDatasetVariant="), ResolutionDatasetVariant);
    if (bResolutionDataset)
    {
        SimulatorLog(FString::Printf(
            TEXT("resolution_dataset enabled=true variant=%s phases=G,A,R native_frames_per_phase=10 phase_seconds=12 slow_motion_cm_s=7 white_floor=%s hide_workers=%s hide_parcels=%s"),
            *ResolutionDatasetVariant,
            bResolutionDatasetWhiteFloor ? TEXT("true") : TEXT("false"),
            bResolutionDatasetHideWorkers ? TEXT("true") : TEXT("false"),
            bResolutionDatasetHideParcels ? TEXT("true") : TEXT("false")));
    }
    if (bPresentation4K)
    {
        SimulatorLog(TEXT("presentation_capture mode=4k output=3840x2160 temporal_convergence_seconds=8"));
    }
    bDrawCollisionDebug = FParse::Param(FCommandLine::Get(), TEXT("QaiCollisionDebug"));
    if (bDrawCollisionDebug)
    {
        SimulatorLog(TEXT("collision_debug state=on source=commandline"));
    }
    LoadRuntimeConfig();
    BindNativeComponents();
    ApplyResolutionDatasetVariant();
    SetupInferenceCapture();
    if (bInferenceEnabled)
    {
        ProbeBackend();
    }
}

void AQaiConveyorWorld::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    // Render readback and PNG compression finish on other threads. Mark the
    // runtime unavailable before cancelling HTTP or releasing textures so no
    // late game-thread continuation can recreate a transient render resource
    // while Unreal is already dismantling the renderer.
    bIsShuttingDown = true;
    SimulatorLog(FString::Printf(TEXT("session_end reason=%d"), static_cast<int32>(EndPlayReason)));
    bInferenceEnabled = false;
    bCaptureReadbackPending = false;
    bEncodingFrame = false;
    if (InferenceRequest)
    {
        InferenceRequest->OnProcessRequestComplete().Unbind();
        InferenceRequest->CancelRequest();
        InferenceRequest.Reset();
    }
    if (ProbeRequest)
    {
        ProbeRequest->OnProcessRequestComplete().Unbind();
        ProbeRequest->CancelRequest();
        ProbeRequest.Reset();
    }
    EncodedFrames.Reset();
    EncodedForkliftFrameTransforms.Reset();
    EncodedFrameTextures.Reset();
    EncodedFrameTimes.Reset();
    SubmittedEncodedFrames.Reset();
    SubmittedFrameTextures.Reset();
    SubmittedFrameTimes.Reset();
    if (CaptureActor && CaptureActor->GetCaptureComponent2D())
    {
        CaptureActor->GetCaptureComponent2D()->TextureTarget = nullptr;
    }
    // QueueCapture owns the readback on the render thread. Drain any final
    // enqueue before releasing the shared object during map/game shutdown.
    FlushRenderingCommands();
    CaptureReadback.Reset();
    CaptureTarget = nullptr;
    CaptureActor = nullptr;
    Super::EndPlay(EndPlayReason);
}

void AQaiConveyorWorld::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);

    if (!bStageReady)
    {
        ComponentBindAccumulator += DeltaSeconds;
        if (ComponentBindAccumulator >= 0.25f)
        {
            ComponentBindAccumulator = 0.0f;
            BindNativeComponents();
        }
    }

    if (bStageReady)
    {
        if (bRuntimeMotionTest)
        {
            RuntimeMotionTestElapsed += DeltaSeconds;
            if (bPhysicsContactTest && !bPhysicsContactTestInitialized
                && Parcels.IsValidIndex(0) && ParcelHalfExtents.IsValidIndex(0)
                && ParcelLinearVelocities.IsValidIndex(0) && Forklifts[0].Root.IsValid())
            {
                FForkliftRuntime& Forklift = Forklifts[0];
                const FQuat ForkliftQuat = Forklift.Root->GetComponentQuat();
                // The authored demo starts with a pallet and carton already
                // occupying the tines. Move that preloaded cargo behind the
                // vehicle only in this offscreen test so the test parcel can
                // exercise direct fork contact without an impossible overlap.
                int32 CargoSide = -1;
                for (const int32 BodyIndex : {Forklift.PalletDynamicBody, Forklift.CartonDynamicBody})
                {
                    if (!DynamicBoxes.IsValidIndex(BodyIndex))
                    {
                        continue;
                    }
                    FDynamicBoxRuntime& Cargo = DynamicBoxes[BodyIndex];
                    if (USceneComponent* CargoRoot = Cargo.Root.Get())
                    {
                        FVector CargoLocation = Forklift.Root->GetComponentLocation()
                            - Forklift.Root->GetForwardVector() * 185.0f
                            + Forklift.Root->GetRightVector() * static_cast<float>(CargoSide) * 70.0f;
                        CargoLocation.Z = Cargo.HalfExtent.Z;
                        CargoRoot->SetWorldLocation(
                            CargoLocation,
                            false,
                            nullptr,
                            ETeleportType::TeleportPhysics);
                        Cargo.SupportedForklift = INDEX_NONE;
                        Cargo.LinearVelocity = FVector::ZeroVector;
                        Cargo.bAwake = false;
                        CargoSide *= -1;
                    }
                }
                FVector ForkCenters = FVector::ZeroVector;
                float ForkTop = -TNumericLimits<float>::Max();
                int32 ForkCount = 0;
                for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
                {
                    if (Shape.IsWedge()
                        || !Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase))
                    {
                        continue;
                    }
                    FVector LocalCenter = Shape.LocalCenter;
                    LocalCenter.Z += Forklift.LiftCm;
                    const FVector ShapeCenter = Forklift.Root->GetComponentLocation()
                        + ForkliftQuat.RotateVector(LocalCenter);
                    ForkCenters += ShapeCenter;
                    ForkTop = FMath::Max(ForkTop, ShapeCenter.Z + Shape.HalfExtent.Z);
                    ++ForkCount;
                }
                if (ForkCount > 0)
                {
                    FVector ParcelCenter = ForkCenters / static_cast<float>(ForkCount);
                    ParcelCenter.Z = ForkTop + ParcelHalfExtents[0].Z + 0.15f;
                    const FQuat TestParcelRotation = ForkliftQuat
                        * FQuat(FVector::UpVector, PI * 0.5f);
                    Parcels[0]->SetWorldLocationAndRotation(
                        ParcelCenter,
                        TestParcelRotation,
                        false,
                        nullptr,
                        ETeleportType::TeleportPhysics);
                    ParcelLinearVelocities[0] = FVector::ZeroVector;
                    ParcelGrounded[0] = true;
                    ParcelSupportedForklifts[0] = 0;
                    ParcelForkContactsLogged[0] = false;
                    PhysicsContactInitialParcelZ = ParcelCenter.Z;
                    bPhysicsContactTestInitialized = true;
                    SimulatorLog(FString::Printf(
                        TEXT("physics_contact_test_initialized parcel=1 fork_top_z_cm=%.2f parcel_z_cm=%.2f half_extent_cm=(%.2f,%.2f,%.2f) rotated_across_tines=true"),
                        ForkTop,
                        ParcelCenter.Z,
                        ParcelHalfExtents[0].X,
                        ParcelHalfExtents[0].Y,
                        ParcelHalfExtents[0].Z));
                }
            }
            if (bShelfForkTest && !bShelfForkTestInitialized && Forklifts[0].Root.IsValid())
            {
                for (int32 BodyIndex = 0; BodyIndex < DynamicBoxes.Num(); ++BodyIndex)
                {
                    FDynamicBoxRuntime& Body = DynamicBoxes[BodyIndex];
                    if (!Body.bShelfParcel || !Body.Root.IsValid())
                    {
                        continue;
                    }
                    const FForkliftRuntime& Forklift = Forklifts[0];
                    const FQuat ForkliftQuat = Forklift.Root->GetComponentQuat();
                    FVector TineCenters = FVector::ZeroVector;
                    float TineTop = -TNumericLimits<float>::Max();
                    int32 TineCount = 0;
                    for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
                    {
                        if (Shape.IsWedge()
                            || !Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase))
                        {
                            continue;
                        }
                        const FVector ShapeCenter = Forklift.Root->GetComponentLocation()
                            + ForkliftQuat.RotateVector(Shape.LocalCenter);
                        TineCenters += ShapeCenter;
                        TineTop = FMath::Max(TineTop, ShapeCenter.Z + Shape.HalfExtent.Z);
                        ++TineCount;
                    }
                    if (TineCount > 0)
                    {
                        const FQuat TestBodyRotation = ForkliftQuat
                            * FQuat(FVector::UpVector, PI * 0.5f);
                        FVector BodyCenter = TineCenters / static_cast<float>(TineCount);
                        BodyCenter.Z = TineTop
                            + ConveyorTuning::ProjectedVerticalHalfExtent(TestBodyRotation, Body.HalfExtent)
                            + 0.15f;
                        Body.Root->SetWorldLocationAndRotation(
                            BodyCenter,
                            TestBodyRotation,
                            false,
                            nullptr,
                            ETeleportType::TeleportPhysics);
                        Body.SupportedForklift = INDEX_NONE;
                        Body.SupportBodyIndex = INDEX_NONE;
                        Body.LinearVelocity = FVector::ZeroVector;
                        Body.AngularVelocityDegrees = FVector::ZeroVector;
                        Body.bAwake = true;
                        ShelfForkTestBodyIndex = BodyIndex;
                        ShelfForkTestInitialBodyZ = BodyCenter.Z;
                        ShelfForkTestInitialBodyLocation = BodyCenter;
                        bShelfForkTestInitialized = true;
                        SimulatorLog(FString::Printf(
                            TEXT("shelf_fork_test_initialized body=%s fork_top_z_cm=%.2f body_z_cm=%.2f"),
                            *Body.Name,
                            TineTop,
                            BodyCenter.Z));
                    }
                    break;
                }
            }
            if (bEvkPhysicsTest && !bEvkPhysicsTestInitialized
                && RuntimeMotionTestElapsed >= 0.35f
                && DynamicBoxes.IsValidIndex(EvkDynamicBody))
            {
                FDynamicBoxRuntime& EvkBody = DynamicBoxes[EvkDynamicBody];
                if (USceneComponent* EvkRoot = EvkBody.Root.Get())
                {
                    EvkPhysicsTestInitialZ = EvkRoot->GetComponentLocation().Z;
                    EvkPhysicsTestInitialLocation = EvkRoot->GetComponentLocation();
                    EvkBody.SupportedForklift = INDEX_NONE;
                    EvkBody.SupportBodyIndex = INDEX_NONE;
                    EvkBody.LinearVelocity = FVector(0.0f, 650.0f, 0.0f);
                    EvkBody.AngularVelocityDegrees = FVector(0.0f, 0.0f, 24.0f);
                    EvkBody.bAwake = true;
                    bEvkPhysicsTestInitialized = true;
                    SimulatorLog(FString::Printf(
                        TEXT("iq9_evk_physics_test_initialized center=(%.2f,%.2f,%.2f) impulse_velocity_cm_s=(0,650,0)"),
                        EvkRoot->GetComponentLocation().X,
                        EvkRoot->GetComponentLocation().Y,
                        EvkRoot->GetComponentLocation().Z));
                }
            }
            if (bForkliftClimbTest && !bForkliftClimbTestInitialized
                && RuntimeMotionTestElapsed >= 0.20f
                && Forklifts[0].Root.IsValid()
                && DynamicBoxes.IsValidIndex(Forklifts[0].PalletDynamicBody))
            {
                FForkliftRuntime& Forklift = Forklifts[0];
                ActiveForklift = 0;
                FDynamicBoxRuntime& Pallet = DynamicBoxes[Forklift.PalletDynamicBody];
                USceneComponent* PalletRoot = Pallet.Root.Get();
                if (PalletRoot)
                {
                    float RearWheelX = 0.0f;
                    float RearWheelRadius = 0.0f;
                    int32 RearWheelCount = 0;
                    for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
                    {
                        if (Shape.IsCylinder() && Shape.LocalCenter.X < 0.0f)
                        {
                            RearWheelX += Shape.LocalCenter.X;
                            RearWheelRadius = FMath::Max(RearWheelRadius, Shape.Radius);
                            ++RearWheelCount;
                        }
                    }
                    if (RearWheelCount > 0)
                    {
                        RearWheelX /= static_cast<float>(RearWheelCount);
                        const FQuat ForkliftQuat = Forklift.Root->GetComponentQuat();
                        const float PalletHalfAlongVehicle = Pallet.HalfExtent.Y;
                        const FVector LocalPalletCenter(
                            RearWheelX - RearWheelRadius - PalletHalfAlongVehicle - 4.0f,
                            0.0f,
                            0.0f);
                        FVector PalletCenter = Forklift.Root->GetComponentLocation()
                            + ForkliftQuat.RotateVector(LocalPalletCenter);
                        PalletCenter.Z = Pallet.HalfExtent.Z;
                        PalletRoot->SetWorldLocationAndRotation(
                            PalletCenter,
                            ForkliftQuat * FQuat(FVector::UpVector, PI * 0.5f),
                            false,
                            nullptr,
                            ETeleportType::TeleportPhysics);
                        Pallet.SupportedForklift = INDEX_NONE;
                        Pallet.SupportBodyIndex = INDEX_NONE;
                        Pallet.LinearVelocity = FVector::ZeroVector;
                        Pallet.AngularVelocityDegrees = FVector::ZeroVector;
                        Pallet.bAwake = false;
                        ForkliftClimbTestPalletBody = Forklift.PalletDynamicBody;

                        if (DynamicBoxes.IsValidIndex(Forklift.CartonDynamicBody))
                        {
                            FDynamicBoxRuntime& Carton = DynamicBoxes[Forklift.CartonDynamicBody];
                            if (USceneComponent* CartonRoot = Carton.Root.Get())
                            {
                                FVector CartonCenter = Forklift.Root->GetComponentLocation()
                                    + Forklift.Root->GetRightVector() * 220.0f;
                                CartonCenter.Z = Carton.HalfExtent.Z;
                                CartonRoot->SetWorldLocation(
                                    CartonCenter,
                                    false,
                                    nullptr,
                                    ETeleportType::TeleportPhysics);
                                Carton.SupportedForklift = INDEX_NONE;
                                Carton.SupportBodyIndex = INDEX_NONE;
                                Carton.LinearVelocity = FVector::ZeroVector;
                                Carton.bAwake = false;
                            }
                        }
                        bForkliftClimbTestInitialized = true;
                        SimulatorLog(FString::Printf(
                            TEXT("forklift_climb_test_initialized pallet=%s rear_wheel_x=%.1f radius=%.1f pallet_center=(%.1f,%.1f,%.1f)"),
                            *Pallet.Name,
                            RearWheelX,
                            RearWheelRadius,
                            PalletCenter.X,
                            PalletCenter.Y,
                            PalletCenter.Z));
                    }
                }
            }
            if (bForkEdgeBalanceTest && !bForkEdgeBalanceTestInitialized
                && RuntimeMotionTestElapsed >= 0.20f
                && Forklifts[0].Root.IsValid()
                && DynamicBoxes.IsValidIndex(Forklifts[0].PalletDynamicBody))
            {
                FForkliftRuntime& Forklift = Forklifts[0];
                FDynamicBoxRuntime& Pallet = DynamicBoxes[Forklift.PalletDynamicBody];
                USceneComponent* ForkliftRoot = Forklift.Root.Get();
                USceneComponent* PalletRoot = Pallet.Root.Get();
                if (ForkliftRoot && PalletRoot)
                {
                    FVector2D SupportMin(TNumericLimits<float>::Max(), TNumericLimits<float>::Max());
                    FVector2D SupportMax(-TNumericLimits<float>::Max(), -TNumericLimits<float>::Max());
                    float TineTop = -TNumericLimits<float>::Max();
                    int32 TineCount = 0;
                    const FTransform ForkliftTransform = ForkliftRoot->GetComponentTransform();
                    const FQuat ForkliftQuat = ForkliftTransform.GetRotation();
                    for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
                    {
                        if (Shape.IsWedge()
                            || !Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase))
                        {
                            continue;
                        }
                        FVector LocalCenter = Shape.LocalCenter;
                        LocalCenter.Z += Shape.bLiftDriven ? Forklift.LiftCm : 0.0f;
                        const FVector ShapeExtent = Shape.GetCollisionHalfExtent();
                        SupportMin.X = FMath::Min(SupportMin.X, LocalCenter.X - ShapeExtent.X);
                        SupportMin.Y = FMath::Min(SupportMin.Y, LocalCenter.Y - ShapeExtent.Y);
                        SupportMax.X = FMath::Max(SupportMax.X, LocalCenter.X + ShapeExtent.X);
                        SupportMax.Y = FMath::Max(SupportMax.Y, LocalCenter.Y + ShapeExtent.Y);
                        const FVector ShapeCenter = ForkliftTransform.TransformPositionNoScale(LocalCenter);
                        TineTop = FMath::Max(
                            TineTop,
                            ShapeCenter.Z + ConveyorTuning::ProjectedVerticalHalfExtent(
                                ForkliftQuat,
                                ShapeExtent));
                        ++TineCount;
                    }
                    if (TineCount > 0)
                    {
                        const FQuat PalletRotation = ForkliftQuat;
                        const float BodyVerticalHalfExtent = ConveyorTuning::ProjectedVerticalHalfExtent(
                            PalletRotation,
                            Pallet.HalfExtent);
                        FVector TargetCenterOfMassLocal(
                            SupportMax.X + 10.0f,
                            (SupportMin.Y + SupportMax.Y) * 0.5f,
                            0.0f);
                        FVector PalletCenterLocal = TargetCenterOfMassLocal
                            - PalletRotation.UnrotateVector(
                                PalletRotation.RotateVector(Pallet.Physics.CenterOfMassLocalOffset));
                        FVector PalletCenter = ForkliftTransform.TransformPositionNoScale(PalletCenterLocal);
                        PalletCenter.Z = TineTop + BodyVerticalHalfExtent + 0.15f;
                        const FTransform PalletWorld(PalletRotation, PalletCenter);
                        PalletRoot->SetWorldTransform(
                            PalletWorld,
                            false,
                            nullptr,
                            ETeleportType::TeleportPhysics);
                        FTransform BaseWorld = PalletWorld;
                        BaseWorld.AddToTranslation(FVector(0.0f, 0.0f, -Forklift.LiftCm));
                        Pallet.ForkliftRelativeTransform = BaseWorld.GetRelativeTransform(ForkliftTransform);
                        Pallet.SupportedForklift = 0;
                        Pallet.SupportBodyIndex = INDEX_NONE;
                        Pallet.ForkUnsupportedSeconds = 0.0f;
                        Pallet.ForkSupportCooldownSeconds = 0.0f;
                        Pallet.LinearVelocity = FVector::ZeroVector;
                        Pallet.AngularVelocityDegrees = FVector::ZeroVector;
                        Pallet.bAwake = false;
                        ForkEdgeBalanceTestPalletBody = Forklift.PalletDynamicBody;
                        ForkEdgeBalanceTestInitialZ = PalletCenter.Z;
                        bForkEdgeBalanceTestInitialized = true;

                        if (DynamicBoxes.IsValidIndex(Forklift.CartonDynamicBody))
                        {
                            FDynamicBoxRuntime& Carton = DynamicBoxes[Forklift.CartonDynamicBody];
                            if (USceneComponent* CartonRoot = Carton.Root.Get())
                            {
                                FVector CartonCenter = ForkliftRoot->GetComponentLocation()
                                    - ForkliftRoot->GetForwardVector() * 180.0f
                                    + ForkliftRoot->GetRightVector() * 170.0f;
                                CartonCenter.Z = Carton.HalfExtent.Z;
                                CartonRoot->SetWorldLocation(
                                    CartonCenter,
                                    false,
                                    nullptr,
                                    ETeleportType::TeleportPhysics);
                                Carton.SupportedForklift = INDEX_NONE;
                                Carton.SupportBodyIndex = INDEX_NONE;
                                Carton.bAwake = false;
                            }
                        }
                        SimulatorLog(FString::Printf(
                            TEXT("fork_edge_balance_test_initialized pallet=%s center_of_mass_local_x_cm=%.2f support_max_x_cm=%.2f overlap_cm=%.2f initial_z_cm=%.2f"),
                            *Pallet.Name,
                            TargetCenterOfMassLocal.X,
                            SupportMax.X,
                            Pallet.HalfExtent.X - 10.0f,
                            ForkEdgeBalanceTestInitialZ));
                    }
                }
            }
            if (bForkWedgeTest && !bForkWedgeTestInitialized
                && RuntimeMotionTestElapsed >= 0.20f
                && Forklifts[0].Root.IsValid())
            {
                FForkliftRuntime& Forklift = Forklifts[0];
                USceneComponent* ForkliftRoot = Forklift.Root.Get();
                const FFittedCollisionBox* ReferenceWedge = nullptr;
                int32 WedgeCount = 0;
                for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
                {
                    if (Shape.IsWedge())
                    {
                        ReferenceWedge = ReferenceWedge ? ReferenceWedge : &Shape;
                        ++WedgeCount;
                    }
                }
                int32 TestBodyIndex = INDEX_NONE;
                for (int32 BodyIndex = 0; BodyIndex < DynamicBoxes.Num(); ++BodyIndex)
                {
                    if (DynamicBoxes[BodyIndex].bShelfParcel
                        && DynamicBoxes[BodyIndex].Root.IsValid())
                    {
                        TestBodyIndex = BodyIndex;
                        break;
                    }
                }
                if (ForkliftRoot && ReferenceWedge && WedgeCount >= 2
                    && DynamicBoxes.IsValidIndex(TestBodyIndex))
                {
                    // Clear the authored load so only the test carton contacts
                    // the advancing wedge and subsequent flat tine.
                    int32 CargoSide = -1;
                    for (const int32 BodyIndex : {
                        Forklift.PalletDynamicBody,
                        Forklift.CartonDynamicBody})
                    {
                        if (!DynamicBoxes.IsValidIndex(BodyIndex))
                        {
                            continue;
                        }
                        FDynamicBoxRuntime& Cargo = DynamicBoxes[BodyIndex];
                        if (USceneComponent* CargoRoot = Cargo.Root.Get())
                        {
                            FVector CargoLocation = ForkliftRoot->GetComponentLocation()
                                - ForkliftRoot->GetForwardVector() * 190.0f
                                + ForkliftRoot->GetRightVector()
                                    * static_cast<float>(CargoSide) * 80.0f;
                            CargoLocation.Z = Cargo.HalfExtent.Z;
                            CargoRoot->SetWorldLocation(
                                CargoLocation,
                                false,
                                nullptr,
                                ETeleportType::TeleportPhysics);
                            Cargo.SupportedForklift = INDEX_NONE;
                            Cargo.SupportBodyIndex = INDEX_NONE;
                            Cargo.LinearVelocity = FVector::ZeroVector;
                            Cargo.bAwake = false;
                            CargoSide *= -1;
                        }
                    }

                    FDynamicBoxRuntime& Body = DynamicBoxes[TestBodyIndex];
                    const FTransform ForkliftTransform = ForkliftRoot->GetComponentTransform();
                    const FQuat ForkliftQuat = ForkliftTransform.GetRotation();
                    const FVector ReferenceWedgeCenter = ReferenceWedge->LocalCenter;
                    constexpr float InitialOverlapCm = 2.0f;
                    const float TipX = ReferenceWedge->bWedgeTipAtPositiveX
                        ? ReferenceWedge->HalfExtent.X
                        : -ReferenceWedge->HalfExtent.X;
                    const float ContactX = ReferenceWedge->bWedgeTipAtPositiveX
                        ? TipX - InitialOverlapCm
                        : TipX + InitialOverlapCm;
                    FVector BodyCenterLocal = ReferenceWedgeCenter;
                    BodyCenterLocal.X += TipX
                        + (ReferenceWedge->bWedgeTipAtPositiveX ? 1.0f : -1.0f)
                            * (Body.HalfExtent.X - InitialOverlapCm);
                    const FVector SurfaceWorld = ForkliftTransform.TransformPositionNoScale(
                        ReferenceWedgeCenter + FVector(
                            ContactX,
                            0.0f,
                            ReferenceWedge->GetWedgeTopLocalZ(ContactX)));
                    FVector BodyCenter = ForkliftTransform.TransformPositionNoScale(BodyCenterLocal);
                    BodyCenter.Z = SurfaceWorld.Z
                        + ConveyorTuning::ProjectedVerticalHalfExtent(ForkliftQuat, Body.HalfExtent)
                        + 0.1f;
                    Body.Root->SetWorldLocationAndRotation(
                        BodyCenter,
                        ForkliftQuat,
                        false,
                        nullptr,
                        ETeleportType::TeleportPhysics);
                    Body.SupportedForklift = INDEX_NONE;
                    Body.SupportBodyIndex = INDEX_NONE;
                    Body.ForkSupportCooldownSeconds = 0.0f;
                    Body.LinearVelocity = FVector::ZeroVector;
                    Body.AngularVelocityDegrees = FVector::ZeroVector;
                    Body.bAwake = true;
                    Body.bLoggedContact = false;
                    ForkWedgeTestBodyIndex = TestBodyIndex;
                    ForkWedgeTestInitialZ = BodyCenter.Z;
                    ForkWedgeTestInitialLocation = BodyCenter;
                    ForkWedgeTestMaximumZ = BodyCenter.Z;
                    bForkWedgeTestInitialized = true;
                    SimulatorLog(FString::Printf(
                        TEXT("fork_wedge_test_initialized body=%s wedges=%d overlap_cm=%.2f tip_surface_z_cm=%.2f body_z_cm=%.2f"),
                        *Body.Name,
                        WedgeCount,
                        InitialOverlapCm,
                        SurfaceWorld.Z,
                        BodyCenter.Z));
                }
            }
            if (bPhysicsContactTest || bShelfForkTest)
            {
                const bool bLifting = RuntimeMotionTestElapsed >= 0.25f && RuntimeMotionTestElapsed < 1.55f;
                const bool bDriving = RuntimeMotionTestElapsed >= 1.65f && RuntimeMotionTestElapsed < 2.65f;
                DriveActiveForklift(
                    bDriving ? 0.28f : 0.0f,
                    bDriving ? 0.12f : 0.0f,
                    bLifting ? 0.75f : 0.0f,
                    !bDriving && !bLifting,
                    DeltaSeconds);
            }
            else if (bEvkPhysicsTest)
            {
                DriveActiveForklift(0.0f, 0.0f, 0.0f, true, DeltaSeconds);
            }
            else if (bForkliftClimbTest)
            {
                const bool bDriving = RuntimeMotionTestElapsed >= 0.35f
                    && RuntimeMotionTestElapsed < 2.75f;
                DriveActiveForklift(
                    bDriving ? -0.58f : 0.0f,
                    0.0f,
                    0.0f,
                    !bDriving,
                    DeltaSeconds);
            }
            else if (bForkEdgeBalanceTest)
            {
                DriveActiveForklift(0.0f, 0.0f, 0.0f, true, DeltaSeconds);
            }
            else if (bForkWedgeTest)
            {
                const bool bDriving = RuntimeMotionTestElapsed >= 0.30f
                    && RuntimeMotionTestElapsed < 2.05f;
                DriveActiveForklift(
                    bDriving ? 0.34f : 0.0f,
                    0.0f,
                    0.0f,
                    !bDriving,
                    DeltaSeconds);
            }
            else
            {
                const bool bDriving = RuntimeMotionTestElapsed < 2.5f;
                const bool bLifting = RuntimeMotionTestElapsed >= 0.35f && RuntimeMotionTestElapsed < 1.65f;
                DriveActiveForklift(
                    bDriving ? 0.7f : 0.0f,
                    bDriving ? 0.18f : 0.0f,
                    bLifting ? 0.55f : 0.0f,
                    !bDriving,
                    DeltaSeconds);
            }
        }
        FixedAccumulator += FMath::Min(DeltaSeconds, 0.25f);
        int32 StepCount = 0;
        while (FixedAccumulator >= ConveyorTuning::FixedStep && StepCount < ConveyorTuning::MaxFixedStepsPerFrame)
        {
            FixedSimulationStep(ConveyorTuning::FixedStep);
            FixedAccumulator -= ConveyorTuning::FixedStep;
            ++StepCount;
        }
        if (StepCount == ConveyorTuning::MaxFixedStepsPerFrame)
        {
            FixedAccumulator = 0.0f;
        }
        if (bForkWedgeTestInitialized
            && DynamicBoxes.IsValidIndex(ForkWedgeTestBodyIndex)
            && DynamicBoxes[ForkWedgeTestBodyIndex].Root.IsValid())
        {
            ForkWedgeTestMaximumZ = FMath::Max(
                ForkWedgeTestMaximumZ,
                DynamicBoxes[ForkWedgeTestBodyIndex].Root->GetComponentLocation().Z);
        }

        TickResolutionDataset(DeltaSeconds);
        TickStackLightRig(DeltaSeconds);
        TickEvkLeds(DeltaSeconds);

        if (bWorkerSoakTest)
        {
            WorkerSoakTestElapsed += DeltaSeconds;
            for (int32 WorkerIndex = 0; WorkerIndex < UE_ARRAY_COUNT(Workers); ++WorkerIndex)
            {
                if (const USceneComponent* WorkerRoot = Workers[WorkerIndex].Root.Get())
                {
                    WorkerMinimumStaticClearanceCm[WorkerIndex] = FMath::Min(
                        WorkerMinimumStaticClearanceCm[WorkerIndex],
                        GetWorkerStaticClearanceCm(WorkerRoot->GetComponentLocation()));
                }
            }
            if (WorkerSoakTestElapsed >= 30.0f && !bWorkerSoakTestReported)
            {
                bWorkerSoakTestReported = true;
                const FVector Worker1Location = Workers[0].Root.IsValid()
                    ? Workers[0].Root->GetComponentLocation()
                    : FVector::ZeroVector;
                const FVector Worker2Location = Workers[1].Root.IsValid()
                    ? Workers[1].Root->GetComponentLocation()
                    : FVector::ZeroVector;
                const bool bPassed = WorkerMinimumStaticClearanceCm[0] >= -0.5f
                && WorkerMinimumStaticClearanceCm[1] >= -0.5f
                && Workers[0].MaximumVisualTurnRateDegreesPerSecond
                    <= ConveyorTuning::WorkerMaximumVisualTurnRateDegreesPerSecond + 1.0f
                && Workers[1].MaximumVisualTurnRateDegreesPerSecond
                    <= ConveyorTuning::WorkerMaximumVisualTurnRateDegreesPerSecond + 1.0f;
            const FString Result = FString::Printf(
                TEXT("QAI_WORKER_SOAK_TEST passed=%s duration_seconds=%.1f minimum_static_clearance_cm=(%.1f,%.1f) route_reversals=(%d,%d) maximum_visual_turn_deg_s=(%.1f,%.1f) worker1=(%.1f,%.1f) worker2=(%.1f,%.1f)"),
                    bPassed ? TEXT("true") : TEXT("false"),
                    WorkerSoakTestElapsed,
                    WorkerMinimumStaticClearanceCm[0],
                    WorkerMinimumStaticClearanceCm[1],
                Workers[0].RouteReversalCount,
                Workers[1].RouteReversalCount,
                Workers[0].MaximumVisualTurnRateDegreesPerSecond,
                Workers[1].MaximumVisualTurnRateDegreesPerSecond,
                Worker1Location.X,
                    Worker1Location.Y,
                    Worker2Location.X,
                    Worker2Location.Y);
                UE_LOG(LogTemp, Display, TEXT("%s"), *Result);
                SimulatorLog(Result);
            }
            if (WorkerSoakTestElapsed >= 31.0f)
            {
                FPlatformMisc::RequestExit(true);
            }
        }

        const float RuntimeCaptureTime = bHudScreenshotTest ? 6.0f : 3.0f;
        if (bRuntimeMotionTest && RuntimeMotionTestElapsed >= RuntimeCaptureTime && !bRuntimeMotionTestCaptured)
        {
            bRuntimeMotionTestCaptured = true;
            const FVector Worker1Location = Workers[0].Root.IsValid()
                ? Workers[0].Root->GetComponentLocation()
                : FVector::ZeroVector;
            const FVector Worker2Location = Workers[1].Root.IsValid()
                ? Workers[1].Root->GetComponentLocation()
                : FVector::ZeroVector;
            const float RootDeltaCm = Forklifts[0].Root.IsValid()
                ? FVector::Dist(
                    Forklifts[0].InitialRoot.GetLocation(),
                    Forklifts[0].Root->GetComponentLocation())
                : 0.0f;
            const float LiftAssemblyDeltaCm = Forklifts[0].LiftAssembly.IsValid()
                ? Forklifts[0].LiftAssembly->GetRelativeLocation().Z
                    - Forklifts[0].InitialLiftRelativeLocation.Z
                : 0.0f;
            const float LiftVisualDeltaCm = Forklifts[0].LiftVisual.IsValid()
                ? Forklifts[0].LiftVisual->Bounds.Origin.Z - Forklifts[0].InitialLiftVisualCenterZ
                : 0.0f;
            int32 GroundedParcels = 0;
            float MaximumRollerGapCm = 0.0f;
            for (const bool bGrounded : ParcelGrounded)
            {
                GroundedParcels += bGrounded ? 1 : 0;
            }
            for (int32 ParcelIndex = 0; ParcelIndex < Parcels.Num(); ++ParcelIndex)
            {
                if (const USceneComponent* Parcel = Parcels[ParcelIndex].Get();
                    Parcel && ParcelHalfExtents.IsValidIndex(ParcelIndex)
                    && ParcelGrounded.IsValidIndex(ParcelIndex)
                    && ParcelGrounded[ParcelIndex]
                    && ParcelSupportedForklifts.IsValidIndex(ParcelIndex)
                    && ParcelSupportedForklifts[ParcelIndex] == INDEX_NONE)
                {
                    const float Bottom = Parcel->GetComponentLocation().Z - ParcelHalfExtents[ParcelIndex].Z;
                    float Distance = 0.0f;
                    float LateralDistance = 0.0f;
                    FVector Projection;
                    FVector Tangent;
                    ConveyorTuning::ProjectToConveyor(
                        Parcel->GetComponentLocation(),
                        Distance,
                        Projection,
                        Tangent,
                        LateralDistance);
                    if (LateralDistance <= ConveyorTuning::BeltHalfWidth - 3.0f)
                    {
                        MaximumRollerGapCm = FMath::Max(
                            MaximumRollerGapCm,
                            FMath::Abs(Bottom - ConveyorSurfaceZCm));
                    }
                }
            }
            const float ShelfBodyLiftCm = bShelfForkTestInitialized
                && DynamicBoxes.IsValidIndex(ShelfForkTestBodyIndex)
                && DynamicBoxes[ShelfForkTestBodyIndex].Root.IsValid()
                ? DynamicBoxes[ShelfForkTestBodyIndex].Root->GetComponentLocation().Z - ShelfForkTestInitialBodyZ
                : 0.0f;
            const int32 ShelfBodySupport = bShelfForkTestInitialized
                && DynamicBoxes.IsValidIndex(ShelfForkTestBodyIndex)
                ? DynamicBoxes[ShelfForkTestBodyIndex].SupportedForklift + 1
                : 0;
            const float ShelfBodyPlanarDisplacementCm = bShelfForkTestInitialized
                && DynamicBoxes.IsValidIndex(ShelfForkTestBodyIndex)
                && DynamicBoxes[ShelfForkTestBodyIndex].Root.IsValid()
                ? FVector::Dist2D(
                    ShelfForkTestInitialBodyLocation,
                    DynamicBoxes[ShelfForkTestBodyIndex].Root->GetComponentLocation())
                : 0.0f;
            const float EvkDropCm = bEvkPhysicsTestInitialized
                && DynamicBoxes.IsValidIndex(EvkDynamicBody)
                && DynamicBoxes[EvkDynamicBody].Root.IsValid()
                ? EvkPhysicsTestInitialZ - DynamicBoxes[EvkDynamicBody].Root->GetComponentLocation().Z
                : 0.0f;
            const FVector EvkFinalLocation = DynamicBoxes.IsValidIndex(EvkDynamicBody)
                && DynamicBoxes[EvkDynamicBody].Root.IsValid()
                ? DynamicBoxes[EvkDynamicBody].Root->GetComponentLocation()
                : FVector::ZeroVector;
            const float EvkPlanarDisplacementCm = FVector::Dist2D(
                EvkPhysicsTestInitialLocation,
                EvkFinalLocation);
            const bool bEvkPhysicsPassed = !bEvkPhysicsTest
                || (EvkDropCm >= 40.0f && EvkPlanarDisplacementCm >= 60.0f);
            int32 WheelCylinderCount = 0;
            for (const FFittedCollisionBox& Shape : Forklifts[0].CollisionBoxes)
            {
                WheelCylinderCount += Shape.IsCylinder() ? 1 : 0;
            }
            const bool bForkliftClimbPassed = !bForkliftClimbTest
                || (bForkliftClimbTestInitialized
                    && Forklifts[0].MaximumRideHeightCm >= 5.0f
                    && WheelCylinderCount == 4);
            if (bForkEdgeBalanceTest)
            {
                const bool bPalletValid = bForkEdgeBalanceTestInitialized
                    && DynamicBoxes.IsValidIndex(ForkEdgeBalanceTestPalletBody)
                    && DynamicBoxes[ForkEdgeBalanceTestPalletBody].Root.IsValid();
                const float DropCm = bPalletValid
                    ? ForkEdgeBalanceTestInitialZ
                        - DynamicBoxes[ForkEdgeBalanceTestPalletBody].Root->GetComponentLocation().Z
                    : 0.0f;
                const float FinalBottomCm = bPalletValid
                    ? DynamicBoxes[ForkEdgeBalanceTestPalletBody].Root->GetComponentLocation().Z
                        - ConveyorTuning::ProjectedVerticalHalfExtent(
                            DynamicBoxes[ForkEdgeBalanceTestPalletBody].Root->GetComponentQuat(),
                            DynamicBoxes[ForkEdgeBalanceTestPalletBody].HalfExtent)
                    : TNumericLimits<float>::Max();
                const bool bReleased = bPalletValid
                    && DynamicBoxes[ForkEdgeBalanceTestPalletBody].SupportedForklift == INDEX_NONE;
                const bool bPassed = bReleased && FinalBottomCm <= 0.75f;
                const FString ForkEdgeResult = FString::Printf(
                    TEXT("QAI_FORK_EDGE_BALANCE_TEST passed=%s released=%s center_drop_cm=%.2f final_bottom_cm=%.2f support_forklift=%d"),
                    bPassed ? TEXT("true") : TEXT("false"),
                    bReleased ? TEXT("true") : TEXT("false"),
                    DropCm,
                    FinalBottomCm,
                    bPalletValid
                        ? DynamicBoxes[ForkEdgeBalanceTestPalletBody].SupportedForklift + 1
                        : 0);
                UE_LOG(LogTemp, Display, TEXT("%s"), *ForkEdgeResult);
                SimulatorLog(ForkEdgeResult);
            }
            if (bForkWedgeTest)
            {
                const float RampClimbCm = ForkWedgeTestMaximumZ - ForkWedgeTestInitialZ;
                const FVector FinalBodyLocation = bForkWedgeTestInitialized
                    && DynamicBoxes.IsValidIndex(ForkWedgeTestBodyIndex)
                    && DynamicBoxes[ForkWedgeTestBodyIndex].Root.IsValid()
                    ? DynamicBoxes[ForkWedgeTestBodyIndex].Root->GetComponentLocation()
                    : ForkWedgeTestInitialLocation;
                const float PlanarTravelCm = FVector::Dist2D(
                    ForkWedgeTestInitialLocation, FinalBodyLocation);
                const int32 FinalSupport = bForkWedgeTestInitialized
                    && DynamicBoxes.IsValidIndex(ForkWedgeTestBodyIndex)
                    ? DynamicBoxes[ForkWedgeTestBodyIndex].SupportedForklift + 1
                    : 0;
                if (bForkWedgeTestInitialized
                    && DynamicBoxes.IsValidIndex(ForkWedgeTestBodyIndex)
                    && DynamicBoxes[ForkWedgeTestBodyIndex].Root.IsValid()
                    && Forklifts[0].Root.IsValid())
                {
                    const FDynamicBoxRuntime& TestBody = DynamicBoxes[ForkWedgeTestBodyIndex];
                    const FVector WorldLocation = TestBody.Root->GetComponentLocation();
                    const FVector ForkLocalLocation = Forklifts[0].Root->GetComponentTransform()
                        .InverseTransformPositionNoScale(WorldLocation);
                    SimulatorLog(FString::Printf(
                        TEXT("fork_wedge_test_final world=(%.2f,%.2f,%.2f) fork_local=(%.2f,%.2f,%.2f) velocity=(%.2f,%.2f,%.2f)"),
                        WorldLocation.X, WorldLocation.Y, WorldLocation.Z,
                        ForkLocalLocation.X, ForkLocalLocation.Y, ForkLocalLocation.Z,
                        TestBody.LinearVelocity.X, TestBody.LinearVelocity.Y, TestBody.LinearVelocity.Z));
                }
                const bool bPriedUnder = RampClimbCm >= 4.0f && FinalSupport == 1;
                const bool bFrictionTransferred = RampClimbCm >= 1.0f
                    && PlanarTravelCm >= 20.0f;
                const bool bPassed = bForkWedgeTestInitialized
                    && (bPriedUnder || bFrictionTransferred);
                const FString ForkWedgeResult = FString::Printf(
                    TEXT("QAI_FORK_WEDGE_TEST passed=%s mode=%s ramp_climb_cm=%.2f planar_cm=%.2f support_forklift=%d"),
                    bPassed ? TEXT("true") : TEXT("false"),
                    bPriedUnder ? TEXT("pried_under") : TEXT("friction_transfer"),
                    RampClimbCm,
                    PlanarTravelCm,
                    FinalSupport);
                UE_LOG(LogTemp, Display, TEXT("%s"), *ForkWedgeResult);
                SimulatorLog(ForkWedgeResult);
            }
            if (bShelfForkTest)
            {
                const bool bPassed = bShelfForkTestInitialized
                    && ShelfBodyLiftCm >= 20.0f
                    && ShelfBodyPlanarDisplacementCm >= 15.0f
                    && ShelfBodySupport == 1;
                const FString ShelfForkResult = FString::Printf(
                    TEXT("QAI_SHELF_FORK_TEST passed=%s lift_cm=%.2f planar_cm=%.2f support_forklift=%d"),
                    bPassed ? TEXT("true") : TEXT("false"),
                    ShelfBodyLiftCm,
                    ShelfBodyPlanarDisplacementCm,
                    ShelfBodySupport);
                UE_LOG(LogTemp, Display, TEXT("%s"), *ShelfForkResult);
                SimulatorLog(ShelfForkResult);
            }
            const FString MotionResult = FString::Printf(
                TEXT("QAI_RUNTIME_MOTION_TEST root_delta_cm=%0.1f lift_delta_cm=%0.1f lift_visual_delta_cm=%0.1f wheels=%0.2f parcels_grounded=%d parcel_impacts=%d fork_contacts=%d maximum_grounded_roller_gap_cm=%.3f contact_parcel_lift_cm=%.2f shelf_body_lift_cm=%.2f shelf_body_support_forklift=%d evk_drop_cm=%.2f evk_planar_cm=%.2f evk_test_passed=%s evk_final=(%.2f,%.2f,%.2f) forklift_shapes=%d wheel_cylinders=%d forklift_climb_cm=%.2f forklift_climb_test_passed=%s maximum_tip_degrees=%.2f worker1=(%0.1f,%0.1f) worker2=(%0.1f,%0.1f)"),
                RootDeltaCm,
                LiftAssemblyDeltaCm,
                LiftVisualDeltaCm,
                Forklifts[0].WheelAngleDegrees,
                GroundedParcels,
                ParcelImpactCount,
                ParcelForkContactCount,
                MaximumRollerGapCm,
                bPhysicsContactTestInitialized && Parcels[0].IsValid()
                    ? Parcels[0]->GetComponentLocation().Z - PhysicsContactInitialParcelZ
                    : 0.0f,
                ShelfBodyLiftCm,
                ShelfBodySupport,
                EvkDropCm,
                EvkPlanarDisplacementCm,
                bEvkPhysicsPassed ? TEXT("true") : TEXT("false"),
                EvkFinalLocation.X,
                EvkFinalLocation.Y,
                EvkFinalLocation.Z,
                Forklifts[0].CollisionBoxes.Num(),
                WheelCylinderCount,
                Forklifts[0].MaximumRideHeightCm,
                bForkliftClimbPassed ? TEXT("true") : TEXT("false"),
                Forklifts[0].MaximumAbsoluteTipDegrees,
                Worker1Location.X,
                Worker1Location.Y,
                Worker2Location.X,
                Worker2Location.Y);
            UE_LOG(LogTemp, Display, TEXT("%s"), *MotionResult);
            SimulatorLog(MotionResult);
            const FString ScreenshotPath = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("Screenshots/runtime-motion.png"));
            IFileManager::Get().MakeDirectory(*FPaths::GetPath(ScreenshotPath), true);
            FScreenshotRequest::RequestScreenshot(ScreenshotPath, bHudScreenshotTest, false);
        }
        const float RuntimeExitTime = bHudScreenshotTest ? 8.0f : 5.0f;
        if (bRuntimeMotionTest && RuntimeMotionTestElapsed >= RuntimeExitTime)
        {
            // This path is used only by the off-screen build validator. A
            // forced exit avoids an UnrealEditor 5.8 typed-element teardown
            // crash after the screenshot has already been flushed to disk.
            FPlatformMisc::RequestExit(true);
        }
        if (bPresentation4K)
        {
            Presentation4KElapsed += DeltaSeconds;
            if (Presentation4KElapsed >= 8.0f && !bPresentation4KCaptured)
            {
                bPresentation4KCaptured = true;
                const FString ScreenshotPath = FPaths::Combine(
                    FPaths::ProjectSavedDir(),
                    TEXT("Screenshots/presentation-4k.png"));
                IFileManager::Get().MakeDirectory(*FPaths::GetPath(ScreenshotPath), true);
                FHighResScreenshotConfig& CaptureConfig = GetHighResScreenshotConfig();
                const bool bResolutionConfigured = CaptureConfig.SetResolution(3840, 2160, 1.0f);
                CaptureConfig.SetFilename(ScreenshotPath);
                const bool bCaptureQueued = bResolutionConfigured
                    && GEngine
                    && GEngine->GameViewport
                    && GEngine->GameViewport->Viewport
                    && GEngine->GameViewport->Viewport->TakeHighResScreenShot();
                SimulatorLog(FString::Printf(
                    TEXT("presentation_capture requested path=\"%s\" resolution=3840x2160 queued=%s elapsed_seconds=%.2f"),
                    *SanitizeLogField(ScreenshotPath),
                    bCaptureQueued ? TEXT("true") : TEXT("false"),
                    Presentation4KElapsed));
            }
            if (Presentation4KElapsed >= 11.0f)
            {
                FPlatformMisc::RequestExit(true);
            }
        }
    }

    DrawCollisionDebug();
    TickInferenceCapture(DeltaSeconds);
}


void AQaiConveyorWorld::TickResolutionDataset(float DeltaSeconds)
{
    if (!bResolutionDataset || !Forklifts[0].Root.IsValid())
    {
        return;
    }

    if (!bResolutionDatasetInitialized)
    {
        bResolutionDatasetInitialized = true;
        ResolutionDatasetInitialForkliftTransform = Forklifts[0].Root->GetComponentTransform();
        ResolutionDatasetElapsed = 0.0f;
        ResolutionDatasetPhase = INDEX_NONE;
        SimulatorLog(FString::Printf(
            TEXT("resolution_dataset initialized capture=%dx%d"),
            HostCaptureWidth,
            HostCaptureHeight));
    }

    ResolutionDatasetElapsed += DeltaSeconds;
    const int32 NewPhase = ResolutionDatasetElapsed < 12.0f
        ? 0
        : (ResolutionDatasetElapsed < 24.0f ? 1 : 2);
    if (NewPhase != ResolutionDatasetPhase)
    {
        ResolutionDatasetPhase = NewPhase;
        CancelActiveInference();
        const TCHAR* PhaseSignal = NewPhase == 0 ? TEXT("G") : (NewPhase == 1 ? TEXT("A") : TEXT("R"));
        SimulatorLog(FString::Printf(
            TEXT("resolution_dataset phase=%s elapsed_seconds=%.2f"),
            PhaseSignal,
            ResolutionDatasetElapsed));
    }

    FForkliftRuntime& Forklift = Forklifts[0];
    FTransform TestTransform = ResolutionDatasetInitialForkliftTransform;
    if (ResolutionDatasetPhase == 1)
    {
        const float PhaseSeconds = ResolutionDatasetElapsed - 12.0f;
        FVector AwayFromBelt = ResolutionDatasetInitialForkliftTransform.GetLocation()
            - FVector(ConveyorTuning::BeltCenterX, ConveyorTuning::BeltCenterY, 0.0f);
        AwayFromBelt.Z = 0.0f;
        AwayFromBelt = AwayFromBelt.GetSafeNormal();
        TestTransform.AddToTranslation(AwayFromBelt * (PhaseSeconds * 7.0f));
        Forklift.SpeedCmPerSecond = 7.0f;
    }
    else if (ResolutionDatasetPhase == 2)
    {
        FVector RedLocation = ResolutionDatasetInitialForkliftTransform.GetLocation();
        RedLocation.X = ConveyorTuning::BeltCenterX;
        RedLocation.Y = ConveyorTuning::BeltCenterY;
        TestTransform.SetLocation(RedLocation);
        Forklift.SpeedCmPerSecond = 0.0f;
    }
    else
    {
        Forklift.SpeedCmPerSecond = 0.0f;
    }


    Forklift.Root->SetWorldTransform(TestTransform, false, nullptr, ETeleportType::TeleportPhysics);
    UpdateSafetySignal();
}

void AQaiConveyorWorld::ApplyResolutionDatasetVariant()
{
    if (!bResolutionDataset || !GetWorld())
    {
        return;
    }

    int32 WhiteFloorComponents = 0;
    int32 HiddenWorkers = 0;
    int32 HiddenParcels = 0;

    if (bResolutionDatasetWhiteFloor)
    {
        UTexture* WhiteTexture = LoadObject<UTexture>(
            nullptr,
            TEXT("/Engine/EngineResources/WhiteSquareTexture.WhiteSquareTexture"));
        for (TActorIterator<AActor> It(GetWorld()); It; ++It)
        {
            TInlineComponentArray<UStaticMeshComponent*> Meshes(*It);
            for (UStaticMeshComponent* Mesh : Meshes)
            {
                if (!Mesh || !Mesh->GetStaticMesh())
                {
                    continue;
                }
                const FString ComponentName = Mesh->GetName();
                const FString MeshName = Mesh->GetStaticMesh()->GetName();
                const bool bWarehouseFloor = MeshName.Contains(TEXT("SM_floor02"), ESearchCase::IgnoreCase)
                    || ComponentName.StartsWith(TEXT("FloorFinish"), ESearchCase::IgnoreCase)
                    || ComponentName.StartsWith(TEXT("BehindWallFloor"), ESearchCase::IgnoreCase);
                if (!bWarehouseFloor)
                {
                    continue;
                }
                for (int32 Slot = 0; Slot < Mesh->GetNumMaterials(); ++Slot)
                {
                    if (UMaterialInterface* Source = Mesh->GetMaterial(Slot))
                    {
                        UMaterialInstanceDynamic* WhiteFloor = UMaterialInstanceDynamic::Create(Source, this);
                        if (WhiteFloor && WhiteTexture)
                        {
                            WhiteFloor->SetTextureParameterValue(TEXT("BaseColorTexture"), WhiteTexture);
                            Mesh->SetMaterial(Slot, WhiteFloor);
                        }
                    }
                }
                ++WhiteFloorComponents;
            }
        }
    }

    const auto HideComponentTree = [](USceneComponent* Root)
    {
        if (Root)
        {
            Root->SetVisibility(false, true);
            Root->SetHiddenInGame(true, true);
        }
    };
    if (bResolutionDatasetHideWorkers)
    {
        for (FWorkerRuntime& Worker : Workers)
        {
            if (USceneComponent* Root = Worker.Root.Get())
            {
                HideComponentTree(Root);
                ++HiddenWorkers;
            }
        }
    }
    if (bResolutionDatasetHideParcels)
    {
        for (const TWeakObjectPtr<USceneComponent>& Parcel : Parcels)
        {
            if (USceneComponent* Root = Parcel.Get())
            {
                HideComponentTree(Root);
                ++HiddenParcels;
            }
        }
        for (FDynamicBoxRuntime& Body : DynamicBoxes)
        {
            if ((Body.bShelfParcel
                    || Body.Name.Contains(TEXT("parcel"), ESearchCase::IgnoreCase)
                    || Body.Name.Contains(TEXT("carton"), ESearchCase::IgnoreCase))
                && Body.Root.IsValid())
            {
                HideComponentTree(Body.Root.Get());
                ++HiddenParcels;
            }
        }
    }
    SimulatorLog(FString::Printf(
        TEXT("resolution_dataset variant_applied name=%s white_floor_components=%d hidden_workers=%d hidden_parcels=%d"),
        *ResolutionDatasetVariant,
        WhiteFloorComponents,
        HiddenWorkers,
        HiddenParcels));
}

void AQaiConveyorWorld::LoadRuntimeConfig()
{
#if PLATFORM_MAC
    const FString SettingsRoot = FPaths::Combine(
        FPlatformProcess::UserHomeDir(),
        TEXT("Library/Application Support/QaiConveyorDemo"));
#else
    const FString SettingsRoot = FPaths::Combine(FPlatformProcess::UserSettingsDir(), TEXT("QaiConveyorDemo"));
#endif
    const FString ConfigPath = FPaths::Combine(SettingsRoot, TEXT("runtime.json"));
    FString Text;
    if (FFileHelper::LoadFileToString(Text, *ConfigPath))
    {
        TSharedPtr<FJsonObject> Json;
        const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Text);
        if (FJsonSerializer::Deserialize(Reader, Json) && Json.IsValid())
        {
            Json->TryGetStringField(TEXT("host_server_url"), HostServerUrl);
            Json->TryGetStringField(TEXT("host_model"), HostModel);
            Json->TryGetStringField(TEXT("evk_server_url"), EvkServerUrl);
            Json->TryGetStringField(TEXT("evk_model"), EvkModel);
            Json->TryGetStringField(TEXT("backend"), ActiveBackend);
            Json->TryGetBoolField(TEXT("auto_inference"), bInferenceEnabled);
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Reason2 runtime configuration was not found: %s"), *ConfigPath);
        SimulatorLog(FString::Printf(TEXT("runtime_config_missing path=%s"), *ConfigPath));
    }

    FParse::Value(FCommandLine::Get(), TEXT("HostServer="), HostServerUrl);
    FParse::Value(FCommandLine::Get(), TEXT("HostModel="), HostModel);
    FParse::Value(FCommandLine::Get(), TEXT("EvkServer="), EvkServerUrl);
    FParse::Value(FCommandLine::Get(), TEXT("EvkModel="), EvkModel);
    FParse::Value(FCommandLine::Get(), TEXT("Backend="), ActiveBackend);
    bInferenceEnabled &= !FParse::Param(FCommandLine::Get(), TEXT("NoInference"));
    ActiveBackend = ActiveBackend.Equals(TEXT("evk"), ESearchCase::IgnoreCase) ? TEXT("evk") : TEXT("host");
    int32 RequestedHostCaptureWidth = ConveyorTuning::DefaultHostCaptureWidth;
    FParse::Value(FCommandLine::Get(), TEXT("QaiHostCaptureWidth="), RequestedHostCaptureWidth);
    HostCaptureWidth = FMath::Clamp(
        FMath::RoundToInt(static_cast<float>(RequestedHostCaptureWidth) / 16.0f) * 16,
        ConveyorTuning::DefaultHostCaptureWidth,
        ConveyorTuning::MaximumHostCaptureWidth);
    HostCaptureHeight = HostCaptureWidth * 9 / 16;
    CaptureRenderWidth = FMath::Max(HostCaptureWidth, ConveyorTuning::EvkCaptureWidth);
    CaptureRenderHeight = CaptureRenderWidth * 9 / 16;
    ActiveModel = CurrentModelName();
    UE_LOG(
        LogTemp,
        Display,
        TEXT("Reason2 runtime configured: backend=%s model=%s host=%s evk=%s inference=%s"),
        *ActiveBackend,
        *ActiveModel,
        *HostServerUrl,
        *EvkServerUrl,
        bInferenceEnabled ? TEXT("on") : TEXT("off"));
    SimulatorLog(FString::Printf(
        TEXT("runtime_configured backend=%s model=%s host=%s evk=%s inference=%s"),
        *ActiveBackend,
        *ActiveModel,
        *HostServerUrl,
        *EvkServerUrl,
        bInferenceEnabled ? TEXT("on") : TEXT("off")));
}

AActor* AQaiConveyorWorld::FindTaggedActor(const FName Tag) const
{
    if (!GetWorld())
    {
        return nullptr;
    }
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        if (It->ActorHasTag(Tag))
        {
            return *It;
        }
    }
    return nullptr;
}

USceneComponent* AQaiConveyorWorld::FindTaggedComponent(const FName Tag) const
{
    if (!GetWorld())
    {
        return nullptr;
    }
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        TInlineComponentArray<USceneComponent*> Components(*It);
        for (USceneComponent* Component : Components)
        {
            if (Component && Component->ComponentHasTag(Tag))
            {
                return Component;
            }
        }
    }
    return nullptr;
}

void AQaiConveyorWorld::MakeMovable(USceneComponent* Component)
{
    if (!Component)
    {
        return;
    }
    const auto NormalizeMovableTransform = [](USceneComponent* Value)
    {
        if (!Value)
        {
            return;
        }
        const FTransform WorldTransform = Value->GetComponentTransform();
        Value->SetMobility(EComponentMobility::Movable);
        if (Value->IsUsingAbsoluteLocation()
            || Value->IsUsingAbsoluteRotation()
            || Value->IsUsingAbsoluteScale())
        {
            // USD imports can retain absolute child transforms. They render
            // correctly at rest but ignore a moving lift/pivot parent. Convert
            // them to relative transforms while preserving the authored pose.
            Value->SetUsingAbsoluteLocation(false);
            Value->SetUsingAbsoluteRotation(false);
            Value->SetUsingAbsoluteScale(false);
            Value->SetWorldTransform(
                WorldTransform,
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
        }
    };
    NormalizeMovableTransform(Component);
    TArray<USceneComponent*> Descendants;
    Component->GetChildrenComponents(true, Descendants);
    for (USceneComponent* Descendant : Descendants)
    {
        if (Descendant)
        {
            NormalizeMovableTransform(Descendant);
        }
    }
}

bool AQaiConveyorWorld::ApplyWestShelfPlacement()
{
    if (bWestShelfPlacementApplied)
    {
        return true;
    }
    if (!GetWorld())
    {
        return false;
    }

    UPrimitiveComponent* RackVisual = nullptr;
    UPrimitiveComponent* WestWall = nullptr;
    TSet<USceneComponent*> ShelfRoots;
    int32 ShelfParcelCount = 0;
    int32 ShelfColliderCount = 0;
    bool bMovedRackStackLight = false;

    const auto LogicalComponentName = [](const USceneComponent* Component)
    {
        FString Name = Component ? Component->GetName() : FString();
        int32 LastUnderscore = INDEX_NONE;
        if (Name.FindLastChar(TEXT('_'), LastUnderscore)
            && Name.Mid(LastUnderscore + 1).IsNumeric())
        {
            Name.LeftInline(LastUnderscore, EAllowShrinking::No);
        }
        return Name;
    };

    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        TInlineComponentArray<USceneComponent*> Components(*It);
        for (USceneComponent* Component : Components)
        {
            if (!Component)
            {
                continue;
            }

            const FString LogicalName = LogicalComponentName(Component);
            if (LogicalName == TEXT("WestRack"))
            {
                RackVisual = Cast<UPrimitiveComponent>(Component);
                ShelfRoots.Add(Component);
            }
            else if (LogicalName == TEXT("WestWall"))
            {
                WestWall = Cast<UPrimitiveComponent>(Component);
            }
            else if (LogicalName.StartsWith(TEXT("ShelfPlane"))
                || LogicalName == TEXT("RearGuard")
                || LogicalName.Contains(TEXT("Upright")))
            {
                ShelfRoots.Add(Component);
                ++ShelfColliderCount;
            }
            else if (LogicalName == TEXT("StackLight"))
            {
                // StackLight_0 is the signal mounted on the west rack. The
                // other two roots retain their _01/_02 logical suffixes.
                ShelfRoots.Add(Component);
                bMovedRackStackLight = true;
            }

            if (Component->ComponentHasTag(TEXT("Qai.ShelfParcel")))
            {
                ShelfRoots.Add(Component);
                ++ShelfParcelCount;
            }
        }
    }

    if (!RackVisual || !WestWall)
    {
        return false;
    }

    // Move the rack toward the room-facing surface of the west wall, retaining
    // a small physical clearance. The authored gap is about 43 cm; moving a
    // literal 50 cm would put the rear frame several centimetres into the wall.
    constexpr float DesiredWallClearanceCm = 3.0f;
    constexpr float MaximumCorrectionCm = 50.0f;
    const float DirectionToWall = FMath::Sign(WestWall->Bounds.Origin.X - RackVisual->Bounds.Origin.X);
    const float WallInnerFaceX = WestWall->Bounds.Origin.X - DirectionToWall * WestWall->Bounds.BoxExtent.X;
    const float RackWallFaceX = RackVisual->Bounds.Origin.X + DirectionToWall * RackVisual->Bounds.BoxExtent.X;
    const float GapBeforeCm = FMath::Abs(RackWallFaceX - WallInnerFaceX);
    const float CorrectionCm = FMath::Clamp(
        GapBeforeCm - DesiredWallClearanceCm,
        0.0f,
        MaximumCorrectionCm);
    const FVector Delta(DirectionToWall * CorrectionCm, 0.0f, 0.0f);

    for (USceneComponent* Root : ShelfRoots)
    {
        MakeMovable(Root);
        Root->AddWorldOffset(Delta, false, nullptr, ETeleportType::TeleportPhysics);
    }

    bWestShelfPlacementApplied = true;
    SimulatorLog(FString::Printf(
        TEXT("west_shelf_placement gap_before_cm=%.1f clearance_after_cm=%.1f delta_x_cm=%.1f roots=%d shelf_colliders=%d shelf_parcels=%d rack_stack_light=%s"),
        GapBeforeCm,
        FMath::Max(0.0f, GapBeforeCm - CorrectionCm),
        Delta.X,
        ShelfRoots.Num(),
        ShelfColliderCount,
        ShelfParcelCount,
        bMovedRackStackLight ? TEXT("moved") : TEXT("missing")));
    return true;
}

void AQaiConveyorWorld::ConfigureForkliftMastMaterials()
{
    if (bForkliftMastMaterialsConfigured || !GetWorld())
    {
        return;
    }

    int32 RoughenedComponents = 0;
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        TInlineComponentArray<UStaticMeshComponent*> MeshComponents(*It);
        for (UStaticMeshComponent* MeshComponent : MeshComponents)
        {
            if (!MeshComponent)
            {
                continue;
            }

            const FString Name = MeshComponent->GetName();
            const bool bMastRail = Name.Contains(TEXT("MastStage"), ESearchCase::IgnoreCase)
                && !Name.Contains(TEXT("MastStageWheel"), ESearchCase::IgnoreCase);
            const bool bLiftCarriage = Name.Contains(TEXT("MastForcer"), ESearchCase::IgnoreCase)
                || Name.Contains(TEXT("LiftCore"), ESearchCase::IgnoreCase)
                || Name.Contains(TEXT("Carriage"), ESearchCase::IgnoreCase)
                || Name.Contains(TEXT("ForkFrame"), ESearchCase::IgnoreCase);
            if (!bMastRail && !bLiftCarriage)
            {
                continue;
            }

            for (int32 MaterialIndex = 0; MaterialIndex < MeshComponent->GetNumMaterials(); ++MaterialIndex)
            {
                UMaterialInterface* ExistingMaterial = MeshComponent->GetMaterial(MaterialIndex);
                UMaterialInstanceDynamic* MastMaterial = ExistingMaterial
                    ? UMaterialInstanceDynamic::Create(ExistingMaterial, this)
                    : nullptr;
                if (!MastMaterial)
                {
                    continue;
                }

                // Preserve the imported forklift atlas and normal detail.
                // Override every material slot on the moving
                // mast/carriage assembly: the remaining mirror patch lived on
                // a secondary LiftCore slot and came from its metallic mask,
                // not roughness alone. The mast is painted/coated steel, so it
                // should read like the surrounding dark frame rather than a
                // polished mirror.
                MastMaterial->SetScalarParameterValue(TEXT("UseRoughnessTexture"), 0.0f);
                MastMaterial->SetScalarParameterValue(TEXT("Roughness"), 0.72f);
                MastMaterial->SetScalarParameterValue(TEXT("RoughnessScale"), 2.75f);
                MastMaterial->SetScalarParameterValue(TEXT("UseMetallicTexture"), 0.0f);
                MastMaterial->SetScalarParameterValue(TEXT("Metallic"), 0.12f);
                MastMaterial->SetScalarParameterValue(TEXT("MetallicScale"), 0.12f);
                MeshComponent->SetMaterial(MaterialIndex, MastMaterial);
                ++RoughenedComponents;
            }
        }
    }

    bForkliftMastMaterialsConfigured = RoughenedComponents > 0;
    SimulatorLog(FString::Printf(
        TEXT("forklift_mast_material components=%d roughness=0.72 metallic=0.12 preserve_albedo=true preserve_normals=true"),
        RoughenedComponents));
}

AQaiConveyorWorld::FPropPhysicsProfile AQaiConveyorWorld::BuildPropPhysicsProfile(
    const FString& Kind,
    const FVector& HalfExtent,
    int32 Variant) const
{
    FPropPhysicsProfile Profile;
    const FString LowerKind = Kind.ToLower();
    const float VolumeCm3 = FMath::Max(
        1.0f,
        8.0f * HalfExtent.X * HalfExtent.Y * HalfExtent.Z);

    if (LowerKind.Contains(TEXT("pallet")))
    {
        Profile.Surface = TEXT("wood_pallet");
        Profile.MassKg = 22.0f;
        Profile.CenterOfMassLocalOffset = FVector(0.0f, 0.0f, -HalfExtent.Z * 0.18f);
        Profile.StaticFriction = 0.74f;
        Profile.DynamicFriction = 0.58f;
        Profile.Restitution = 0.035f;
        Profile.GroundAngularDamping = 5.8f;
        Profile.SleepLinearSpeedCm = 0.65f;
        Profile.SleepAngularSpeedDegrees = 0.55f;
    }
    else if (LowerKind.Contains(TEXT("evk")) || LowerKind.Contains(TEXT("iq9")))
    {
        Profile.Surface = TEXT("electronics_enclosure");
        Profile.MassKg = ConveyorTuning::IQ9EvkMassKg;
        Profile.CenterOfMassLocalOffset = FVector(0.18f, -0.12f, -0.42f);
        Profile.StaticFriction = 0.52f;
        Profile.DynamicFriction = 0.39f;
        Profile.Restitution = 0.10f;
        Profile.GroundAngularDamping = 3.6f;
        Profile.SleepLinearSpeedCm = 0.75f;
        Profile.SleepAngularSpeedDegrees = 0.70f;
    }
    else if (LowerKind.Contains(TEXT("desktop tray")) || LowerKind.Contains(TEXT("plastic tray")))
    {
        // The authored H20 container is a thin-walled, empty polypropylene
        // sorting tray. Treating its enclosing volume as solid plastic makes
        // it unrealistically heavy, so use a measured empty-bin class mass.
        Profile.Surface = TEXT("rigid_plastic_tray");
        Profile.MassKg = 1.25f;
        Profile.CenterOfMassLocalOffset = FVector(0.0f, 0.0f, -HalfExtent.Z * 0.22f);
        Profile.StaticFriction = 0.44f;
        Profile.DynamicFriction = 0.31f;
        Profile.Restitution = 0.14f;
        Profile.GroundAngularDamping = 3.2f;
        Profile.SleepLinearSpeedCm = 0.72f;
        Profile.SleepAngularSpeedDegrees = 0.66f;
    }
    else
    {
        const bool bShelfCarton = LowerKind.Contains(TEXT("shelf"));
        const bool bForkliftCarton = LowerKind.Contains(TEXT("forklift"));
        Profile.Surface = TEXT("loaded_cardboard");
        // The large forklift carton represents a palletized industrial load,
        // not an empty corrugated shell. Shelf and conveyor cartons remain
        // light enough to be displaced individually.
        const float BulkDensityKgPerCm3 = bForkliftCarton ? 0.00085f : 0.000050f;
        Profile.MassKg = FMath::Clamp(
            VolumeCm3 * BulkDensityKgPerCm3,
            bForkliftCarton ? 120.0f : 1.4f,
            bForkliftCarton ? 450.0f : 14.0f);
        // Loaded cartons are rarely perfectly balanced. A small deterministic
        // offset makes different boxes tip and settle differently without
        // introducing random, non-reproducible inference frames.
        const int32 XCode = ((Variant * 37 + 1) % 5) - 2;
        const int32 YCode = ((Variant * 53 + 3) % 5) - 2;
        Profile.CenterOfMassLocalOffset = FVector(
            HalfExtent.X * 0.035f * static_cast<float>(XCode),
            HalfExtent.Y * 0.035f * static_cast<float>(YCode),
            -HalfExtent.Z * (bShelfCarton ? 0.14f : 0.10f));
        Profile.StaticFriction = 0.66f;
        Profile.DynamicFriction = 0.50f;
        Profile.Restitution = 0.075f;
        Profile.GroundAngularDamping = 4.4f;
        Profile.SleepLinearSpeedCm = 0.85f;
        Profile.SleepAngularSpeedDegrees = 0.80f;
    }

    const FVector FullSize = HalfExtent * 2.0f;
    Profile.InertiaTensorKgCm2 = FVector(
        Profile.MassKg * (FMath::Square(FullSize.Y) + FMath::Square(FullSize.Z)) / 12.0f,
        Profile.MassKg * (FMath::Square(FullSize.X) + FMath::Square(FullSize.Z)) / 12.0f,
        Profile.MassKg * (FMath::Square(FullSize.X) + FMath::Square(FullSize.Y)) / 12.0f)
        .ComponentMax(FVector(1.0f));
    Profile.ImpactResponseScale = FMath::Clamp(
        FMath::Sqrt(4.0f / FMath::Max(Profile.MassKg, 0.1f)),
        0.34f,
        1.75f);
    const float MeanInertia = (
        Profile.InertiaTensorKgCm2.X
        + Profile.InertiaTensorKgCm2.Y
        + Profile.InertiaTensorKgCm2.Z) / 3.0f;
    Profile.AngularResponseScale = FMath::Clamp(
        FMath::Sqrt(4500.0f / FMath::Max(MeanInertia, 1.0f)),
        0.28f,
        1.65f);
    return Profile;
}

int32 AQaiConveyorWorld::BuildCollisionGuard()
{
    AuthoredCollisionComponents.Reset();
    CollisionObstacles.Reset();

    FBox ConveyorBounds(ForceInit);
    auto AddObstacle = [this](const FString& Name, const FBoxSphereBounds& Bounds)
    {
        if (Bounds.BoxExtent.X <= UE_KINDA_SMALL_NUMBER || Bounds.BoxExtent.Y <= UE_KINDA_SMALL_NUMBER)
        {
            return;
        }
        FCollisionObstacle& Obstacle = CollisionObstacles.AddDefaulted_GetRef();
        Obstacle.Name = Name;
        Obstacle.Center = Bounds.Origin;
        Obstacle.HalfExtent = Bounds.BoxExtent;
        Obstacle.Rotation = FQuat::Identity;
    };

    if (GetWorld())
    {
        for (TActorIterator<AActor> It(GetWorld()); It; ++It)
        {
            TInlineComponentArray<UPrimitiveComponent*> Components(*It);
            for (UPrimitiveComponent* Primitive : Components)
            {
                if (!Primitive || !Primitive->IsRegistered())
                {
                    continue;
                }

                FString LogicalName = Primitive->GetName();
                int32 LastUnderscore = INDEX_NONE;
                if (LogicalName.FindLastChar(TEXT('_'), LastUnderscore))
                {
                    const FString Suffix = LogicalName.Mid(LastUnderscore + 1);
                    if (Suffix.IsNumeric())
                    {
                        LogicalName.LeftInline(LastUnderscore, EAllowShrinking::No);
                    }
                }

                const bool bAuthoredCollider = LogicalName == TEXT("Surface")
                    || LogicalName.StartsWith(TEXT("ShelfPlane"))
                    || LogicalName == TEXT("RearGuard")
                    || LogicalName.Contains(TEXT("Upright"))
                    || LogicalName == TEXT("PackingTable")
                    || LogicalName.StartsWith(TEXT("NorthWall"))
                    || LogicalName == TEXT("WestWall");
                if (bAuthoredCollider)
                {
                    AuthoredCollisionComponents.Add(Primitive);
                }

                if (LogicalName == TEXT("Surface"))
                {
                    ConveyorBounds += Primitive->Bounds.GetBox();
                }
                else if (LogicalName.StartsWith(TEXT("ShelfPlane")))
                {
                    AddObstacle(TEXT("shelf plane"), Primitive->Bounds);
                }
                else if (LogicalName == TEXT("RearGuard") || LogicalName.Contains(TEXT("Upright")))
                {
                    AddObstacle(TEXT("shelf frame"), Primitive->Bounds);
                }
                else if (LogicalName == TEXT("PackingTable"))
                {
                    AddObstacle(TEXT("packing table"), Primitive->Bounds);
                    if (PackingTableDesktopBounds.IsValid && CollisionObstacles.Num() > 0)
                    {
                        // The authored proxy includes bolts/upper trim and is
                        // several centimetres taller than the visible worktop.
                        // Preserve its sides but make top support coincide
                        // with the metal desktop so lightweight props do not
                        // appear to hover.
                        FCollisionObstacle& TableObstacle = CollisionObstacles.Last();
                        const float Bottom = TableObstacle.Center.Z - TableObstacle.HalfExtent.Z;
                        const float VisibleTop = PackingTableDesktopBounds.Max.Z;
                        if (VisibleTop > Bottom)
                        {
                            TableObstacle.Center.Z = 0.5f * (Bottom + VisibleTop);
                            TableObstacle.HalfExtent.Z = 0.5f * (VisibleTop - Bottom);
                        }
                        SimulatorLog(FString::Printf(
                            TEXT("packing_table_support visible_top_z_cm=%.2f collider_top_z_cm=%.2f"),
                            VisibleTop,
                            TableObstacle.Center.Z + TableObstacle.HalfExtent.Z));
                    }
                }
                else if (LogicalName.StartsWith(TEXT("NorthWall")) || LogicalName == TEXT("WestWall"))
                {
                    AddObstacle(TEXT("factory wall"), Primitive->Bounds);
                }
                else if (LogicalName == TEXT("Pole"))
                {
                    // These visible signal poles were not part of the original
                    // software guard even though a vehicle can reach them.
                    AddObstacle(TEXT("stack light"), Primitive->Bounds);
                }
            }
        }
    }

    if (ConveyorBounds.IsValid)
    {
        // Follow the analytic belt centreline with tangent-aligned boxes. This
        // is close to the retained roller mesh, rotates correctly through both
        // curves, and remains far cheaper than per-triangle Chaos collision.
        constexpr int32 ConveyorSegments = 40;
        constexpr float StraightHalf = 235.1374f;
        constexpr float Radius = 150.0f;
        constexpr float Perimeter = 4.0f * StraightHalf + 2.0f * PI * Radius;
        const FVector Extent3 = ConveyorBounds.GetExtent();
        const float TrackHalfWidth = FMath::Min(55.0f, Extent3.X * 0.25f);
        const float SegmentHalfLength = Perimeter / static_cast<float>(ConveyorSegments) * 0.58f;
        for (int32 SegmentIndex = 0; SegmentIndex < ConveyorSegments; ++SegmentIndex)
        {
            FVector Position;
            FVector Tangent;
            ConveyorTuning::EvaluateConveyor(
                (static_cast<float>(SegmentIndex) + 0.5f) * Perimeter / static_cast<float>(ConveyorSegments),
                Position,
                Tangent);
            FCollisionObstacle& Obstacle = CollisionObstacles.AddDefaulted_GetRef();
            Obstacle.Name = TEXT("conveyor");
            // Keep the coarse guard under the roller crown. Module bounds
            // include raised rails; using their full height made loose cargo
            // and shelf cartons visibly hover above the rollers.
            constexpr float SupportDepthCm = 18.0f;
            Obstacle.Center = FVector(
                Position.X,
                Position.Y,
                ConveyorSurfaceZCm - SupportDepthCm * 0.5f);
            Obstacle.HalfExtent = FVector(SegmentHalfLength, TrackHalfWidth, SupportDepthCm * 0.5f);
            Obstacle.Rotation = FRotator(0.0f, Tangent.Rotation().Yaw, 0.0f).Quaternion();
        }
    }

    CollisionStatus = FString::Printf(TEXT("ready | %d collision volumes"), CollisionObstacles.Num());
    UE_LOG(
        LogTemp,
        Display,
        TEXT("Qai collision guard ready: %d authored components, %d effective volumes"),
        AuthoredCollisionComponents.Num(),
        CollisionObstacles.Num());
    SimulatorLog(FString::Printf(
        TEXT("collision_ready authored=%d proxies=%d"),
        AuthoredCollisionComponents.Num(),
        CollisionObstacles.Num()));
    return AuthoredCollisionComponents.Num();
}

bool AQaiConveyorWorld::WouldForkliftCollide(
    int32 MovingIndex,
    const FVector& CandidateLocation,
    const FRotator& CandidateRotation,
    FString& OutObstacle) const
{
    OutObstacle.Reset();
    const FForkliftRuntime& MovingForklift = Forklifts[MovingIndex];
    const FQuat CandidateQuat = CandidateRotation.Quaternion();
    const int32 ShapeCount = MovingForklift.CollisionBoxes.Num();
    for (int32 ShapeIndex = 0; ShapeIndex < FMath::Max(1, ShapeCount); ++ShapeIndex)
    {
        FFittedCollisionBox Fallback;
        Fallback.Name = TEXT("vehicle body");
        Fallback.LocalCenter = FVector(0.0f, 0.0f, 75.0f);
        Fallback.HalfExtent = FVector(
            ConveyorTuning::ForkliftHalfLength,
            ConveyorTuning::ForkliftHalfWidth,
            75.0f);
        const FFittedCollisionBox& Shape = ShapeCount > 0
            ? MovingForklift.CollisionBoxes[ShapeIndex]
            : Fallback;
        FVector LocalCenter = Shape.LocalCenter;
        LocalCenter.Z += Shape.bLiftDriven ? MovingForklift.LiftCm : 0.0f;
        const FVector ShapeCenter = CandidateLocation + CandidateQuat.RotateVector(LocalCenter);
        const FVector Forward = CandidateQuat.GetForwardVector();
        const FVector Right = CandidateQuat.GetRightVector();
        const FVector CollisionExtent = Shape.GetCollisionHalfExtent();
        const float WorldHalfX = FMath::Abs(Forward.X) * CollisionExtent.X
            + FMath::Abs(Right.X) * CollisionExtent.Y;
        const float WorldHalfY = FMath::Abs(Forward.Y) * CollisionExtent.X
            + FMath::Abs(Right.Y) * CollisionExtent.Y;
        if (ShapeCenter.X - WorldHalfX < ConveyorTuning::WorldMinimumX
            || ShapeCenter.X + WorldHalfX > ConveyorTuning::WorldMaximumX
            || ShapeCenter.Y - WorldHalfY < ConveyorTuning::WorldMinimumY
            || ShapeCenter.Y + WorldHalfY > ConveyorTuning::WorldMaximumY)
        {
            OutObstacle = TEXT("factory boundary");
            return true;
        }

        for (const FCollisionObstacle& Obstacle : CollisionObstacles)
        {
            // Rack cartons need a real gap for the horizontal tines. The
            // shelf support plane still supports every dynamic carton, while
            // the mast, heel, chassis and wheels continue to collide with the
            // rack. Treating the full shelf deck as a blocker for the thin
            // tine made it impossible to enter the pickup clearance.
            if (Obstacle.Name == TEXT("shelf plane")
                && Shape.Name.Contains(TEXT("tine"), ESearchCase::IgnoreCase))
            {
                continue;
            }
            // The lightweight packing table is represented by one solid box,
            // but the visible desktop/feet leave access from its front edge.
            // Let only the thin tines enter that pickup clearance so they can
            // nudge or lift the movable tabletop EVK; body, mast, heels and
            // wheels still collide with the table.
            if (Obstacle.Name == TEXT("packing table")
                && Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase))
            {
                continue;
            }
            const bool bOverlaps = ConveyorTuning::ObbOverlapsObb(
                ShapeCenter,
                CandidateQuat,
                CollisionExtent,
                Obstacle.Center,
                Obstacle.Rotation,
                Obstacle.HalfExtent);
            if (bOverlaps)
            {
                OutObstacle = Obstacle.Name;
                return true;
            }
        }

        for (int32 WorkerIndex = 0; WorkerIndex < UE_ARRAY_COUNT(Workers); ++WorkerIndex)
        {
            if (const USceneComponent* Worker = Workers[WorkerIndex].Root.Get())
            {
                const FVector WorkerLocation = Worker->GetComponentLocation();
                const bool bOverlaps = ConveyorTuning::CircleOverlapsBox2D(
                    FVector2D(WorkerLocation.X, WorkerLocation.Y),
                    ConveyorTuning::WorkerRadiusCm,
                    ShapeCenter,
                    CandidateQuat,
                    CollisionExtent);
                if (bOverlaps)
                {
                    OutObstacle = FString::Printf(TEXT("worker %d"), WorkerIndex + 1);
                    return true;
                }
            }
        }
    }

    if (ConveyorTuning::EnabledForkliftCount > 1
        && WouldForkliftsOverlap(MovingIndex, CandidateLocation, CandidateRotation))
    {
        OutObstacle = FString::Printf(TEXT("forklift %d"), MovingIndex == 0 ? 2 : 1);
        return true;
    }

    return false;
}

bool AQaiConveyorWorld::CanWorkerOccupy(int32 MovingIndex, const FVector& CandidateLocation) const
{
    const FVector2D Center(CandidateLocation.X, CandidateLocation.Y);
    constexpr float Radius = ConveyorTuning::WorkerRadiusCm;
    if (Center.X - Radius < ConveyorTuning::WorldMinimumX
        || Center.X + Radius > ConveyorTuning::WorldMaximumX
        || Center.Y - Radius < ConveyorTuning::WorldMinimumY
        || Center.Y + Radius > ConveyorTuning::WorldMaximumY)
    {
        return false;
    }
    for (const FCollisionObstacle& Obstacle : CollisionObstacles)
    {
        if (ConveyorTuning::CircleOverlapsBox2D(
                Center,
                Radius,
                Obstacle.Center,
                Obstacle.Rotation,
                Obstacle.HalfExtent,
                2.0f))
        {
            return false;
        }
    }
    for (int32 ForkliftIndex = 0; ForkliftIndex < ConveyorTuning::EnabledForkliftCount; ++ForkliftIndex)
    {
        const FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        if (const USceneComponent* Root = Forklift.Root.Get())
        {
            const FQuat RootQuat = Root->GetComponentQuat();
            for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
            {
                FVector LocalCenter = Shape.LocalCenter;
                LocalCenter.Z += Shape.bLiftDriven ? Forklift.LiftCm : 0.0f;
                const FVector ShapeCenter = Root->GetComponentLocation() + RootQuat.RotateVector(LocalCenter);
                const bool bOverlaps = ConveyorTuning::CircleOverlapsBox2D(
                    Center,
                    Radius,
                    ShapeCenter,
                    RootQuat,
                    Shape.GetCollisionHalfExtent(),
                    2.0f);
                if (bOverlaps)
                {
                    return false;
                }
            }
        }
    }
    for (int32 WorkerIndex = 0; WorkerIndex < UE_ARRAY_COUNT(Workers); ++WorkerIndex)
    {
        if (WorkerIndex == MovingIndex)
        {
            continue;
        }
        if (const USceneComponent* OtherWorker = Workers[WorkerIndex].Root.Get())
        {
            const FVector Other = OtherWorker->GetComponentLocation();
            if (FVector2D::DistSquared(Center, FVector2D(Other.X, Other.Y)) < FMath::Square(2.0f * Radius))
            {
                return false;
            }
        }
    }
    return true;
}

float AQaiConveyorWorld::GetWorkerStaticClearanceCm(const FVector& CandidateLocation) const
{
    const FVector2D Center(CandidateLocation.X, CandidateLocation.Y);
    constexpr float Radius = ConveyorTuning::WorkerRadiusCm;
    float MinimumClearance = FMath::Min(
        FMath::Min(
            Center.X - ConveyorTuning::WorldMinimumX,
            ConveyorTuning::WorldMaximumX - Center.X),
        FMath::Min(
            Center.Y - ConveyorTuning::WorldMinimumY,
            ConveyorTuning::WorldMaximumY - Center.Y)) - Radius;
    for (const FCollisionObstacle& Obstacle : CollisionObstacles)
    {
        const FVector2D ExpandedExtent(Obstacle.HalfExtent.X + 2.0f, Obstacle.HalfExtent.Y + 2.0f);
        const FVector Forward3 = Obstacle.Rotation.GetForwardVector();
        const FVector Right3 = Obstacle.Rotation.GetRightVector();
        const FVector2D Relative = Center - FVector2D(Obstacle.Center.X, Obstacle.Center.Y);
        const FVector2D Q(
            FMath::Abs(FVector2D::DotProduct(Relative, FVector2D(Forward3.X, Forward3.Y))) - ExpandedExtent.X,
            FMath::Abs(FVector2D::DotProduct(Relative, FVector2D(Right3.X, Right3.Y))) - ExpandedExtent.Y);
        const float OutsideDistance = FVector2D(FMath::Max(Q.X, 0.0f), FMath::Max(Q.Y, 0.0f)).Size();
        const float InsideDistance = FMath::Min(FMath::Max(Q.X, Q.Y), 0.0f);
        MinimumClearance = FMath::Min(MinimumClearance, OutsideDistance + InsideDistance - Radius);
    }
    return MinimumClearance;
}

bool AQaiConveyorWorld::ValidateCollisionGuard()
{
    const FCollisionObstacle* PackingTable = CollisionObstacles.FindByPredicate([](const FCollisionObstacle& Obstacle)
    {
        return Obstacle.Name == TEXT("packing table");
    });
    const bool bStatic = PackingTable && Forklifts[0].Root.IsValid()
        && ConveyorTuning::ObbOverlapsObb(
            PackingTable->Center,
            FQuat::Identity,
            FVector(40.0f, 40.0f, 40.0f),
            PackingTable->Center,
            PackingTable->Rotation,
            PackingTable->HalfExtent);
    const bool bWorker = Workers[0].Root.IsValid() && Forklifts[0].Root.IsValid()
        && ConveyorTuning::CircleOverlapsObb(
            FVector2D(Workers[0].Root->GetComponentLocation().X, Workers[0].Root->GetComponentLocation().Y),
            ConveyorTuning::WorkerRadiusCm,
            Workers[0].Root->GetComponentLocation(),
            Forklifts[0].Root->GetComponentRotation());
    const bool bForklift = Forklifts[1].Root.IsValid() && Forklifts[0].Root.IsValid()
        && WouldForkliftsOverlap(
            0,
            Forklifts[1].Root->GetComponentLocation(),
            Forklifts[1].Root->GetComponentRotation());
    const FCollisionObstacle* Conveyor = CollisionObstacles.FindByPredicate([](const FCollisionObstacle& Obstacle)
    {
        return Obstacle.Name == TEXT("conveyor");
    });
    const FFittedCollisionBox* LowerChassis = Forklifts[0].CollisionBoxes.FindByPredicate(
        [](const FFittedCollisionBox& Shape)
        {
            return Shape.Name == TEXT("lower chassis");
        });
    bool bConveyor = false;
    if (Conveyor && LowerChassis && Forklifts[0].Root.IsValid())
    {
        const FRotator TestRotation = Conveyor->Rotation.Rotator();
        FVector TestLocation = Forklifts[0].InitialRoot.GetLocation();
        const FVector RotatedChassisCenter = TestRotation.Quaternion().RotateVector(
            LowerChassis->LocalCenter);
        TestLocation.X = Conveyor->Center.X - RotatedChassisCenter.X;
        TestLocation.Y = Conveyor->Center.Y - RotatedChassisCenter.Y;
        FString TestObstacle;
        bConveyor = WouldForkliftCollide(0, TestLocation, TestRotation, TestObstacle)
            && TestObstacle == TEXT("conveyor");
    }
    const bool bComplete = AuthoredCollisionComponents.Num() >= 16 && CollisionObstacles.Num() >= 32;
    SimulatorLog(FString::Printf(
        TEXT("collision_self_test static=%s conveyor=%s worker=%s forklift=%s complete=%s authored=%d proxies=%d"),
        bStatic ? TEXT("true") : TEXT("false"),
        bConveyor ? TEXT("true") : TEXT("false"),
        bWorker ? TEXT("true") : TEXT("false"),
        bForklift ? TEXT("true") : TEXT("false"),
        bComplete ? TEXT("true") : TEXT("false"),
        AuthoredCollisionComponents.Num(),
        CollisionObstacles.Num()));
    return bStatic && bConveyor && bWorker && bForklift && bComplete;
}

void AQaiConveyorWorld::UpdateCollisionStatus(int32 ForkliftIndex, const FString& Obstacle)
{
    if (!Obstacle.IsEmpty())
    {
        CollisionStatus = FString::Printf(
            TEXT("%s: %s"),
            FMath::Abs(Forklifts[ForkliftIndex].SpeedCmPerSecond) > 2.0f
                ? TEXT("BRAKING")
                : TEXT("PROTECTIVE STOP"),
            *Obstacle);
        if (LastCollisionObstacle != Obstacle)
        {
            LastCollisionObstacle = Obstacle;
            ++CollisionBlockCount;
            SimulatorLog(FString::Printf(
                TEXT("collision_block forklift=%d obstacle=%s count=%d"),
                ForkliftIndex + 1,
                *Obstacle,
                CollisionBlockCount));
        }
        return;
    }

    if (!LastCollisionObstacle.IsEmpty())
    {
        SimulatorLog(FString::Printf(
            TEXT("collision_clear forklift=%d obstacle=%s"),
            ForkliftIndex + 1,
            *LastCollisionObstacle));
        LastCollisionObstacle.Reset();
    }
    CollisionStatus = FString::Printf(TEXT("ready | %d collision volumes"), CollisionObstacles.Num());
}

void AQaiConveyorWorld::RecordControllerInput(const FString& Details)
{
    SimulatorLog(Details);
}

void AQaiConveyorWorld::LogStageBindingFailure(const FString& Details)
{
    const double Now = FPlatformTime::Seconds();
    if (Details == LastLoggedStageFailure && Now - LastStageFailureLogSeconds < 60.0)
    {
        return;
    }
    LastLoggedStageFailure = Details;
    LastStageFailureLogSeconds = Now;
    UE_LOG(LogTemp, Error, TEXT("Qai stage binding failed: %s"), *Details);
    SimulatorLog(TEXT("stage_binding_failed ") + Details);
}

void AQaiConveyorWorld::BindNativeComponents()
{
    DynamicBoxes.Reset();
    Forklifts[0].Root = FindTaggedComponent(TEXT("Qai.Forklift1"));
    Forklifts[1].Root = FindTaggedComponent(TEXT("Qai.Forklift2"));
    if (!Forklifts[0].Root.IsValid() || !Forklifts[1].Root.IsValid())
    {
        StageStatus = TEXT("Binding optimized native warehouse assets");
        StageError = TEXT("Required tagged forklift roots are missing from WarehouseConveyor.");
        LogStageBindingFailure(TEXT("reason=missing_forklift_roots"));
        return;
    }
    if (!ApplyWestShelfPlacement())
    {
        StageStatus = TEXT("Binding optimized native warehouse assets");
        StageError = TEXT("West rack or west wall placement component is not ready.");
        LogStageBindingFailure(TEXT("reason=missing_west_shelf_placement_components"));
        return;
    }
    ConfigureForkliftMastMaterials();

    Forklifts[0].Pallet = FindTaggedComponent(TEXT("Qai.Forklift1Pallet"));
    Forklifts[0].Carton = FindTaggedComponent(TEXT("Qai.Forklift1Carton"));
    Forklifts[1].Pallet = FindTaggedComponent(TEXT("Qai.Forklift2Pallet"));
    Forklifts[1].Carton = FindTaggedComponent(TEXT("Qai.Forklift2Carton"));
    const TCHAR* WheelRoles[] = {TEXT("LeftFront"), TEXT("RightFront"), TEXT("LeftRear"), TEXT("RightRear")};
    int32 BoundWheels = 0;
    for (int32 ForkliftIndex = 0; ForkliftIndex < 2; ++ForkliftIndex)
    {
        FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        MakeMovable(Forklift.Root.Get());
        Forklift.LiftAssembly.Reset();
        Forklift.LiftDrivenComponents.Reset();
        Forklift.LiftDrivenRelativeTransforms.Reset();
        TArray<USceneComponent*> RootChildren;
        Forklift.Root->GetChildrenComponents(false, RootChildren);
        for (USceneComponent* Child : RootChildren)
        {
            FString LogicalName = Child ? Child->GetName() : FString();
            int32 LastUnderscore = INDEX_NONE;
            if (LogicalName.FindLastChar(TEXT('_'), LastUnderscore)
                && LogicalName.Mid(LastUnderscore + 1).IsNumeric())
            {
                LogicalName.LeftInline(LastUnderscore, EAllowShrinking::No);
            }
            if (LogicalName.Equals(TEXT("lift"), ESearchCase::IgnoreCase))
            {
                Forklift.LiftAssembly = Child;
                MakeMovable(Child);
                Forklift.InitialLiftRelativeLocation = Child->GetRelativeLocation();
                break;
            }
        }
        Forklift.LiftVisual.Reset();
        if (USceneComponent* LiftAssembly = Forklift.LiftAssembly.Get())
        {
            TInlineComponentArray<UPrimitiveComponent*> LiftPrimitives(LiftAssembly->GetOwner());
            for (UPrimitiveComponent* Primitive : LiftPrimitives)
            {
                if (!Primitive || (Primitive != LiftAssembly && !Primitive->IsAttachedTo(LiftAssembly)))
                {
                    continue;
                }
                const FString Name = Primitive->GetName();
                if (Name.Contains(TEXT("Fork01"), ESearchCase::IgnoreCase)
                    || Name.Contains(TEXT("Fork02"), ESearchCase::IgnoreCase))
                {
                    Forklift.LiftVisual = Primitive;
                    Forklift.InitialLiftVisualCenterZ = Primitive->Bounds.Origin.Z;
                    break;
                }
            }
        }
        MakeMovable(Forklift.Pallet.Get());
        MakeMovable(Forklift.Carton.Get());
        float LowestWheelZ = TNumericLimits<float>::Max();
        for (int32 WheelIndex = 0; WheelIndex < UE_ARRAY_COUNT(WheelRoles); ++WheelIndex)
        {
            Forklift.Wheels[WheelIndex] = FindTaggedComponent(FName(*FString::Printf(
                TEXT("Qai.Forklift%d.Wheel.%s"), ForkliftIndex + 1, WheelRoles[WheelIndex])));
            if (Forklift.Wheels[WheelIndex].IsValid())
            {
                USceneComponent* Wheel = Forklift.Wheels[WheelIndex].Get();
                MakeMovable(Wheel);

                // The USD wheel transform is authored at a forklift-wide
                // assembly origin. Rotating it directly therefore swings the
                // wheel around the vehicle. Insert a vehicle-aligned pivot at
                // the rendered wheel's own bounds centre and preserve the
                // complete imported mesh subtree below it.
                FBox WheelBounds(ForceInit);
                TInlineComponentArray<UPrimitiveComponent*> PrimitiveComponents(Wheel->GetOwner());
                for (UPrimitiveComponent* Primitive : PrimitiveComponents)
                {
                    if (Primitive && (Primitive == Wheel || Primitive->IsAttachedTo(Wheel)))
                    {
                        WheelBounds += Primitive->Bounds.GetBox();
                    }
                }
                if (WheelBounds.IsValid && Wheel->GetAttachParent())
                {
                    LowestWheelZ = FMath::Min(LowestWheelZ, WheelBounds.Min.Z);
                    const FName PivotName(*FString::Printf(
                        TEXT("QaiWheelPivot_F%d_W%d"), ForkliftIndex + 1, WheelIndex + 1));
                    USceneComponent* Pivot = NewObject<USceneComponent>(Wheel->GetOwner(), PivotName);
                    Wheel->GetOwner()->AddInstanceComponent(Pivot);
                    Pivot->SetMobility(EComponentMobility::Movable);
                    Pivot->RegisterComponentWithWorld(GetWorld());
                    Pivot->AttachToComponent(Wheel->GetAttachParent(), FAttachmentTransformRules::KeepWorldTransform);
                    Pivot->SetWorldLocationAndRotation(WheelBounds.GetCenter(), Forklift.Root->GetComponentQuat());
                    Pivot->SetWorldScale3D(FVector::OneVector);
                    Wheel->AttachToComponent(Pivot, FAttachmentTransformRules::KeepWorldTransform);
                    Forklift.WheelPivots[WheelIndex] = Pivot;
                    Forklift.WheelPivotInitialRelative[WheelIndex] = Pivot->GetRelativeRotation().Quaternion();
                }
                ++BoundWheels;
            }
        }
        if (LowestWheelZ < TNumericLimits<float>::Max())
        {
            // The USD root is authored 10 cm above the floor. Ground from the
            // actual rendered tire bounds so both vehicle variants sit on the
            // concrete without embedding or hovering.
            Forklift.GroundingOffsetCm = ConveyorTuning::ForkliftTireClearanceCm - LowestWheelZ;
            Forklift.Root->AddWorldOffset(
                FVector(0.0f, 0.0f, Forklift.GroundingOffsetCm),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
        }
        if (UPrimitiveComponent* LiftVisual = Forklift.LiftVisual.Get())
        {
            LiftVisual->UpdateBounds();
            Forklift.InitialLiftVisualCenterZ = LiftVisual->Bounds.Origin.Z;
        }
        Forklift.InitialRoot = Forklift.Root->GetComponentTransform();
        Forklift.CollisionBoxes.Reset();

        // Fit a small compound collider to the retained render hierarchy.
        // Grouping the chassis/cab and mast keeps the hot 120 Hz test cheap;
        // wheels and both fork tines remain independent so empty space under
        // and between the forks is no longer treated as solid vehicle body.
        FBox BodyBounds(ForceInit);
        FBox LiftCoreBounds(ForceInit);
        FBox ForkBounds[2] = {FBox(ForceInit), FBox(ForceInit)};
        FBox WheelBounds[4] = {
            FBox(ForceInit), FBox(ForceInit), FBox(ForceInit), FBox(ForceInit)};
        TInlineComponentArray<UPrimitiveComponent*> ForkliftPrimitives(Forklift.Root->GetOwner());
        for (UPrimitiveComponent* Primitive : ForkliftPrimitives)
        {
            if (!Primitive || (Primitive != Forklift.Root && !Primitive->IsAttachedTo(Forklift.Root.Get())))
            {
                continue;
            }
            Primitive->UpdateBounds();
            const auto IsUnder = [Primitive](const USceneComponent* Parent)
            {
                return Parent && (Primitive == Parent || Primitive->IsAttachedTo(Parent));
            };
            if (IsUnder(Forklift.Pallet.Get()) || IsUnder(Forklift.Carton.Get()))
            {
                continue;
            }
            bool bWheel = false;
            for (int32 WheelIndex = 0; WheelIndex < 4; ++WheelIndex)
            {
                if (IsUnder(Forklift.Wheels[WheelIndex].Get()))
                {
                    WheelBounds[WheelIndex] += Primitive->Bounds.GetBox();
                    bWheel = true;
                    break;
                }
            }
            if (bWheel)
            {
                continue;
            }
            if (IsUnder(Forklift.LiftAssembly.Get()))
            {
                const FString PrimitiveName = Primitive->GetName();
                if (PrimitiveName.Contains(TEXT("Fork01"), ESearchCase::IgnoreCase))
                {
                    ForkBounds[0] += Primitive->Bounds.GetBox();
                }
                else if (PrimitiveName.Contains(TEXT("Fork02"), ESearchCase::IgnoreCase))
                {
                    ForkBounds[1] += Primitive->Bounds.GetBox();
                }
                else
                {
                    LiftCoreBounds += Primitive->Bounds.GetBox();
                }
                continue;
            }
            BodyBounds += Primitive->Bounds.GetBox();
        }

        const auto ToLocalBounds = [&Forklift](const FBox& WorldBounds)
        {
            FBox LocalBounds(ForceInit);
            if (!WorldBounds.IsValid)
            {
                return LocalBounds;
            }
            const FVector Minimum = WorldBounds.Min;
            const FVector Maximum = WorldBounds.Max;
            for (int32 Corner = 0; Corner < 8; ++Corner)
            {
                const FVector WorldCorner(
                    (Corner & 1) != 0 ? Maximum.X : Minimum.X,
                    (Corner & 2) != 0 ? Maximum.Y : Minimum.Y,
                    (Corner & 4) != 0 ? Maximum.Z : Minimum.Z);
                // USD assembly roots carry a 0.01 metres-to-centimetres scale.
                // Bounds are already in world centimetres, so removing scale
                // here would inflate every fitted collider by 100x.
                LocalBounds += Forklift.InitialRoot.InverseTransformPositionNoScale(WorldCorner);
            }
            return LocalBounds;
        };
        const auto AddLocalFittedBox = [&Forklift, ForkliftIndex](
            const FString& Name, const FBox& LocalBounds, bool bLiftDriven)
        {
            if (!LocalBounds.IsValid)
            {
                return;
            }
            FFittedCollisionBox& Shape = Forklift.CollisionBoxes.AddDefaulted_GetRef();
            Shape.Name = Name;
            Shape.LocalCenter = LocalBounds.GetCenter();
            Shape.HalfExtent = LocalBounds.GetExtent().ComponentMax(FVector(2.5f));
            Shape.Shape = FFittedCollisionBox::EShape::Box;
            Shape.Radius = 0.0f;
            Shape.CylinderHalfLength = 0.0f;
            Shape.bLiftDriven = bLiftDriven;
            SimulatorLog(FString::Printf(
                TEXT("forklift_collision_shape forklift=%d type=box name=%s center=(%.1f,%.1f,%.1f) extent=(%.1f,%.1f,%.1f) lift_driven=%s"),
                ForkliftIndex + 1,
                *Name,
                Shape.LocalCenter.X,
                Shape.LocalCenter.Y,
                Shape.LocalCenter.Z,
                Shape.HalfExtent.X,
                Shape.HalfExtent.Y,
                Shape.HalfExtent.Z,
                bLiftDriven ? TEXT("true") : TEXT("false")));
        };
        const auto AddLocalFittedCylinder = [&Forklift, ForkliftIndex](
            const FString& Name, const FBox& LocalBounds)
        {
            if (!LocalBounds.IsValid)
            {
                return;
            }
            const FVector SourceExtent = LocalBounds.GetExtent();
            // Tire axles run along forklift-local Y. Preserve the actual
            // rendered width along that axle and fit a circular X/Z profile;
            // this is a finite wheel disk, not a sphere that can collide far
            // outside the sidewall.
            const float Radius = FMath::Clamp(
                FMath::Min(SourceExtent.X, SourceExtent.Z) * 0.92f,
                18.0f,
                ConveyorTuning::ForkliftWheelRadiusCm);
            const float HalfLength = FMath::Clamp(SourceExtent.Y * 0.96f, 7.0f, 22.0f);
            FFittedCollisionBox& Shape = Forklift.CollisionBoxes.AddDefaulted_GetRef();
            Shape.Name = Name;
            Shape.LocalCenter = LocalBounds.GetCenter();
            Shape.HalfExtent = FVector(Radius, HalfLength, Radius);
            Shape.Radius = Radius;
            Shape.CylinderHalfLength = HalfLength;
            Shape.Shape = FFittedCollisionBox::EShape::Cylinder;
            Shape.bLiftDriven = false;
            SimulatorLog(FString::Printf(
                TEXT("forklift_collision_shape forklift=%d type=cylinder name=%s center=(%.1f,%.1f,%.1f) radius=%.1f half_length=%.1f axis=local_y source_extent=(%.1f,%.1f,%.1f)"),
                ForkliftIndex + 1,
                *Name,
                Shape.LocalCenter.X,
                Shape.LocalCenter.Y,
                Shape.LocalCenter.Z,
                Radius,
                HalfLength,
                SourceExtent.X,
                SourceExtent.Y,
                SourceExtent.Z));
        };
        const auto AddLocalFittedWedge = [&Forklift, ForkliftIndex](
            const FString& Name,
            const FBox& LocalBounds,
            bool bLiftDriven,
            bool bTipAtPositiveX,
            float TipThicknessCm)
        {
            if (!LocalBounds.IsValid)
            {
                return;
            }
            FFittedCollisionBox& Shape = Forklift.CollisionBoxes.AddDefaulted_GetRef();
            Shape.Name = Name;
            Shape.LocalCenter = LocalBounds.GetCenter();
            Shape.HalfExtent = LocalBounds.GetExtent().ComponentMax(FVector(1.0f, 1.0f, 0.5f));
            Shape.Shape = FFittedCollisionBox::EShape::Wedge;
            Shape.Radius = 0.0f;
            Shape.CylinderHalfLength = 0.0f;
            Shape.WedgeTipThicknessCm = FMath::Clamp(
                TipThicknessCm,
                0.2f,
                Shape.HalfExtent.Z * 1.5f);
            Shape.bLiftDriven = bLiftDriven;
            Shape.bWedgeTipAtPositiveX = bTipAtPositiveX;
            SimulatorLog(FString::Printf(
                TEXT("forklift_collision_shape forklift=%d type=wedge name=%s center=(%.1f,%.1f,%.1f) extent=(%.1f,%.1f,%.1f) tip_thickness_cm=%.2f tip_direction=%s lift_driven=%s"),
                ForkliftIndex + 1,
                *Name,
                Shape.LocalCenter.X,
                Shape.LocalCenter.Y,
                Shape.LocalCenter.Z,
                Shape.HalfExtent.X,
                Shape.HalfExtent.Y,
                Shape.HalfExtent.Z,
                Shape.WedgeTipThicknessCm,
                bTipAtPositiveX ? TEXT("positive_x") : TEXT("negative_x"),
                bLiftDriven ? TEXT("true") : TEXT("false")));
        };
        const auto AddFittedBox = [&ToLocalBounds, &AddLocalFittedBox](
            const FString& Name, const FBox& WorldBounds, bool bLiftDriven)
        {
            AddLocalFittedBox(Name, ToLocalBounds(WorldBounds), bLiftDriven);
        };
        const FBox BodyLocal = ToLocalBounds(BodyBounds);
        int32 HullBoxes = 0;
        if (BodyLocal.IsValid)
        {
            const FVector Minimum = BodyLocal.Min;
            const FVector Maximum = BodyLocal.Max;
            const FVector Center = BodyLocal.GetCenter();
            const FVector Size = BodyLocal.GetSize();
            const auto XAt = [&Minimum, &Size](float Alpha)
            {
                return Minimum.X + Size.X * Alpha;
            };
            const auto ZAt = [&Minimum, &Size](float Alpha)
            {
                return Minimum.Z + Size.Z * Alpha;
            };
            const float InnerHalfWidth = Size.Y * 0.44f;
            const float HoodHalfWidth = Size.Y * 0.46f;
            const float PostHalf = FMath::Clamp(Size.Y * 0.038f, 3.5f, 5.5f);
            const float PostY = Size.Y * 0.5f - PostHalf;
            const float PostBottom = ZAt(0.53f);
            const float PostTop = ZAt(0.90f);

            // Follow the visual profile instead of filling its entire AABB:
            // low counterweight/chassis/hood, an open operator bay, then a
            // thin roof carried by four narrow guard posts.
            AddLocalFittedBox(
                TEXT("rear counterweight"),
                FBox(
                    FVector(Minimum.X, Minimum.Y, ZAt(0.13f)),
                    FVector(XAt(0.45f), Maximum.Y, ZAt(0.49f))),
                false);
            AddLocalFittedBox(
                TEXT("lower chassis"),
                FBox(
                    FVector(XAt(0.29f), Center.Y - InnerHalfWidth, ZAt(0.10f)),
                    FVector(Maximum.X, Center.Y + InnerHalfWidth, ZAt(0.43f))),
                false);
            AddLocalFittedBox(
                TEXT("engine hood"),
                FBox(
                    FVector(XAt(0.05f), Center.Y - HoodHalfWidth, ZAt(0.36f)),
                    FVector(XAt(0.58f), Center.Y + HoodHalfWidth, ZAt(0.57f))),
                false);
            AddLocalFittedBox(
                TEXT("overhead guard roof"),
                FBox(
                    FVector(XAt(0.23f), Minimum.Y, ZAt(0.88f)),
                    FVector(XAt(0.93f), Maximum.Y, Maximum.Z)),
                false);
            for (int32 Longitudinal = 0; Longitudinal < 2; ++Longitudinal)
            {
                const float PostX = XAt(Longitudinal == 0 ? 0.27f : 0.87f);
                for (int32 Side = 0; Side < 2; ++Side)
                {
                    const float SideY = Center.Y + (Side == 0 ? -PostY : PostY);
                    AddLocalFittedBox(
                        FString::Printf(
                            TEXT("%s %s guard post"),
                            Longitudinal == 0 ? TEXT("rear") : TEXT("front"),
                            Side == 0 ? TEXT("left") : TEXT("right")),
                        FBox(
                            FVector(PostX - PostHalf, SideY - PostHalf, PostBottom),
                            FVector(PostX + PostHalf, SideY + PostHalf, PostTop)),
                        false);
                }
            }
            HullBoxes = 8;
        }
        AddFittedBox(TEXT("mast"), LiftCoreBounds, true);
        const FBox LiftCoreLocal = ToLocalBounds(LiftCoreBounds);
        for (int32 ForkIndex = 0; ForkIndex < 2; ++ForkIndex)
        {
            const FBox ForkLocal = ToLocalBounds(ForkBounds[ForkIndex]);
            if (!ForkLocal.IsValid)
            {
                continue;
            }
            const FString Side = ForkIndex == 0 ? TEXT("left") : TEXT("right");
            const FVector Size = ForkLocal.GetSize();
            const float TineThickness = FMath::Clamp(Size.Z * 0.16f, 3.0f, 7.0f);
            const float HeelDepth = FMath::Clamp(Size.X * 0.14f, 6.0f, 15.0f);
            const float MastX = LiftCoreLocal.IsValid ? LiftCoreLocal.GetCenter().X : ForkLocal.Min.X;
            const bool bHeelAtMinimumX = FMath::Abs(ForkLocal.Min.X - MastX)
                <= FMath::Abs(ForkLocal.Max.X - MastX);
            const bool bTipAtPositiveX = bHeelAtMinimumX;
            const float WedgeLength = FMath::Clamp(Size.X * 0.17f, 16.0f, 22.0f);
            FBox TineBounds = ForkLocal;
            TineBounds.Max.Z = TineBounds.Min.Z + TineThickness;
            FBox WedgeBounds = TineBounds;
            if (bTipAtPositiveX)
            {
                WedgeBounds.Min.X = WedgeBounds.Max.X - WedgeLength;
                TineBounds.Max.X = WedgeBounds.Min.X;
            }
            else
            {
                WedgeBounds.Max.X = WedgeBounds.Min.X + WedgeLength;
                TineBounds.Min.X = WedgeBounds.Max.X;
            }
            AddLocalFittedBox(Side + TEXT(" fork tine"), TineBounds, true);
            AddLocalFittedWedge(
                Side + TEXT(" fork tine tip wedge"),
                WedgeBounds,
                true,
                bTipAtPositiveX,
                FMath::Clamp(TineThickness * 0.12f, 0.35f, 0.9f));

            // Fit the short vertical heel independently instead of filling
            // the empty inside of the imported L-shaped fork with one AABB.
            FBox HeelBounds = ForkLocal;
            if (bHeelAtMinimumX)
            {
                HeelBounds.Max.X = HeelBounds.Min.X + HeelDepth;
            }
            else
            {
                HeelBounds.Min.X = HeelBounds.Max.X - HeelDepth;
            }
            AddLocalFittedBox(Side + TEXT(" fork heel"), HeelBounds, true);
        }
        for (int32 WheelIndex = 0; WheelIndex < 4; ++WheelIndex)
        {
            AddLocalFittedCylinder(
                FString::Printf(TEXT("wheel %d"), WheelIndex + 1),
                ToLocalBounds(WheelBounds[WheelIndex]));
        }
        if (BodyLocal.IsValid)
        {
            const FVector BodySize = BodyLocal.GetSize();
            // The rear counterweight keeps the unloaded vehicle's centre of
            // mass low and slightly behind centre. Cargo mass is combined at
            // runtime, including its current lift height.
            Forklift.BaseCenterOfMassLocalCm = FVector(
                FMath::Lerp(BodyLocal.Min.X, BodyLocal.Max.X, 0.40f),
                BodyLocal.GetCenter().Y,
                FMath::Lerp(BodyLocal.Min.Z, BodyLocal.Max.Z, 0.33f));
            Forklift.CombinedCenterOfMassLocalCm = Forklift.BaseCenterOfMassLocalCm;
            Forklift.CombinedMassKg = Forklift.ChassisMassKg;
            Forklift.ChassisInertiaTensorKgCm2 = FVector(
                Forklift.ChassisMassKg
                    * (FMath::Square(BodySize.Y) + FMath::Square(BodySize.Z)) / 12.0f,
                Forklift.ChassisMassKg
                    * (FMath::Square(BodySize.X) + FMath::Square(BodySize.Z)) / 12.0f,
                Forklift.ChassisMassKg
                    * (FMath::Square(BodySize.X) + FMath::Square(BodySize.Y)) / 12.0f);
        }
        SimulatorLog(FString::Printf(
            TEXT("forklift_physics_profile forklift=%d mass_kg=%.0f base_com_cm=(%.1f,%.1f,%.1f) inertia_kg_cm2=(%.0f,%.0f,%.0f) tires=industrial_rubber static_friction=0.90 dynamic_friction=0.72"),
            ForkliftIndex + 1,
            Forklift.ChassisMassKg,
            Forklift.BaseCenterOfMassLocalCm.X,
            Forklift.BaseCenterOfMassLocalCm.Y,
            Forklift.BaseCenterOfMassLocalCm.Z,
            Forklift.ChassisInertiaTensorKgCm2.X,
            Forklift.ChassisInertiaTensorKgCm2.Y,
            Forklift.ChassisInertiaTensorKgCm2.Z));
        SimulatorLog(FString::Printf(
            TEXT("forklift_collision_fit forklift=%d shapes=%d hull_boxes=%d mast=%s forks=%d wheel_cylinders=%d"),
            ForkliftIndex + 1,
            Forklift.CollisionBoxes.Num(),
            HullBoxes,
            LiftCoreBounds.IsValid ? TEXT("true") : TEXT("false"),
            (ForkBounds[0].IsValid ? 1 : 0) + (ForkBounds[1].IsValid ? 1 : 0),
            (WheelBounds[0].IsValid ? 1 : 0) + (WheelBounds[1].IsValid ? 1 : 0)
                + (WheelBounds[2].IsValid ? 1 : 0) + (WheelBounds[3].IsValid ? 1 : 0)));
        if (USceneComponent* LiftAssembly = Forklift.LiftAssembly.Get())
        {
            TArray<USceneComponent*> LiftDescendants;
            LiftAssembly->GetChildrenComponents(true, LiftDescendants);
            for (USceneComponent* Descendant : LiftDescendants)
            {
                if (Descendant)
                {
                    Forklift.LiftDrivenComponents.Add(Descendant);
                    Forklift.LiftDrivenRelativeTransforms.Add(
                        Descendant->GetComponentTransform().GetRelativeTransform(Forklift.InitialRoot));
                }
            }
        }
        if (Forklift.Pallet.IsValid())
        {
            Forklift.PalletRelative = Forklift.Pallet->GetComponentTransform().GetRelativeTransform(Forklift.InitialRoot);
        }
        if (Forklift.Carton.IsValid())
        {
            Forklift.CartonRelative = Forklift.Carton->GetComponentTransform().GetRelativeTransform(Forklift.InitialRoot);
        }
    }

    if (ConveyorTuning::EnabledForkliftCount < UE_ARRAY_COUNT(Forklifts))
    {
        const auto DisableAuthoredComponent = [](USceneComponent* Component)
        {
            if (!Component)
            {
                return;
            }

            // USD imports place most of the warehouse under one actor. Never
            // hide or disable that owner just to remove one authored subtree;
            // doing so also removes forklift 1, racks, conveyors and floors.
            TArray<USceneComponent*> Descendants;
            Component->GetChildrenComponents(true, Descendants);
            TArray<USceneComponent*> Components;
            Components.Reserve(Descendants.Num() + 1);
            Components.Add(Component);
            Components.Append(Descendants);
            Component->SetVisibility(false, true);
            for (USceneComponent* Child : Components)
            {
                if (!Child)
                {
                    continue;
                }
                Child->SetComponentTickEnabled(false);
                if (UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Child))
                {
                    Primitive->SetVisibility(false, false);
                    Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                    Primitive->SetGenerateOverlapEvents(false);
                    Primitive->SetSimulatePhysics(false);
                }
            }
        };
        DisableAuthoredComponent(Forklifts[1].Root.Get());
        DisableAuthoredComponent(Forklifts[1].Pallet.Get());
        DisableAuthoredComponent(Forklifts[1].Carton.Get());
        SimulatorLog(TEXT("forklift_disabled forklift=2 mode=hidden_collisionless_not_selectable"));
    }

    const auto AddDynamicBox = [this](
        USceneComponent* VisualRoot,
        const FString& Name,
        int32 SupportedForklift,
        bool bShelfParcel,
        const FBox* WorldBoundsOverride) -> int32
    {
        if (!VisualRoot || !VisualRoot->GetOwner())
        {
            return INDEX_NONE;
        }
        MakeMovable(VisualRoot);
        FBox WorldBounds = WorldBoundsOverride && WorldBoundsOverride->IsValid
            ? *WorldBoundsOverride
            : FBox(ForceInit);
        if (!WorldBounds.IsValid)
        {
            TInlineComponentArray<UPrimitiveComponent*> Primitives(VisualRoot->GetOwner());
            for (UPrimitiveComponent* Primitive : Primitives)
            {
                if (Primitive && (Primitive == VisualRoot || Primitive->IsAttachedTo(VisualRoot)))
                {
                    Primitive->UpdateBounds();
                    WorldBounds += Primitive->Bounds.GetBox();
                }
            }
        }
        if (!WorldBounds.IsValid)
        {
            return INDEX_NONE;
        }

        USceneComponent* Pivot = NewObject<USceneComponent>(
            VisualRoot->GetOwner(),
            FName(*FString::Printf(TEXT("QaiDynamicPivot_%d"), DynamicBoxes.Num() + 1)));
        VisualRoot->GetOwner()->AddInstanceComponent(Pivot);
        Pivot->SetMobility(EComponentMobility::Movable);
        Pivot->RegisterComponentWithWorld(GetWorld());
        if (USceneComponent* Parent = VisualRoot->GetAttachParent())
        {
            Pivot->AttachToComponent(Parent, FAttachmentTransformRules::KeepWorldTransform);
        }
        else
        {
            VisualRoot->GetOwner()->SetRootComponent(Pivot);
        }
        Pivot->SetWorldLocationAndRotation(WorldBounds.GetCenter(), VisualRoot->GetComponentQuat());
        Pivot->SetWorldScale3D(FVector::OneVector);
        VisualRoot->AttachToComponent(Pivot, FAttachmentTransformRules::KeepWorldTransform);

        FBox LocalBounds(ForceInit);
        const FTransform PivotTransform = Pivot->GetComponentTransform();
        for (int32 Corner = 0; Corner < 8; ++Corner)
        {
            LocalBounds += PivotTransform.InverseTransformPositionNoScale(FVector(
                (Corner & 1) != 0 ? WorldBounds.Max.X : WorldBounds.Min.X,
                (Corner & 2) != 0 ? WorldBounds.Max.Y : WorldBounds.Min.Y,
                (Corner & 4) != 0 ? WorldBounds.Max.Z : WorldBounds.Min.Z));
        }
        FDynamicBoxRuntime& Body = DynamicBoxes.AddDefaulted_GetRef();
        Body.Name = Name;
        Body.Root = Pivot;
        Body.InitialTransform = PivotTransform;
        Body.HalfExtent = LocalBounds.GetExtent().ComponentMax(FVector(2.0f));
        Body.Physics = BuildPropPhysicsProfile(Name, Body.HalfExtent, DynamicBoxes.Num());
        Body.SupportedForklift = SupportedForklift;
        Body.InitialSupportedForklift = SupportedForklift;
        Body.bShelfParcel = bShelfParcel;
        Body.bAwake = false;
        if (SupportedForklift != INDEX_NONE && Forklifts[SupportedForklift].Root.IsValid())
        {
            Body.ForkliftRelativeTransform = PivotTransform.GetRelativeTransform(
                Forklifts[SupportedForklift].Root->GetComponentTransform());
        }
        SimulatorLog(FString::Printf(
            TEXT("dynamic_box_bound name=%s center=(%.1f,%.1f,%.1f) extent=(%.1f,%.1f,%.1f) support_forklift=%d shelf=%s surface=%s mass_kg=%.2f com_cm=(%.2f,%.2f,%.2f) friction=(%.2f,%.2f) restitution=%.3f"),
            *Name,
            PivotTransform.GetLocation().X,
            PivotTransform.GetLocation().Y,
            PivotTransform.GetLocation().Z,
            Body.HalfExtent.X,
            Body.HalfExtent.Y,
            Body.HalfExtent.Z,
            SupportedForklift + 1,
            bShelfParcel ? TEXT("true") : TEXT("false"),
            *Body.Physics.Surface.ToString(),
            Body.Physics.MassKg,
            Body.Physics.CenterOfMassLocalOffset.X,
            Body.Physics.CenterOfMassLocalOffset.Y,
            Body.Physics.CenterOfMassLocalOffset.Z,
            Body.Physics.StaticFriction,
            Body.Physics.DynamicFriction,
            Body.Physics.Restitution));
        return DynamicBoxes.Num() - 1;
    };

    for (int32 ForkliftIndex = 0; ForkliftIndex < ConveyorTuning::EnabledForkliftCount; ++ForkliftIndex)
    {
        FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        Forklift.PalletDynamicBody = AddDynamicBox(
            Forklift.Pallet.Get(),
            FString::Printf(TEXT("forklift %d pallet"), ForkliftIndex + 1),
            ForkliftIndex,
            false,
            nullptr);
        Forklift.CartonDynamicBody = AddDynamicBox(
            Forklift.Carton.Get(),
            FString::Printf(TEXT("forklift %d carton"), ForkliftIndex + 1),
            ForkliftIndex,
            false,
            nullptr);
        if (DynamicBoxes.IsValidIndex(Forklift.PalletDynamicBody))
        {
            Forklift.Pallet = DynamicBoxes[Forklift.PalletDynamicBody].Root;
            Forklift.PalletRelative = DynamicBoxes[Forklift.PalletDynamicBody].ForkliftRelativeTransform;
        }
        if (DynamicBoxes.IsValidIndex(Forklift.CartonDynamicBody))
        {
            Forklift.Carton = DynamicBoxes[Forklift.CartonDynamicBody].Root;
            Forklift.CartonRelative = DynamicBoxes[Forklift.CartonDynamicBody].ForkliftRelativeTransform;
            DynamicBoxes[Forklift.CartonDynamicBody].SupportBodyIndex = Forklift.PalletDynamicBody;
            DynamicBoxes[Forklift.CartonDynamicBody].InitialSupportBodyIndex = Forklift.PalletDynamicBody;
        }
    }

    TArray<USceneComponent*> ShelfParcelComponents;
    if (GetWorld())
    {
        for (TActorIterator<AActor> It(GetWorld()); It; ++It)
        {
            TInlineComponentArray<USceneComponent*> Components(*It);
            for (USceneComponent* Component : Components)
            {
                if (Component && Component->ComponentHasTag(TEXT("Qai.ShelfParcel")))
                {
                    ShelfParcelComponents.Add(Component);
                }
            }
        }
    }
    ShelfParcelComponents.Sort([](const USceneComponent& A, const USceneComponent& B)
    {
        return A.GetPathName() < B.GetPathName();
    });
    int32 BoundShelfParcels = 0;
    for (USceneComponent* ShelfParcel : ShelfParcelComponents)
    {
        BoundShelfParcels += AddDynamicBox(
            ShelfParcel,
            FString::Printf(TEXT("shelf parcel %d"), BoundShelfParcels + 1),
            INDEX_NONE,
            true,
            nullptr) != INDEX_NONE ? 1 : 0;
    }
    PackingTableDesktopBounds = FBox(ForceInit);
    PackingTableTrayBounds = FBox(ForceInit);
    PackingTableTrayVisualRoot.Reset();
    if (GetWorld())
    {
        for (TActorIterator<AActor> It(GetWorld()); It; ++It)
        {
            if (!It->ActorHasTag(TEXT("Qai.DetailedPackingTable")))
            {
                continue;
            }
            TInlineComponentArray<UStaticMeshComponent*> TableMeshes(*It);
            for (UStaticMeshComponent* TableMesh : TableMeshes)
            {
                if (!TableMesh || !TableMesh->GetStaticMesh())
                {
                    continue;
                }
                const FString MeshName = TableMesh->GetStaticMesh()->GetName();
                if (MeshName.Contains(TEXT("HeavyDutyPackingTable_C01_TableTop"))
                    && !MeshName.Contains(TEXT("TableTopSide")))
                {
                    TableMesh->UpdateBounds();
                    PackingTableDesktopBounds += TableMesh->Bounds.GetBox();
                }
                else if (MeshName.Equals(TEXT("SM_Container_H20_01"), ESearchCase::IgnoreCase))
                {
                    TableMesh->UpdateBounds();
                    PackingTableTrayBounds += TableMesh->Bounds.GetBox();
                    PackingTableTrayVisualRoot = TableMesh;
                }
            }
        }
    }

    EvkVisualRoot = FindTaggedComponent(TEXT("Qai.IQ9EVK"));
    if (EvkVisualRoot.IsValid() && PackingTableDesktopBounds.IsValid)
    {
        // Put the board halfway across the clear span from the desktop's left
        // edge to the empty H20 tray. Keep its front face exactly 10 cm back
        // from the aisle-facing desktop edge. All measurements come from the
        // rendered meshes rather than the coarse table collision obstacle.
        const float GapRightX = PackingTableTrayBounds.IsValid
            ? PackingTableTrayBounds.Min.X
            : PackingTableDesktopBounds.Max.X;
        const float GapMidpointX = 0.5f * (PackingTableDesktopBounds.Min.X + GapRightX);
        const FVector DesiredCenter(
            FMath::Clamp(
                GapMidpointX,
                PackingTableDesktopBounds.Min.X + ConveyorTuning::IQ9EvkHalfExtentCm.X,
                PackingTableDesktopBounds.Max.X - ConveyorTuning::IQ9EvkHalfExtentCm.X),
            PackingTableDesktopBounds.Max.Y - ConveyorTuning::IQ9EvkHalfExtentCm.Y - 10.0f,
            PackingTableDesktopBounds.Max.Z + ConveyorTuning::IQ9EvkHalfExtentCm.Z);
        const FVector PlacementDelta = DesiredCenter - EvkVisualRoot->GetComponentLocation();
        if (AActor* EvkActor = EvkVisualRoot->GetOwner())
        {
            EvkActor->AddActorWorldOffset(PlacementDelta, false, nullptr, ETeleportType::TeleportPhysics);
        }
        else
        {
            EvkVisualRoot->SetWorldLocation(DesiredCenter, false, nullptr, ETeleportType::TeleportPhysics);
        }
        SimulatorLog(FString::Printf(
            TEXT("iq9_evk_placement location=(%.2f,%.2f,%.2f) desktop_min=(%.2f,%.2f,%.2f) desktop_max=(%.2f,%.2f,%.2f) tray_min=(%.2f,%.2f,%.2f) tray_max=(%.2f,%.2f,%.2f) horizontal=midpoint_left_edge_to_tray front_edge_inset_cm=10.00 visual_gap_cm=0.00"),
            DesiredCenter.X,
            DesiredCenter.Y,
            DesiredCenter.Z,
            PackingTableDesktopBounds.Min.X,
            PackingTableDesktopBounds.Min.Y,
            PackingTableDesktopBounds.Min.Z,
            PackingTableDesktopBounds.Max.X,
            PackingTableDesktopBounds.Max.Y,
            PackingTableDesktopBounds.Max.Z,
            PackingTableTrayBounds.Min.X,
            PackingTableTrayBounds.Min.Y,
            PackingTableTrayBounds.Min.Z,
            PackingTableTrayBounds.Max.X,
            PackingTableTrayBounds.Max.Y,
            PackingTableTrayBounds.Max.Z));
    }
    PackingTableTrayDynamicBody = AddDynamicBox(
        PackingTableTrayVisualRoot.Get(),
        TEXT("desktop tray"),
        INDEX_NONE,
        false,
        PackingTableTrayBounds.IsValid ? &PackingTableTrayBounds : nullptr);
    if (DynamicBoxes.IsValidIndex(PackingTableTrayDynamicBody))
    {
        const FDynamicBoxRuntime& TrayBody = DynamicBoxes[PackingTableTrayDynamicBody];
        SimulatorLog(FString::Printf(
            TEXT("desktop_tray_physics status=ready collider=box size_cm=(%.1f,%.1f,%.1f) mass_kg=%.2f surface=%s gravity=true movable=true forklift_contact=true reset=true"),
            TrayBody.HalfExtent.X * 2.0f,
            TrayBody.HalfExtent.Y * 2.0f,
            TrayBody.HalfExtent.Z * 2.0f,
            TrayBody.Physics.MassKg,
            *TrayBody.Physics.Surface.ToString()));
    }
    EvkDynamicBody = AddDynamicBox(
        EvkVisualRoot.Get(),
        TEXT("IQ9 EVK"),
        INDEX_NONE,
        false,
        nullptr);
    if (DynamicBoxes.IsValidIndex(EvkDynamicBody))
    {
        FDynamicBoxRuntime& EvkBody = DynamicBoxes[EvkDynamicBody];
        // Use the measured enclosure dimensions, not the render bounds. The
        // top/side image planes and runtime LED/fog components are cosmetic
        // and must never enlarge or offset the contact volume.
        EvkBody.HalfExtent = ConveyorTuning::IQ9EvkHalfExtentCm;
        EvkBody.Physics = BuildPropPhysicsProfile(TEXT("IQ9 EVK"), EvkBody.HalfExtent, 0);
        EvkBody.bEvkProp = true;
        SimulatorLog(FString::Printf(
            TEXT("iq9_evk_physics status=ready collider=box size_cm=(%.2f,%.2f,%.2f) mass_kg=%.2f com_cm=(%.2f,%.2f,%.2f) surface=%s friction=(%.2f,%.2f) restitution=%.2f gravity=true knockable=true f8_debug=true"),
            EvkBody.HalfExtent.X * 2.0f,
            EvkBody.HalfExtent.Y * 2.0f,
            EvkBody.HalfExtent.Z * 2.0f,
            EvkBody.Physics.MassKg,
            EvkBody.Physics.CenterOfMassLocalOffset.X,
            EvkBody.Physics.CenterOfMassLocalOffset.Y,
            EvkBody.Physics.CenterOfMassLocalOffset.Z,
            *EvkBody.Physics.Surface.ToString(),
            EvkBody.Physics.StaticFriction,
            EvkBody.Physics.DynamicFriction,
            EvkBody.Physics.Restitution));
    }
    ConfigureEvkProp();
    SimulatorLog(FString::Printf(
        TEXT("dynamic_boxes_bound total=%d forklift_cargo=%d shelf_parcels=%d dynamic_props=%d"),
        DynamicBoxes.Num(),
        ConveyorTuning::EnabledForkliftCount * 2,
        BoundShelfParcels,
        (DynamicBoxes.IsValidIndex(EvkDynamicBody) ? 1 : 0)
            + (DynamicBoxes.IsValidIndex(PackingTableTrayDynamicBody) ? 1 : 0)));

    Parcels.Reset();
    ParcelInitialTransforms.Reset();
    ParcelPhases.Reset();
    ParcelHeadingOffsets.Reset();
    ParcelDistances.Reset();
    ParcelVerticalPositions.Reset();
    ParcelVerticalVelocities.Reset();
    ParcelHalfHeights.Reset();
    ParcelHalfExtents.Reset();
    ParcelPhysicsProfiles.Reset();
    ParcelPitchDegrees.Reset();
    ParcelRollDegrees.Reset();
    ParcelPitchVelocities.Reset();
    ParcelRollVelocities.Reset();
    ParcelYawVelocities.Reset();
    ParcelGrounded.Reset();
    ParcelLinearVelocities.Reset();
    ParcelSupportedForklifts.Reset();
    ParcelForkContactsLogged.Reset();
    ParcelImpactCount = 0;
    ParcelForkContactCount = 0;
    bLoggedFirstParcelContact = false;
    ConveyorSurfaceZCm = -TNumericLimits<float>::Max();
    if (GetWorld())
    {
        for (TActorIterator<AActor> It(GetWorld()); It; ++It)
        {
            TInlineComponentArray<UPrimitiveComponent*> Components(*It);
            for (UPrimitiveComponent* Primitive : Components)
            {
                const FString Name = Primitive ? Primitive->GetName() : FString();
                if (Primitive && Name.Contains(TEXT("Roller"), ESearchCase::IgnoreCase))
                {
                    // Measure the actual roller crown. The old module-level
                    // bound included side rails and put every carton almost
                    // 40 cm above the physical contact surface.
                    ConveyorSurfaceZCm = FMath::Max(ConveyorSurfaceZCm, Primitive->Bounds.GetBox().Max.Z);
                }
            }
        }
    }
    if (ConveyorSurfaceZCm < -100000.0f)
    {
        ConveyorSurfaceZCm = ConveyorTuning::ConveyorRollerTopFallbackCm;
    }
    constexpr float ConveyorPerimeter = 2.0f * (2.0f * 235.1374f) + 2.0f * PI * 150.0f;
    for (int32 Index = 1; Index <= 8; ++Index)
    {
        if (USceneComponent* ParcelRoot = FindTaggedComponent(FName(*FString::Printf(TEXT("Qai.Parcel%d"), Index))))
        {
            MakeMovable(ParcelRoot);

            // USD parcel roots carry authored assembly offsets, so placing
            // those roots directly on the analytic belt path can leave the
            // rendered carton beside the conveyor. Drive a pivot at the
            // rendered bounds centre and preserve the imported subtree below
            // it, just as the forklift wheels use their own local pivots.
            USceneComponent* Parcel = ParcelRoot;
            FBox ParcelBounds(ForceInit);
            FBox ParcelOrientedBounds(ForceInit);
            int32 ParcelRenderMeshCount = 0;
            const FVector ParcelRootOrigin = ParcelRoot->GetComponentLocation();
            const FQuat ParcelRootRotation = ParcelRoot->GetComponentQuat();
            TInlineComponentArray<UStaticMeshComponent*> StaticMeshComponents(ParcelRoot->GetOwner());
            for (UStaticMeshComponent* Mesh : StaticMeshComponents)
            {
                if (!Mesh || !Mesh->GetStaticMesh() || Mesh->bHiddenInGame || !Mesh->IsVisible()
                    || (Mesh != ParcelRoot && !Mesh->IsAttachedTo(ParcelRoot)))
                {
                    continue;
                }

                ++ParcelRenderMeshCount;
                ParcelBounds += Mesh->Bounds.GetBox();

                // Build the contact box in the parcel's own orientation from
                // the visible mesh-local bounds. Converting a rotated world
                // AABB back to parcel space produces a second, much larger
                // AABB for diagonal cartons; that was the source of the
                // oversized cyan boxes seen on only some belt parcels.
                FVector MeshMinimum;
                FVector MeshMaximum;
                Mesh->GetLocalBounds(MeshMinimum, MeshMaximum);
                for (int32 Corner = 0; Corner < 8; ++Corner)
                {
                    const FVector MeshLocalCorner(
                        (Corner & 1) != 0 ? MeshMaximum.X : MeshMinimum.X,
                        (Corner & 2) != 0 ? MeshMaximum.Y : MeshMinimum.Y,
                        (Corner & 4) != 0 ? MeshMaximum.Z : MeshMinimum.Z);
                    const FVector WorldCorner = Mesh->GetComponentTransform().TransformPosition(MeshLocalCorner);
                    ParcelOrientedBounds += ParcelRootRotation.UnrotateVector(WorldCorner - ParcelRootOrigin);
                }
            }
            if (ParcelOrientedBounds.IsValid && ParcelRoot->GetAttachParent())
            {
                const FName PivotName(*FString::Printf(TEXT("QaiParcelPivot_%d"), Index));
                USceneComponent* Pivot = NewObject<USceneComponent>(ParcelRoot->GetOwner(), PivotName);
                ParcelRoot->GetOwner()->AddInstanceComponent(Pivot);
                Pivot->SetMobility(EComponentMobility::Movable);
                Pivot->RegisterComponentWithWorld(GetWorld());
                Pivot->AttachToComponent(ParcelRoot->GetAttachParent(), FAttachmentTransformRules::KeepWorldTransform);
                Pivot->SetWorldLocationAndRotation(
                    ParcelRootOrigin + ParcelRootRotation.RotateVector(ParcelOrientedBounds.GetCenter()),
                    ParcelRootRotation);
                Pivot->SetWorldScale3D(FVector::OneVector);
                ParcelRoot->AttachToComponent(Pivot, FAttachmentTransformRules::KeepWorldTransform);
                Parcel = Pivot;
            }
            Parcels.Add(Parcel);
            ParcelInitialTransforms.Add(Parcel->GetComponentTransform());
            const float Phase = 0.015f + static_cast<float>(Index - 1) * 0.125f;
            ParcelPhases.Add(Phase);
            ParcelDistances.Add(Phase * ConveyorPerimeter);
            const FVector ParcelHalfExtent = ParcelOrientedBounds.IsValid
                ? ParcelOrientedBounds.GetExtent()
                : FVector(28.0f, 20.0f, 14.0f);
            const float HalfHeight = ParcelHalfExtent.Z;
            ParcelHalfHeights.Add(HalfHeight);
            ParcelHalfExtents.Add(ParcelHalfExtent);
            ParcelPhysicsProfiles.Add(BuildPropPhysicsProfile(
                TEXT("conveyor carton"),
                ParcelHalfExtent,
                Index));
            ParcelVerticalPositions.Add(ConveyorSurfaceZCm + HalfHeight);
            ParcelVerticalVelocities.Add(0.0f);
            ParcelPitchDegrees.Add(0.0f);
            ParcelRollDegrees.Add(0.0f);
            ParcelPitchVelocities.Add(0.0f);
            ParcelRollVelocities.Add(0.0f);
            ParcelYawVelocities.Add(0.0f);
            ParcelGrounded.Add(true);
            FVector AuthoredPosition;
            FVector AuthoredTangent;
            ConveyorTuning::EvaluateConveyor(Phase * ConveyorPerimeter, AuthoredPosition, AuthoredTangent);
            const FQuat Heading = FRotator(0.0f, AuthoredTangent.Rotation().Yaw, 0.0f).Quaternion();
            ParcelHeadingOffsets.Add(Heading.Inverse() * Parcel->GetComponentQuat());
            AuthoredPosition.Z = ConveyorSurfaceZCm + HalfHeight;
            Parcel->SetWorldLocationAndRotation(
                AuthoredPosition,
                Heading * ParcelHeadingOffsets.Last(),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            ParcelInitialTransforms.Last() = Parcel->GetComponentTransform();
            ParcelLinearVelocities.Add(AuthoredTangent * ConveyorTuning::ConveyorSpeedCm);
            ParcelSupportedForklifts.Add(INDEX_NONE);
            ParcelForkContactsLogged.Add(false);
            const FPropPhysicsProfile& ParcelPhysics = ParcelPhysicsProfiles.Last();
            SimulatorLog(FString::Printf(
                TEXT("conveyor_parcel_physics parcel=%d meshes=%d collider_cm=(%.2f,%.2f,%.2f) world_aabb_cm=(%.2f,%.2f,%.2f) surface=%s mass_kg=%.2f com_cm=(%.2f,%.2f,%.2f) inertia_kg_cm2=(%.1f,%.1f,%.1f) friction=(%.2f,%.2f) restitution=%.3f"),
                Index,
                ParcelRenderMeshCount,
                ParcelHalfExtent.X * 2.0f,
                ParcelHalfExtent.Y * 2.0f,
                ParcelHalfExtent.Z * 2.0f,
                ParcelBounds.GetSize().X,
                ParcelBounds.GetSize().Y,
                ParcelBounds.GetSize().Z,
                *ParcelPhysics.Surface.ToString(),
                ParcelPhysics.MassKg,
                ParcelPhysics.CenterOfMassLocalOffset.X,
                ParcelPhysics.CenterOfMassLocalOffset.Y,
                ParcelPhysics.CenterOfMassLocalOffset.Z,
                ParcelPhysics.InertiaTensorKgCm2.X,
                ParcelPhysics.InertiaTensorKgCm2.Y,
                ParcelPhysics.InertiaTensorKgCm2.Z,
                ParcelPhysics.StaticFriction,
                ParcelPhysics.DynamicFriction,
                ParcelPhysics.Restitution));
        }
    }
    SimulatorLog(TEXT("conveyor_contact_model drive=coulomb_moving_surface centering_force=disabled heading=torque_limited fixed_step_hz=120"));

    Workers[0].Root = FindTaggedComponent(TEXT("Qai.Worker1"));
    Workers[0].Waypoints = {
        FVector(170.0f, 410.0f, 0.0f), FVector(-295.0f, 410.0f, 0.0f),
        FVector(-295.0f, -180.0f, 0.0f), FVector(55.0f, -180.0f, 0.0f),
        FVector(170.0f, -230.0f, 0.0f)};
    Workers[0].DwellSeconds = {0.0f, 0.0f, 0.0f, 0.0f, 0.8f};
    Workers[1].Root = FindTaggedComponent(TEXT("Qai.Worker2"));
    Workers[1].Waypoints = {
        FVector(-310.0f, -250.0f, 0.0f), FVector(-295.0f, -180.0f, 0.0f),
        FVector(55.0f, -180.0f, 0.0f), FVector(160.0f, -230.0f, 0.0f)};
    Workers[1].DwellSeconds = {0.0f, 0.0f, 0.0f, 2.0f};
    int32 BoundWorkers = 0;
    for (int32 WorkerIndex = 0; WorkerIndex < UE_ARRAY_COUNT(Workers); ++WorkerIndex)
    {
        FWorkerRuntime& Worker = Workers[WorkerIndex];
        if (!Worker.Root.IsValid() || Worker.Waypoints.Num() < 2)
        {
            continue;
        }
        MakeMovable(Worker.Root.Get());
        const FVector InitialDirection = Worker.Waypoints[1] - Worker.Waypoints[0];
        Worker.InitialHeadingYawDegrees = InitialDirection.Rotation().Yaw;
        Worker.TargetHeadingYawDegrees = Worker.InitialHeadingYawDegrees;
        Worker.PendingHeadingYawDegrees = Worker.InitialHeadingYawDegrees;
        Worker.PendingHeadingSeconds = 0.0f;
        Worker.VisualHeadingYawDegrees = Worker.InitialHeadingYawDegrees;
        Worker.MaximumVisualTurnRateDegreesPerSecond = 0.0f;
        const FQuat InitialHeading = FRotator(
            0.0f,
            Worker.InitialHeadingYawDegrees,
            0.0f).Quaternion();
        Worker.HeadingOffset = InitialHeading.Inverse() * Worker.Root->GetComponentQuat();
        if (WorkerIndex == 0)
        {
            // Worker1's source skeleton faces -X relative to its authored
            // scene root; compensate so the walk clip faces route travel.
            Worker.HeadingOffset *= FQuat(FVector::UpVector, PI);
        }
        Worker.DestinationIndex = 1;
        Worker.RouteDirection = 1;
        Worker.RouteReversalCount = 0;
        Worker.AvoidanceTurnSign = WorkerIndex == 0 ? 1 : -1;
        Worker.RouteReversalCooldownSeconds = 0.0f;
        Worker.BlockedSeconds = 0.0f;
        Worker.InitialTransform = Worker.Root->GetComponentTransform();
        ++BoundWorkers;
    }
    SimulatorLog(FString::Printf(
        TEXT("worker_motion_ready workers=%d visual_turn_rate_deg_s=%.1f heading_commit_ms=%.0f reversal_cooldown_ms=%.0f sticky_avoidance=true"),
        BoundWorkers,
        ConveyorTuning::WorkerMaximumVisualTurnRateDegreesPerSecond,
        ConveyorTuning::WorkerHeadingCommitSeconds * 1000.0f,
        ConveyorTuning::WorkerRouteReversalCooldownSeconds * 1000.0f));

    for (int32 ColorIndex = 0; ColorIndex < 3; ++ColorIndex)
    {
        StackLensComponents[ColorIndex].Reset();
        StackLensMaterials[ColorIndex].Reset();
        StackBillboardComponents[ColorIndex].Reset();
        StackBillboardMaterials[ColorIndex].Reset();
        StackHaloFogComponents[ColorIndex].Reset();
        StackWallSpotLights[ColorIndex].Reset();
        StackRoomSpotLights[ColorIndex].Reset();
        StackEmitterLights[ColorIndex].Reset();
    }
    const TCHAR* LightNames[] = {TEXT("Green"), TEXT("Amber"), TEXT("Red")};
    for (int32 ColorIndex = 0; ColorIndex < 3; ++ColorIndex)
    {
        for (int32 LightIndex = 0; LightIndex < 3; ++LightIndex)
        {
            const FString BaseTag = FString::Printf(TEXT("Qai.%s.%d"), LightNames[ColorIndex], LightIndex);
            UPrimitiveComponent* Lens = Cast<UPrimitiveComponent>(FindTaggedComponent(FName(*BaseTag)));
            StackLensComponents[ColorIndex].Add(Lens);
            if (Lens)
            {
                Lens->SetVisibility(true, true);
                UMaterialInstanceDynamic* LensMaterial = Lens->CreateDynamicMaterialInstance(0);
                if (LensMaterial)
                {
                    LensMaterial->SetScalarParameterValue(TEXT("EmissiveStrength"), 0.0f);
                }
                StackLensMaterials[ColorIndex].Add(LensMaterial);
            }
            if (AActor* GlowActor = FindTaggedActor(FName(*(BaseTag + TEXT(".Glow")))))
            {
                StackEmitterLights[ColorIndex].Add(GlowActor->FindComponentByClass<UPointLightComponent>());
            }
        }
    }
    ConfigureStackLightRig();

    if (AActor* Detector = FindTaggedActor(TEXT("Qai.DetectorEndline")))
    {
        DetectorCamera = Detector->GetRootComponent();
    }
    const int32 BoundColliders = BuildCollisionGuard();
    const bool bCollisionReady = ValidateCollisionGuard();
    const bool bRequiredComponentsReady = Parcels.Num() == 8
        && Forklifts[0].LiftAssembly.IsValid() && Forklifts[1].LiftAssembly.IsValid()
        && Forklifts[0].Pallet.IsValid() && Forklifts[0].Carton.IsValid()
        && Forklifts[1].Pallet.IsValid() && Forklifts[1].Carton.IsValid()
        && DetectorCamera.IsValid() && BoundWheels == 8 && BoundWorkers == 2
        && BoundShelfParcels == 24 && DynamicBoxes.Num() == 26 + ConveyorTuning::EnabledForkliftCount * 2
        && EvkVisualRoot.IsValid() && DynamicBoxes.IsValidIndex(EvkDynamicBody)
        && PackingTableTrayVisualRoot.IsValid()
        && DynamicBoxes.IsValidIndex(PackingTableTrayDynamicBody)
        && BoundColliders >= 16 && bCollisionReady;
    if (!bRequiredComponentsReady)
    {
        StageStatus = TEXT("NATIVE ASSET IMPORT INCOMPLETE");
        StageError = FString::Printf(
            TEXT("Expected 8 belt parcels, 24 dynamic shelf parcels, one movable IQ9 EVK, one movable desktop tray, 8 authored wheels, 2 authored lift assemblies, 2 workers, %d active cargo components, DetectorEndline, and 16 collision components; found %d belt parcels, %d shelf parcels, %d wheels, %d workers, %d dynamic bodies, and %d collision components."),
            ConveyorTuning::EnabledForkliftCount * 2,
            Parcels.Num(), BoundShelfParcels, BoundWheels, BoundWorkers, DynamicBoxes.Num(), BoundColliders);
        LogStageBindingFailure(FString::Printf(
            TEXT("reason=incomplete_scene belt_parcels=%d shelf_parcels=%d wheels=%d workers=%d dynamic_bodies=%d colliders=%d"),
            Parcels.Num(),
            BoundShelfParcels,
            BoundWheels,
            BoundWorkers,
            DynamicBoxes.Num(),
            BoundColliders));
        return;
    }

    bStageReady = true;
    StageStatus = TEXT("Optimized warehouse simulation ready");
    StageError.Reset();
    UE_LOG(
        LogTemp,
        Display,
        TEXT("Qai conveyor native binding ready: %d active forklift, 2 authored, %d belt parcels, %d dynamic bodies, %d collision components"),
        ConveyorTuning::EnabledForkliftCount,
        Parcels.Num(),
        DynamicBoxes.Num(),
        BoundColliders);
    SimulatorLog(FString::Printf(
        TEXT("physics_solver_ready mode=deterministic_obb fixed_hz=120 max_substeps=8 dynamic_props=%d conveyor_props=%d wheel_disks=4 sleeping=enabled broadphase=bounded gpu_cost=none"),
        DynamicBoxes.Num(),
        Parcels.Num()));
    SimulatorLog(FString::Printf(
        TEXT("stage_ready forklifts_active=%d forklifts_authored=2 lifts=2 lift_visuals=%d wheels=%d workers=%d parcels=%d colliders=%d proxies=%d forklift1=(%.1f,%.1f,%.1f) forklift2=disabled grounding_offsets_cm=(%.1f,%.1f) conveyor_surface_z_cm=%.1f worker1=(%.1f,%.1f) worker2=(%.1f,%.1f)"),
        ConveyorTuning::EnabledForkliftCount,
        (Forklifts[0].LiftVisual.IsValid() ? 1 : 0) + (Forklifts[1].LiftVisual.IsValid() ? 1 : 0),
        BoundWheels,
        BoundWorkers,
        Parcels.Num(),
        BoundColliders,
        CollisionObstacles.Num(),
        Forklifts[0].Root->GetComponentLocation().X,
        Forklifts[0].Root->GetComponentLocation().Y,
        Forklifts[0].Root->GetComponentLocation().Z,
        Forklifts[0].GroundingOffsetCm,
        Forklifts[1].GroundingOffsetCm,
        ConveyorSurfaceZCm,
        Workers[0].Root->GetComponentLocation().X,
        Workers[0].Root->GetComponentLocation().Y,
        Workers[1].Root->GetComponentLocation().X,
        Workers[1].Root->GetComponentLocation().Y));
    UpdateSafetySignal();
    UpdateStackLights();
}

void AQaiConveyorWorld::ConfigureStackLightRig()
{
    const bool bShadowedEmitters = AQaiConveyorGameMode::IsUltraRenderTier();
    int32 BoundEmitters = 0;
    int32 BoundLenses = 0;
    int32 BoundHalos = 0;
    int32 BoundWallSpots = 0;
    int32 ZeroedImportedLights = 0;

    // USD supplies a white SceneWash light as a direct child of each stack in
    // addition to the three tagged colored lens emitters. It is not a model
    // state, so leaving its imported intensity intact shows a white lamp while
    // Reason2 is offline or still starting. Zero every light beneath a stack
    // first; tagged emitters are rebound and recolored below, while untagged
    // white helpers deliberately remain dark.
    if (GetWorld())
    {
        for (TActorIterator<AActor> It(GetWorld()); It; ++It)
        {
            TInlineComponentArray<ULightComponent*> Lights(*It);
            for (ULightComponent* Light : Lights)
            {
                if (!Light)
                {
                    continue;
                }
                bool bAttachedToStack = false;
                for (USceneComponent* Parent = Light->GetAttachParent(); Parent; Parent = Parent->GetAttachParent())
                {
                    FString LogicalName = Parent->GetName();
                    int32 LastUnderscore = INDEX_NONE;
                    if (LogicalName.FindLastChar(TEXT('_'), LastUnderscore)
                        && LogicalName.Mid(LastUnderscore + 1).IsNumeric())
                    {
                        LogicalName.LeftInline(LastUnderscore, EAllowShrinking::No);
                    }
                    if (LogicalName.StartsWith(TEXT("StackLight")))
                    {
                        bAttachedToStack = true;
                        break;
                    }
                }
                if (bAttachedToStack)
                {
                    Light->SetIntensity(0.0f);
                    Light->SetLightColor(FLinearColor::Black, false);
                    ++ZeroedImportedLights;
                }
            }
        }
    }

    for (ALocalFogVolume* Halo : RuntimeStackHaloActors)
    {
        if (Halo)
        {
            Halo->Destroy();
        }
    }
    RuntimeStackHaloActors.Reset();
    for (ASpotLight* WallSpot : RuntimeStackWallSpotActors)
    {
        if (WallSpot)
        {
            WallSpot->Destroy();
        }
    }
    RuntimeStackWallSpotActors.Reset();
    for (ASpotLight* RoomSpot : RuntimeStackRoomSpotActors)
    {
        if (RoomSpot)
        {
            RoomSpot->Destroy();
        }
    }
    RuntimeStackRoomSpotActors.Reset();
    for (UMaterialBillboardComponent* Billboard : RuntimeStackBillboards)
    {
        if (Billboard)
        {
            Billboard->DestroyComponent();
        }
    }
    RuntimeStackBillboards.Reset();

    UMaterialInterface* BillboardMaterial = LoadObject<UMaterialInterface>(
        nullptr,
        TEXT("/Game/ConveyorRuntime/VisualFinish/M_StackHaloTint.M_StackHaloTint"));

    for (int32 ColorIndex = 0; ColorIndex < 3; ++ColorIndex)
    {
        for (int32 StackIndex = 0; StackIndex < StackLensComponents[ColorIndex].Num(); ++StackIndex)
        {
            TWeakObjectPtr<UPrimitiveComponent>& WeakLens = StackLensComponents[ColorIndex][StackIndex];
            UPrimitiveComponent* Lens = WeakLens.Get();
            BoundLenses += Lens ? 1 : 0;
            if (Lens && GetWorld())
            {
                const float LensRadius = FMath::Clamp(Lens->Bounds.SphereRadius, 4.0f, 12.0f);
                const float HaloDiameter = FMath::Max(225.0f, LensRadius * 22.0f);
                const FVector LensOrigin = Lens->Bounds.Origin;
                // The USD scene contains two wall-mounted stacks and one
                // conveyor-side stack. Position magnitude is not a reliable
                // wall-normal test (the rear-wall stack has a larger X than
                // Y coordinate), so preserve their authored index mapping.
                static const FVector InwardNormals[] = {
                    // Rack-front stack points away from the shelves.
                    FVector(0.0f, -1.0f, 0.0f),
                    // Rear-wall stack points into the occupied room.
                    FVector(0.0f, 1.0f, 0.0f),
                    // Free-standing conveyor stack points across the line.
                    FVector(-1.0f, 0.0f, 0.0f),
                };
                const FVector Inward = InwardNormals[FMath::Clamp(StackIndex, 0, 2)];
                if (BillboardMaterial)
                {
                    UMaterialInstanceDynamic* BillboardMID = UMaterialInstanceDynamic::Create(BillboardMaterial, this);
                    UMaterialBillboardComponent* Billboard = NewObject<UMaterialBillboardComponent>(this);
                    if (BillboardMID && Billboard)
                    {
                        BillboardMID->SetScalarParameterValue(TEXT("HaloStrength"), 0.0f);
                        Billboard->AddElement(BillboardMID, nullptr, false, 142.0f, 142.0f, nullptr);
                        // A camera-facing halo quad placed almost on the wall
                        // can intersect that wall obliquely and reveal a hard
                        // diagonal clip. Keep the soft card entirely in front
                        // of the wall/lens assembly.
                        Billboard->SetWorldLocation(LensOrigin + Inward * 14.0f);
                        Billboard->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                        Billboard->SetCastShadow(false);
                        // Appearance only: physical point lights provide the
                        // colored illumination and Lumen bounce. Never let the
                        // camera-facing card itself enter an indirect-lighting
                        // representation as a large white emissive surface.
                        Billboard->SetAffectDynamicIndirectLighting(false);
                        Billboard->SetAffectDistanceFieldLighting(false);
                        Billboard->SetTranslucentSortPriority(18);
                        // A zero-strength modulate material still evaluates to
                        // emissive white. Hardware Lumen can capture that
                        // neutral card and bounce a white pool onto the wall,
                        // even though it is visually neutral in the main pass.
                        // Remove inactive cards from rendering altogether.
                        Billboard->SetVisibility(false, true);
                        Billboard->SetHiddenInGame(true, true);
                        Billboard->RegisterComponent();
                        RuntimeStackBillboards.Add(Billboard);
                        StackBillboardComponents[ColorIndex].Add(Billboard);
                        StackBillboardMaterials[ColorIndex].Add(BillboardMID);
                    }
                }
                ALocalFogVolume* Halo = GetWorld()->SpawnActor<ALocalFogVolume>(
                    LensOrigin + Inward * (HaloDiameter * 0.5f),
                    FRotationMatrix::MakeFromX(Inward).Rotator());
                if (Halo)
                {
                    Halo->Tags.Add(FName(*FString::Printf(TEXT("Qai.StackHalo.%d"), ColorIndex)));
                    // Start the volume exactly at the lens plane and extend
                    // it inward by one diameter. The previous 2.45x ellipsoid
                    // reached far behind wall-mounted lenses, so its clipped
                    // intersection looked like a diagonally displaced beam.
                    const float BaseVolumeSize = ULocalFogVolumeComponent::GetBaseVolumeSize();
                    Halo->SetActorScale3D(FVector(
                        HaloDiameter / BaseVolumeSize,
                        (HaloDiameter * 0.78f) / BaseVolumeSize,
                        (HaloDiameter * 0.78f) / BaseVolumeSize));
                    if (ULocalFogVolumeComponent* HaloFog = Halo->GetComponent())
                    {
                        HaloFog->SetRadialFogExtinction(0.0f);
                        HaloFog->SetHeightFogExtinction(0.0f);
                        HaloFog->SetFogPhaseG(0.35f);
                        HaloFog->SetFogAlbedo(FLinearColor::Black);
                        HaloFog->SetFogEmissive(FLinearColor::Black);
                        HaloFog->SetFogStartDistance(0.0f);
                        StackHaloFogComponents[ColorIndex].Add(HaloFog);
                    }
                    RuntimeStackHaloActors.Add(Halo);
                    ++BoundHalos;
                    SimulatorLog(FString::Printf(
                        TEXT("stack_light_alignment stack=%d color=%d lens=(%.1f,%.1f,%.1f) fog=(%.1f,%.1f,%.1f) inward=(%.1f,%.1f,%.1f)"),
                        StackIndex + 1,
                        ColorIndex,
                        LensOrigin.X,
                        LensOrigin.Y,
                        LensOrigin.Z,
                        Halo->GetActorLocation().X,
                        Halo->GetActorLocation().Y,
                        Halo->GetActorLocation().Z,
                        Inward.X,
                        Inward.Y,
                        Inward.Z));
                }
                const FVector ProjectorLocation = LensOrigin + Inward * 105.0f;
                const FRotator ProjectorRotation = FRotationMatrix::MakeFromX(-Inward).Rotator();
                if (ASpotLight* WallSpot = GetWorld()->SpawnActor<ASpotLight>(ProjectorLocation, ProjectorRotation))
                {
                    WallSpot->Tags.Add(FName(*FString::Printf(TEXT("Qai.StackWallSpot.%d"), ColorIndex)));
                    if (USpotLightComponent* Spot = WallSpot->SpotLightComponent)
                    {
                        Spot->SetMobility(EComponentMobility::Movable);
                        Spot->SetIntensityUnits(ELightUnits::Lumens);
                        Spot->SetAttenuationRadius(410.0f);
                        // Keep the cone boundary far outside the nearby wall.
                        // The point emitter and rect washes shape the visible
                        // pool; this almost-hemispherical spot only fills the
                        // lens-adjacent wall without a triangular cutoff.
                        Spot->SetInnerConeAngle(60.0f);
                        Spot->SetOuterConeAngle(88.0f);
                        Spot->SetSourceRadius(35.0f);
                        Spot->SetSoftSourceRadius(70.0f);
                        Spot->SetCastShadows(false);
                        Spot->SetIndirectLightingIntensity(1.35f);
                        Spot->SetVolumetricScatteringIntensity(1.1f);
                        Spot->SetIntensity(0.0f);
                        StackWallSpotLights[ColorIndex].Add(Spot);
                    }
                    RuntimeStackWallSpotActors.Add(WallSpot);
                    ++BoundWallSpots;
                }

                // Retain a zero-energy room projector as a diagnostic rig
                // element, but do not use a bounded cone for atmosphere. Even
                // with low surface energy its edge could strike a perpendicular
                // wall as a colored diagonal. The point emitter and local fog
                // ellipsoid below now provide the camera-visible volume.
                // Start the visible beam well in front of the wall and tilt
                // it slightly toward the floor. This keeps the cone boundary
                // from slicing the wall as a hard diagonal while the local
                // fog ellipsoid supplies the broad lens-adjacent glow.
                const FVector RoomDirection = (Inward + FVector(0.0f, 0.0f, -0.14f)).GetSafeNormal();
                const FVector RoomProjectorLocation = LensOrigin + Inward * 58.0f;
                const FRotator RoomProjectorRotation = FRotationMatrix::MakeFromX(RoomDirection).Rotator();
                if (ASpotLight* RoomSpot = GetWorld()->SpawnActor<ASpotLight>(RoomProjectorLocation, RoomProjectorRotation))
                {
                    RoomSpot->Tags.Add(FName(*FString::Printf(TEXT("Qai.StackRoomVolume.%d"), ColorIndex)));
                    if (USpotLightComponent* Spot = RoomSpot->SpotLightComponent)
                    {
                        Spot->SetMobility(EComponentMobility::Movable);
                        Spot->SetIntensityUnits(ELightUnits::Lumens);
                        Spot->SetAttenuationRadius(470.0f);
                        Spot->SetInnerConeAngle(23.0f);
                        Spot->SetOuterConeAngle(38.0f);
                        Spot->SetSourceRadius(18.0f);
                        Spot->SetSoftSourceRadius(42.0f);
                        Spot->SetCastShadows(false);
                        Spot->SetIndirectLightingIntensity(0.75f);
                        // Make this projector fog-dominant. A high-lumen
                        // surface spotlight left a visible triangular edge on
                        // perpendicular walls; low direct energy multiplied
                        // in the fog preserves the colored air volume without
                        // painting that hard cone onto geometry.
                        Spot->SetVolumetricScatteringIntensity(60.0f);
                        Spot->SetIntensity(0.0f);
                        StackRoomSpotLights[ColorIndex].Add(Spot);
                    }
                    RuntimeStackRoomSpotActors.Add(RoomSpot);
                }
            }
        }
        for (int32 StackIndex = 0; StackIndex < StackEmitterLights[ColorIndex].Num(); ++StackIndex)
        {
            TWeakObjectPtr<UPointLightComponent>& WeakEmitter = StackEmitterLights[ColorIndex][StackIndex];
            if (UPointLightComponent* Emitter = WeakEmitter.Get())
            {
                // Imported emitters can sit fractionally inside the wall
                // shell. Move wall-mounted lights into the room so their
                // shadow rays do not begin inside the portal/wall mesh and
                // produce a diagonal self-shadow.
                static const FVector EmitterInwardNormals[] = {
                    FVector(0.0f, -1.0f, 0.0f),
                    FVector(0.0f, 1.0f, 0.0f),
                    FVector(-1.0f, 0.0f, 0.0f),
                };
                if (StackLensComponents[ColorIndex].IsValidIndex(StackIndex))
                {
                    if (const UPrimitiveComponent* Lens = StackLensComponents[ColorIndex][StackIndex].Get())
                    {
                        Emitter->SetWorldLocation(
                            Lens->Bounds.Origin
                            + EmitterInwardNormals[FMath::Clamp(StackIndex, 0, 2)] * 12.0f);
                    }
                }
                Emitter->SetMobility(EComponentMobility::Movable);
                Emitter->SetIntensityUnits(ELightUnits::Lumens);
                Emitter->SetUseInverseSquaredFalloff(true);
                // Keep the inverse-square fade boundary outside the compact
                // warehouse walls; an in-frame attenuation sphere can read as
                // a diagonal cut where it intersects a perpendicular panel.
                Emitter->SetAttenuationRadius(760.0f);
                Emitter->SetSourceRadius(8.0f);
                Emitter->SetSoftSourceRadius(32.0f);
                Emitter->SetCastShadows(bShadowedEmitters);
                Emitter->SetIndirectLightingIntensity(
                    AQaiConveyorGameMode::UsesHardwareLumen() ? 1.65f : 1.15f);
                Emitter->SetVolumetricScatteringIntensity(16.0f);
                Emitter->SetIntensity(0.0f);
                Emitter->SetVisibility(true);
                ++BoundEmitters;
            }
        }
    }

    StackWashLights.Reset();
    StackWashMultipliers.Reset();
    for (int32 Index = 0; Index < 3; ++Index)
    {
        AActor* BounceActor = FindTaggedActor(FName(*FString::Printf(TEXT("Qai.StackBounce.%d"), Index)));
        URectLightComponent* Wash = BounceActor ? BounceActor->FindComponentByClass<URectLightComponent>() : nullptr;
        if (Wash)
        {
            Wash->SetMobility(EComponentMobility::Movable);
            Wash->SetIntensityUnits(ELightUnits::Lumens);
            Wash->SetAttenuationRadius(900.0f);
            Wash->SetSourceWidth(280.0f);
            Wash->SetSourceHeight(180.0f);
            Wash->SetCastShadows(false);
            Wash->SetIndirectLightingIntensity(1.45f);
            Wash->SetVolumetricScatteringIntensity(0.55f);
            Wash->SetIntensity(0.0f);
            Wash->SetVisibility(true);
        }
        StackWashLights.Add(Wash);
        // The imported bounce rect is very close to its wall. At full energy
        // its center clips to white under the filmic tonemapper even when the
        // light color is saturated. Keep it as local/floor fill and let the
        // larger runtime rect create the broad visible wall pool.
        StackWashMultipliers.Add(0.20f);
    }

    // The imported rects face the floor. Two additional broad, shadowless
    // emitters paint the nearby walls, recreating the large Omniverse wash
    // without a camera-wide color grade. The third stack is floor-only: its
    // nearest useful wall is outside the requested 450-650 cm range.
    if (RuntimeStackWashActors.IsEmpty() && GetWorld())
    {
        struct FWallWashSpec
        {
            FVector Origin;
            FVector Target;
            const TCHAR* Tag;
        };
        const FWallWashSpec Specs[] = {
            {FVector(-340.0f, 351.0f, 310.0f), FVector(-525.0f, 351.0f, 235.0f), TEXT("Qai.StackWallWash.0")},
            // Keep the rear-wall wash well in front of the signal. The old
            // -320 cm origin sat almost on the wall/lens plane and compressed
            // the colored pool into a clipped white hotspot.
            {FVector(435.0f, -210.0f, 215.0f), FVector(435.0f, -520.0f, 190.0f), TEXT("Qai.StackWallWash.1")},
        };
        for (const FWallWashSpec& Spec : Specs)
        {
            if (ARectLight* Actor = GetWorld()->SpawnActor<ARectLight>(Spec.Origin, FRotationMatrix::MakeFromX(Spec.Target - Spec.Origin).Rotator()))
            {
                Actor->Tags.Add(FName(Spec.Tag));
                RuntimeStackWashActors.Add(Actor);
            }
        }
    }
    for (ARectLight* Actor : RuntimeStackWashActors)
    {
        URectLightComponent* Wash = Actor ? Actor->FindComponentByClass<URectLightComponent>() : nullptr;
        if (Wash)
        {
            Wash->SetMobility(EComponentMobility::Movable);
            Wash->SetIntensityUnits(ELightUnits::Lumens);
            Wash->SetAttenuationRadius(900.0f);
            Wash->SetSourceWidth(280.0f);
            Wash->SetSourceHeight(210.0f);
            Wash->SetCastShadows(false);
            Wash->SetIndirectLightingIntensity(1.55f);
            Wash->SetVolumetricScatteringIntensity(0.48f);
            Wash->SetIntensity(0.0f);
            Wash->SetVisibility(true);
        }
        StackWashLights.Add(Wash);
        // Runtime wall washes supplement the authored floor washes. Keeping
        // them near half energy preserves signal hue on white walls while
        // Lumen supplies the secondary bounce on nearby glossy surfaces.
        StackWashMultipliers.Add(0.62f);
    }

    if (!RuntimeSignalFogActor && GetWorld())
    {
        RuntimeSignalFogActor = GetWorld()->SpawnActor<AExponentialHeightFog>(
            FVector(0.0f, 0.0f, -220.0f),
            FRotator::ZeroRotator);
        if (RuntimeSignalFogActor)
        {
            RuntimeSignalFogActor->Tags.Add(TEXT("Qai.SignalVolumetricFog"));
        }
    }
    if (RuntimeSignalFogActor)
    {
        if (UExponentialHeightFogComponent* Fog = RuntimeSignalFogActor->GetComponent())
        {
            // A very low neutral density keeps the warehouse crisp while
            // allowing the active local signal light to draw a saturated,
            // camera-visible colored volume around its lens.
            Fog->SetFogDensity(0.0048f);
            Fog->SetFogHeightFalloff(0.001f);
            Fog->SetFogInscatteringColor(FLinearColor(0.72f, 0.75f, 0.80f));
            Fog->SetFogMaxOpacity(0.11f);
            Fog->SetStartDistance(35.0f);
            Fog->SetVolumetricFog(true);
            Fog->SetVolumetricFogScatteringDistribution(0.78f);
            Fog->SetVolumetricFogExtinctionScale(0.62f);
            Fog->SetVolumetricFogAlbedo(FColor::White);
            Fog->SetVolumetricFogEmissive(FLinearColor::Black);
            Fog->SetVolumetricFogDistance(2200.0f);
            Fog->SetVolumetricFogStartDistance(0.0f);
            Fog->SetVolumetricFogNearFadeInDistance(45.0f);
        }
    }

    bStackLightRigInitialized = false;
    SimulatorLog(FString::Printf(
        TEXT("stack_light_rig lenses=%d lens_billboards=%d halos=%d emitters=%d washes=%d wall_spots=%d room_volume_spots=%d imported_lights_zeroed=%d dark_until_model=true shadowed_emitters=%s volumetric_fog=%s local_radius_cm=760 volume_length_cm=470 wash_radius_cm=900 fade_ms=200 hardware_lumen=%s"),
        BoundLenses,
        RuntimeStackBillboards.Num(),
        BoundHalos,
        BoundEmitters,
        StackWashLights.Num(),
        BoundWallSpots,
        RuntimeStackRoomSpotActors.Num(),
        ZeroedImportedLights,
        bShadowedEmitters ? TEXT("true") : TEXT("false"),
        RuntimeSignalFogActor ? TEXT("true") : TEXT("false"),
        AQaiConveyorGameMode::UsesHardwareLumen() ? TEXT("true") : TEXT("false")));
}

void AQaiConveyorWorld::ConfigureEvkProp()
{
    for (UStaticMeshComponent* LedMesh : RuntimeEvkLedMeshes)
    {
        if (LedMesh)
        {
            LedMesh->DestroyComponent();
        }
    }
    for (UPointLightComponent* LedLight : RuntimeEvkLedLights)
    {
        if (LedLight)
        {
            LedLight->DestroyComponent();
        }
    }
    for (ALocalFogVolume* FogActor : RuntimeEvkLedFogActors)
    {
        if (FogActor)
        {
            FogActor->Destroy();
        }
    }
    RuntimeEvkLedMeshes.Reset();
    RuntimeEvkLedLights.Reset();
    RuntimeEvkLedMaterials.Reset();
    RuntimeEvkLedFogActors.Reset();
    EvkLedFogComponents.Reset();
    EvkLedTimers.Reset();
    EvkLedStates.Reset();

    USceneComponent* VisualRoot = EvkVisualRoot.Get();
    if (!VisualRoot || !GetWorld())
    {
        SimulatorLog(TEXT("iq9_evk_led_rig status=missing_visual"));
        return;
    }
    // The impostor body uses a 100 cm engine cube scaled to the measured
    // 10.6 x 10.6 x 4.3 cm enclosure.
    // AddDynamicBox has already inserted an unscaled centre pivot; anchor the
    // physical LED rig there so relative centimetres and emitter size are not
    // multiplied by the visual cube scale.
    USceneComponent* LedAnchor = VisualRoot->GetAttachParent();
    if (!LedAnchor)
    {
        LedAnchor = VisualRoot;
    }

    UStaticMesh* SphereMesh = LoadObject<UStaticMesh>(
        nullptr,
        TEXT("/Engine/BasicShapes/Sphere.Sphere"));
    UMaterialInterface* LedMaterial = LoadObject<UMaterialInterface>(
        nullptr,
        TEXT("/Game/IQ9EVK/Runtime/M_IQ9_LED.M_IQ9_LED"));
    if (!SphereMesh || !LedMaterial)
    {
        SimulatorLog(FString::Printf(
            TEXT("iq9_evk_led_rig status=missing_assets sphere=%s material=%s"),
            SphereMesh ? TEXT("true") : TEXT("false"),
            LedMaterial ? TEXT("true") : TEXT("false")));
        return;
    }

    static const FVector LedPositions[] = {
        // The five-view impostor is centred at its physics pivot. Keep the two
        // real emitters just above the exposed top-board image.
        FVector(3.20f, 3.30f, 2.55f),
        FVector(1.85f, 3.30f, 2.55f),
    };
    static const FLinearColor LedColors[] = {
        FLinearColor(0.01f, 1.0f, 0.05f),
        FLinearColor(1.0f, 0.24f, 0.005f),
    };
    EvkLedRandom.Initialize(0x19E9);
    const float FogScale = 11.0f / ULocalFogVolumeComponent::GetBaseVolumeSize();
    for (int32 LedIndex = 0; LedIndex < UE_ARRAY_COUNT(LedPositions); ++LedIndex)
    {
        UStaticMeshComponent* LedMesh = NewObject<UStaticMeshComponent>(
            this,
            FName(*FString::Printf(TEXT("IQ9LedMesh_%d"), LedIndex + 1)));
        AddInstanceComponent(LedMesh);
        LedMesh->SetStaticMesh(SphereMesh);
        LedMesh->SetMobility(EComponentMobility::Movable);
        LedMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        LedMesh->SetCastShadow(false);
        LedMesh->AttachToComponent(LedAnchor, FAttachmentTransformRules::KeepRelativeTransform);
        LedMesh->SetRelativeLocation(LedPositions[LedIndex]);
        LedMesh->SetRelativeScale3D(FVector(0.008f));
        UMaterialInstanceDynamic* LedMID = UMaterialInstanceDynamic::Create(LedMaterial, this);
        if (LedMID)
        {
            LedMID->SetVectorParameterValue(TEXT("LedColor"), LedColors[LedIndex]);
            LedMID->SetScalarParameterValue(TEXT("LedStrength"), LedIndex == 0 ? 180.0f : 0.0f);
            LedMesh->SetMaterial(0, LedMID);
        }
        LedMesh->RegisterComponentWithWorld(GetWorld());
        RuntimeEvkLedMeshes.Add(LedMesh);
        RuntimeEvkLedMaterials.Add(LedMID);

        UPointLightComponent* LedLight = NewObject<UPointLightComponent>(
            this,
            FName(*FString::Printf(TEXT("IQ9LedLight_%d"), LedIndex + 1)));
        AddInstanceComponent(LedLight);
        LedLight->SetMobility(EComponentMobility::Movable);
        LedLight->SetIntensityUnits(ELightUnits::Lumens);
        LedLight->SetIntensity(LedIndex == 0 ? 28.0f : 0.0f);
        LedLight->SetLightColor(LedColors[LedIndex]);
        LedLight->SetAttenuationRadius(24.0f);
        LedLight->SetSourceRadius(0.35f);
        LedLight->SetSoftSourceRadius(1.0f);
        LedLight->SetCastShadows(false);
        LedLight->SetVolumetricScatteringIntensity(8.0f);
        LedLight->AttachToComponent(LedAnchor, FAttachmentTransformRules::KeepRelativeTransform);
        LedLight->SetRelativeLocation(LedPositions[LedIndex] + FVector(0.0f, 0.0f, 0.35f));
        LedLight->RegisterComponentWithWorld(GetWorld());
        RuntimeEvkLedLights.Add(LedLight);

        ALocalFogVolume* FogActor = GetWorld()->SpawnActor<ALocalFogVolume>(
            LedAnchor->GetComponentTransform().TransformPosition(LedPositions[LedIndex]),
            LedAnchor->GetComponentRotation());
        if (FogActor)
        {
            FogActor->SetActorScale3D(FVector(FogScale));
            FogActor->AttachToComponent(LedAnchor, FAttachmentTransformRules::KeepWorldTransform);
            if (ULocalFogVolumeComponent* Fog = FogActor->GetComponent())
            {
                Fog->SetRadialFogExtinction(0.0f);
                Fog->SetHeightFogExtinction(0.0f);
                Fog->SetFogPhaseG(0.12f);
                Fog->SetFogAlbedo(FLinearColor::Black);
                Fog->SetFogEmissive(LedIndex == 0 ? LedColors[LedIndex] * 0.48f : FLinearColor::Black);
                Fog->SetFogStartDistance(0.0f);
                EvkLedFogComponents.Add(Fog);
            }
            RuntimeEvkLedFogActors.Add(FogActor);
        }
        EvkLedTimers.Add(LedIndex == 0 ? 0.0f : 0.32f);
        EvkLedStates.Add(LedIndex == 0);
    }
    SimulatorLog(FString::Printf(
        TEXT("iq9_evk_led_rig status=ready leds=%d mode=steady_green+random_blink_amber local_fog_volumes=%d point_radius_cm=24 random_seed=0x19E9"),
        RuntimeEvkLedMeshes.Num(),
        RuntimeEvkLedFogActors.Num()));
}

void AQaiConveyorWorld::TickEvkLeds(float DeltaSeconds)
{
    static const FLinearColor LedColors[] = {
        FLinearColor(0.01f, 1.0f, 0.05f),
        FLinearColor(1.0f, 0.24f, 0.005f),
    };
    for (int32 LedIndex = 0; LedIndex < EvkLedTimers.Num(); ++LedIndex)
    {
        bool bEnabled = LedIndex == 0;
        if (LedIndex > 0)
        {
            EvkLedTimers[LedIndex] -= DeltaSeconds;
            bEnabled = EvkLedStates.IsValidIndex(LedIndex)
                && EvkLedStates[LedIndex];
            if (EvkLedTimers[LedIndex] <= 0.0f)
            {
                bEnabled = !bEnabled;
                EvkLedTimers[LedIndex] = bEnabled
                    ? EvkLedRandom.FRandRange(0.07f, 0.42f)
                    : EvkLedRandom.FRandRange(0.18f, 1.15f);
            }
        }
        if (EvkLedStates.IsValidIndex(LedIndex))
        {
            EvkLedStates[LedIndex] = bEnabled;
        }
        const FLinearColor Color = LedColors[FMath::Clamp(LedIndex, 0, UE_ARRAY_COUNT(LedColors) - 1)];
        if (RuntimeEvkLedMaterials.IsValidIndex(LedIndex) && RuntimeEvkLedMaterials[LedIndex])
        {
            RuntimeEvkLedMaterials[LedIndex]->SetVectorParameterValue(TEXT("LedColor"), Color);
            RuntimeEvkLedMaterials[LedIndex]->SetScalarParameterValue(
                TEXT("LedStrength"),
                bEnabled ? 180.0f : 0.0f);
        }
        if (RuntimeEvkLedLights.IsValidIndex(LedIndex) && RuntimeEvkLedLights[LedIndex])
        {
            RuntimeEvkLedLights[LedIndex]->SetLightColor(Color);
            RuntimeEvkLedLights[LedIndex]->SetIntensity(bEnabled ? 28.0f : 0.0f);
        }
        if (EvkLedFogComponents.IsValidIndex(LedIndex))
        {
            if (ULocalFogVolumeComponent* Fog = EvkLedFogComponents[LedIndex].Get())
            {
                Fog->SetFogEmissive(bEnabled ? Color * 0.48f : FLinearColor::Black);
            }
        }
    }
}

void AQaiConveyorWorld::DriveActiveForklift(float Throttle, float Steer, float Lift, bool bBrake, float DeltaSeconds)
{
    CommandThrottle = FMath::Clamp(Throttle, -1.0f, 1.0f);
    CommandSteer = FMath::Clamp(Steer, -1.0f, 1.0f);
    CommandLift = FMath::Clamp(Lift, -1.0f, 1.0f);
    bCommandBrake = bBrake;
}

void AQaiConveyorWorld::CycleForklift(int32 Direction)
{
    const int32 Count = ConveyorTuning::EnabledForkliftCount;
    SelectForklift((ActiveForklift + (Direction < 0 ? Count - 1 : 1)) % Count);
}

void AQaiConveyorWorld::SelectForklift(int32 Index)
{
    ActiveForklift = FMath::Clamp(Index, 0, ConveyorTuning::EnabledForkliftCount - 1);
    CommandThrottle = 0.0f;
    CommandSteer = 0.0f;
    CommandLift = 0.0f;
    LastCollisionObstacle.Reset();
    CollisionStatus = FString::Printf(TEXT("ready | %d collision volumes"), CollisionObstacles.Num());
}

void AQaiConveyorWorld::ResetScene()
{
    if (!bStageReady)
    {
        return;
    }

    CancelActiveInference();
    CommandThrottle = 0.0f;
    CommandSteer = 0.0f;
    CommandLift = 0.0f;
    bCommandBrake = false;
    ActiveForklift = 0;
    FixedAccumulator = 0.0f;
    ConveyorDistanceCm = 0.0f;
    CollisionBlockCount = 0;
    LastCollisionObstacle.Reset();
    CollisionStatus = FString::Printf(TEXT("ready | %d collision volumes"), CollisionObstacles.Num());

    for (int32 ForkliftIndex = 0; ForkliftIndex < UE_ARRAY_COUNT(Forklifts); ++ForkliftIndex)
    {
        FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        if (!Forklift.Root.IsValid())
        {
            continue;
        }
        Forklift.Root->SetWorldTransform(
            Forklift.InitialRoot, false, nullptr, ETeleportType::TeleportPhysics);
        Forklift.SpeedCmPerSecond = 0.0f;
        Forklift.SteeringInput = 0.0f;
        Forklift.SurfaceLinearVelocityCmPerSecond = FVector::ZeroVector;
        Forklift.SurfaceYawVelocityDegreesPerSecond = 0.0f;
        Forklift.PreviousSpeedCmPerSecond = 0.0f;
        Forklift.CombinedMassKg = Forklift.ChassisMassKg;
        Forklift.CombinedCenterOfMassLocalCm = Forklift.BaseCenterOfMassLocalCm;
        Forklift.LiftCm = 0.0f;
        Forklift.LiftDeltaCm = 0.0f;
        Forklift.WheelAngleDegrees = 0.0f;
        Forklift.RideHeightOffsetCm = 0.0f;
        Forklift.MaximumRideHeightCm = 0.0f;
        for (float& WheelOffset : Forklift.WheelSupportOffsetsCm)
        {
            WheelOffset = 0.0f;
        }
        Forklift.BodyPitchDegrees = 0.0f;
        Forklift.BodyRollDegrees = 0.0f;
        Forklift.PitchVelocityDegrees = 0.0f;
        Forklift.RollVelocityDegrees = 0.0f;
        Forklift.MaximumAbsoluteTipDegrees = 0.0f;
        Forklift.bWheelClimbActive = false;
        Forklift.bForkReactionActive = false;
        Forklift.LastBlockedDriveSign = 0.0f;
        Forklift.LastBlockedObstacle.Reset();
        for (int32 WheelIndex = 0; WheelIndex < 4; ++WheelIndex)
        {
            if (USceneComponent* WheelPivot = Forklift.WheelPivots[WheelIndex].Get())
            {
                WheelPivot->SetRelativeRotation(Forklift.WheelPivotInitialRelative[WheelIndex]);
            }
        }
        UpdateCargo(Forklift);
    }

    for (FDynamicBoxRuntime& Body : DynamicBoxes)
    {
        Body.SupportedForklift = Body.InitialSupportedForklift;
        Body.SupportBodyIndex = Body.InitialSupportBodyIndex;
        Body.LinearVelocity = FVector::ZeroVector;
        Body.AngularVelocityDegrees = FVector::ZeroVector;
        Body.ForkSupportCooldownSeconds = 0.0f;
        Body.ForkUnsupportedSeconds = 0.0f;
        Body.LastSupportCoverage = 1.0f;
        Body.bAwake = false;
        Body.bLoggedContact = false;
        Body.bLoggedLedgeRelease = false;
        if (USceneComponent* Root = Body.Root.Get())
        {
            Root->SetWorldTransform(Body.InitialTransform, false, nullptr, ETeleportType::TeleportPhysics);
        }
    }

    constexpr float ConveyorPerimeter = 2.0f * (2.0f * 235.1374f) + 2.0f * PI * 150.0f;
    for (int32 Index = 0; Index < Parcels.Num(); ++Index)
    {
        if (USceneComponent* Parcel = Parcels[Index].Get(); Parcel && ParcelInitialTransforms.IsValidIndex(Index))
        {
            Parcel->SetWorldTransform(
                ParcelInitialTransforms[Index], false, nullptr, ETeleportType::TeleportPhysics);
        }
        if (ParcelPhases.IsValidIndex(Index) && ParcelDistances.IsValidIndex(Index))
        {
            ParcelDistances[Index] = ParcelPhases[Index] * ConveyorPerimeter;
        }
        if (ParcelVerticalPositions.IsValidIndex(Index) && ParcelHalfHeights.IsValidIndex(Index))
        {
            ParcelVerticalPositions[Index] = ConveyorSurfaceZCm + ParcelHalfHeights[Index];
        }
        if (ParcelVerticalVelocities.IsValidIndex(Index)) ParcelVerticalVelocities[Index] = 0.0f;
        if (ParcelPitchDegrees.IsValidIndex(Index)) ParcelPitchDegrees[Index] = 0.0f;
        if (ParcelRollDegrees.IsValidIndex(Index)) ParcelRollDegrees[Index] = 0.0f;
        if (ParcelPitchVelocities.IsValidIndex(Index)) ParcelPitchVelocities[Index] = 0.0f;
        if (ParcelRollVelocities.IsValidIndex(Index)) ParcelRollVelocities[Index] = 0.0f;
        if (ParcelYawVelocities.IsValidIndex(Index)) ParcelYawVelocities[Index] = 0.0f;
        if (ParcelGrounded.IsValidIndex(Index)) ParcelGrounded[Index] = true;
        if (ParcelSupportedForklifts.IsValidIndex(Index)) ParcelSupportedForklifts[Index] = INDEX_NONE;
        if (ParcelForkContactsLogged.IsValidIndex(Index)) ParcelForkContactsLogged[Index] = false;
        if (ParcelLinearVelocities.IsValidIndex(Index))
        {
            FVector Position;
            FVector Tangent;
            ConveyorTuning::EvaluateConveyor(ParcelDistances[Index], Position, Tangent);
            ParcelLinearVelocities[Index] = Tangent * ConveyorTuning::ConveyorSpeedCm;
        }
    }
    ParcelImpactCount = 0;
    ParcelForkContactCount = 0;
    bLoggedFirstParcelContact = false;

    for (int32 WorkerIndex = 0; WorkerIndex < UE_ARRAY_COUNT(Workers); ++WorkerIndex)
    {
        FWorkerRuntime& Worker = Workers[WorkerIndex];
        if (USceneComponent* Root = Worker.Root.Get())
        {
            Root->SetWorldTransform(Worker.InitialTransform, false, nullptr, ETeleportType::TeleportPhysics);
        }
        Worker.DestinationIndex = 1;
        Worker.RouteDirection = 1;
        Worker.RouteReversalCount = 0;
        Worker.AvoidanceTurnSign = WorkerIndex == 0 ? 1 : -1;
        Worker.DwellRemaining = 0.0f;
        Worker.BlockedSeconds = 0.0f;
        Worker.RouteReversalCooldownSeconds = 0.0f;
        Worker.TargetHeadingYawDegrees = Worker.InitialHeadingYawDegrees;
        Worker.PendingHeadingYawDegrees = Worker.InitialHeadingYawDegrees;
        Worker.PendingHeadingSeconds = 0.0f;
        Worker.VisualHeadingYawDegrees = Worker.InitialHeadingYawDegrees;
        Worker.MaximumVisualTurnRateDegreesPerSecond = 0.0f;
        Worker.SeparationEvents = 0;
    }

    EncodedFrames.Reset();
    EncodedForkliftFrameTransforms.Reset();
    EncodedFrameTextures.Reset();
    EncodedFrameTimes.Reset();
    SubmittedEncodedFrames.Reset();
    SubmittedFrameTextures.Reset();
    SubmittedFrameTimes.Reset();
    SubmittedGroundTruthSignal = TEXT("G");
    bSubmittedForkliftMotion = false;
    bSubmittedRedOverlap = false;
    RawModelSignal = TEXT("-");
    ModelSignal = TEXT("-");
    GroundTruthSignal = TEXT("G");
    LastLoggedStackSignal.Reset();
    bStackLightRigInitialized = false;
    EvkLedRandom.Initialize(0x19E9);
    for (int32 LedIndex = 0; LedIndex < EvkLedTimers.Num(); ++LedIndex)
    {
        EvkLedTimers[LedIndex] = 0.06f + static_cast<float>(LedIndex) * 0.09f;
        if (EvkLedStates.IsValidIndex(LedIndex))
        {
            EvkLedStates[LedIndex] = false;
        }
        if (RuntimeEvkLedMaterials.IsValidIndex(LedIndex) && RuntimeEvkLedMaterials[LedIndex])
        {
            RuntimeEvkLedMaterials[LedIndex]->SetScalarParameterValue(TEXT("LedStrength"), 0.0f);
        }
        if (RuntimeEvkLedLights.IsValidIndex(LedIndex) && RuntimeEvkLedLights[LedIndex])
        {
            RuntimeEvkLedLights[LedIndex]->SetIntensity(0.0f);
        }
        if (EvkLedFogComponents.IsValidIndex(LedIndex))
        {
            if (ULocalFogVolumeComponent* Fog = EvkLedFogComponents[LedIndex].Get())
            {
                Fog->SetFogEmissive(FLinearColor::Black);
            }
        }
    }
    TickEvkLeds(0.0f);
    UpdateStackLights();
    UpdateSafetySignal();
    SimulatorLog(FString::Printf(
        TEXT("scene_reset forklifts=2 dynamic_boxes=%d conveyor_parcels=%d workers=2 inference_frames=0"),
        DynamicBoxes.Num(),
        Parcels.Num()));
}

void AQaiConveyorWorld::ToggleBackend()
{
    CancelActiveInference();
    ActiveBackend = ActiveBackend == TEXT("host") ? TEXT("evk") : TEXT("host");
    ActiveModel = CurrentModelName();
    bBackendHealthy = false;
    bHasLoggedBackendProbe = false;
    bLoggedFirstInferenceSubmission = false;
    LastLoggedInferenceState.Reset();
    LastInferenceSummaryLogSeconds = -60.0;
    BackendStatus = TEXT("checking");
    const bool bEvkCapture = ActiveBackend == TEXT("evk");
    SimulatorLog(FString::Printf(
        TEXT("backend_selected backend=%s model=%s capture_output=%dx%d"),
        *ActiveBackend,
        *ActiveModel,
        bEvkCapture ? ConveyorTuning::EvkCaptureWidth : HostCaptureWidth,
        bEvkCapture ? ConveyorTuning::EvkCaptureHeight : HostCaptureHeight));
    ProbeBackend();
}

void AQaiConveyorWorld::ToggleInference()
{
    bInferenceEnabled = !bInferenceEnabled;
    if (!bInferenceEnabled)
    {
        CancelActiveInference();
        BackendStatus = TEXT("paused");
    }
    else if (!bBackendHealthy)
    {
        ProbeBackend();
    }
    else
    {
        BackendStatus = TEXT("ready");
    }
}

void AQaiConveyorWorld::ToggleCollisionDebug()
{
    bDrawCollisionDebug = !bDrawCollisionDebug;
    if (!bDrawCollisionDebug && GetWorld())
    {
        if (ULineBatchComponent* Lines = GetWorld()->GetLineBatcher(UWorld::ELineBatcherType::Foreground))
        {
            Lines->Flush();
        }
    }
    const FString State = bDrawCollisionDebug ? TEXT("on") : TEXT("off");
    UE_LOG(LogTemp, Display, TEXT("Qai collision debug %s"), *State);
    SimulatorLog(FString::Printf(TEXT("collision_debug state=%s"), *State));
}

void AQaiConveyorWorld::DrawCollisionDebug() const
{
    if (!bDrawCollisionDebug || !GetWorld())
    {
        return;
    }
    ULineBatchComponent* Lines = GetWorld()->GetLineBatcher(UWorld::ELineBatcherType::Foreground);
    if (!Lines)
    {
        return;
    }
    constexpr float LifeTime = 0.05f;
    constexpr uint8 DepthPriority = 1;

    // Magenta identifies fitted static blockers. Every proxy retains its real
    // height and orientation; the conveyor uses short tangent-aligned pieces.
    for (const FCollisionObstacle& Obstacle : CollisionObstacles)
    {
        Lines->DrawBox(
            Obstacle.Center,
            Obstacle.HalfExtent,
            Obstacle.Rotation,
            FLinearColor(1.0f, 0.0f, 1.0f),
            LifeTime,
            DepthPriority,
            2.0f);
    }

    // Green identifies compound moving bodies: chassis/cab, mast, individual
    // tines, and wheels. Capsules keep walking workers from snagging edges.
    for (int32 ForkliftIndex = 0; ForkliftIndex < ConveyorTuning::EnabledForkliftCount; ++ForkliftIndex)
    {
        const FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        if (const USceneComponent* Root = Forklift.Root.Get())
        {
            const FQuat RootQuat = Root->GetComponentQuat();
            for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
            {
                FVector LocalCenter = Shape.LocalCenter;
                LocalCenter.Z += Shape.bLiftDriven ? Forklift.LiftCm : 0.0f;
                const FVector ShapeCenter = Root->GetComponentLocation()
                    + RootQuat.RotateVector(LocalCenter);
                if (Shape.IsCylinder())
                {
                    const FVector Axle = RootQuat.GetRightVector() * Shape.CylinderHalfLength;
                    DrawDebugCylinder(
                        GetWorld(),
                        ShapeCenter - Axle,
                        ShapeCenter + Axle,
                        Shape.Radius,
                        18,
                        FColor(0, 255, 31),
                        false,
                        LifeTime,
                        DepthPriority,
                        3.0f);
                }
                else if (Shape.IsWedge())
                {
                    // Draw the actual tapered prism instead of its broad-phase
                    // box so F8 shows the same ramp used by contact solving.
                    const float TipX = Shape.bWedgeTipAtPositiveX
                        ? Shape.HalfExtent.X
                        : -Shape.HalfExtent.X;
                    const float BackX = -TipX;
                    const float BottomZ = -Shape.HalfExtent.Z;
                    const float BackTopZ = Shape.HalfExtent.Z;
                    const float TipTopZ = BottomZ + Shape.WedgeTipThicknessCm;
                    const auto ToWorld = [&RootQuat, Root, &LocalCenter](
                        float X,
                        float Y,
                        float Z)
                    {
                        return Root->GetComponentLocation()
                            + RootQuat.RotateVector(LocalCenter + FVector(X, Y, Z));
                    };
                    FVector Vertices[8] = {
                        ToWorld(BackX, -Shape.HalfExtent.Y, BottomZ),
                        ToWorld(BackX, -Shape.HalfExtent.Y, BackTopZ),
                        ToWorld(TipX, -Shape.HalfExtent.Y, TipTopZ),
                        ToWorld(TipX, -Shape.HalfExtent.Y, BottomZ),
                        ToWorld(BackX, Shape.HalfExtent.Y, BottomZ),
                        ToWorld(BackX, Shape.HalfExtent.Y, BackTopZ),
                        ToWorld(TipX, Shape.HalfExtent.Y, TipTopZ),
                        ToWorld(TipX, Shape.HalfExtent.Y, BottomZ),
                    };
                    constexpr int32 EdgePairs[12][2] = {
                        {0, 1}, {1, 2}, {2, 3}, {3, 0},
                        {4, 5}, {5, 6}, {6, 7}, {7, 4},
                        {0, 4}, {1, 5}, {2, 6}, {3, 7},
                    };
                    for (const auto& Edge : EdgePairs)
                    {
                        DrawDebugLine(
                            GetWorld(),
                            Vertices[Edge[0]],
                            Vertices[Edge[1]],
                            FColor(0, 255, 31),
                            false,
                            LifeTime,
                            DepthPriority,
                            3.0f);
                    }
                }
                else
                {
                    Lines->DrawBox(
                        ShapeCenter,
                        Shape.HalfExtent,
                        RootQuat,
                        FLinearColor(0.0f, 1.0f, 0.12f),
                        LifeTime,
                        DepthPriority,
                        3.0f);
                }
            }
            const FVector ForkliftCenterOfMass = Root->GetComponentTransform()
                .TransformPositionNoScale(Forklift.CombinedCenterOfMassLocalCm);
            DrawDebugSphere(
                GetWorld(),
                ForkliftCenterOfMass,
                6.0f,
                10,
                FColor(150, 80, 255),
                false,
                LifeTime,
                DepthPriority,
                2.5f);
        }
    }
    for (const FWorkerRuntime& Worker : Workers)
    {
        if (const USceneComponent* Root = Worker.Root.Get())
        {
            Lines->DrawCapsule(
                Root->GetComponentLocation() + FVector(0.0f, 0.0f, 90.0f),
                90.0f,
                ConveyorTuning::WorkerRadiusCm,
                FQuat::Identity,
                FLinearColor(0.0f, 1.0f, 0.12f),
                LifeTime,
                DepthPriority,
                3.0f);
        }
    }

    // Yellow identifies independent gravity bodies: pallet/carton pairs and
    // the 24 cartons reconstructed from the original shelf rigid bodies.
    for (const FDynamicBoxRuntime& Body : DynamicBoxes)
    {
        if (const USceneComponent* Root = Body.Root.Get())
        {
            Lines->DrawBox(
                Root->GetComponentLocation(),
                Body.HalfExtent,
                Root->GetComponentQuat(),
                FLinearColor(1.0f, 0.72f, 0.0f),
                LifeTime,
                DepthPriority,
                2.5f);
            DrawDebugSphere(
                GetWorld(),
                Root->GetComponentLocation()
                    + Root->GetComponentQuat().RotateVector(
                        Body.Physics.CenterOfMassLocalOffset),
                2.5f,
                8,
                FColor(255, 245, 120),
                false,
                LifeTime,
                DepthPriority,
                2.0f);
        }
    }

    // Cyan parcel boxes use the same moving pivot and rotation as the visual.
    for (int32 ParcelIndex = 0; ParcelIndex < Parcels.Num(); ++ParcelIndex)
    {
        const USceneComponent* ParcelRoot = Parcels[ParcelIndex].Get();
        if (!ParcelRoot || !ParcelHalfExtents.IsValidIndex(ParcelIndex))
        {
            continue;
        }
        Lines->DrawBox(
            ParcelRoot->GetComponentLocation(),
            ParcelHalfExtents[ParcelIndex],
            ParcelRoot->GetComponentQuat(),
            FLinearColor(0.0f, 0.9f, 1.0f),
            LifeTime,
            DepthPriority,
            2.0f);
        if (ParcelPhysicsProfiles.IsValidIndex(ParcelIndex))
        {
            DrawDebugSphere(
                GetWorld(),
                ParcelRoot->GetComponentLocation()
                    + ParcelRoot->GetComponentQuat().RotateVector(
                        ParcelPhysicsProfiles[ParcelIndex].CenterOfMassLocalOffset),
                2.5f,
                8,
                FColor(170, 255, 255),
                false,
                LifeTime,
                DepthPriority,
                2.0f);
        }
    }
}

FVector AQaiConveyorWorld::GetActiveForkliftLocation() const
{
    return Forklifts[ActiveForklift].Root.IsValid() ? Forklifts[ActiveForklift].Root->GetComponentLocation() : FVector::ZeroVector;
}

FVector AQaiConveyorWorld::GetEvkLocation() const
{
    return EvkVisualRoot.IsValid() ? EvkVisualRoot->GetComponentLocation() : FVector::ZeroVector;
}

float AQaiConveyorWorld::GetActiveSpeedMetersPerSecond() const
{
    return Forklifts[ActiveForklift].SpeedCmPerSecond / 100.0f;
}

float AQaiConveyorWorld::GetActiveLiftMeters() const
{
    return Forklifts[ActiveForklift].LiftCm / 100.0f;
}

void AQaiConveyorWorld::FixedSimulationStep(float StepSeconds)
{
    SimulateForklifts(StepSeconds);
    SimulateDynamicBoxes(StepSeconds);
    SimulateConveyor(StepSeconds);
    SimulateWorkers(StepSeconds);
    UpdateSafetySignal();
}

bool AQaiConveyorWorld::WouldForkliftsOverlap(
    int32 MovingIndex,
    const FVector& CandidateLocation,
    const FRotator& CandidateRotation) const
{
    const int32 OtherIndex = MovingIndex == 0 ? 1 : 0;
    const USceneComponent* OtherRoot = Forklifts[OtherIndex].Root.Get();
    if (!OtherRoot)
    {
        return false;
    }

    const FForkliftRuntime& Moving = Forklifts[MovingIndex];
    const FForkliftRuntime& Other = Forklifts[OtherIndex];
    const FQuat CandidateQuat = CandidateRotation.Quaternion();
    const FQuat OtherQuat = OtherRoot->GetComponentQuat();
    for (const FFittedCollisionBox& MovingShape : Moving.CollisionBoxes)
    {
        FVector MovingLocalCenter = MovingShape.LocalCenter;
        MovingLocalCenter.Z += MovingShape.bLiftDriven ? Moving.LiftCm : 0.0f;
        const FVector MovingCenter = CandidateLocation + CandidateQuat.RotateVector(MovingLocalCenter);
        for (const FFittedCollisionBox& OtherShape : Other.CollisionBoxes)
        {
            FVector OtherLocalCenter = OtherShape.LocalCenter;
            OtherLocalCenter.Z += OtherShape.bLiftDriven ? Other.LiftCm : 0.0f;
            const FVector OtherCenter = OtherRoot->GetComponentLocation() + OtherQuat.RotateVector(OtherLocalCenter);
            const bool bOverlaps = ConveyorTuning::ObbOverlapsObb(
                MovingCenter,
                CandidateQuat,
                MovingShape.GetCollisionHalfExtent(),
                OtherCenter,
                OtherQuat,
                OtherShape.GetCollisionHalfExtent());
            if (bOverlaps)
            {
                return true;
            }
        }
    }
    return false;
}

void AQaiConveyorWorld::SimulateForklifts(float StepSeconds)
{
    const auto CalculateWheelRidePose = [this](
        int32 ForkliftIndex,
        const FVector& CandidateLocation,
        const FRotator& CandidateRotation,
        float (&OutWheelOffsets)[4],
        FString& OutSupportName,
        int32& OutWheelIndex)
    {
        OutSupportName.Reset();
        OutWheelIndex = INDEX_NONE;
        const FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        const float GroundRootZ = Forklift.InitialRoot.GetLocation().Z;
        FVector GroundLocation = CandidateLocation;
        GroundLocation.Z = GroundRootZ;
        const FQuat CandidateQuat = CandidateRotation.Quaternion();
        float MaximumOffset = 0.0f;
        int32 WheelOrdinal = 0;
        for (float& Offset : OutWheelOffsets)
        {
            Offset = 0.0f;
        }

        for (int32 ShapeIndex = 0; ShapeIndex < Forklift.CollisionBoxes.Num(); ++ShapeIndex)
        {
            const FFittedCollisionBox& Wheel = Forklift.CollisionBoxes[ShapeIndex];
            if (!Wheel.IsCylinder())
            {
                continue;
            }
            const int32 CurrentWheel = FMath::Clamp(WheelOrdinal++, 0, 3);
            const FVector GroundWheelCenter = GroundLocation
                + CandidateQuat.RotateVector(Wheel.LocalCenter);
            const float GroundContactZ = GroundWheelCenter.Z - Wheel.Radius;
            const auto ConsiderSupport = [
                &MaximumOffset,
                &OutWheelOffsets,
                &OutSupportName,
                &OutWheelIndex,
                &GroundWheelCenter,
                &GroundContactZ,
                &CandidateQuat,
                &Wheel,
                ShapeIndex,
                CurrentWheel](
                    const FString& Name,
                    const FVector& BoxCenter,
                    const FQuat& BoxRotation,
                    const FVector& BoxHalfExtent)
            {
                const float BoxTop = BoxCenter.Z + BoxHalfExtent.Z;
                const float StepHeight = BoxTop - GroundContactZ;
                // Vertical faces significantly taller than the wheel radius
                // remain blockers/push contacts. Low pallets, boards and
                // parcels can become rolling wheel supports.
                if (StepHeight <= 0.5f || StepHeight > Wheel.Radius * 0.92f)
                {
                    return;
                }
                float LongitudinalGap = 0.0f;
                if (!ConveyorTuning::WheelDiskOverlapsSupport2D(
                    GroundWheelCenter,
                    CandidateQuat,
                    Wheel.Radius,
                    Wheel.CylinderHalfLength,
                    BoxCenter,
                    BoxRotation,
                    BoxHalfExtent,
                    LongitudinalGap))
                {
                    return;
                }
                const float VerticalArc = FMath::Sqrt(FMath::Max(
                    0.0f,
                    FMath::Square(Wheel.Radius) - FMath::Square(LongitudinalGap)));
                const float RequiredOffset = BoxTop + VerticalArc - GroundWheelCenter.Z;
                OutWheelOffsets[CurrentWheel] = FMath::Max(
                    OutWheelOffsets[CurrentWheel],
                    RequiredOffset);
                if (RequiredOffset > MaximumOffset)
                {
                    MaximumOffset = RequiredOffset;
                    OutSupportName = Name;
                    OutWheelIndex = ShapeIndex;
                }
            };

            for (int32 BodyIndex = 0; BodyIndex < DynamicBoxes.Num(); ++BodyIndex)
            {
                const FDynamicBoxRuntime& Body = DynamicBoxes[BodyIndex];
                const USceneComponent* Root = Body.Root.Get();
                if (!Root || Body.SupportedForklift == ForkliftIndex)
                {
                    continue;
                }
                ConsiderSupport(
                    Body.Name,
                    Root->GetComponentLocation(),
                    Root->GetComponentQuat(),
                    Body.HalfExtent);
            }
            for (int32 ParcelIndex = 0; ParcelIndex < Parcels.Num(); ++ParcelIndex)
            {
                const USceneComponent* Parcel = Parcels[ParcelIndex].Get();
                if (!Parcel || !ParcelHalfExtents.IsValidIndex(ParcelIndex)
                    || (ParcelSupportedForklifts.IsValidIndex(ParcelIndex)
                        && ParcelSupportedForklifts[ParcelIndex] == ForkliftIndex))
                {
                    continue;
                }
                ConsiderSupport(
                    FString::Printf(TEXT("conveyor parcel %d"), ParcelIndex + 1),
                    Parcel->GetComponentLocation(),
                    Parcel->GetComponentQuat(),
                    ParcelHalfExtents[ParcelIndex]);
            }
        }
        return FMath::Max(0.0f, MaximumOffset);
    };

    const auto FindForkLiftContact = [this](
        int32 ForkliftIndex,
        float ProposedLiftCm,
        FString& OutObstacle)
    {
        OutObstacle.Reset();
        const FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        const USceneComponent* Root = Forklift.Root.Get();
        if (!Root)
        {
            return false;
        }
        const FQuat RootQuat = Root->GetComponentQuat();
        for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
        {
            if (!Shape.bLiftDriven
                || (!Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase)
                    && !Shape.Name.Contains(TEXT("fork heel"), ESearchCase::IgnoreCase)))
            {
                continue;
            }
            FVector LocalCenter = Shape.LocalCenter;
            LocalCenter.Z += ProposedLiftCm;
            const FVector ShapeCenter = Root->GetComponentLocation() + RootQuat.RotateVector(LocalCenter);
            for (const FCollisionObstacle& Obstacle : CollisionObstacles)
            {
                // ShelfPlane is a hidden carton support sheet, not the rack's
                // physical beams. The visible shelf frame, packing table and
                // segmented conveyor guards are the hydraulic contact bodies.
                if (Obstacle.Name != TEXT("shelf frame")
                    && Obstacle.Name != TEXT("packing table")
                    && Obstacle.Name != TEXT("conveyor"))
                {
                    continue;
                }
                if (ConveyorTuning::ObbOverlapsObb(
                        ShapeCenter,
                        RootQuat,
                        Shape.HalfExtent,
                        Obstacle.Center,
                        Obstacle.Rotation,
                        Obstacle.HalfExtent,
                        0.5f))
                {
                    OutObstacle = Obstacle.Name;
                    return true;
                }
            }
        }
        return false;
    };

    const auto MeasureCollisionPenetration = [this](
        int32 ForkliftIndex,
        const FVector& CandidateLocation,
        const FRotator& CandidateRotation,
        const FString& ObstacleName)
    {
        const FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        const FQuat CandidateQuat = CandidateRotation.Quaternion();
        float MaximumPenetration = 0.0f;
        for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
        {
            FVector LocalCenter = Shape.LocalCenter;
            LocalCenter.Z += Shape.bLiftDriven ? Forklift.LiftCm : 0.0f;
            const FVector ShapeCenter = CandidateLocation + CandidateQuat.RotateVector(LocalCenter);
            const FVector ShapeExtent = Shape.GetCollisionHalfExtent();

            if (ObstacleName == TEXT("factory boundary"))
            {
                const FVector Forward = CandidateQuat.GetForwardVector();
                const FVector Right = CandidateQuat.GetRightVector();
                const float WorldHalfX = FMath::Abs(Forward.X) * ShapeExtent.X
                    + FMath::Abs(Right.X) * ShapeExtent.Y;
                const float WorldHalfY = FMath::Abs(Forward.Y) * ShapeExtent.X
                    + FMath::Abs(Right.Y) * ShapeExtent.Y;
                const float BoundaryPenetration = FMath::Max(
                    FMath::Max(
                        ConveyorTuning::WorldMinimumX - (ShapeCenter.X - WorldHalfX),
                        (ShapeCenter.X + WorldHalfX) - ConveyorTuning::WorldMaximumX),
                    FMath::Max(
                        ConveyorTuning::WorldMinimumY - (ShapeCenter.Y - WorldHalfY),
                        (ShapeCenter.Y + WorldHalfY) - ConveyorTuning::WorldMaximumY));
                MaximumPenetration = FMath::Max(MaximumPenetration, BoundaryPenetration);
                continue;
            }

            for (const FCollisionObstacle& Obstacle : CollisionObstacles)
            {
                if (Obstacle.Name != ObstacleName
                    || (Obstacle.Name == TEXT("shelf plane")
                        && Shape.Name.Contains(TEXT("tine"), ESearchCase::IgnoreCase))
                    || (Obstacle.Name == TEXT("packing table")
                        && Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase)))
                {
                    continue;
                }
                MaximumPenetration = FMath::Max(
                    MaximumPenetration,
                    ConveyorTuning::ObbPlanarPenetrationCm(
                        ShapeCenter,
                        CandidateQuat,
                        ShapeExtent,
                        Obstacle.Center,
                        Obstacle.Rotation,
                        Obstacle.HalfExtent));
            }

            for (int32 WorkerIndex = 0; WorkerIndex < UE_ARRAY_COUNT(Workers); ++WorkerIndex)
            {
                if (ObstacleName != FString::Printf(TEXT("worker %d"), WorkerIndex + 1))
                {
                    continue;
                }
                if (const USceneComponent* Worker = Workers[WorkerIndex].Root.Get())
                {
                    const FVector LocalWorker = CandidateQuat.Inverse().RotateVector(
                        Worker->GetComponentLocation() - ShapeCenter);
                    const float PenetrationX = ShapeExtent.X
                        + ConveyorTuning::WorkerRadiusCm + ConveyorTuning::CollisionMarginCm
                        - FMath::Abs(LocalWorker.X);
                    const float PenetrationY = ShapeExtent.Y
                        + ConveyorTuning::WorkerRadiusCm + ConveyorTuning::CollisionMarginCm
                        - FMath::Abs(LocalWorker.Y);
                    if (PenetrationX > 0.0f && PenetrationY > 0.0f)
                    {
                        MaximumPenetration = FMath::Max(
                            MaximumPenetration,
                            FMath::Min(PenetrationX, PenetrationY));
                    }
                }
            }
        }
        return FMath::Max(0.0f, MaximumPenetration);
    };

    for (int32 Index = 0; Index < ConveyorTuning::EnabledForkliftCount; ++Index)
    {
        FForkliftRuntime& Forklift = Forklifts[Index];
        if (!Forklift.Root.IsValid())
        {
            continue;
        }

        const FTransform ForkliftTransform = Forklift.Root->GetComponentTransform();
        float CombinedMassKg = Forklift.ChassisMassKg;
        FVector WeightedCenterOfMass = Forklift.BaseCenterOfMassLocalCm
            * Forklift.ChassisMassKg;
        const auto AddSupportedMass = [
            &ForkliftTransform,
            &CombinedMassKg,
            &WeightedCenterOfMass](
                const USceneComponent* BodyRoot,
                const FQuat& BodyRotation,
                const FPropPhysicsProfile& Profile)
        {
            if (!BodyRoot)
            {
                return;
            }
            const FVector WorldCenterOfMass = BodyRoot->GetComponentLocation()
                + BodyRotation.RotateVector(Profile.CenterOfMassLocalOffset);
            const FVector LocalCenterOfMass = ForkliftTransform.InverseTransformPositionNoScale(
                WorldCenterOfMass);
            CombinedMassKg += Profile.MassKg;
            WeightedCenterOfMass += LocalCenterOfMass * Profile.MassKg;
        };
        for (const FDynamicBoxRuntime& Body : DynamicBoxes)
        {
            if (Body.SupportedForklift == Index)
            {
                if (const USceneComponent* BodyRoot = Body.Root.Get())
                {
                    AddSupportedMass(BodyRoot, BodyRoot->GetComponentQuat(), Body.Physics);
                }
            }
        }
        for (int32 ParcelIndex = 0; ParcelIndex < Parcels.Num(); ++ParcelIndex)
        {
            if (ParcelSupportedForklifts.IsValidIndex(ParcelIndex)
                && ParcelSupportedForklifts[ParcelIndex] == Index
                && ParcelPhysicsProfiles.IsValidIndex(ParcelIndex))
            {
                if (const USceneComponent* Parcel = Parcels[ParcelIndex].Get())
                {
                    AddSupportedMass(
                        Parcel,
                        Parcel->GetComponentQuat(),
                        ParcelPhysicsProfiles[ParcelIndex]);
                }
            }
        }
        Forklift.CombinedMassKg = CombinedMassKg;
        Forklift.CombinedCenterOfMassLocalCm = WeightedCenterOfMass
            / FMath::Max(CombinedMassKg, 1.0f);

        const bool bControlled = Index == ActiveForklift;
        const float Throttle = bControlled ? CommandThrottle : 0.0f;
        const bool bBrake = bControlled && bCommandBrake;
        const float TargetSpeed = bBrake ? 0.0f : Throttle * ConveyorTuning::MaxSpeedCm;
        const float ChangeRate = bBrake ? ConveyorTuning::BrakingCm : ConveyorTuning::AccelerationCm;
        const float PreviousSpeedCmPerSecond = Forklift.PreviousSpeedCmPerSecond;
        Forklift.SpeedCmPerSecond = FMath::FInterpConstantTo(
            Forklift.SpeedCmPerSecond,
            TargetSpeed,
            StepSeconds,
            ChangeRate);
        const float RequestedSteering = bControlled ? CommandSteer : 0.0f;
        const float SteeringRate = FMath::Abs(RequestedSteering) > UE_SMALL_NUMBER
            ? ConveyorTuning::SteeringRisePerSecond
            : ConveyorTuning::SteeringReturnPerSecond;
        Forklift.SteeringInput = FMath::FInterpConstantTo(
            Forklift.SteeringInput,
            RequestedSteering,
            StepSeconds,
            SteeringRate);

        const float PreviousLiftCm = Forklift.LiftCm;
        const float ProposedLiftCm = bControlled
            ? FMath::Clamp(
                Forklift.LiftCm + CommandLift * ConveyorTuning::LiftSpeedCm * StepSeconds,
                0.0f,
                ConveyorTuning::MaxLiftCm)
            : Forklift.LiftCm;
        FString ForkReactionObstacle;
        const bool bForkReaction = bControlled
            && FMath::Abs(CommandLift) > 0.05f
            && FindForkLiftContact(Index, ProposedLiftCm, ForkReactionObstacle);
        if (!bForkReaction)
        {
            Forklift.LiftCm = ProposedLiftCm;
        }
        else
        {
            // The lift is kinematic, but its hydraulic reaction is not free:
            // pressing the tines into a shelf/conveyor pushes the front of
            // the chassis in the opposite vertical direction and can tip it.
            const float InertiaScale = FMath::Clamp(
                Forklift.ChassisMassKg / FMath::Max(Forklift.CombinedMassKg, 1.0f),
                0.55f,
                1.0f);
            Forklift.PitchVelocityDegrees += -CommandLift
                * 135.0f
                * InertiaScale
                * StepSeconds;
        }
        if (bForkReaction != Forklift.bForkReactionActive)
        {
            Forklift.bForkReactionActive = bForkReaction;
            SimulatorLog(FString::Printf(
                TEXT("forklift_hydraulic_reaction forklift=%d state=%s obstacle=%s lift_command=%+.2f"),
                Index + 1,
                bForkReaction ? TEXT("on") : TEXT("off"),
                ForkReactionObstacle.IsEmpty() ? TEXT("none") : *ForkReactionObstacle,
                CommandLift));
        }
        Forklift.LiftDeltaCm = Forklift.LiftCm - PreviousLiftCm;

        const float SteerRadians = Forklift.SteeringInput * ConveyorTuning::MaxSteerRadians;
        const FVector CurrentLocation = Forklift.Root->GetComponentLocation();
        const FRotator CurrentRotation = Forklift.Root->GetComponentRotation();
        const FRotator CurrentBaseRotation(0.0f, CurrentRotation.Yaw, 0.0f);
        FString ExistingOverlap;
        const bool bStartsOverlapping = WouldForkliftCollide(
            Index,
            CurrentLocation,
            CurrentBaseRotation,
            ExistingOverlap);
        const float ExistingPenetration = bStartsOverlapping
            ? MeasureCollisionPenetration(Index, CurrentLocation, CurrentBaseRotation, ExistingOverlap)
            : 0.0f;
        float YawRadiansPerSecond = (Forklift.SpeedCmPerSecond / ConveyorTuning::WheelbaseCm) * FMath::Tan(SteerRadians);

        FString CollisionObstacle;
        const float RequestedDriveSign = FMath::Abs(Throttle) > 0.05f ? FMath::Sign(Throttle) : 0.0f;
        // A forklift's rear wheels steer while its front wheels stay aligned
        // with the chassis. Integrate the front axle forward and derive the
        // chassis center from it; this makes the rear swing around the fixed
        // front axle instead of following passenger-car/front-steer geometry.
        const float HalfWheelbaseCm = ConveyorTuning::WheelbaseCm * 0.5f;
        const FVector CurrentFrontAxle = CurrentLocation
            + CurrentBaseRotation.Vector() * HalfWheelbaseCm;

        FRotator CandidateRotation = CurrentBaseRotation;
        CandidateRotation.Yaw = FRotator::NormalizeAxis(
            CandidateRotation.Yaw + FMath::RadiansToDegrees(YawRadiansPerSecond * StepSeconds));
        FRotator CandidateHeading = CurrentBaseRotation;
        CandidateHeading.Yaw = FRotator::NormalizeAxis(
            CurrentBaseRotation.Yaw
                + 0.5f * FRotator::NormalizeAxis(CandidateRotation.Yaw - CurrentBaseRotation.Yaw));
        const FVector CandidateFrontAxle = CurrentFrontAxle
            + CandidateHeading.Vector() * Forklift.SpeedCmPerSecond * StepSeconds;
        FVector Next = CandidateFrontAxle - CandidateRotation.Vector() * HalfWheelbaseCm;
        Next.Z = CurrentLocation.Z;
        FString ImmediateObstacle;
        const bool bCandidateCollides = WouldForkliftCollide(Index, Next, CandidateRotation, ImmediateObstacle);
        const float CandidatePenetration = bCandidateCollides
            ? MeasureCollisionPenetration(Index, Next, CandidateRotation, ImmediateObstacle)
            : 0.0f;
        // Imported authored layouts can begin with conservative 2-D proxies
        // touching by a few centimetres. Never trap a vehicle in that state:
        // allow only motion that holds or reduces the penetration. The former
        // name-only rule allowed a vehicle to continue deeper through another
        // segment of the same conveyor because every segment is named
        // "conveyor". This depth constraint preserves tangential sliding and
        // backing out without permitting progress through a solid obstacle.
        const bool bEscapingExistingOverlap = bStartsOverlapping
            && bCandidateCollides
            && ImmediateObstacle == ExistingOverlap
            && CandidatePenetration <= ExistingPenetration + 0.001f
            && FVector::DistSquared2D(CurrentLocation, Next) > UE_KINDA_SMALL_NUMBER;
        if (bCandidateCollides && !bEscapingExistingOverlap)
        {
            CollisionObstacle = ImmediateObstacle;
            // A contact constrains this integration step, not the throttle.
            // Keeping signed wheel speed alive means the opposite trigger can
            // pull the truck away instead of waiting for a collision latch or
            // an artificial automatic brake to release.
            Next = CurrentLocation;
            CandidateRotation = CurrentBaseRotation;
            if (RequestedDriveSign != 0.0f)
            {
                Forklift.LastBlockedDriveSign = RequestedDriveSign;
                Forklift.LastBlockedObstacle = ImmediateObstacle;
            }
        }
        else if (!bCandidateCollides && Forklift.LastBlockedDriveSign != 0.0f)
        {
            SimulatorLog(FString::Printf(
                TEXT("forklift_collision_escape forklift=%d obstacle=%s throttle_sign=%.0f"),
                Index + 1,
                *Forklift.LastBlockedObstacle,
                RequestedDriveSign));
            Forklift.LastBlockedDriveSign = 0.0f;
            Forklift.LastBlockedObstacle.Reset();
        }

        float TargetWheelOffsets[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        FString RideSupportName;
        int32 RideWheelIndex = INDEX_NONE;
        const float MaximumTargetRideHeight = CalculateWheelRidePose(
            Index,
            Next,
            CandidateRotation,
            TargetWheelOffsets,
            RideSupportName,
            RideWheelIndex);
        for (int32 WheelIndex = 0; WheelIndex < 4; ++WheelIndex)
        {
            Forklift.WheelSupportOffsetsCm[WheelIndex] = FMath::FInterpTo(
                Forklift.WheelSupportOffsetsCm[WheelIndex],
                TargetWheelOffsets[WheelIndex],
                StepSeconds,
                TargetWheelOffsets[WheelIndex] > Forklift.WheelSupportOffsetsCm[WheelIndex]
                    ? 24.0f
                    : 10.0f);
        }
        const float FrontRide = 0.5f * (
            Forklift.WheelSupportOffsetsCm[0] + Forklift.WheelSupportOffsetsCm[1]);
        const float RearRide = 0.5f * (
            Forklift.WheelSupportOffsetsCm[2] + Forklift.WheelSupportOffsetsCm[3]);
        const float LeftRide = 0.5f * (
            Forklift.WheelSupportOffsetsCm[0] + Forklift.WheelSupportOffsetsCm[2]);
        const float RightRide = 0.5f * (
            Forklift.WheelSupportOffsetsCm[1] + Forklift.WheelSupportOffsetsCm[3]);
        Forklift.RideHeightOffsetCm = 0.25f * (
            FrontRide + RearRide + LeftRide + RightRide);
        Next.Z = Forklift.InitialRoot.GetLocation().Z + Forklift.RideHeightOffsetCm;

        float TargetPitch = FMath::RadiansToDegrees(FMath::Atan2(
            FrontRide - RearRide,
            ConveyorTuning::WheelbaseCm));
        constexpr float ApproximateTrackWidthCm = 112.0f;
        float TargetRoll = FMath::RadiansToDegrees(FMath::Atan2(
            RightRide - LeftRide,
            ApproximateTrackWidthCm));
        const float LongitudinalAccelerationCm = StepSeconds > UE_SMALL_NUMBER
            ? (Forklift.SpeedCmPerSecond - PreviousSpeedCmPerSecond) / StepSeconds
            : 0.0f;
        const float LateralAccelerationCm = Forklift.SpeedCmPerSecond * YawRadiansPerSecond;
        const float CenterOfMassHeightScale = FMath::Clamp(
            FMath::Abs(Forklift.CombinedCenterOfMassLocalCm.Z)
                / FMath::Max(FMath::Abs(Forklift.BaseCenterOfMassLocalCm.Z), 20.0f),
            0.70f,
            2.50f);
        // Industrial forklifts use a stiff front axle and a constrained rear
        // articulation. Keep acceleration weight transfer subtle; the prior
        // passenger-car-like values visibly lifted an axle even though all
        // four tire contacts were on flat concrete.
        TargetPitch += -FMath::RadiansToDegrees(FMath::Atan2(
            LongitudinalAccelerationCm,
            ConveyorTuning::GravityCmPerSecondSquared))
            * CenterOfMassHeightScale
            * 0.10f;
        TargetRoll += FMath::RadiansToDegrees(FMath::Atan2(
            LateralAccelerationCm,
            ConveyorTuning::GravityCmPerSecondSquared))
            * CenterOfMassHeightScale
            * 0.18f;
        const FVector CargoCenterShift = Forklift.CombinedCenterOfMassLocalCm
            - Forklift.BaseCenterOfMassLocalCm;
        if (FMath::Abs(CargoCenterShift.X) > 45.0f)
        {
            TargetPitch += FMath::Sign(CargoCenterShift.X)
                * (FMath::Abs(CargoCenterShift.X) - 45.0f)
                * 0.055f;
        }
        if (FMath::Abs(CargoCenterShift.Y) > 30.0f)
        {
            TargetRoll += FMath::Sign(CargoCenterShift.Y)
                * (FMath::Abs(CargoCenterShift.Y) - 30.0f)
                * 0.070f;
        }
        const float LoadedResponse = FMath::Clamp(
            Forklift.ChassisMassKg / FMath::Max(Forklift.CombinedMassKg, 1.0f),
            0.55f,
            1.0f);
        Forklift.PitchVelocityDegrees += (
            (TargetPitch - Forklift.BodyPitchDegrees) * (25.0f + 5.0f * LoadedResponse)
            - Forklift.PitchVelocityDegrees * (8.5f + LoadedResponse)) * StepSeconds;
        Forklift.RollVelocityDegrees += (
            (TargetRoll - Forklift.BodyRollDegrees) * (23.0f + 5.0f * LoadedResponse)
            - Forklift.RollVelocityDegrees * (8.0f + LoadedResponse)) * StepSeconds;
        Forklift.BodyPitchDegrees = FMath::Clamp(
            Forklift.BodyPitchDegrees + Forklift.PitchVelocityDegrees * StepSeconds,
            -18.0f,
            18.0f);
        Forklift.BodyRollDegrees = FMath::Clamp(
            Forklift.BodyRollDegrees + Forklift.RollVelocityDegrees * StepSeconds,
            -14.0f,
            14.0f);
        Forklift.MaximumAbsoluteTipDegrees = FMath::Max(
            Forklift.MaximumAbsoluteTipDegrees,
            FMath::Max(FMath::Abs(Forklift.BodyPitchDegrees), FMath::Abs(Forklift.BodyRollDegrees)));
        CandidateRotation.Pitch = Forklift.BodyPitchDegrees;
        CandidateRotation.Roll = Forklift.BodyRollDegrees;
        Forklift.MaximumRideHeightCm = FMath::Max(
            Forklift.MaximumRideHeightCm,
            MaximumTargetRideHeight);
        Forklift.PreviousSpeedCmPerSecond = Forklift.SpeedCmPerSecond;
        const bool bNowClimbing = MaximumTargetRideHeight > 0.75f;
        if (bNowClimbing != Forklift.bWheelClimbActive)
        {
            Forklift.bWheelClimbActive = bNowClimbing;
            SimulatorLog(FString::Printf(
                TEXT("forklift_wheel_climb forklift=%d state=%s height_cm=%.2f pitch=%.2f roll=%.2f support=%s wheel_shape=%d"),
                Index + 1,
                bNowClimbing ? TEXT("on") : TEXT("off"),
                MaximumTargetRideHeight,
                Forklift.BodyPitchDegrees,
                Forklift.BodyRollDegrees,
                RideSupportName.IsEmpty() ? TEXT("floor") : *RideSupportName,
                RideWheelIndex + 1));
        }

        const float TravelCm = FVector::Dist2D(CurrentLocation, Next);
        Forklift.SurfaceLinearVelocityCmPerSecond = StepSeconds > UE_SMALL_NUMBER
            ? (Next - CurrentLocation) / StepSeconds
            : FVector::ZeroVector;
        Forklift.SurfaceLinearVelocityCmPerSecond.Z += StepSeconds > UE_SMALL_NUMBER
            ? Forklift.LiftDeltaCm / StepSeconds
            : 0.0f;
        Forklift.SurfaceYawVelocityDegreesPerSecond = StepSeconds > UE_SMALL_NUMBER
            ? FRotator::NormalizeAxis(CandidateRotation.Yaw - CurrentRotation.Yaw) / StepSeconds
            : 0.0f;
        if (FVector::DistSquared(CurrentLocation, Next) > UE_KINDA_SMALL_NUMBER
            || !CandidateRotation.Equals(CurrentRotation, UE_KINDA_SMALL_NUMBER))
        {
            Forklift.Root->SetWorldLocationAndRotation(Next, CandidateRotation, false, nullptr, ETeleportType::TeleportPhysics);
            Forklift.WheelAngleDegrees = FMath::Fmod(
                Forklift.WheelAngleDegrees + FMath::RadiansToDegrees(TravelCm / ConveyorTuning::ForkliftWheelRadiusCm)
                    * FMath::Sign(Forklift.SpeedCmPerSecond),
                360.0f);
        }
        if (bControlled)
        {
            UpdateCollisionStatus(Index, CollisionObstacle);
        }
        for (int32 WheelIndex = 0; WheelIndex < 4; ++WheelIndex)
        {
            if (USceneComponent* WheelPivot = Forklift.WheelPivots[WheelIndex].Get())
            {
                // Forklifts steer at the rear axle. The visual rear wheels
                // angle opposite the desired turn because the back of the
                // vehicle swings outward while the nose turns inward.
                const float WheelSteer = WheelIndex >= 2 ? -SteerRadians : 0.0f;
                const FQuat Steering(FVector::UpVector, WheelSteer);
                const FQuat Rolling(FVector::RightVector, FMath::DegreesToRadians(Forklift.WheelAngleDegrees));
                WheelPivot->SetRelativeRotation(Forklift.WheelPivotInitialRelative[WheelIndex] * Steering * Rolling);
            }
        }
        UpdateCargo(Forklift);
    }
}

void AQaiConveyorWorld::UpdateCargo(FForkliftRuntime& Forklift)
{
    if (!Forklift.Root.IsValid())
    {
        return;
    }
    const FTransform RootTransform = Forklift.Root->GetComponentTransform();
    if (USceneComponent* LiftAssembly = Forklift.LiftAssembly.Get())
    {
        LiftAssembly->SetRelativeLocation(
            Forklift.InitialLiftRelativeLocation + FVector(0.0f, 0.0f, Forklift.LiftCm),
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
    }
    for (int32 Index = 0; Index < Forklift.LiftDrivenComponents.Num(); ++Index)
    {
        USceneComponent* Driven = Forklift.LiftDrivenComponents[Index].Get();
        if (!Driven || !Forklift.LiftDrivenRelativeTransforms.IsValidIndex(Index))
        {
            continue;
        }
        FTransform Desired = Forklift.LiftDrivenRelativeTransforms[Index] * RootTransform;
        Desired.AddToTranslation(FVector(0.0f, 0.0f, Forklift.LiftCm));
        Driven->SetWorldTransform(Desired, false, nullptr, ETeleportType::TeleportPhysics);
        Driven->UpdateBounds();
    }
}

void AQaiConveyorWorld::SimulateDynamicBoxes(float StepSeconds)
{
    const auto HorizontalOverlap = [](const FDynamicBoxRuntime& Body, const FVector& BodyCenter, const FQuat& BodyQuat, const FCollisionObstacle& Obstacle)
    {
        FVector FlatBodyCenter = BodyCenter;
        FVector FlatObstacleCenter = Obstacle.Center;
        FlatBodyCenter.Z = 0.0f;
        FlatObstacleCenter.Z = 0.0f;
        FVector FlatBodyExtent = Body.HalfExtent;
        FVector FlatObstacleExtent = Obstacle.HalfExtent;
        FlatBodyExtent.Z = 10.0f;
        FlatObstacleExtent.Z = 10.0f;
        return ConveyorTuning::ObbOverlapsObb(
            FlatBodyCenter,
            BodyQuat,
            FlatBodyExtent,
            FlatObstacleCenter,
            Obstacle.Rotation,
            FlatObstacleExtent,
            0.0f);
    };

    const auto CarrierVelocityAtPoint = [](
        const FForkliftRuntime& Forklift,
        const FVector& WorldPoint)
    {
        FVector Velocity = Forklift.SurfaceLinearVelocityCmPerSecond;
        if (Forklift.Root.IsValid())
        {
            const FVector AngularVelocityRadians(
                0.0f,
                0.0f,
                FMath::DegreesToRadians(Forklift.SurfaceYawVelocityDegreesPerSecond));
            Velocity += FVector::CrossProduct(
                AngularVelocityRadians,
                WorldPoint - Forklift.Root->GetComponentLocation());
        }
        return Velocity;
    };

    // Contact with a moving fork surface uses finite Coulomb friction rather
    // than an attachment. Static friction can make a lightly accelerating
    // load stick; dynamic friction still lets it slide, overhang, and fall.
    // The carrier velocity is the resolved chassis/lift motion, so spinning
    // wheels against an obstacle cannot move a parcel through the world.
    const auto ApplyForkSurfaceFriction = [&CarrierVelocityAtPoint](
        FDynamicBoxRuntime& Body,
        const FForkliftRuntime& Forklift,
        float DeltaSeconds,
        float NormalLoadScale)
    {
        const float SurfaceStaticFriction = FMath::Min(Body.Physics.StaticFriction, 0.72f);
        const float SurfaceDynamicFriction = FMath::Min(Body.Physics.DynamicFriction, 0.56f);
        const FVector CarrierVelocity = CarrierVelocityAtPoint(
            Forklift,
            Body.Root.IsValid()
                ? Body.Root->GetComponentLocation()
                : FVector::ZeroVector);
        FVector RelativeVelocity = Body.LinearVelocity - CarrierVelocity;
        RelativeVelocity.Z = 0.0f;
        const float RelativeSpeed = RelativeVelocity.Size2D();
        const float StaticDeltaSpeed = SurfaceStaticFriction
            * ConveyorTuning::GravityCmPerSecondSquared
            * DeltaSeconds
            * NormalLoadScale;
        if (RelativeSpeed <= StaticDeltaSpeed)
        {
            Body.LinearVelocity.X = CarrierVelocity.X;
            Body.LinearVelocity.Y = CarrierVelocity.Y;
        }
        else if (RelativeSpeed > UE_SMALL_NUMBER)
        {
            const float NewRelativeSpeed = FMath::Max(
                0.0f,
                RelativeSpeed - SurfaceDynamicFriction
                    * ConveyorTuning::GravityCmPerSecondSquared
                    * DeltaSeconds
                    * NormalLoadScale);
            const FVector NewRelativeVelocity = RelativeVelocity
                * (NewRelativeSpeed / RelativeSpeed);
            Body.LinearVelocity.X = CarrierVelocity.X
                + NewRelativeVelocity.X;
            Body.LinearVelocity.Y = CarrierVelocity.Y
                + NewRelativeVelocity.Y;
        }
        Body.AngularVelocityDegrees.Z = FMath::FInterpConstantTo(
            Body.AngularVelocityDegrees.Z,
            Forklift.SurfaceYawVelocityDegreesPerSecond,
            DeltaSeconds,
            120.0f * NormalLoadScale);
    };

    struct FForkSupportEvaluation
    {
        bool bHasVerticalContact = false;
        bool bCenterOfMassSupported = false;
        int32 ContactTineCount = 0;
        float Coverage = 0.0f;
        float SupportTop = 0.0f;
        FVector OverhangDirection = FVector::ZeroVector;
    };

    // Fork cargo is only kinematically carried while its projected center of
    // mass remains inside the convex hull of the tines that are actually in
    // contact. A load bridging both tines may balance over their gap; a load
    // touching only one tine must balance over that tine rather than over an
    // invisible full-width platform.
    const auto EvaluateForkSupport = [&HorizontalOverlap](
        const FDynamicBoxRuntime& Body,
        const FTransform& BodyTransform,
        const FForkliftRuntime& Forklift,
        bool bAttachedCargo)
    {
        FForkSupportEvaluation Result;
        const USceneComponent* ForkliftRoot = Forklift.Root.Get();
        if (!ForkliftRoot)
        {
            return Result;
        }

        const FTransform ForkliftTransform = ForkliftRoot->GetComponentTransform();
        const FQuat ForkliftQuat = ForkliftTransform.GetRotation();
        const FVector BodyCenter = BodyTransform.GetLocation();
        const FQuat BodyQuat = BodyTransform.GetRotation();
        const float BodyBottom = BodyCenter.Z
            - ConveyorTuning::ProjectedVerticalHalfExtent(BodyQuat, Body.HalfExtent);
        TArray<FVector4> ContactRectangles;
        FVector2D SupportMin(TNumericLimits<float>::Max(), TNumericLimits<float>::Max());
        FVector2D SupportMax(-TNumericLimits<float>::Max(), -TNumericLimits<float>::Max());

        for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
        {
            if (Shape.IsWedge()
                || !Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase))
            {
                continue;
            }
            FVector LocalCenter = Shape.LocalCenter;
            LocalCenter.Z += Shape.bLiftDriven ? Forklift.LiftCm : 0.0f;
            const FVector ShapeCenter = ForkliftTransform.TransformPositionNoScale(LocalCenter);
            const FVector ShapeExtent = Shape.GetCollisionHalfExtent();
            FCollisionObstacle TineObstacle;
            TineObstacle.Center = ShapeCenter;
            TineObstacle.HalfExtent = ShapeExtent;
            TineObstacle.Rotation = ForkliftQuat;
            const float TineTop = ShapeCenter.Z
                + ConveyorTuning::ProjectedVerticalHalfExtent(ForkliftQuat, ShapeExtent);
            const float LowerContactToleranceCm = bAttachedCargo ? 10.0f : 4.0f;
            const float UpperContactToleranceCm = bAttachedCargo ? 14.0f : 9.0f;
            const bool bAtContactHeight = BodyBottom >= TineTop - LowerContactToleranceCm
                && BodyBottom <= TineTop + UpperContactToleranceCm;
            if (!bAtContactHeight
                || !HorizontalOverlap(Body, BodyCenter, BodyQuat, TineObstacle))
            {
                continue;
            }

            const FVector2D TineMin(LocalCenter.X - ShapeExtent.X, LocalCenter.Y - ShapeExtent.Y);
            const FVector2D TineMax(LocalCenter.X + ShapeExtent.X, LocalCenter.Y + ShapeExtent.Y);
            ContactRectangles.Emplace(TineMin.X, TineMax.X, TineMin.Y, TineMax.Y);
            SupportMin.X = FMath::Min(SupportMin.X, TineMin.X);
            SupportMin.Y = FMath::Min(SupportMin.Y, TineMin.Y);
            SupportMax.X = FMath::Max(SupportMax.X, TineMax.X);
            SupportMax.Y = FMath::Max(SupportMax.Y, TineMax.Y);
            Result.SupportTop = FMath::Max(Result.SupportTop, TineTop);
            ++Result.ContactTineCount;
        }

        Result.bHasVerticalContact = Result.ContactTineCount > 0;
        if (!Result.bHasVerticalContact)
        {
            return Result;
        }

        const FVector CenterOfMassWorld = BodyCenter
            + BodyQuat.RotateVector(Body.Physics.CenterOfMassLocalOffset);
        const FVector CenterOfMassLocal = ForkliftTransform.InverseTransformPositionNoScale(
            CenterOfMassWorld);
        constexpr float SupportMarginCm = 1.5f;
        Result.bCenterOfMassSupported = CenterOfMassLocal.X >= SupportMin.X - SupportMarginCm
            && CenterOfMassLocal.X <= SupportMax.X + SupportMarginCm
            && CenterOfMassLocal.Y >= SupportMin.Y - SupportMarginCm
            && CenterOfMassLocal.Y <= SupportMax.Y + SupportMarginCm;

        const FVector2D NearestSupportPoint(
            FMath::Clamp(CenterOfMassLocal.X, SupportMin.X, SupportMax.X),
            FMath::Clamp(CenterOfMassLocal.Y, SupportMin.Y, SupportMax.Y));
        const FVector LocalOverhang(
            CenterOfMassLocal.X - NearestSupportPoint.X,
            CenterOfMassLocal.Y - NearestSupportPoint.Y,
            0.0f);
        Result.OverhangDirection = ForkliftQuat.RotateVector(LocalOverhang).GetSafeNormal2D();

        constexpr int32 SamplesPerAxis = 13;
        int32 SupportedSamples = 0;
        const FVector BodyForward = BodyQuat.GetForwardVector();
        const FVector BodyRight = BodyQuat.GetRightVector();
        for (int32 X = 0; X < SamplesPerAxis; ++X)
        {
            const float LocalX = FMath::Lerp(
                -Body.HalfExtent.X,
                Body.HalfExtent.X,
                (static_cast<float>(X) + 0.5f) / SamplesPerAxis);
            for (int32 Y = 0; Y < SamplesPerAxis; ++Y)
            {
                const float LocalY = FMath::Lerp(
                    -Body.HalfExtent.Y,
                    Body.HalfExtent.Y,
                    (static_cast<float>(Y) + 0.5f) / SamplesPerAxis);
                const FVector SampleWorld = BodyCenter + BodyForward * LocalX + BodyRight * LocalY;
                const FVector SampleLocal = ForkliftTransform.InverseTransformPositionNoScale(SampleWorld);
                bool bSampleSupported = false;
                for (const FVector4& Rectangle : ContactRectangles)
                {
                    if (SampleLocal.X >= Rectangle.X && SampleLocal.X <= Rectangle.Y
                        && SampleLocal.Y >= Rectangle.Z && SampleLocal.Y <= Rectangle.W)
                    {
                        bSampleSupported = true;
                        break;
                    }
                }
                SupportedSamples += bSampleSupported ? 1 : 0;
            }
        }
        Result.Coverage = static_cast<float>(SupportedSamples)
            / static_cast<float>(SamplesPerAxis * SamplesPerAxis);
        return Result;
    };

    for (int32 BodyIndex = 0; BodyIndex < DynamicBoxes.Num(); ++BodyIndex)
    {
        FDynamicBoxRuntime& Body = DynamicBoxes[BodyIndex];
        USceneComponent* BodyRoot = Body.Root.Get();
        if (!BodyRoot)
        {
            continue;
        }
        const FPropPhysicsProfile& Physics = Body.Physics;
        Body.ForkSupportCooldownSeconds = FMath::Max(
            0.0f,
            Body.ForkSupportCooldownSeconds - StepSeconds);

        // The carton initially rides on the pallet, not on an invisible
        // socket above the tines. If that pallet falls, release the carton as
        // an independent rigid body so it cannot remain floating in mid-air.
        if (Body.SupportedForklift != INDEX_NONE
            && DynamicBoxes.IsValidIndex(Body.SupportBodyIndex)
            && DynamicBoxes[Body.SupportBodyIndex].SupportedForklift != Body.SupportedForklift)
        {
            const int32 PreviousForklift = Body.SupportedForklift;
            const int32 PreviousSupportBody = Body.SupportBodyIndex;
            const FForkliftRuntime& Forklift = Forklifts[PreviousForklift];
            Body.SupportedForklift = INDEX_NONE;
            Body.SupportBodyIndex = INDEX_NONE;
            Body.bAwake = true;
            Body.ForkSupportCooldownSeconds = 0.25f;
            Body.ForkUnsupportedSeconds = 0.0f;
            Body.LinearVelocity = Forklift.Root.IsValid()
                ? Forklift.SurfaceLinearVelocityCmPerSecond
                : FVector::ZeroVector;
            SimulatorLog(FString::Printf(
                TEXT("cargo_support_lost body=%s former_support_body=%d"),
                *Body.Name,
                PreviousSupportBody));
        }

        if (Body.SupportedForklift != INDEX_NONE && Forklifts[Body.SupportedForklift].Root.IsValid())
        {
            FForkliftRuntime& SupportForklift = Forklifts[Body.SupportedForklift];
            FTransform Desired = Body.ForkliftRelativeTransform * SupportForklift.Root->GetComponentTransform();
            Desired.AddToTranslation(FVector(0.0f, 0.0f, SupportForklift.LiftCm));
            bool bDetachedFromUnsupportedFork = false;
            if (Body.SupportBodyIndex == INDEX_NONE)
            {
                const FForkSupportEvaluation ForkSupport = EvaluateForkSupport(
                    Body,
                    Desired,
                    SupportForklift,
                    true);
                Body.LastSupportCoverage = ForkSupport.Coverage;
                if (ForkSupport.bHasVerticalContact && ForkSupport.bCenterOfMassSupported)
                {
                    Body.ForkUnsupportedSeconds = 0.0f;
                }
                else
                {
                    Body.ForkUnsupportedSeconds += StepSeconds;
                    if (Body.ForkUnsupportedSeconds >= 0.06f)
                    {
                        const FVector TipDirection = ForkSupport.OverhangDirection.IsNearlyZero()
                            ? SupportForklift.Root->GetForwardVector()
                            : ForkSupport.OverhangDirection;
                        const FVector TipAxis = FVector::CrossProduct(
                            FVector::UpVector,
                            TipDirection).GetSafeNormal();
                        Body.SupportedForklift = INDEX_NONE;
                        Body.SupportBodyIndex = INDEX_NONE;
                        Body.bAwake = true;
                        Body.ForkSupportCooldownSeconds = 0.35f;
                        Body.ForkUnsupportedSeconds = 0.0f;
                        Body.LinearVelocity = SupportForklift.SurfaceLinearVelocityCmPerSecond;
                        Body.LinearVelocity.Z = FMath::Min(Body.LinearVelocity.Z, 0.0f);
                        Body.AngularVelocityDegrees += TipAxis
                            * FMath::Lerp(38.0f, 82.0f, 1.0f - ForkSupport.Coverage)
                            * Body.Physics.AngularResponseScale;
                        bDetachedFromUnsupportedFork = true;
                        SimulatorLog(FString::Printf(
                            TEXT("fork_support_lost body=%s reason=%s contacting_tines=%d support_fraction=%.2f overhang=(%.2f,%.2f)"),
                            *Body.Name,
                            ForkSupport.bHasVerticalContact
                                ? TEXT("center_of_mass_outside")
                                : TEXT("no_tine_contact"),
                            ForkSupport.ContactTineCount,
                            ForkSupport.Coverage,
                            TipDirection.X,
                            TipDirection.Y));
                    }
                }
            }
            else
            {
                Body.ForkUnsupportedSeconds = 0.0f;
            }

            if (bDetachedFromUnsupportedFork)
            {
                // Continue into the ordinary rigid-body integration below;
                // the cooldown prevents immediate reattachment at the edge.
            }
            else
            {
                bool bBlocked = false;
                FString BlockingName;
                for (const FCollisionObstacle& Obstacle : CollisionObstacles)
                {
                    if (Body.bShelfParcel && Obstacle.Name == TEXT("shelf plane"))
                    {
                        continue;
                    }
                    if (ConveyorTuning::ObbOverlapsObb(
                            Desired.GetLocation(),
                            Desired.GetRotation(),
                            Body.HalfExtent,
                            Obstacle.Center,
                            Obstacle.Rotation,
                            Obstacle.HalfExtent,
                            1.0f))
                    {
                        bBlocked = true;
                        BlockingName = Obstacle.Name;
                        break;
                    }
                }
                if (!bBlocked)
                {
                    for (int32 OtherIndex = 0; OtherIndex < DynamicBoxes.Num(); ++OtherIndex)
                    {
                        if (OtherIndex == BodyIndex || !DynamicBoxes[OtherIndex].bShelfParcel)
                        {
                            continue;
                        }
                        FDynamicBoxRuntime& Other = DynamicBoxes[OtherIndex];
                        USceneComponent* OtherRoot = Other.Root.Get();
                        if (!OtherRoot
                            || Body.SupportBodyIndex == OtherIndex
                            || Other.SupportBodyIndex == BodyIndex
                            || (Other.SupportedForklift == Body.SupportedForklift
                                && Other.SupportedForklift != INDEX_NONE))
                        {
                            continue;
                        }
                        FVector OtherSeparation;
                        if (ConveyorTuning::FindPlanarObbSeparation(
                                OtherRoot->GetComponentLocation(),
                                OtherRoot->GetComponentQuat(),
                                Other.HalfExtent,
                                Desired.GetLocation(),
                                Desired.GetRotation(),
                                Body.HalfExtent,
                                OtherSeparation))
                        {
                            // Shelf cartons are independent rigid bodies. A
                            // carried carton must push and wake its neighbour,
                            // not lose its fork support merely because the two
                            // boxes began in light contact on the shelf.
                            OtherRoot->AddWorldOffset(
                                OtherSeparation,
                                false,
                                nullptr,
                                ETeleportType::TeleportPhysics);
                            const FVector CarrierVelocity = SupportForklift.SurfaceLinearVelocityCmPerSecond;
                            Other.LinearVelocity.X = CarrierVelocity.X;
                            Other.LinearVelocity.Y = CarrierVelocity.Y;
                            Other.LinearVelocity.Z = FMath::Max(Other.LinearVelocity.Z, 0.0f);
                            Other.AngularVelocityDegrees += FVector(
                                -OtherSeparation.Y,
                                OtherSeparation.X,
                                0.0f).GetSafeNormal() * 8.0f;
                            Other.SupportedForklift = INDEX_NONE;
                            Other.SupportBodyIndex = INDEX_NONE;
                            Other.bAwake = true;
                        }
                    }
                }
                const float RequiredLateralAcceleration =
                    FMath::Abs(SupportForklift.SpeedCmPerSecond)
                    * FMath::DegreesToRadians(
                        FMath::Abs(SupportForklift.SurfaceYawVelocityDegreesPerSecond));
                const float AvailableStaticAcceleration =
                    FMath::Min(Body.Physics.StaticFriction, 0.72f)
                    * ConveyorTuning::GravityCmPerSecondSquared;
                const bool bTractionExceeded = RequiredLateralAcceleration
                    > AvailableStaticAcceleration;
                if (!bBlocked && !bTractionExceeded)
                {
                    BodyRoot->SetWorldTransform(Desired, false, nullptr, ETeleportType::TeleportPhysics);
                    Body.LinearVelocity = CarrierVelocityAtPoint(
                        SupportForklift,
                        Desired.GetLocation());
                    continue;
                }

                Body.SupportedForklift = INDEX_NONE;
                Body.SupportBodyIndex = INDEX_NONE;
                Body.bAwake = true;
                Body.ForkSupportCooldownSeconds = 0.35f;
                Body.ForkUnsupportedSeconds = 0.0f;
                Body.LinearVelocity = CarrierVelocityAtPoint(
                    SupportForklift,
                    BodyRoot->GetComponentLocation());
                Body.LinearVelocity.Z = FMath::Min(Body.LinearVelocity.Z, 0.0f);
                Body.AngularVelocityDegrees.Z = FMath::Clamp(
                    SupportForklift.SurfaceYawVelocityDegreesPerSecond,
                    -30.0f,
                    30.0f);
                SimulatorLog(FString::Printf(
                    TEXT("cargo_detached body=%s reason=%s lateral_accel_cm_s2=%.1f static_limit_cm_s2=%.1f"),
                    *Body.Name,
                    bTractionExceeded ? TEXT("friction_limit") : *BlockingName,
                    RequiredLateralAcceleration,
                    AvailableStaticAcceleration));
            }
        }

        // Independently sleeping shelf cartons wake when a wheel, body, mast,
        // or fork volume touches them at the matching height.
        bool bCapturedByFork = false;
        for (int32 ForkliftIndex = 0; ForkliftIndex < ConveyorTuning::EnabledForkliftCount && !bCapturedByFork; ++ForkliftIndex)
        {
            const FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
            const USceneComponent* ForkliftRoot = Forklift.Root.Get();
            if (!ForkliftRoot)
            {
                continue;
            }
            const FQuat ForkliftQuat = ForkliftRoot->GetComponentQuat();
            const FForkSupportEvaluation ForkSupport = EvaluateForkSupport(
                Body,
                BodyRoot->GetComponentTransform(),
                Forklift,
                false);
            const bool bCanRestOnFork = Body.ForkSupportCooldownSeconds <= 0.0f
                && ForkSupport.bHasVerticalContact
                && ForkSupport.bCenterOfMassSupported;
            if (bCanRestOnFork)
            {
                FTransform BaseWorld = BodyRoot->GetComponentTransform();
                BaseWorld.AddToTranslation(FVector(0.0f, 0.0f, -Forklift.LiftCm));
                Body.ForkliftRelativeTransform = BaseWorld.GetRelativeTransform(
                    ForkliftRoot->GetComponentTransform());
                Body.SupportedForklift = ForkliftIndex;
                Body.SupportBodyIndex = INDEX_NONE;
                Body.ForkUnsupportedSeconds = 0.0f;
                Body.LastSupportCoverage = ForkSupport.Coverage;
                Body.LinearVelocity = FVector::ZeroVector;
                Body.AngularVelocityDegrees = FVector::ZeroVector;
                Body.bAwake = false;
                bCapturedByFork = true;
                SimulatorLog(FString::Printf(
                    TEXT("fork_support_acquired body=%s forklift=%d contacting_tines=%d support_fraction=%.2f clearance_cm=%.1f"),
                    *Body.Name,
                    ForkliftIndex + 1,
                    ForkSupport.ContactTineCount,
                    ForkSupport.Coverage,
                    BodyRoot->GetComponentLocation().Z - Body.HalfExtent.Z - ForkSupport.SupportTop));
                break;
            }
            for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
            {
                if (Body.ForkSupportCooldownSeconds > 0.0f
                    && Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase))
                {
                    continue;
                }
                FVector LocalCenter = Shape.LocalCenter;
                LocalCenter.Z += Shape.bLiftDriven ? Forklift.LiftCm : 0.0f;
                const FVector ShapeCenter = ForkliftRoot->GetComponentLocation() + ForkliftQuat.RotateVector(LocalCenter);
                FCollisionObstacle ShapeObstacle;
                ShapeObstacle.Center = ShapeCenter;
                ShapeObstacle.HalfExtent = Shape.GetCollisionHalfExtent();
                ShapeObstacle.Rotation = ForkliftQuat;
                if (!Shape.IsWedge()
                    && Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase)
                    && HorizontalOverlap(
                        Body,
                        BodyRoot->GetComponentLocation(),
                        BodyRoot->GetComponentQuat(),
                        ShapeObstacle))
                {
                    const FQuat BodyQuat = BodyRoot->GetComponentQuat();
                    const float BodyBottom = BodyRoot->GetComponentLocation().Z
                        - ConveyorTuning::ProjectedVerticalHalfExtent(
                            BodyQuat,
                            Body.HalfExtent);
                    const float TineTop = ShapeCenter.Z
                        + ConveyorTuning::ProjectedVerticalHalfExtent(
                            ForkliftQuat,
                            Shape.GetCollisionHalfExtent());
                    if (BodyBottom >= TineTop - 6.0f
                        && BodyBottom <= TineTop + 5.0f)
                    {
                        // The flat tine begins exactly where the wedge reaches
                        // full height. Its front face is one-way in this narrow
                        // top band so the ramp can finish sliding underneath;
                        // finite steel/cardboard friction carries the load as
                        // the contact patch grows. EvaluateForkSupport captures
                        // it only after its center of mass has genuine support.
                        ApplyForkSurfaceFriction(Body, Forklift, StepSeconds, 1.0f);
                        Body.bAwake = true;
                        continue;
                    }
                }
                if (Shape.IsWedge()
                    && HorizontalOverlap(
                        Body,
                        BodyRoot->GetComponentLocation(),
                        BodyRoot->GetComponentQuat(),
                        ShapeObstacle))
                {
                    // The broad phase uses the wedge's enclosing OBB, but the
                    // upper half of that OBB is empty. Sample the deepest part
                    // of the actual overlap footprint so the contact height
                    // climbs smoothly as the tapered tip advances under a box.
                    const FTransform ForkliftTransform = ForkliftRoot->GetComponentTransform();
                    const FVector BodyCenter = BodyRoot->GetComponentLocation();
                    const FQuat BodyQuat = BodyRoot->GetComponentQuat();
                    const FVector RelativeCenter = ForkliftTransform.InverseTransformPositionNoScale(
                        BodyCenter) - LocalCenter;
                    const FVector ForkX = ForkliftQuat.GetForwardVector();
                    const float BodyHalfAlongForkX =
                        FMath::Abs(FVector::DotProduct(BodyQuat.GetForwardVector(), ForkX)) * Body.HalfExtent.X
                        + FMath::Abs(FVector::DotProduct(BodyQuat.GetRightVector(), ForkX)) * Body.HalfExtent.Y
                        + FMath::Abs(FVector::DotProduct(BodyQuat.GetUpVector(), ForkX)) * Body.HalfExtent.Z;
                    const float OverlapMinX = FMath::Max(
                        -Shape.HalfExtent.X,
                        RelativeCenter.X - BodyHalfAlongForkX);
                    const float OverlapMaxX = FMath::Min(
                        Shape.HalfExtent.X,
                        RelativeCenter.X + BodyHalfAlongForkX);
                    if (OverlapMinX <= OverlapMaxX)
                    {
                        const float ContactX = Shape.bWedgeTipAtPositiveX
                            ? OverlapMinX
                            : OverlapMaxX;
                        const float ContactY = FMath::Clamp(
                            RelativeCenter.Y,
                            -Shape.HalfExtent.Y,
                            Shape.HalfExtent.Y);
                        const float SurfaceLocalZ = Shape.GetWedgeTopLocalZ(ContactX);
                        const float SurfaceWorldZ = ForkliftTransform.TransformPositionNoScale(
                            LocalCenter + FVector(ContactX, ContactY, SurfaceLocalZ)).Z;
                        const float WedgeBottomWorldZ = ForkliftTransform.TransformPositionNoScale(
                            LocalCenter + FVector(ContactX, ContactY, -Shape.HalfExtent.Z)).Z;
                        const float BodyBottom = BodyCenter.Z
                            - ConveyorTuning::ProjectedVerticalHalfExtent(BodyQuat, Body.HalfExtent);

                        // A parcel above the sloping face occupies the empty
                        // triangular half and must not hit an invisible wall.
                        // Near the face, resolve upward so the fork pries under
                        // it; only contacts below the wedge base fall through
                        // to the ordinary solid-body response.
                        if (BodyBottom > SurfaceWorldZ + 5.0f)
                        {
                            continue;
                        }
                        if (BodyBottom >= WedgeBottomWorldZ - 5.0f)
                        {
                            const float Penetration = FMath::Max(
                                0.0f,
                                SurfaceWorldZ - BodyBottom + 0.2f);
                            if (Penetration > UE_KINDA_SMALL_NUMBER)
                            {
                                BodyRoot->AddWorldOffset(
                                    FVector(0.0f, 0.0f, Penetration),
                                    false,
                                    nullptr,
                                    ETeleportType::TeleportPhysics);
                            }
                            // A sloping tip has a smaller normal load than a
                            // flat tine, so it transfers less planar traction
                            // while still being able to pry under the carton.
                            ApplyForkSurfaceFriction(Body, Forklift, StepSeconds, 0.18f);
                            const float LiftVelocity = StepSeconds > UE_SMALL_NUMBER
                                ? Forklift.LiftDeltaCm / StepSeconds
                                : 0.0f;
                            Body.LinearVelocity.Z = FMath::Max(
                                Body.LinearVelocity.Z,
                                FMath::Max(0.0f, LiftVelocity));
                            Body.bAwake = true;
                            if (!Body.bLoggedContact)
                            {
                                Body.bLoggedContact = true;
                                SimulatorLog(FString::Printf(
                                    TEXT("fork_wedge_pry body=%s forklift=%d part=%s surface_z_cm=%.2f penetration_cm=%.2f"),
                                    *Body.Name,
                                    ForkliftIndex + 1,
                                    *Shape.Name,
                                    SurfaceWorldZ,
                                    Penetration));
                            }
                            break;
                        }
                    }
                }
                const bool bShapeOverlap = ConveyorTuning::ObbOverlapsObb(
                    BodyRoot->GetComponentLocation(),
                    BodyRoot->GetComponentQuat(),
                    Body.HalfExtent,
                    ShapeCenter,
                    ForkliftQuat,
                    Shape.GetCollisionHalfExtent(),
                    0.0f);
                if (!bShapeOverlap)
                {
                    continue;
                }
                if (Shape.IsCylinder())
                {
                    const float BodyTop = BodyRoot->GetComponentLocation().Z + Body.HalfExtent.Z;
                    const float WheelBottom = ShapeCenter.Z - Shape.Radius;
                    const float StepHeight = BodyTop - WheelBottom;
                    float LongitudinalGap = 0.0f;
                    const bool bDiskCanRollOntoBody = ConveyorTuning::WheelDiskOverlapsSupport2D(
                        ShapeCenter,
                        ForkliftQuat,
                        Shape.Radius,
                        Shape.CylinderHalfLength,
                        BodyRoot->GetComponentLocation(),
                        BodyRoot->GetComponentQuat(),
                        Body.HalfExtent,
                        LongitudinalGap);
                    if (StepHeight > 0.0f
                        && StepHeight <= Shape.Radius * 0.92f
                        && bDiskCanRollOntoBody)
                    {
                        // Resolve this as tire support in SimulateForklifts.
                        // Giving the light prop a lateral impulse first made a
                        // loose pallet skate away before the circular contact
                        // could lift the much heavier forklift.
                        continue;
                    }
                }
                FVector Push = BodyRoot->GetComponentLocation() - ShapeCenter;
                Push.Z = 0.0f;
                if (!Push.Normalize())
                {
                    Push = ForkliftRoot->GetForwardVector();
                }
                Body.LinearVelocity += Push
                    * FMath::Max(90.0f, FMath::Abs(Forklift.SpeedCmPerSecond) * 1.4f)
                    * Physics.ImpactResponseScale;
                Body.LinearVelocity.Z = FMath::Max(Body.LinearVelocity.Z, 18.0f);
                Body.AngularVelocityDegrees.Z += (BodyIndex % 2 == 0 ? 55.0f : -55.0f)
                    * Physics.AngularResponseScale;
                Body.bAwake = true;
                if (!Body.bLoggedContact)
                {
                    Body.bLoggedContact = true;
                    SimulatorLog(FString::Printf(
                        TEXT("dynamic_body_hit body=%s forklift=%d part=%s"),
                        *Body.Name,
                        ForkliftIndex + 1,
                        *Shape.Name));
                }
                break;
            }
        }
        if (bCapturedByFork)
        {
            continue;
        }
        if (!Body.bAwake)
        {
            continue;
        }

        const FVector CurrentCenter = BodyRoot->GetComponentLocation();
        const FQuat CurrentRotation = BodyRoot->GetComponentQuat();
        const FVector CurrentCenterOfMass = CurrentCenter
            + CurrentRotation.RotateVector(Physics.CenterOfMassLocalOffset);
        const float LinearAirDrag = FMath::Exp(-Physics.AirLinearDamping * StepSeconds);
        Body.LinearVelocity *= LinearAirDrag;
        Body.LinearVelocity.Z -= ConveyorTuning::GravityCmPerSecondSquared * StepSeconds;
        FVector NextCenterOfMass = CurrentCenterOfMass + Body.LinearVelocity * StepSeconds;
        FQuat NextRotation = CurrentRotation;
        Body.AngularVelocityDegrees *= FMath::Exp(-Physics.AirAngularDamping * StepSeconds);
        const FVector AngularStepDegrees = Body.AngularVelocityDegrees * StepSeconds;
        if (AngularStepDegrees.SizeSquared() > KINDA_SMALL_NUMBER)
        {
            NextRotation = FQuat(
                AngularStepDegrees.GetSafeNormal(),
                FMath::DegreesToRadians(AngularStepDegrees.Size())) * NextRotation;
            NextRotation.Normalize();
        }
        FVector NextCenter = NextCenterOfMass
            - NextRotation.RotateVector(Physics.CenterOfMassLocalOffset);

        float SupportTop = 0.0f;
        float SupportCoverage = 1.0f;
        FString SupportName = TEXT("floor");
        float PartialSupportTop = 0.0f;
        float PartialSupportCoverage = 0.0f;
        FVector PartialOverhangDirection = FVector::ZeroVector;
        const float CurrentVerticalHalfExtent = ConveyorTuning::ProjectedVerticalHalfExtent(
            BodyRoot->GetComponentQuat(),
            Body.HalfExtent);
        const float NextVerticalHalfExtent = ConveyorTuning::ProjectedVerticalHalfExtent(
            NextRotation,
            Body.HalfExtent);
        for (const FCollisionObstacle& Obstacle : CollisionObstacles)
        {
            if (Obstacle.Name != TEXT("shelf plane")
                && Obstacle.Name != TEXT("packing table")
                && Obstacle.Name != TEXT("conveyor"))
            {
                continue;
            }
            const float CandidateTop = Obstacle.Center.Z + Obstacle.HalfExtent.Z;
            const float CurrentBottom = CurrentCenter.Z - CurrentVerticalHalfExtent;
            const float NextBottom = NextCenter.Z - NextVerticalHalfExtent;
            if (CandidateTop >= SupportTop
                && CurrentBottom >= CandidateTop - 4.0f
                && NextBottom <= CandidateTop + 2.0f
                && HorizontalOverlap(Body, NextCenter, NextRotation, Obstacle))
            {
                FVector OverhangDirection;
                const float Coverage = ConveyorTuning::EvaluateSupportCoverage(
                    NextCenter,
                    NextRotation,
                    Body.HalfExtent,
                    Obstacle.Center,
                    Obstacle.Rotation,
                    Obstacle.HalfExtent,
                    OverhangDirection);
                const bool bCenterOfMassSupported = ConveyorTuning::IsCenterOfMassSupported(
                    NextCenter + NextRotation.RotateVector(Physics.CenterOfMassLocalOffset),
                    Obstacle.Center,
                    Obstacle.Rotation,
                    Obstacle.HalfExtent);
                if (Coverage > 0.0f && bCenterOfMassSupported)
                {
                    SupportTop = CandidateTop;
                    SupportCoverage = Coverage;
                    SupportName = Obstacle.Name;
                }
                else if (Coverage > 0.0f && CandidateTop >= PartialSupportTop)
                {
                    PartialSupportTop = CandidateTop;
                    PartialSupportCoverage = Coverage;
                    PartialOverhangDirection = OverhangDirection;
                }
            }
        }
        for (int32 OtherIndex = 0; OtherIndex < DynamicBoxes.Num(); ++OtherIndex)
        {
            if (OtherIndex == BodyIndex)
            {
                continue;
            }
            const FDynamicBoxRuntime& Other = DynamicBoxes[OtherIndex];
            const USceneComponent* OtherRoot = Other.Root.Get();
            if (!OtherRoot)
            {
                continue;
            }
            const float CandidateTop = OtherRoot->GetComponentLocation().Z + Other.HalfExtent.Z;
            const float CurrentBottom = CurrentCenter.Z - CurrentVerticalHalfExtent;
            const float NextBottom = NextCenter.Z - NextVerticalHalfExtent;
            FCollisionObstacle OtherBox;
            OtherBox.Center = OtherRoot->GetComponentLocation();
            OtherBox.HalfExtent = Other.HalfExtent;
            OtherBox.Rotation = OtherRoot->GetComponentQuat();
            if (CandidateTop >= SupportTop
                && CurrentBottom >= CandidateTop - 3.0f
                && NextBottom <= CandidateTop + 2.0f
                && HorizontalOverlap(Body, NextCenter, NextRotation, OtherBox))
            {
                FVector OverhangDirection;
                const float Coverage = ConveyorTuning::EvaluateSupportCoverage(
                    NextCenter,
                    NextRotation,
                    Body.HalfExtent,
                    OtherBox.Center,
                    OtherBox.Rotation,
                    OtherBox.HalfExtent,
                    OverhangDirection);
                const bool bCenterOfMassSupported = ConveyorTuning::IsCenterOfMassSupported(
                    NextCenter + NextRotation.RotateVector(Physics.CenterOfMassLocalOffset),
                    OtherBox.Center,
                    OtherBox.Rotation,
                    OtherBox.HalfExtent);
                if (Coverage > 0.0f && bCenterOfMassSupported)
                {
                    SupportTop = CandidateTop;
                    SupportCoverage = Coverage;
                    SupportName = Other.Name;
                }
                else if (Coverage > 0.0f && CandidateTop >= PartialSupportTop)
                {
                    PartialSupportTop = CandidateTop;
                    PartialSupportCoverage = Coverage;
                    PartialOverhangDirection = OverhangDirection;
                }
            }
        }

        if (PartialSupportTop > SupportTop + 1.0f
            && PartialSupportCoverage > 0.0f
            && !PartialOverhangDirection.IsNearlyZero())
        {
            Body.LastSupportCoverage = PartialSupportCoverage;
            const FVector TipAxis = FVector::CrossProduct(
                FVector::UpVector,
                PartialOverhangDirection).GetSafeNormal();
            const float CenterOfMassHeightScale = FMath::Clamp(
                1.0f + Physics.CenterOfMassLocalOffset.Z / FMath::Max(Body.HalfExtent.Z, 1.0f) * 0.35f,
                0.65f,
                1.35f);
            Body.AngularVelocityDegrees += TipAxis
                * 105.0f
                * Physics.AngularResponseScale
                * CenterOfMassHeightScale
                * StepSeconds;
            Body.bAwake = true;
            if (!Body.bLoggedLedgeRelease)
            {
                Body.bLoggedLedgeRelease = true;
                SimulatorLog(FString::Printf(
                    TEXT("dynamic_body_ledge_release body=%s support_fraction=%.2f overhang=(%.2f,%.2f)"),
                    *Body.Name,
                    PartialSupportCoverage,
                    PartialOverhangDirection.X,
                    PartialOverhangDirection.Y));
            }
        }
        else
        {
            Body.LastSupportCoverage = SupportCoverage;
            Body.bLoggedLedgeRelease = false;
        }

        const float ImpactSpeed = FMath::Max(0.0f, -Body.LinearVelocity.Z);
        if (NextCenter.Z - NextVerticalHalfExtent <= SupportTop && Body.LinearVelocity.Z <= 0.0f)
        {
            NextCenter.Z = SupportTop + NextVerticalHalfExtent;
            Body.LinearVelocity.Z = ImpactSpeed > 115.0f
                ? ImpactSpeed * Physics.Restitution
                : 0.0f;
            ConveyorTuning::ApplyCoulombFriction(
                Body.LinearVelocity,
                FVector::ZeroVector,
                Physics.StaticFriction,
                Physics.DynamicFriction,
                Physics.SleepLinearSpeedCm,
                StepSeconds);
            Body.AngularVelocityDegrees *= FMath::Exp(
                -Physics.GroundAngularDamping * StepSeconds);
            if (Body.LinearVelocity.SizeSquared2D()
                    < FMath::Square(Physics.SleepLinearSpeedCm)
                && FMath::Abs(Body.LinearVelocity.Z) <= Physics.SleepLinearSpeedCm
                && Body.AngularVelocityDegrees.SizeSquared()
                    < FMath::Square(Physics.SleepAngularSpeedDegrees))
            {
                Body.LinearVelocity = FVector::ZeroVector;
                Body.AngularVelocityDegrees = FVector::ZeroVector;
                Body.bAwake = false;
                SimulatorLog(FString::Printf(
                    TEXT("dynamic_body_sleep body=%s support=%s"),
                    *Body.Name,
                    *SupportName));
            }
        }

        for (const FCollisionObstacle& Obstacle : CollisionObstacles)
        {
            const float ObstacleTop = Obstacle.Center.Z + Obstacle.HalfExtent.Z;
            const float NextBottom = NextCenter.Z - NextVerticalHalfExtent;
            const bool bLeavingTopSupport = PartialSupportTop >= ObstacleTop - 1.0f
                && (Obstacle.Name == TEXT("shelf plane")
                    || Obstacle.Name == TEXT("packing table")
                    || Obstacle.Name == TEXT("conveyor"))
                && NextBottom >= ObstacleTop - 4.0f;
            if (bLeavingTopSupport)
            {
                // Once the centre of mass crosses an edge, the residual top
                // contact is a tipping pivot, not a side wall that can pin the
                // object forever at the lip.
                continue;
            }
            if (!ConveyorTuning::ObbOverlapsObb(
                    NextCenter,
                    NextRotation,
                    Body.HalfExtent,
                    Obstacle.Center,
                    Obstacle.Rotation,
                    Obstacle.HalfExtent,
                    0.0f))
            {
                continue;
            }
            // A top contact has already been resolved above. Any remaining
            // overlap is a side/underside impact, so reject horizontal motion.
            NextCenter.X = CurrentCenter.X;
            NextCenter.Y = CurrentCenter.Y;
            Body.LinearVelocity.X *= -Physics.Restitution;
            Body.LinearVelocity.Y *= -Physics.Restitution;
            Body.AngularVelocityDegrees.Z *= 0.5f;
            break;
        }
        for (int32 OtherIndex = 0; OtherIndex < DynamicBoxes.Num(); ++OtherIndex)
        {
            if (OtherIndex == BodyIndex)
            {
                continue;
            }
            FDynamicBoxRuntime& Other = DynamicBoxes[OtherIndex];
            USceneComponent* OtherRoot = Other.Root.Get();
            if (!OtherRoot)
            {
                continue;
            }
            FVector ContactSeparation;
            if (!ConveyorTuning::FindPlanarObbSeparation(
                    NextCenter,
                    NextRotation,
                    Body.HalfExtent,
                    OtherRoot->GetComponentLocation(),
                    OtherRoot->GetComponentQuat(),
                    Other.HalfExtent,
                    ContactSeparation))
            {
                continue;
            }
            const FVector Normal = ContactSeparation.GetSafeNormal2D();
            if (Normal.IsNearlyZero())
            {
                continue;
            }
            const float BodyInverseMass = 1.0f / FMath::Max(Body.Physics.MassKg, 0.05f);
            const float OtherInverseMass = 1.0f / FMath::Max(Other.Physics.MassKg, 0.05f);
            const float InverseMassSum = BodyInverseMass + OtherInverseMass;
            NextCenter += ContactSeparation * (BodyInverseMass / InverseMassSum);
            OtherRoot->AddWorldOffset(
                -ContactSeparation * (OtherInverseMass / InverseMassSum),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            const float RelativeNormalSpeed = FVector::DotProduct(
                Body.LinearVelocity - Other.LinearVelocity,
                Normal);
            if (RelativeNormalSpeed < 0.0f)
            {
                const float Restitution = FMath::Min(
                    Body.Physics.Restitution,
                    Other.Physics.Restitution);
                const float ImpulseMagnitude = -(1.0f + Restitution)
                    * RelativeNormalSpeed / InverseMassSum;
                Body.LinearVelocity += Normal * ImpulseMagnitude * BodyInverseMass;
                Other.LinearVelocity -= Normal * ImpulseMagnitude * OtherInverseMass;
            }
            Other.bAwake = true;
            break;
        }

        // Keep fallen bodies inside the demo room and dissipate energy rather
        // than allowing them to tunnel permanently beyond the presentation.
        if (NextCenter.X - Body.HalfExtent.X < ConveyorTuning::WorldMinimumX
            || NextCenter.X + Body.HalfExtent.X > ConveyorTuning::WorldMaximumX)
        {
            NextCenter.X = FMath::Clamp(
                NextCenter.X,
                ConveyorTuning::WorldMinimumX + Body.HalfExtent.X,
                ConveyorTuning::WorldMaximumX - Body.HalfExtent.X);
            Body.LinearVelocity.X *= -0.25f;
        }
        if (NextCenter.Y - Body.HalfExtent.Y < ConveyorTuning::WorldMinimumY
            || NextCenter.Y + Body.HalfExtent.Y > ConveyorTuning::WorldMaximumY)
        {
            NextCenter.Y = FMath::Clamp(
                NextCenter.Y,
                ConveyorTuning::WorldMinimumY + Body.HalfExtent.Y,
                ConveyorTuning::WorldMaximumY - Body.HalfExtent.Y);
            Body.LinearVelocity.Y *= -0.25f;
        }
        // Contact correction is sequential and can briefly compound impulses
        // when a stacked carton, pallet, and fork all touch during a turn.
        // Bound that numerical energy to physically credible warehouse-prop
        // speeds; gravity, sliding and tipping remain fully simulated.
        const bool bForkliftCargo = Body.Name.Contains(TEXT("forklift"), ESearchCase::IgnoreCase);
        const float MaximumPlanarSpeedCmPerSecond = bForkliftCargo ? 180.0f : 260.0f;
        const float PlanarSpeed = Body.LinearVelocity.Size2D();
        if (PlanarSpeed > MaximumPlanarSpeedCmPerSecond)
        {
            const float Scale = MaximumPlanarSpeedCmPerSecond / PlanarSpeed;
            Body.LinearVelocity.X *= Scale;
            Body.LinearVelocity.Y *= Scale;
        }
        Body.LinearVelocity.Z = FMath::Clamp(Body.LinearVelocity.Z, -450.0f, 180.0f);
        Body.AngularVelocityDegrees.X = FMath::Clamp(Body.AngularVelocityDegrees.X, -180.0f, 180.0f);
        Body.AngularVelocityDegrees.Y = FMath::Clamp(Body.AngularVelocityDegrees.Y, -180.0f, 180.0f);
        Body.AngularVelocityDegrees.Z = FMath::Clamp(Body.AngularVelocityDegrees.Z, -150.0f, 150.0f);
        BodyRoot->SetWorldLocationAndRotation(
            NextCenter,
            NextRotation,
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
    }
}

void AQaiConveyorWorld::SimulateConveyor(float StepSeconds)
{
    const auto BeltSupportCoverage = [](
        const FVector& Center,
        const FQuat& Rotation,
        const FVector& HalfExtent,
        const FVector& Projection,
        const FVector& Tangent)
    {
        const FVector BeltNormal = FVector::CrossProduct(FVector::UpVector, Tangent).GetSafeNormal2D();
        const float CenterOffset = FMath::Abs(FVector::DotProduct(Center - Projection, BeltNormal));
        const float ProjectedHalfWidth = FMath::Max(
            1.0f,
            FMath::Abs(FVector::DotProduct(Rotation.GetForwardVector(), BeltNormal)) * HalfExtent.X
                + FMath::Abs(FVector::DotProduct(Rotation.GetRightVector(), BeltNormal)) * HalfExtent.Y);
        const float ParcelMinimum = CenterOffset - ProjectedHalfWidth;
        const float ParcelMaximum = CenterOffset + ProjectedHalfWidth;
        const float Overlap = FMath::Max(
            0.0f,
            FMath::Min(ParcelMaximum, ConveyorTuning::BeltHalfWidth)
                - FMath::Max(ParcelMinimum, -ConveyorTuning::BeltHalfWidth));
        return FMath::Clamp(Overlap / (2.0f * ProjectedHalfWidth), 0.0f, 1.0f);
    };

    for (int32 Index = 0; Index < Parcels.Num(); ++Index)
    {
        USceneComponent* Parcel = Parcels[Index].Get();
        if (!Parcel
            || !ParcelDistances.IsValidIndex(Index)
            || !ParcelHalfExtents.IsValidIndex(Index)
            || !ParcelPhysicsProfiles.IsValidIndex(Index)
            || !ParcelLinearVelocities.IsValidIndex(Index)
            || !ParcelGrounded.IsValidIndex(Index)
            || !ParcelPitchVelocities.IsValidIndex(Index)
            || !ParcelRollVelocities.IsValidIndex(Index)
            || !ParcelYawVelocities.IsValidIndex(Index))
        {
            continue;
        }

        const FVector HalfExtent = ParcelHalfExtents[Index];
        const FPropPhysicsProfile& Physics = ParcelPhysicsProfiles[Index];
        FVector CurrentCenter = Parcel->GetComponentLocation();
        FQuat CurrentRotation = Parcel->GetComponentQuat();
        FVector& Velocity = ParcelLinearVelocities[Index];
        const FVector CurrentCenterOfMass = CurrentCenter
            + CurrentRotation.RotateVector(Physics.CenterOfMassLocalOffset);

        float CurrentDistance = 0.0f;
        float CurrentLateralDistance = 0.0f;
        FVector CurrentProjection;
        FVector CurrentTangent;
        ConveyorTuning::ProjectToConveyor(
            CurrentCenter,
            CurrentDistance,
            CurrentProjection,
            CurrentTangent,
            CurrentLateralDistance);
        float CurrentCenterOfMassDistance = 0.0f;
        float CurrentCenterOfMassLateralDistance = 0.0f;
        FVector CurrentCenterOfMassProjection;
        FVector CurrentCenterOfMassTangent;
        ConveyorTuning::ProjectToConveyor(
            CurrentCenterOfMass,
            CurrentCenterOfMassDistance,
            CurrentCenterOfMassProjection,
            CurrentCenterOfMassTangent,
            CurrentCenterOfMassLateralDistance);
        const float CurrentVerticalHalfExtent = ConveyorTuning::ProjectedVerticalHalfExtent(
            CurrentRotation,
            HalfExtent);
        const float CurrentBottom = CurrentCenter.Z - CurrentVerticalHalfExtent;
        const float CurrentBeltCoverage = BeltSupportCoverage(
            CurrentCenter,
            CurrentRotation,
            HalfExtent,
            CurrentProjection,
            CurrentTangent);
        const bool bWasOnBelt = ParcelGrounded[Index]
            && CurrentBeltCoverage > 0.0f
            && CurrentCenterOfMassLateralDistance <= ConveyorTuning::BeltHalfWidth
            && FMath::Abs(CurrentBottom - ConveyorSurfaceZCm) <= 7.0f;
        if (bWasOnBelt)
        {
            const FVector TargetVelocity = CurrentTangent * ConveyorTuning::ConveyorSpeedCm;
            ConveyorTuning::ApplyCoulombFriction(
                Velocity,
                TargetVelocity,
                Physics.StaticFriction,
                Physics.DynamicFriction,
                Physics.SleepLinearSpeedCm,
                StepSeconds);
        }

        Velocity *= FMath::Exp(-Physics.AirLinearDamping * StepSeconds);
        Velocity.Z -= ConveyorTuning::GravityCmPerSecondSquared * StepSeconds;
        FVector NextCenterOfMass = CurrentCenterOfMass + Velocity * StepSeconds;
        FQuat NextRotation = CurrentRotation;
        float& PitchVelocity = ParcelPitchVelocities[Index];
        float& RollVelocity = ParcelRollVelocities[Index];
        float& YawVelocity = ParcelYawVelocities[Index];
        PitchVelocity *= FMath::Exp(-Physics.AirAngularDamping * StepSeconds);
        RollVelocity *= FMath::Exp(-Physics.AirAngularDamping * StepSeconds);
        YawVelocity *= FMath::Exp(-Physics.AirAngularDamping * StepSeconds);
        if (FMath::Abs(PitchVelocity) > UE_KINDA_SMALL_NUMBER
            || FMath::Abs(RollVelocity) > UE_KINDA_SMALL_NUMBER
            || FMath::Abs(YawVelocity) > UE_KINDA_SMALL_NUMBER)
        {
            const FQuat PitchStep(
                NextRotation.GetRightVector(),
                FMath::DegreesToRadians(PitchVelocity * StepSeconds));
            const FQuat RollStep(
                NextRotation.GetForwardVector(),
                FMath::DegreesToRadians(RollVelocity * StepSeconds));
            const FQuat YawStep(
                FVector::UpVector,
                FMath::DegreesToRadians(YawVelocity * StepSeconds));
            NextRotation = (YawStep * RollStep * PitchStep * NextRotation).GetNormalized();
        }
        FVector NextCenter = NextCenterOfMass
            - NextRotation.RotateVector(Physics.CenterOfMassLocalOffset);
        float NextDistance = 0.0f;
        float NextLateralDistance = 0.0f;
        FVector NextProjection;
        FVector NextTangent;
        ConveyorTuning::ProjectToConveyor(
            NextCenter,
            NextDistance,
            NextProjection,
            NextTangent,
            NextLateralDistance);
        const FVector NextResolvedCenterOfMass = NextCenter
            + NextRotation.RotateVector(Physics.CenterOfMassLocalOffset);
        float NextCenterOfMassDistance = 0.0f;
        float NextCenterOfMassLateralDistance = 0.0f;
        FVector NextCenterOfMassProjection;
        FVector NextCenterOfMassTangent;
        ConveyorTuning::ProjectToConveyor(
            NextResolvedCenterOfMass,
            NextCenterOfMassDistance,
            NextCenterOfMassProjection,
            NextCenterOfMassTangent,
            NextCenterOfMassLateralDistance);
        const float NextBeltCoverage = BeltSupportCoverage(
            NextCenter,
            NextRotation,
            HalfExtent,
            NextProjection,
            NextTangent);

        float SupportTop = 0.0f;
        FVector SupportVelocity = FVector::ZeroVector;
        bool bOnBelt = false;
        int32 SupportedForklift = INDEX_NONE;
        const float NextVerticalHalfExtent = ConveyorTuning::ProjectedVerticalHalfExtent(
            NextRotation,
            HalfExtent);
        const float NextBottom = NextCenter.Z - NextVerticalHalfExtent;
        if (NextBeltCoverage > 0.0f
            && NextCenterOfMassLateralDistance <= ConveyorTuning::BeltHalfWidth
            && CurrentBottom >= ConveyorSurfaceZCm - 6.0f
            && (NextBottom <= ConveyorSurfaceZCm + 3.0f || bWasOnBelt))
        {
            SupportTop = ConveyorSurfaceZCm;
            SupportVelocity = NextTangent * ConveyorTuning::ConveyorSpeedCm;
            bOnBelt = true;
        }
        else if (NextBeltCoverage > 0.0f
            && NextBottom <= ConveyorSurfaceZCm + 3.0f
            && CurrentBottom >= ConveyorSurfaceZCm - 6.0f)
        {
            // Once the centre of mass passes the belt edge, retain the small
            // remaining contact as a pivot instead of an invisible support.
            const FVector OverhangDirection = (
                NextResolvedCenterOfMass - NextCenterOfMassProjection).GetSafeNormal2D();
            const FVector TipAxis = FVector::CrossProduct(
                FVector::UpVector,
                OverhangDirection).GetSafeNormal();
            PitchVelocity += FVector::DotProduct(TipAxis, NextRotation.GetRightVector())
                * 110.0f * Physics.AngularResponseScale * StepSeconds;
            RollVelocity += FVector::DotProduct(TipAxis, NextRotation.GetForwardVector())
                * 110.0f * Physics.AngularResponseScale * StepSeconds;
        }

        // The forklift render hierarchy remains kinematic for predictable
        // controls, but its fitted body, mast, wheels and individual fork
        // tines are moving collision shapes. Resolve those shapes against the
        // independently integrated parcel body instead of letting a scripted
        // belt transform overwrite contacts on the next tick.
        for (int32 ForkliftIndex = 0; ForkliftIndex < ConveyorTuning::EnabledForkliftCount; ++ForkliftIndex)
        {
            const FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
            const USceneComponent* ForkliftRoot = Forklift.Root.Get();
            if (!ForkliftRoot)
            {
                continue;
            }
            const FQuat ForkliftQuat = ForkliftRoot->GetComponentQuat();
            const FVector ForkliftPlanarVelocity = ForkliftRoot->GetForwardVector() * Forklift.SpeedCmPerSecond;
            const float ForkliftVerticalVelocity = StepSeconds > UE_SMALL_NUMBER
                ? Forklift.LiftDeltaCm / StepSeconds
                : 0.0f;
            for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
            {
                FVector LocalCenter = Shape.LocalCenter;
                LocalCenter.Z += Shape.bLiftDriven ? Forklift.LiftCm : 0.0f;
                const FVector ShapeCenter = ForkliftRoot->GetComponentLocation()
                    + ForkliftQuat.RotateVector(LocalCenter);
                FVector FlatParcelCenter = NextCenter;
                FVector FlatShapeCenter = ShapeCenter;
                FlatParcelCenter.Z = 0.0f;
                FlatShapeCenter.Z = 0.0f;
                FVector FlatParcelExtent = HalfExtent;
                FVector FlatShapeExtent = Shape.GetCollisionHalfExtent();
                FlatParcelExtent.Z = 5.0f;
                FlatShapeExtent.Z = 5.0f;
                const bool bHorizontalOverlap = ConveyorTuning::ObbOverlapsObb(
                    FlatParcelCenter,
                    NextRotation,
                    FlatParcelExtent,
                    FlatShapeCenter,
                    ForkliftQuat,
                    FlatShapeExtent,
                    0.0f);
                if (!bHorizontalOverlap)
                {
                    continue;
                }

                const bool bForkTine = Shape.Name.Contains(TEXT("fork"), ESearchCase::IgnoreCase);
                float ShapeTop = ShapeCenter.Z + Shape.GetCollisionHalfExtent().Z;
                if (Shape.IsWedge())
                {
                    const FTransform ForkliftTransform = ForkliftRoot->GetComponentTransform();
                    const FVector RelativeCenter = ForkliftTransform.InverseTransformPositionNoScale(
                        NextCenter) - LocalCenter;
                    const FVector ForkX = ForkliftQuat.GetForwardVector();
                    const float ParcelHalfAlongForkX =
                        FMath::Abs(FVector::DotProduct(NextRotation.GetForwardVector(), ForkX)) * HalfExtent.X
                        + FMath::Abs(FVector::DotProduct(NextRotation.GetRightVector(), ForkX)) * HalfExtent.Y
                        + FMath::Abs(FVector::DotProduct(NextRotation.GetUpVector(), ForkX)) * HalfExtent.Z;
                    const float OverlapMinX = FMath::Max(
                        -Shape.HalfExtent.X,
                        RelativeCenter.X - ParcelHalfAlongForkX);
                    const float OverlapMaxX = FMath::Min(
                        Shape.HalfExtent.X,
                        RelativeCenter.X + ParcelHalfAlongForkX);
                    const float ContactX = Shape.bWedgeTipAtPositiveX
                        ? OverlapMinX
                        : OverlapMaxX;
                    const float ContactY = FMath::Clamp(
                        RelativeCenter.Y,
                        -Shape.HalfExtent.Y,
                        Shape.HalfExtent.Y);
                    ShapeTop = ForkliftTransform.TransformPositionNoScale(
                        LocalCenter + FVector(
                            ContactX,
                            ContactY,
                            Shape.GetWedgeTopLocalZ(ContactX))).Z;
                }
                const bool bCanRestOnShape = CurrentBottom >= ShapeTop - 6.0f
                    && NextBottom <= ShapeTop + 5.0f;
                const bool bForkRisingUnderParcel = bForkTine
                    && ForkliftVerticalVelocity > 0.1f
                    && ShapeTop >= CurrentBottom - 5.0f
                    && ShapeTop <= CurrentCenter.Z + 4.0f;
                if ((bCanRestOnShape || bForkRisingUnderParcel) && ShapeTop >= SupportTop)
                {
                    SupportTop = ShapeTop;
                    SupportVelocity = Shape.IsWedge()
                        ? FVector::ZeroVector
                        : ForkliftPlanarVelocity;
                    SupportVelocity.Z = Shape.bLiftDriven ? ForkliftVerticalVelocity : 0.0f;
                    bOnBelt = false;
                    SupportedForklift = Shape.IsWedge() ? INDEX_NONE : ForkliftIndex;
                    if (ParcelForkContactsLogged.IsValidIndex(Index)
                        && !ParcelForkContactsLogged[Index])
                    {
                        ParcelForkContactsLogged[Index] = true;
                        ++ParcelForkContactCount;
                        SimulatorLog(FString::Printf(
                            TEXT("conveyor_parcel_supported parcel=%d forklift=%d part=%s contact_count=%d"),
                            Index + 1,
                            ForkliftIndex + 1,
                            *Shape.Name,
                            ParcelForkContactCount));
                    }
                }

                if (Shape.IsWedge()
                    || (bForkTine && bCanRestOnShape))
                {
                    // A sloped support uses the wedge surface, while a flat
                    // tine top contact is the continuation of that ramp. Never
                    // run planar separation against either enclosing OBB here:
                    // doing so recreates a blunt invisible step at the join.
                    continue;
                }
                FVector Separation;
                const bool bHasSeparation = ConveyorTuning::FindPlanarObbSeparation(
                    NextCenter,
                    NextRotation,
                    HalfExtent,
                    ShapeCenter,
                    ForkliftQuat,
                    Shape.GetCollisionHalfExtent(),
                    Separation);
                if (bHasSeparation)
                {
                    NextCenter += Separation;
                    FVector Normal = Separation.GetSafeNormal2D();
                    const float CarrierNormalSpeed = FVector::DotProduct(ForkliftPlanarVelocity, Normal);
                    const float ParcelNormalSpeed = FVector::DotProduct(Velocity, Normal);
                    if (ParcelNormalSpeed < CarrierNormalSpeed)
                    {
                        Velocity += Normal * (CarrierNormalSpeed - ParcelNormalSpeed + 10.0f);
                    }
                    Velocity.Z = FMath::Max(Velocity.Z, bForkTine ? 4.0f : 0.0f);
                    if (ParcelForkContactsLogged.IsValidIndex(Index)
                        && !ParcelForkContactsLogged[Index])
                    {
                        ParcelForkContactsLogged[Index] = true;
                        ++ParcelForkContactCount;
                        SimulatorLog(FString::Printf(
                            TEXT("conveyor_parcel_hit parcel=%d forklift=%d part=%s contact_count=%d"),
                            Index + 1,
                            ForkliftIndex + 1,
                            *Shape.Name,
                            ParcelForkContactCount));
                    }
                }
            }
        }

        // Loose forklift cargo and shelf parcels use the same contact model.
        // This lets a parcel pushed off the rollers collide with a pallet or
        // another carton instead of ghosting through it.
        for (FDynamicBoxRuntime& Other : DynamicBoxes)
        {
            USceneComponent* OtherRoot = Other.Root.Get();
            if (!OtherRoot)
            {
                continue;
            }
            const FVector OtherCenter = OtherRoot->GetComponentLocation();
            const float OtherTop = OtherCenter.Z + Other.HalfExtent.Z;
            FVector FlatParcelCenter = NextCenter;
            FVector FlatOtherCenter = OtherCenter;
            FlatParcelCenter.Z = 0.0f;
            FlatOtherCenter.Z = 0.0f;
            FVector FlatParcelExtent = HalfExtent;
            FVector FlatOtherExtent = Other.HalfExtent;
            FlatParcelExtent.Z = 5.0f;
            FlatOtherExtent.Z = 5.0f;
            const bool bHorizontalOverlap = ConveyorTuning::ObbOverlapsObb(
                FlatParcelCenter,
                NextRotation,
                FlatParcelExtent,
                FlatOtherCenter,
                OtherRoot->GetComponentQuat(),
                FlatOtherExtent,
                0.0f);
            if (bHorizontalOverlap
                && CurrentBottom >= OtherTop - 4.0f
                && NextBottom <= OtherTop + 3.0f
                && OtherTop >= SupportTop)
            {
                SupportTop = OtherTop;
                SupportVelocity = Other.LinearVelocity;
                bOnBelt = false;
                SupportedForklift = INDEX_NONE;
            }
            FVector Separation;
            if (ConveyorTuning::FindPlanarObbSeparation(
                    NextCenter,
                    NextRotation,
                    HalfExtent,
                    OtherCenter,
                    OtherRoot->GetComponentQuat(),
                    Other.HalfExtent,
                    Separation))
            {
                const FVector Normal = Separation.GetSafeNormal2D();
                const float ParcelInverseMass = 1.0f / FMath::Max(Physics.MassKg, 0.05f);
                const float OtherInverseMass = 1.0f / FMath::Max(Other.Physics.MassKg, 0.05f);
                const float InverseMassSum = ParcelInverseMass + OtherInverseMass;
                NextCenter += Separation * (ParcelInverseMass / InverseMassSum);
                OtherRoot->AddWorldOffset(
                    -Separation * (OtherInverseMass / InverseMassSum),
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
                const float RelativeNormalSpeed = FVector::DotProduct(
                    Velocity - Other.LinearVelocity,
                    Normal);
                if (RelativeNormalSpeed < 0.0f)
                {
                    const float Restitution = FMath::Min(
                        Physics.Restitution,
                        Other.Physics.Restitution);
                    const float ImpulseMagnitude = -(1.0f + Restitution)
                        * RelativeNormalSpeed / InverseMassSum;
                    const FVector Impulse = Normal * ImpulseMagnitude;
                    Velocity += Impulse * ParcelInverseMass;
                    Other.LinearVelocity -= Impulse * OtherInverseMass;
                    const FVector ContactPoint = (NextCenter + OtherCenter) * 0.5f;
                    const FVector ParcelCenterOfMass = NextCenter
                        + NextRotation.RotateVector(Physics.CenterOfMassLocalOffset);
                    const FVector OtherCenterOfMass = OtherCenter
                        + OtherRoot->GetComponentQuat().RotateVector(
                            Other.Physics.CenterOfMassLocalOffset);
                    YawVelocity = FMath::Clamp(
                        YawVelocity + ConveyorTuning::YawImpulseDeltaDegrees(
                            ContactPoint - ParcelCenterOfMass,
                            Impulse,
                            Physics.InertiaTensorKgCm2.Z,
                            Physics.AngularResponseScale),
                        -ConveyorTuning::ParcelMaximumImpactYawSpeedDegrees,
                        ConveyorTuning::ParcelMaximumImpactYawSpeedDegrees);
                    Other.AngularVelocityDegrees.Z = FMath::Clamp(
                        Other.AngularVelocityDegrees.Z
                            + ConveyorTuning::YawImpulseDeltaDegrees(
                                ContactPoint - OtherCenterOfMass,
                                -Impulse,
                                Other.Physics.InertiaTensorKgCm2.Z,
                                Other.Physics.AngularResponseScale),
                        -ConveyorTuning::ParcelMaximumImpactYawSpeedDegrees,
                        ConveyorTuning::ParcelMaximumImpactYawSpeedDegrees);
                }
                Other.bAwake = true;
            }
        }

        bool bGrounded = false;
        const float ImpactSpeed = FMath::Max(0.0f, -Velocity.Z);
        if (NextCenter.Z - NextVerticalHalfExtent <= SupportTop
            && Velocity.Z <= SupportVelocity.Z + 1.0f)
        {
            NextCenter.Z = SupportTop + NextVerticalHalfExtent;
            const bool bBounced = ImpactSpeed > 115.0f && Physics.Restitution > 0.01f;
            Velocity.Z = bBounced
                ? SupportVelocity.Z + ImpactSpeed * Physics.Restitution
                : SupportVelocity.Z;
            bGrounded = !bBounced;
            if (ImpactSpeed > 12.0f)
            {
                ++ParcelImpactCount;
                if (!bLoggedFirstParcelContact)
                {
                    bLoggedFirstParcelContact = true;
                    SimulatorLog(FString::Printf(
                        TEXT("conveyor_contact parcel=%d impact_speed_cm_s=%.1f roller_top_z_cm=%.3f gravity_cm_s2=%.1f"),
                        Index + 1,
                        ImpactSpeed,
                        ConveyorSurfaceZCm,
                        ConveyorTuning::GravityCmPerSecondSquared));
                }
            }
            if (bOnBelt && !bBounced)
            {
                // Existing contacts were already accelerated before the
                // integration step. Apply once here only for a newly landed
                // parcel, avoiding the old double drive impulse.
                if (!bWasOnBelt)
                {
                    ConveyorTuning::ApplyCoulombFriction(
                        Velocity,
                        SupportVelocity,
                        Physics.StaticFriction,
                        Physics.DynamicFriction,
                        Physics.SleepLinearSpeedCm,
                        StepSeconds);
                }
                const FQuat Heading = FRotator(0.0f, NextTangent.Rotation().Yaw, 0.0f).Quaternion();
                const FQuat TargetRotation = ParcelHeadingOffsets.IsValidIndex(Index)
                    ? Heading * ParcelHeadingOffsets[Index]
                    : Heading;
                const float HeadingError = FMath::FindDeltaAngleDegrees(
                    NextRotation.Rotator().Yaw,
                    TargetRotation.Rotator().Yaw);
                const float DesiredAngularAcceleration = FMath::Clamp(
                    HeadingError * 5.0f,
                    -ConveyorTuning::ConveyorHeadingAngularAccelerationDegrees,
                    ConveyorTuning::ConveyorHeadingAngularAccelerationDegrees);
                YawVelocity = FMath::Clamp(
                    YawVelocity + DesiredAngularAcceleration * StepSeconds,
                    -ConveyorTuning::ConveyorMaximumYawSpeedDegrees,
                    ConveyorTuning::ConveyorMaximumYawSpeedDegrees);
                YawVelocity *= FMath::Exp(
                    -ConveyorTuning::ConveyorYawDampingPerSecond * StepSeconds);
                PitchVelocity *= FMath::Exp(-Physics.GroundAngularDamping * StepSeconds);
                RollVelocity *= FMath::Exp(-Physics.GroundAngularDamping * StepSeconds);
            }
            else if (SupportedForklift != INDEX_NONE && !bBounced)
            {
                ConveyorTuning::ApplyCoulombFriction(
                    Velocity,
                    SupportVelocity,
                    Physics.StaticFriction,
                    Physics.DynamicFriction,
                    Physics.SleepLinearSpeedCm,
                    StepSeconds);
            }
            else if (!bBounced)
            {
                ConveyorTuning::ApplyCoulombFriction(
                    Velocity,
                    SupportVelocity,
                    Physics.StaticFriction,
                    Physics.DynamicFriction,
                    Physics.SleepLinearSpeedCm,
                    StepSeconds);
            }
        }
        else if (NextCenter.Z - NextVerticalHalfExtent <= 0.0f)
        {
            NextCenter.Z = NextVerticalHalfExtent;
            Velocity.Z = ImpactSpeed > 115.0f ? ImpactSpeed * Physics.Restitution : 0.0f;
            ConveyorTuning::ApplyCoulombFriction(
                Velocity,
                FVector::ZeroVector,
                Physics.StaticFriction,
                Physics.DynamicFriction,
                Physics.SleepLinearSpeedCm,
                StepSeconds);
            bGrounded = Velocity.Z <= UE_KINDA_SMALL_NUMBER;
            bOnBelt = false;
            SupportedForklift = INDEX_NONE;
        }

        for (const FCollisionObstacle& Obstacle : CollisionObstacles)
        {
            if (Obstacle.Name == TEXT("conveyor"))
            {
                continue;
            }
            FVector Separation;
            if (ConveyorTuning::FindPlanarObbSeparation(
                    NextCenter,
                    NextRotation,
                    HalfExtent,
                    Obstacle.Center,
                    Obstacle.Rotation,
                    Obstacle.HalfExtent,
                    Separation))
            {
                NextCenter += Separation;
                const FVector Normal = Separation.GetSafeNormal2D();
                const float NormalSpeed = FVector::DotProduct(Velocity, Normal);
                if (NormalSpeed < 0.0f)
                {
                    Velocity -= Normal * ((1.0f + Physics.Restitution) * NormalSpeed);
                }
                break;
            }
        }

        if (NextCenter.Z < -100.0f
            || NextCenter.X < ConveyorTuning::WorldMinimumX - 100.0f
            || NextCenter.X > ConveyorTuning::WorldMaximumX + 100.0f
            || NextCenter.Y < ConveyorTuning::WorldMinimumY - 100.0f
            || NextCenter.Y > ConveyorTuning::WorldMaximumY + 100.0f)
        {
            FVector ResetTangent;
            ConveyorTuning::EvaluateConveyor(ParcelDistances[Index], NextCenter, ResetTangent);
            NextCenter.Z = ConveyorSurfaceZCm + HalfExtent.Z + 35.0f;
            Velocity = ResetTangent * ConveyorTuning::ConveyorSpeedCm;
            Velocity.Z = -20.0f;
            PitchVelocity = 0.0f;
            RollVelocity = 0.0f;
            YawVelocity = 0.0f;
            bGrounded = false;
            bOnBelt = false;
            SupportedForklift = INDEX_NONE;
            SimulatorLog(FString::Printf(TEXT("conveyor_parcel_recycled parcel=%d"), Index + 1));
        }

        ConveyorTuning::ProjectToConveyor(
            NextCenter,
            ParcelDistances[Index],
            NextProjection,
            NextTangent,
            NextLateralDistance);
        if (ParcelVerticalPositions.IsValidIndex(Index))
        {
            ParcelVerticalPositions[Index] = NextCenter.Z;
        }
        if (ParcelVerticalVelocities.IsValidIndex(Index))
        {
            ParcelVerticalVelocities[Index] = Velocity.Z;
        }
        if (ParcelPitchDegrees.IsValidIndex(Index))
        {
            ParcelPitchDegrees[Index] = NextRotation.Rotator().Pitch;
        }
        if (ParcelRollDegrees.IsValidIndex(Index))
        {
            ParcelRollDegrees[Index] = NextRotation.Rotator().Roll;
        }
        ParcelGrounded[Index] = bGrounded;
        if (ParcelSupportedForklifts.IsValidIndex(Index))
        {
            ParcelSupportedForklifts[Index] = SupportedForklift;
        }
        Parcel->SetWorldLocationAndRotation(
            NextCenter,
            NextRotation,
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
    }

    // Resolve carton-to-carton contacts after all bodies have advanced. The
    // two-way correction keeps densely grouped parcels from interpenetrating
    // while preserving deterministic ordering at the 120 Hz fixed step.
    for (int32 LeftIndex = 0; LeftIndex < Parcels.Num(); ++LeftIndex)
    {
        USceneComponent* Left = Parcels[LeftIndex].Get();
            if (!Left || !ParcelHalfExtents.IsValidIndex(LeftIndex)
                || !ParcelPhysicsProfiles.IsValidIndex(LeftIndex)
                || !ParcelLinearVelocities.IsValidIndex(LeftIndex)
                || !ParcelYawVelocities.IsValidIndex(LeftIndex))
        {
            continue;
        }
        for (int32 RightIndex = LeftIndex + 1; RightIndex < Parcels.Num(); ++RightIndex)
        {
            USceneComponent* Right = Parcels[RightIndex].Get();
            if (!Right || !ParcelHalfExtents.IsValidIndex(RightIndex)
                || !ParcelPhysicsProfiles.IsValidIndex(RightIndex)
                || !ParcelLinearVelocities.IsValidIndex(RightIndex)
                || !ParcelYawVelocities.IsValidIndex(RightIndex))
            {
                continue;
            }
            FVector Separation;
            if (!ConveyorTuning::FindPlanarObbSeparation(
                    Left->GetComponentLocation(),
                    Left->GetComponentQuat(),
                    ParcelHalfExtents[LeftIndex],
                    Right->GetComponentLocation(),
                    Right->GetComponentQuat(),
                    ParcelHalfExtents[RightIndex],
                    Separation))
            {
                continue;
            }
            const FPropPhysicsProfile& LeftPhysics = ParcelPhysicsProfiles[LeftIndex];
            const FPropPhysicsProfile& RightPhysics = ParcelPhysicsProfiles[RightIndex];
            const float LeftInverseMass = 1.0f / FMath::Max(LeftPhysics.MassKg, 0.05f);
            const float RightInverseMass = 1.0f / FMath::Max(RightPhysics.MassKg, 0.05f);
            const float InverseMassSum = LeftInverseMass + RightInverseMass;
            Left->AddWorldOffset(
                Separation * (LeftInverseMass / InverseMassSum),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            Right->AddWorldOffset(
                -Separation * (RightInverseMass / InverseMassSum),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            const FVector Normal = Separation.GetSafeNormal2D();
            const float RelativeNormalSpeed = FVector::DotProduct(
                ParcelLinearVelocities[LeftIndex] - ParcelLinearVelocities[RightIndex],
                Normal);
            if (RelativeNormalSpeed < 0.0f)
            {
                const float Restitution = FMath::Min(
                    LeftPhysics.Restitution,
                    RightPhysics.Restitution);
                const float ImpulseMagnitude = -(1.0f + Restitution)
                    * RelativeNormalSpeed / InverseMassSum;
                const FVector Impulse = Normal * ImpulseMagnitude;
                ParcelLinearVelocities[LeftIndex] += Impulse * LeftInverseMass;
                ParcelLinearVelocities[RightIndex] -= Impulse * RightInverseMass;
                const FVector ContactPoint = (
                    Left->GetComponentLocation() + Right->GetComponentLocation()) * 0.5f;
                const FVector LeftCenterOfMass = Left->GetComponentLocation()
                    + Left->GetComponentQuat().RotateVector(
                        LeftPhysics.CenterOfMassLocalOffset);
                const FVector RightCenterOfMass = Right->GetComponentLocation()
                    + Right->GetComponentQuat().RotateVector(
                        RightPhysics.CenterOfMassLocalOffset);
                ParcelYawVelocities[LeftIndex] = FMath::Clamp(
                    ParcelYawVelocities[LeftIndex]
                        + ConveyorTuning::YawImpulseDeltaDegrees(
                            ContactPoint - LeftCenterOfMass,
                            Impulse,
                            LeftPhysics.InertiaTensorKgCm2.Z,
                            LeftPhysics.AngularResponseScale),
                    -ConveyorTuning::ParcelMaximumImpactYawSpeedDegrees,
                    ConveyorTuning::ParcelMaximumImpactYawSpeedDegrees);
                ParcelYawVelocities[RightIndex] = FMath::Clamp(
                    ParcelYawVelocities[RightIndex]
                        + ConveyorTuning::YawImpulseDeltaDegrees(
                            ContactPoint - RightCenterOfMass,
                            -Impulse,
                            RightPhysics.InertiaTensorKgCm2.Z,
                            RightPhysics.AngularResponseScale),
                    -ConveyorTuning::ParcelMaximumImpactYawSpeedDegrees,
                    ConveyorTuning::ParcelMaximumImpactYawSpeedDegrees);
            }
        }
    }
}

void AQaiConveyorWorld::SimulateWorkers(float StepSeconds)
{
    // Prediction prevents new pedestrian overlaps, but an authored/reset pose
    // or a moving prop can still leave two capsules intersecting. Separate an
    // existing overlap before route planning so neither worker deadlocks while
    // waiting for the other to leave the same occupied space.
    for (int32 FirstIndex = 0; FirstIndex < UE_ARRAY_COUNT(Workers); ++FirstIndex)
    {
        USceneComponent* FirstRoot = Workers[FirstIndex].Root.Get();
        if (!FirstRoot)
        {
            continue;
        }
        for (int32 SecondIndex = FirstIndex + 1; SecondIndex < UE_ARRAY_COUNT(Workers); ++SecondIndex)
        {
            USceneComponent* SecondRoot = Workers[SecondIndex].Root.Get();
            if (!SecondRoot)
            {
                continue;
            }
            FVector Delta = SecondRoot->GetComponentLocation() - FirstRoot->GetComponentLocation();
            Delta.Z = 0.0f;
            const float Distance = Delta.Size2D();
            const float RequiredDistance = 2.0f * ConveyorTuning::WorkerRadiusCm + 2.0f;
            if (Distance >= RequiredDistance)
            {
                continue;
            }
            const FVector Normal = Distance > 0.1f
                ? Delta / Distance
                : FVector(FirstIndex == 0 ? 1.0f : -1.0f, 0.0f, 0.0f);
            // Each candidate is validated against the other's current pose,
            // so use the full penetration for that independent test. If both
            // moves are accepted this simply leaves a small safety gap.
            const float Correction = RequiredDistance - Distance + 0.5f;
            const FVector FirstCandidate = FirstRoot->GetComponentLocation() - Normal * Correction;
            const FVector SecondCandidate = SecondRoot->GetComponentLocation() + Normal * Correction;
            const bool bFirstCanMove = CanWorkerOccupy(FirstIndex, FirstCandidate);
            const bool bSecondCanMove = CanWorkerOccupy(SecondIndex, SecondCandidate);
            if (bFirstCanMove)
            {
                FirstRoot->SetWorldLocation(FirstCandidate);
            }
            if (bSecondCanMove)
            {
                SecondRoot->SetWorldLocation(SecondCandidate);
            }
            if (!bFirstCanMove && !bSecondCanMove)
            {
                Workers[FirstIndex].AvoidanceTurnSign = -1;
                Workers[SecondIndex].AvoidanceTurnSign = 1;
            }
            ++Workers[FirstIndex].SeparationEvents;
            ++Workers[SecondIndex].SeparationEvents;
            if (Workers[FirstIndex].SeparationEvents <= 2)
            {
                SimulatorLog(FString::Printf(
                    TEXT("worker_separation workers=%d,%d overlap_cm=%.2f moved=%s,%s"),
                    FirstIndex + 1,
                    SecondIndex + 1,
                    RequiredDistance - Distance,
                    bFirstCanMove ? TEXT("true") : TEXT("false"),
                    bSecondCanMove ? TEXT("true") : TEXT("false")));
            }
        }
    }

    for (int32 WorkerIndex = 0; WorkerIndex < UE_ARRAY_COUNT(Workers); ++WorkerIndex)
    {
        FWorkerRuntime& Worker = Workers[WorkerIndex];
        USceneComponent* WorkerRoot = Worker.Root.Get();
        if (!WorkerRoot || !Worker.Waypoints.IsValidIndex(Worker.DestinationIndex))
        {
            continue;
        }
        Worker.RouteReversalCooldownSeconds = FMath::Max(
            0.0f,
            Worker.RouteReversalCooldownSeconds - StepSeconds);
        const auto AdvanceRoute = [&Worker]()
        {
            const int32 LastWaypoint = Worker.Waypoints.Num() - 1;
            int32 NextDestination = Worker.DestinationIndex + Worker.RouteDirection;
            if (NextDestination > LastWaypoint)
            {
                Worker.RouteDirection = -1;
                NextDestination = FMath::Max(0, LastWaypoint - 1);
            }
            else if (NextDestination < 0)
            {
                Worker.RouteDirection = 1;
                NextDestination = FMath::Min(LastWaypoint, 1);
            }
            Worker.DestinationIndex = NextDestination;
        };
        const auto ReverseRoute = [&Worker, &AdvanceRoute]()
        {
            if (Worker.RouteReversalCooldownSeconds > 0.0f)
            {
                return false;
            }
            Worker.RouteDirection *= -1;
            ++Worker.RouteReversalCount;
            Worker.RouteReversalCooldownSeconds =
                ConveyorTuning::WorkerRouteReversalCooldownSeconds;
            AdvanceRoute();
            return true;
        };
        if (Worker.DwellRemaining > 0.0f)
        {
            Worker.DwellRemaining = FMath::Max(0.0f, Worker.DwellRemaining - StepSeconds);
            continue;
        }
        FVector Current = WorkerRoot->GetComponentLocation();
        FVector Destination = Worker.Waypoints[Worker.DestinationIndex];
        Destination.Z = Current.Z;
        const FVector Delta = Destination - Current;
        const float Distance = Delta.Size2D();
        if (Distance <= 0.5f)
        {
            Worker.DwellRemaining = Worker.DwellSeconds.IsValidIndex(Worker.DestinationIndex)
                ? Worker.DwellSeconds[Worker.DestinationIndex]
                : 0.0f;
            AdvanceRoute();
            Worker.BlockedSeconds = 0.0f;
            continue;
        }
        const float StepCm = ConveyorTuning::WorkerSpeedCm * StepSeconds;
        const FVector Direction = Delta.GetSafeNormal2D();
        const bool bWouldArrive = Distance <= StepCm + 0.1f;
        FVector Next = bWouldArrive ? Destination : Current + Direction * StepCm;
        if (!CanWorkerOccupy(WorkerIndex, Next))
        {
            Worker.BlockedSeconds += StepSeconds;
            bool bFoundDetour = false;
            // Probe small steering arcs so a pedestrian turns around walls,
            // portals, vehicles, and the other worker instead of walking in
            // place against the collision boundary.
            const float PreferredSign = static_cast<float>(Worker.AvoidanceTurnSign);
            const float AvoidanceAngles[] = {
                30.0f * PreferredSign,
                60.0f * PreferredSign,
                90.0f * PreferredSign,
                -30.0f * PreferredSign,
                -60.0f * PreferredSign,
                -90.0f * PreferredSign,
            };
            for (const float AngleDegrees : AvoidanceAngles)
            {
                const FVector AvoidanceDirection = Direction.RotateAngleAxis(AngleDegrees, FVector::UpVector);
                const FVector Candidate = Current + AvoidanceDirection * StepCm;
                if (CanWorkerOccupy(WorkerIndex, Candidate))
                {
                    Next = Candidate;
                    Worker.AvoidanceTurnSign = AngleDegrees >= 0.0f ? 1 : -1;
                    bFoundDetour = true;
                    break;
                }
            }
            if (!bFoundDetour)
            {
                if (Worker.BlockedSeconds >= 0.35f)
                {
                    if (ReverseRoute())
                    {
                        Worker.DwellRemaining = 0.15f;
                        Worker.BlockedSeconds = 0.0f;
                        SimulatorLog(FString::Printf(
                            TEXT("worker_route_reversed worker=%d reason=blocked destination=%d cooldown_ms=%.0f"),
                            WorkerIndex + 1,
                            Worker.DestinationIndex,
                            ConveyorTuning::WorkerRouteReversalCooldownSeconds * 1000.0f));
                    }
                }
                continue;
            }
            // A usable detour is forward progress, not a blocked state. Decay
            // the timer and retain the chosen side so the probes do not swap
            // left/right every simulation step near an obstacle boundary.
            Worker.BlockedSeconds = FMath::Max(
                0.0f,
                Worker.BlockedSeconds - StepSeconds * 4.0f);
        }
        else
        {
            Worker.BlockedSeconds = 0.0f;
        }

        const FVector ActualDelta = Next - Current;
        if (ActualDelta.SizeSquared2D() <= UE_KINDA_SMALL_NUMBER)
        {
            continue;
        }
        const float InstantHeadingYaw = ActualDelta.Rotation().Yaw;
        const float TargetError = FMath::FindDeltaAngleDegrees(
            Worker.TargetHeadingYawDegrees,
            InstantHeadingYaw);
        if (FMath::Abs(TargetError) <= ConveyorTuning::WorkerHeadingDeadbandDegrees)
        {
            Worker.TargetHeadingYawDegrees = FMath::UnwindDegrees(InstantHeadingYaw);
            Worker.PendingHeadingYawDegrees = Worker.TargetHeadingYawDegrees;
            Worker.PendingHeadingSeconds = 0.0f;
        }
        else if (FMath::Abs(TargetError) >= 150.0f)
        {
            // Commit genuine route reversals immediately, but the visual still
            // turns at the bounded rate below rather than snapping 180 degrees.
            Worker.TargetHeadingYawDegrees = FMath::UnwindDegrees(InstantHeadingYaw);
            Worker.PendingHeadingYawDegrees = Worker.TargetHeadingYawDegrees;
            Worker.PendingHeadingSeconds = 0.0f;
        }
        else
        {
            const float PendingError = FMath::Abs(FMath::FindDeltaAngleDegrees(
                Worker.PendingHeadingYawDegrees,
                InstantHeadingYaw));
            if (Worker.PendingHeadingSeconds <= 0.0f || PendingError > 10.0f)
            {
                Worker.PendingHeadingYawDegrees = FMath::UnwindDegrees(InstantHeadingYaw);
                Worker.PendingHeadingSeconds = StepSeconds;
            }
            else
            {
                Worker.PendingHeadingSeconds += StepSeconds;
            }
            if (Worker.PendingHeadingSeconds >= ConveyorTuning::WorkerHeadingCommitSeconds)
            {
                Worker.TargetHeadingYawDegrees = Worker.PendingHeadingYawDegrees;
                Worker.PendingHeadingSeconds = 0.0f;
            }
        }

        const float VisualError = FMath::FindDeltaAngleDegrees(
            Worker.VisualHeadingYawDegrees,
            Worker.TargetHeadingYawDegrees);
        const float ResponseAlpha = 1.0f - FMath::Exp(
            -ConveyorTuning::WorkerHeadingResponsePerSecond * StepSeconds);
        const float MaximumTurnStep =
            ConveyorTuning::WorkerMaximumVisualTurnRateDegreesPerSecond * StepSeconds;
        const float VisualTurnStep = FMath::Clamp(
            VisualError * ResponseAlpha,
            -MaximumTurnStep,
            MaximumTurnStep);
        Worker.VisualHeadingYawDegrees = FMath::UnwindDegrees(
            Worker.VisualHeadingYawDegrees + VisualTurnStep);
        Worker.MaximumVisualTurnRateDegreesPerSecond = FMath::Max(
            Worker.MaximumVisualTurnRateDegreesPerSecond,
            FMath::Abs(VisualTurnStep) / FMath::Max(StepSeconds, UE_SMALL_NUMBER));
        const FQuat VisualHeading = FRotator(
            0.0f,
            Worker.VisualHeadingYawDegrees,
            0.0f).Quaternion();
        WorkerRoot->SetWorldLocationAndRotation(Next, VisualHeading * Worker.HeadingOffset);
        if (bWouldArrive && FVector::DistSquared2D(Next, Destination) <= 1.0f)
        {
            Worker.DwellRemaining = Worker.DwellSeconds.IsValidIndex(Worker.DestinationIndex)
                ? Worker.DwellSeconds[Worker.DestinationIndex]
                : 0.0f;
            AdvanceRoute();
        }
    }
}

void AQaiConveyorWorld::UpdateSafetySignal()
{
    bool bMoving = false;
    float MinimumClearance = TNumericLimits<float>::Max();
    const float BeltMinX = ConveyorTuning::BeltCenterX - ConveyorTuning::BeltHalfWidth;
    const float BeltMaxX = ConveyorTuning::BeltCenterX + ConveyorTuning::BeltHalfWidth;
    const float BeltMinY = ConveyorTuning::BeltCenterY - ConveyorTuning::BeltHalfLength;
    const float BeltMaxY = ConveyorTuning::BeltCenterY + ConveyorTuning::BeltHalfLength;

    for (int32 ForkliftIndex = 0; ForkliftIndex < ConveyorTuning::EnabledForkliftCount; ++ForkliftIndex)
    {
        const FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        if (!Forklift.Root.IsValid())
        {
            continue;
        }
        bMoving |= FMath::Abs(Forklift.SpeedCmPerSecond) >= ConveyorTuning::MovingThresholdCm;
        const FVector Center = Forklift.Root->GetComponentLocation();
        const float ClosestX = FMath::Clamp(Center.X, BeltMinX, BeltMaxX);
        const float ClosestY = FMath::Clamp(Center.Y, BeltMinY, BeltMaxY);
        const FVector2D Direction(ClosestX - Center.X, ClosestY - Center.Y);
        const float CenterDistance = Direction.Size();
        float Support = ConveyorTuning::ForkliftHalfWidth;
        if (CenterDistance > KINDA_SMALL_NUMBER)
        {
            const FVector2D Unit = Direction / CenterDistance;
            const FVector Forward3 = Forklift.Root->GetForwardVector();
            const FVector Right3 = Forklift.Root->GetRightVector();
            const FVector2D Forward(Forward3.X, Forward3.Y);
            const FVector2D Right(Right3.X, Right3.Y);
            Support = FMath::Abs(FVector2D::DotProduct(Unit, Forward)) * ConveyorTuning::ForkliftHalfLength
                + FMath::Abs(FVector2D::DotProduct(Unit, Right)) * ConveyorTuning::ForkliftHalfWidth;
        }
        MinimumClearance = FMath::Min(MinimumClearance, CenterDistance - Support);
    }

    const FString NewSignal = MinimumClearance < ConveyorTuning::RedClearanceCm ? TEXT("R") : (bMoving ? TEXT("A") : TEXT("G"));
    if (NewSignal != GroundTruthSignal)
    {
        GroundTruthSignal = NewSignal;
    }
}

void AQaiConveyorWorld::UpdateStackLights()
{
    // The physical lamps expose the model's answer verbatim. Ground truth and
    // visual tracking remain diagnostics only; they never recolor the lamp.
    FString DisplaySignal = (RawModelSignal == TEXT("G")
        || RawModelSignal == TEXT("A")
        || RawModelSignal == TEXT("R")) ? RawModelSignal : TEXT("-");
    FString SignalSource = TEXT("reason2_raw");
    FString ForcedSignal;
    if (FParse::Value(FCommandLine::Get(), TEXT("QaiVisualSignal="), ForcedSignal))
    {
        ForcedSignal = ForcedSignal.Left(1).ToUpper();
        if (ForcedSignal == TEXT("G") || ForcedSignal == TEXT("A") || ForcedSignal == TEXT("R"))
        {
            DisplaySignal = ForcedSignal;
            SignalSource = TEXT("qa_override");
        }
    }

    StackTargetWeights[0] = DisplaySignal == TEXT("G") ? 1.0f : 0.0f;
    StackTargetWeights[1] = DisplaySignal == TEXT("A") ? 1.0f : 0.0f;
    StackTargetWeights[2] = DisplaySignal == TEXT("R") ? 1.0f : 0.0f;
    if (LastLoggedStackSignal != DisplaySignal)
    {
        SimulatorLog(FString::Printf(
            TEXT("stack_signal from=%s to=%s source=%s raw_model=%s ground_truth=%s fade_ms=200 red_wash_guard=enabled"),
            LastLoggedStackSignal.IsEmpty() ? TEXT("-") : *LastLoggedStackSignal,
            *DisplaySignal,
            *SignalSource,
            *RawModelSignal,
            *GroundTruthSignal));
        LastLoggedStackSignal = DisplaySignal;
    }

    if (!bStackLightRigInitialized)
    {
        for (int32 ColorIndex = 0; ColorIndex < 3; ++ColorIndex)
        {
            StackSignalWeights[ColorIndex] = StackTargetWeights[ColorIndex];
        }
        bStackLightRigInitialized = true;
        TickStackLightRig(0.0f);
    }
}

void AQaiConveyorWorld::TickStackLightRig(float DeltaSeconds)
{
    if (!bStackLightRigInitialized)
    {
        return;
    }

    // [green, amber, red]. These are deliberately saturated rather than
    // pastel. The local fog is extinction/scattering dominant: a large
    // emissive value clips through the filmic tonemapper and turns a red halo
    // white, especially when the stack is mounted on a neutral wall.
    static const FLinearColor Colors[] = {
        FLinearColor(0.006f, 0.72f, 0.025f),
        FLinearColor(1.0f, 0.16f, 0.002f),
        FLinearColor(1.0f, 0.008f, 0.002f),
    };
    // Manual EV 7.65 divides pre-exposed radiance substantially. Values in
    // this range put the active glass above the bloom threshold while the
    // fixed exposure and filmic shoulder retain a colored (not white) halo.
    static const float EmissiveStrengths[] = {220.0f, 275.0f, 350.0f};
    static const float HaloStrengths[] = {1.30f, 1.42f, 1.55f};
    static const float LocalLumens[] = {3800.0f, 4600.0f, 5600.0f};
    // Point emitters and large rect washes now cover both local and broad
    // wall illumination. Keep the legacy wall spots at zero energy because a
    // bounded projection can reveal a diagonal edge on adjacent wall panels.
    static const float WallSpotLumens[] = {0.0f, 0.0f, 0.0f};
    static const float RoomVolumeLumens[] = {0.0f, 0.0f, 0.0f};
    // Rect lights emit into one half-space; on the perpendicular portal wall
    // that horizon showed up as a diagonal color cutoff. The widened physical
    // point emitters provide the broad cross-platform surface wash instead.
    static const float WashLumens[] = {0.0f, 0.0f, 0.0f};
    constexpr float FadeRatePerSecond = 5.0f;

    for (int32 ColorIndex = 0; ColorIndex < 3; ++ColorIndex)
    {
        StackSignalWeights[ColorIndex] = FMath::FInterpConstantTo(
            StackSignalWeights[ColorIndex],
            StackTargetWeights[ColorIndex],
            DeltaSeconds,
            FadeRatePerSecond);
        const float Weight = StackSignalWeights[ColorIndex];
        const bool bHaloVisible = Weight > 0.001f;
        for (TWeakObjectPtr<UMaterialBillboardComponent>& WeakBillboard : StackBillboardComponents[ColorIndex])
        {
            if (UMaterialBillboardComponent* Billboard = WeakBillboard.Get())
            {
                Billboard->SetHiddenInGame(!bHaloVisible, true);
                Billboard->SetVisibility(bHaloVisible, true);
            }
        }
        for (TWeakObjectPtr<UMaterialInstanceDynamic>& WeakMaterial : StackLensMaterials[ColorIndex])
        {
            if (UMaterialInstanceDynamic* Material = WeakMaterial.Get())
            {
                Material->SetScalarParameterValue(TEXT("EmissiveStrength"), EmissiveStrengths[ColorIndex] * Weight);
                Material->SetScalarParameterValue(TEXT("BaseBrightness"), 0.035f + 0.22f * Weight);
            }
        }
        for (TWeakObjectPtr<UMaterialInstanceDynamic>& WeakMaterial : StackBillboardMaterials[ColorIndex])
        {
            if (UMaterialInstanceDynamic* Material = WeakMaterial.Get())
            {
                Material->SetVectorParameterValue(TEXT("HaloColor"), Colors[ColorIndex] * 3.4f);
                Material->SetScalarParameterValue(TEXT("HaloStrength"), 1.02f * Weight);
            }
        }
        for (TWeakObjectPtr<ULocalFogVolumeComponent>& WeakFog : StackHaloFogComponents[ColorIndex])
        {
            if (ULocalFogVolumeComponent* HaloFog = WeakFog.Get())
            {
                const float HaloWeight = HaloStrengths[ColorIndex] * Weight;
                HaloFog->SetRadialFogExtinction(6.2f * HaloWeight);
                HaloFog->SetFogAlbedo(Colors[ColorIndex]);
                HaloFog->SetFogEmissive(Colors[ColorIndex] * (1.30f * HaloWeight));
            }
        }
        for (TWeakObjectPtr<USpotLightComponent>& WeakSpot : StackWallSpotLights[ColorIndex])
        {
            if (USpotLightComponent* Spot = WeakSpot.Get())
            {
                Spot->SetLightColor(Colors[ColorIndex], false);
                Spot->SetIntensity(WallSpotLumens[ColorIndex] * Weight);
            }
        }
        for (TWeakObjectPtr<USpotLightComponent>& WeakSpot : StackRoomSpotLights[ColorIndex])
        {
            if (USpotLightComponent* Spot = WeakSpot.Get())
            {
                Spot->SetLightColor(Colors[ColorIndex], false);
                Spot->SetIntensity(RoomVolumeLumens[ColorIndex] * Weight);
            }
        }
        for (TWeakObjectPtr<UPointLightComponent>& WeakEmitter : StackEmitterLights[ColorIndex])
        {
            if (UPointLightComponent* Emitter = WeakEmitter.Get())
            {
                Emitter->SetLightColor(Colors[ColorIndex], false);
                Emitter->SetIntensity(LocalLumens[ColorIndex] * Weight);
            }
        }
    }

    float WashIntensity = 0.0f;
    FLinearColor WashRadiance = FLinearColor::Black;
    for (int32 ColorIndex = 0; ColorIndex < 3; ++ColorIndex)
    {
        const float Contribution = WashLumens[ColorIndex] * StackSignalWeights[ColorIndex];
        WashIntensity += Contribution;
        WashRadiance += Colors[ColorIndex] * Contribution;
    }
    const FLinearColor WashColor = WashIntensity > UE_KINDA_SMALL_NUMBER
        ? WashRadiance / WashIntensity
        : Colors[0];
    for (int32 Index = 0; Index < StackWashLights.Num(); ++Index)
    {
        if (URectLightComponent* Wash = StackWashLights[Index].Get())
        {
            Wash->SetLightColor(WashColor, false);
            Wash->SetIntensity(WashIntensity * StackWashMultipliers[Index]);
        }
    }
}

void AQaiConveyorWorld::SetupInferenceCapture()
{
    CaptureReadback = MakeShared<FRHIGPUTextureReadback>(TEXT("QaiReason2CaptureReadback"));
    CaptureTarget = NewObject<UTextureRenderTarget2D>(this);
    CaptureTarget->RenderTargetFormat = RTF_RGBA8_SRGB;
    CaptureTarget->InitAutoFormat(CaptureRenderWidth, CaptureRenderHeight);
    CaptureTarget->TargetGamma = 2.2f;
    CaptureTarget->UpdateResourceImmediate(true);

    CaptureActor = GetWorld()->SpawnActor<ASceneCapture2D>();
    if (CaptureActor)
    {
        USceneCaptureComponent2D* Capture = CaptureActor->GetCaptureComponent2D();
        // SceneCapture's manual, non-physical path uses a distinct compensation
        // baseline. This calibrated value retains the bright warehouse look
        // while keeping neutral-white clipping below one percent.
        float CaptureExposureBias = 10.0f;
        FParse::Value(FCommandLine::Get(), TEXT("QaiCaptureExposureBias="), CaptureExposureBias);
        Capture->TextureTarget = CaptureTarget;
        Capture->bCaptureEveryFrame = false;
        Capture->bCaptureOnMovement = false;
        Capture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
        // Preserve the authored Omniverse detector optics rather than using
        // Unreal's former 70-degree approximation. The wider shifted sensor
        // keeps the complete forklift and aisle context in frame and prevents
        // nearby workers/parcels from dominating the vision tokens.
        constexpr float DetectorFocalLengthMm = 9.5f;
        constexpr float DetectorHorizontalApertureMm = 20.955f;
        constexpr float DetectorHorizontalApertureOffsetMm = -3.0f;
        const float DetectorHalfFovRadians = FMath::Atan(
            DetectorHorizontalApertureMm / (2.0f * DetectorFocalLengthMm));
        Capture->FOVAngle = FMath::RadiansToDegrees(DetectorHalfFovRadians * 2.0f);
        Capture->bUseCustomProjectionMatrix = true;
        Capture->CustomProjectionMatrix = FReversedZPerspectiveMatrix(
            DetectorHalfFovRadians,
            DetectorHalfFovRadians,
            1.0f,
            static_cast<float>(CaptureRenderWidth) / CaptureRenderHeight,
            GNearClippingPlane,
            GNearClippingPlane);
        const float HorizontalProjectionOffset =
            2.0f * DetectorHorizontalApertureOffsetMm / DetectorHorizontalApertureMm;
        Capture->CustomProjectionMatrix.M[2][0] = -HorizontalProjectionOffset;
        Capture->ShowFlags.SetMotionBlur(false);
        Capture->ShowFlags.SetTemporalAA(true);
        // Match the constrained game camera so inference frames do not change
        // exposure when a bright red source enters view. Bloom is lower than
        // presentation bloom to retain parcel/person edges for Reason2.
        Capture->PostProcessSettings.bOverride_AutoExposureMethod = true;
        Capture->PostProcessSettings.AutoExposureMethod = AEM_Manual;
        Capture->PostProcessSettings.bOverride_AutoExposureApplyPhysicalCameraExposure = true;
        Capture->PostProcessSettings.AutoExposureApplyPhysicalCameraExposure = false;
        Capture->PostProcessSettings.bOverride_AutoExposureMinBrightness = true;
        Capture->PostProcessSettings.AutoExposureMinBrightness = 7.65f;
        Capture->PostProcessSettings.bOverride_AutoExposureMaxBrightness = true;
        Capture->PostProcessSettings.AutoExposureMaxBrightness = 7.65f;
        Capture->PostProcessSettings.bOverride_AutoExposureBias = true;
        Capture->PostProcessSettings.AutoExposureBias = CaptureExposureBias;
        Capture->PostProcessSettings.bOverride_LocalExposureMethod = true;
        Capture->PostProcessSettings.LocalExposureMethod = ELocalExposureMethod::Bilateral;
        Capture->PostProcessSettings.bOverride_LocalExposureHighlightContrastScale = true;
        Capture->PostProcessSettings.LocalExposureHighlightContrastScale = 0.62f;
        Capture->PostProcessSettings.bOverride_LocalExposureShadowContrastScale = true;
        Capture->PostProcessSettings.LocalExposureShadowContrastScale = 0.78f;
        Capture->PostProcessSettings.bOverride_LocalExposureDetailStrength = true;
        Capture->PostProcessSettings.LocalExposureDetailStrength = 1.04f;
        Capture->PostProcessSettings.bOverride_LocalExposureBlurredLuminanceBlend = true;
        Capture->PostProcessSettings.LocalExposureBlurredLuminanceBlend = 0.62f;
        Capture->PostProcessSettings.bOverride_ColorSaturation = true;
        Capture->PostProcessSettings.ColorSaturation = FVector4(1.0f, 1.0f, 1.0f, 1.0f);
        Capture->PostProcessSettings.bOverride_ColorContrast = true;
        Capture->PostProcessSettings.ColorContrast = FVector4(0.94f, 0.94f, 0.94f, 1.0f);
        Capture->PostProcessSettings.bOverride_ColorGain = true;
        Capture->PostProcessSettings.ColorGain = FVector4(0.98f, 0.98f, 0.98f, 1.0f);
        Capture->PostProcessSettings.bOverride_FilmSlope = true;
        Capture->PostProcessSettings.FilmSlope = 0.82f;
        Capture->PostProcessSettings.bOverride_FilmToe = true;
        Capture->PostProcessSettings.FilmToe = 0.42f;
        Capture->PostProcessSettings.bOverride_FilmShoulder = true;
        Capture->PostProcessSettings.FilmShoulder = 0.22f;
        Capture->PostProcessSettings.bOverride_BloomIntensity = true;
        Capture->PostProcessSettings.BloomIntensity = 0.12f;
        Capture->PostProcessSettings.bOverride_BloomThreshold = true;
        Capture->PostProcessSettings.BloomThreshold = 1.35f;
        Capture->PostProcessSettings.bOverride_VignetteIntensity = true;
        Capture->PostProcessSettings.VignetteIntensity = 0.08f;
        const bool bEvkCapture = ActiveBackend == TEXT("evk");
        SimulatorLog(FString::Printf(
            TEXT("capture_initialized render_size=%dx%d output_size=%dx%d backend=%s exposure_mode=manual physical_camera=false exposure_compensation=%.2f highlight_scale=0.62 bloom=0.12 horizontal_fov_deg=%.2f aperture_offset_mm=%.2f"),
            CaptureRenderWidth,
            CaptureRenderHeight,
            bEvkCapture ? ConveyorTuning::EvkCaptureWidth : HostCaptureWidth,
            bEvkCapture ? ConveyorTuning::EvkCaptureHeight : HostCaptureHeight,
            *ActiveBackend,
            CaptureExposureBias,
            Capture->FOVAngle,
            DetectorHorizontalApertureOffsetMm));
    }
    else
    {
        SimulatorLog(TEXT("capture_initialization_failed actor_spawn"));
    }
}

void AQaiConveyorWorld::CancelActiveInference()
{
    // Any queued readback or PNG task belongs to the previous backend/state.
    // Its completion is discarded instead of contaminating the next pair.
    ++InferenceCaptureGeneration;
    if (InferenceRequest)
    {
        InferenceRequest->OnProcessRequestComplete().Unbind();
        InferenceRequest->CancelRequest();
        InferenceRequest.Reset();
    }
    bInferenceBusy = false;
    EncodedFrames.Reset();
    EncodedForkliftFrameTransforms.Reset();
    EncodedFrameTextures.Reset();
    EncodedFrameTimes.Reset();
    SubmittedEncodedFrames.Reset();
    SubmittedFrameTextures.Reset();
    SubmittedFrameTimes.Reset();
    SubmittedGroundTruthSignal = TEXT("G");
    bSubmittedForkliftMotion = false;
    bSubmittedRedOverlap = false;
}

void AQaiConveyorWorld::TickInferenceCapture(float DeltaSeconds)
{
    if (bIsShuttingDown)
    {
        return;
    }
    if (bCaptureReadbackPending)
    {
        ReadbackAndCompressCapture();
    }

    BackendProbeAccumulator += DeltaSeconds;
    if (bInferenceEnabled && !bBackendHealthy && BackendProbeAccumulator >= 5.0f)
    {
        ProbeBackend();
    }

    const bool bOfflineDatasetCapture = bResolutionDataset && !bInferenceEnabled;
    if (!bStageReady
        || (!bInferenceEnabled && !bOfflineDatasetCapture)
        || (bInferenceEnabled && !bBackendHealthy)
        || !CaptureActor
        || bEncodingFrame)
    {
        return;
    }
    CaptureAccumulator += DeltaSeconds;
    if (CaptureAccumulator >= ConveyorTuning::CaptureInterval)
    {
        CaptureAccumulator = 0.0f;
        if (bInferenceBusy)
        {
            ++DroppedInferenceFrames;
            return;
        }
        QueueCapture();
    }
}

void AQaiConveyorWorld::QueueCapture()
{
    if (bIsShuttingDown || !DetectorCamera.IsValid() || !CaptureActor || !CaptureTarget || !CaptureReadback || bCaptureReadbackPending)
    {
        if (!bLoggedCaptureUnavailable && !bCaptureReadbackPending)
        {
            bLoggedCaptureUnavailable = true;
            SimulatorLog(FString::Printf(
                TEXT("capture_unavailable detector=%s actor=%s target=%s readback=%s"),
                DetectorCamera.IsValid() ? TEXT("true") : TEXT("false"),
                CaptureActor ? TEXT("true") : TEXT("false"),
                CaptureTarget ? TEXT("true") : TEXT("false"),
                CaptureReadback ? TEXT("true") : TEXT("false")));
        }
        return;
    }
    PendingCaptureGeneration = InferenceCaptureGeneration;
    const bool bEvkCapture = ActiveBackend == TEXT("evk");
    PendingCaptureOutputWidth = bEvkCapture
        ? ConveyorTuning::EvkCaptureWidth
        : HostCaptureWidth;
    PendingCaptureOutputHeight = bEvkCapture
        ? ConveyorTuning::EvkCaptureHeight
        : HostCaptureHeight;
    CaptureActor->SetActorTransform(DetectorCamera->GetComponentTransform());
    CaptureActor->GetCaptureComponent2D()->CaptureScene();

    FTextureRenderTargetResource* Resource = CaptureTarget->GameThread_GetRenderTargetResource();
    if (!Resource)
    {
        BackendStatus = TEXT("capture render target unavailable");
        if (!bLoggedCaptureUnavailable)
        {
            bLoggedCaptureUnavailable = true;
            SimulatorLog(TEXT("capture_unavailable reason=render_target_resource"));
        }
        return;
    }
    const TSharedRef<FRHIGPUTextureReadback> Readback = CaptureReadback.ToSharedRef();
    const int32 RenderWidth = CaptureRenderWidth;
    const int32 RenderHeight = CaptureRenderHeight;
    ENQUEUE_RENDER_COMMAND(QaiEnqueueCaptureReadback)(
        [Resource, Readback, RenderWidth, RenderHeight](FRHICommandListImmediate& RHICmdList)
        {
            FRHITexture* Texture = Resource->GetRenderTargetTexture();
            if (!Texture)
            {
                return;
            }
            RHICmdList.Transition(FRHITransitionInfo(Texture, ERHIAccess::SRVMask, ERHIAccess::CopySrc));
            Readback->EnqueueCopy(
                RHICmdList,
                Texture,
                FResolveRect(0, 0, RenderWidth, RenderHeight));
            RHICmdList.Transition(FRHITransitionInfo(Texture, ERHIAccess::CopySrc, ERHIAccess::SRVMask));
        });
    bCaptureReadbackPending = true;
    if (!bLoggedFirstCapture)
    {
        bLoggedFirstCapture = true;
        SimulatorLog(TEXT("capture_queued"));
    }
}

void AQaiConveyorWorld::ReadbackAndCompressCapture()
{
    if (!CaptureReadback || !CaptureReadback->IsReady())
    {
        return;
    }
    bCaptureReadbackPending = false;
    bEncodingFrame = true;
    const int32 RenderWidth = CaptureRenderWidth;
    const int32 RenderHeight = CaptureRenderHeight;
    const int32 OutputWidth = PendingCaptureOutputWidth;
    const int32 OutputHeight = PendingCaptureOutputHeight;
    const uint32 CaptureGeneration = PendingCaptureGeneration;
    if (!bLoggedFirstReadback)
    {
        bLoggedFirstReadback = true;
        SimulatorLog(TEXT("capture_readback_ready"));
    }
    TWeakObjectPtr<AQaiConveyorWorld> WeakThis(this);
    const TSharedRef<FRHIGPUTextureReadback> Readback = CaptureReadback.ToSharedRef();
    ENQUEUE_RENDER_COMMAND(QaiCopyCaptureReadback)(
        [WeakThis, Readback, RenderWidth, RenderHeight, OutputWidth, OutputHeight, CaptureGeneration](FRHICommandListImmediate& RHICmdList)
        {
            int32 RowPitchPixels = 0;
            int32 BufferHeight = 0;
            void* Buffer = Readback->Lock(RowPitchPixels, &BufferHeight);
            TArray<FColor> Pixels;
            const bool bValid = Buffer
                && RowPitchPixels >= RenderWidth
                && BufferHeight >= RenderHeight;
            if (bValid)
            {
                Pixels.SetNumUninitialized(
                    RenderWidth * RenderHeight);
                const uint8* Source = static_cast<const uint8*>(Buffer);
                for (int32 Row = 0; Row < RenderHeight; ++Row)
                {
                    FMemory::Memcpy(
                        Pixels.GetData() + Row * RenderWidth,
                        Source + Row * RowPitchPixels * sizeof(FColor),
                        RenderWidth * sizeof(FColor));
                }
            }
            Readback->Unlock();

            if (!bValid)
            {
                AsyncTask(ENamedThreads::GameThread, [WeakThis, RowPitchPixels, BufferHeight]()
                {
                    if (AQaiConveyorWorld* Runtime = WeakThis.Get(); Runtime && !Runtime->bIsShuttingDown)
                    {
                        Runtime->bEncodingFrame = false;
                        Runtime->BackendStatus = TEXT("capture readback failed");
                        SimulatorLog(FString::Printf(
                            TEXT("capture_readback_failed row_pitch=%d buffer_height=%d"),
                            RowPitchPixels,
                            BufferHeight));
                    }
                });
                return;
            }

            Async(EAsyncExecution::ThreadPool,
                [WeakThis, RenderWidth, RenderHeight, OutputWidth, OutputHeight, CaptureGeneration, Pixels = MoveTemp(Pixels)]() mutable
            {
                TArray<FColor> OutputPixels;
                if (OutputWidth != RenderWidth
                    || OutputHeight != RenderHeight)
                {
                    FImageUtils::ImageResize(
                        RenderWidth,
                        RenderHeight,
                        Pixels,
                        OutputWidth,
                        OutputHeight,
                        OutputPixels,
                        true,
                        true);
                }
                else
                {
                    OutputPixels = MoveTemp(Pixels);
                }

                // The thumbnail helper is allowed to emit JPEG data and did
                // so on Windows, despite this pipeline declaring image/png.
                // Preserve every captured pixel with an explicitly lossless
                // PNG before sending the frame to Reason2.
                TArray64<uint8> Png;
                FImageUtils::PNGCompressImageArray(
                    OutputWidth,
                    OutputHeight,
                    TArrayView64<const FColor>(OutputPixels.GetData(), OutputPixels.Num()),
                    Png);
                FString Encoded = Png.IsEmpty()
                    ? FString()
                    : FBase64::Encode(Png.GetData(), static_cast<uint32>(Png.Num()));
                AsyncTask(ENamedThreads::GameThread,
                    [WeakThis,
                     Encoded = MoveTemp(Encoded),
                     OutputPixels = MoveTemp(OutputPixels),
                     OutputWidth,
                     OutputHeight,
                     CaptureGeneration]() mutable
                {
                    if (AQaiConveyorWorld* Runtime = WeakThis.Get(); Runtime && !Runtime->bIsShuttingDown)
                    {
                        Runtime->OnFrameEncoded(
                            MoveTemp(Encoded),
                            MoveTemp(OutputPixels),
                            OutputWidth,
                            OutputHeight,
                            CaptureGeneration);
                    }
                });
            });
        });
}
UTexture2D* AQaiConveyorWorld::CreateInferencePreviewTexture(
    const TArray<FColor>& Pixels,
    int32 FrameWidth,
    int32 FrameHeight)
{
    if (Pixels.Num() != FrameWidth * FrameHeight)
    {
        return nullptr;
    }
    const TConstArrayView64<uint8> ImageData(
        reinterpret_cast<const uint8*>(Pixels.GetData()),
        static_cast<int64>(Pixels.Num()) * sizeof(FColor));
    UTexture2D* Texture = UTexture2D::CreateTransient(
        FrameWidth,
        FrameHeight,
        PF_B8G8R8A8,
        NAME_None,
        ImageData);
    if (!Texture || !Texture->GetPlatformData() || Texture->GetPlatformData()->Mips.IsEmpty())
    {
        return nullptr;
    }
    Texture->SRGB = true;
    Texture->NeverStream = true;
    Texture->Filter = TF_Bilinear;
    // CreateTransient initialized the mip before its first render resource was
    // created. Recreate once after setting the display flags so the HUD never
    // races an empty transient resource under heavy GPU load.
    Texture->UpdateResource();
    return Texture;
}

void AQaiConveyorWorld::OnFrameEncoded(
    FString EncodedPng,
    TArray<FColor> Pixels,
    int32 FrameWidth,
    int32 FrameHeight,
    uint32 CaptureGeneration)
{
    if (bIsShuttingDown)
    {
        return;
    }
    bEncodingFrame = false;
    if (CaptureGeneration != InferenceCaptureGeneration)
    {
        SimulatorLog(FString::Printf(
            TEXT("capture_discarded stale_generation=%u active_generation=%u"),
            CaptureGeneration,
            InferenceCaptureGeneration));
        return;
    }
    ++CaptureFrameCount;
    if (!bLoggedFirstCaptureLuminance || CaptureFrameCount % 60 == 0)
    {
        int32 Histogram[256] = {};
        int64 LuminanceSum = 0;
        int32 WhiteClippedPixels = 0;
        for (const FColor& Pixel : Pixels)
        {
            const uint8 Luminance = static_cast<uint8>((54 * Pixel.R + 183 * Pixel.G + 19 * Pixel.B) >> 8);
            ++Histogram[Luminance];
            LuminanceSum += Luminance;
            WhiteClippedPixels += Pixel.R >= 250 && Pixel.G >= 250 && Pixel.B >= 250 ? 1 : 0;
        }
        const int32 PercentileTarget = FMath::CeilToInt(static_cast<float>(Pixels.Num()) * 0.95f);
        int32 RunningPixels = 0;
        int32 P95 = 255;
        for (int32 Value = 0; Value < UE_ARRAY_COUNT(Histogram); ++Value)
        {
            RunningPixels += Histogram[Value];
            if (RunningPixels >= PercentileTarget)
            {
                P95 = Value;
                break;
            }
        }
        const double Mean = Pixels.IsEmpty() ? 0.0 : static_cast<double>(LuminanceSum) / Pixels.Num();
        const double WhiteClipPercent = Pixels.IsEmpty()
            ? 0.0
            : 100.0 * static_cast<double>(WhiteClippedPixels) / Pixels.Num();
        SimulatorLog(FString::Printf(
            TEXT("capture_luminance frame=%d mean_srgb=%.1f p95_srgb=%d neutral_white_clip_percent=%.2f"),
            CaptureFrameCount,
            Mean,
            P95,
            WhiteClipPercent));
        bLoggedFirstCaptureLuminance = true;
    }
    if (!bLoggedFirstEncodedFrame)
    {
        bLoggedFirstEncodedFrame = true;
        SimulatorLog(FString::Printf(TEXT("capture_encoded size=%dx%d base64_chars=%d"), FrameWidth, FrameHeight, EncodedPng.Len()));
    }
    if (bResolutionDataset)
    {
        const int32 SignalIndex = GroundTruthSignal == TEXT("G")
            ? 0
            : (GroundTruthSignal == TEXT("A") ? 1 : (GroundTruthSignal == TEXT("R") ? 2 : INDEX_NONE));
        if (SignalIndex != INDEX_NONE && ResolutionDatasetFrameCounts[SignalIndex] < 10)
        {
            TArray<uint8> PngBytes;
            if (FBase64::Decode(EncodedPng, PngBytes))
            {
                const FString Signal = SignalIndex == 0 ? TEXT("G") : (SignalIndex == 1 ? TEXT("A") : TEXT("R"));
                const int32 FrameNumber = ResolutionDatasetFrameCounts[SignalIndex];
                const FString DiagnosticRoot = FPaths::Combine(
                    FPaths::ProjectSavedDir(),
                    TEXT("Diagnostics/ResolutionSweepAblations"),
                    ResolutionDatasetVariant,
                    FString::Printf(TEXT("%dx%d"), FrameWidth, FrameHeight),
                    Signal);
                IFileManager::Get().MakeDirectory(*DiagnosticRoot, true);
                const FString OutputPath = FPaths::Combine(
                    DiagnosticRoot,
                    FString::Printf(TEXT("frame-%02d.png"), FrameNumber));
                if (FFileHelper::SaveArrayToFile(PngBytes, *OutputPath))
                {
                    ++ResolutionDatasetFrameCounts[SignalIndex];
                    SimulatorLog(FString::Printf(
                        TEXT("resolution_dataset saved signal=%s frame=%d size=%dx%d path=\"%s\""),
                        *Signal,
                        FrameNumber,
                        FrameWidth,
                        FrameHeight,
                        *SanitizeLogField(OutputPath)));
                }
            }
        }
    }
    if (!bInferenceEnabled)
    {
        EncodedFrames.Reset();
        EncodedForkliftFrameTransforms.Reset();
        EncodedFrameTextures.Reset();
        EncodedFrameTimes.Reset();
        return;
    }
    EncodedFrames.Add(MoveTemp(EncodedPng));
    EncodedForkliftFrameTransforms.Add(
        Forklifts[0].Root.IsValid()
            ? Forklifts[0].Root->GetComponentTransform()
            : FTransform::Identity);
    EncodedFrameTextures.Add(CreateInferencePreviewTexture(Pixels, FrameWidth, FrameHeight));
    EncodedFrameTimes.Add(FDateTime::Now().ToString(TEXT("%H:%M:%S")));
    while (EncodedFrames.Num() > 2)
    {
        EncodedFrames.RemoveAt(0);
        EncodedForkliftFrameTransforms.RemoveAt(0);
        EncodedFrameTextures.RemoveAt(0);
        EncodedFrameTimes.RemoveAt(0);
    }
    if (EncodedFrames.Num() == 1)
    {
        // Motion classification requires two distinct observations. The old
        // warm-up duplicated the first image and elicited AMBER even though no
        // temporal evidence existed.
        return;
    }
    SubmitInference();
}

void AQaiConveyorWorld::SubmitInference()
{
    if (!bInferenceEnabled || !bBackendHealthy || bInferenceBusy || EncodedFrames.Num() < 2)
    {
        return;
    }

    // Keep this byte-for-byte equivalent to the original Omniverse
    // HAZARD_PROMPT. Even whitespace changes alter multimodal tokenization.
    const FString Prompt = TEXT(
        "Review the images in chronological order, with the newest image last.\n"
        "Return R if any part of any forklift is inside the marked red zone.\n"
        "Otherwise, return A if any forklift is moving at all.\n"
        "Otherwise, return G.\n"
        "Ignore human workers, parcels, and the stack light.\n"
        "Return one letter only: R, A, or G.\n");
    TArray<TSharedPtr<FJsonValue>> Content;
    for (const FString& Frame : EncodedFrames)
    {
        TSharedPtr<FJsonObject> Url = MakeShared<FJsonObject>();
        Url->SetStringField(TEXT("url"), TEXT("data:image/png;base64,") + Frame);
        TSharedPtr<FJsonObject> Item = MakeShared<FJsonObject>();
        Item->SetStringField(TEXT("type"), TEXT("image_url"));
        Item->SetObjectField(TEXT("image_url"), Url);
        Content.Add(MakeShared<FJsonValueObject>(Item));
    }
    TSharedPtr<FJsonObject> TextItem = MakeShared<FJsonObject>();
    TextItem->SetStringField(TEXT("type"), TEXT("text"));
    TextItem->SetStringField(TEXT("text"), Prompt);
    Content.Add(MakeShared<FJsonValueObject>(TextItem));

    TSharedPtr<FJsonObject> Message = MakeShared<FJsonObject>();
    Message->SetStringField(TEXT("role"), TEXT("user"));
    Message->SetArrayField(TEXT("content"), Content);
    TSharedPtr<FJsonObject> Body = MakeShared<FJsonObject>();
    Body->SetStringField(TEXT("model"), CurrentModelName());
    Body->SetNumberField(TEXT("temperature"), 0.0);
    Body->SetNumberField(TEXT("top_k"), 1);
    Body->SetNumberField(TEXT("seed"), 42);
    Body->SetNumberField(TEXT("max_completion_tokens"), 32);
    Body->SetBoolField(TEXT("enable_think"), false);
    Body->SetStringField(TEXT("grammar"), TEXT("root ::= [GAR]"));
    Body->SetStringField(TEXT("grammar_string"), TEXT("root ::= [GAR]"));
    Body->SetArrayField(TEXT("messages"), {MakeShared<FJsonValueObject>(Message)});
    FString BodyText;
    const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&BodyText);
    FJsonSerializer::Serialize(Body.ToSharedRef(), Writer);

    TSharedRef<IHttpRequest> Request = FHttpModule::Get().CreateRequest();
    InferenceRequest = Request;
    Request->SetURL(CurrentServerUrl() + TEXT("/v1/chat/completions"));
    Request->SetVerb(TEXT("POST"));
    Request->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    Request->SetTimeout(ActiveBackend == TEXT("evk") ? 120.0f : 60.0f);
    Request->SetContentAsString(BodyText);
    if (!bLoggedFirstInferenceSubmission)
    {
        bLoggedFirstInferenceSubmission = true;
        SimulatorLog(FString::Printf(
            TEXT("inference_submitted backend=%s request_bytes=%d frames=%d prompt_profile=omniverse_compact_r5 detector_projection=authored_endline"),
            *ActiveBackend,
            BodyText.Len(),
            EncodedFrames.Num()));
    }
    TWeakObjectPtr<AQaiConveyorWorld> WeakThis(this);
    Request->OnProcessRequestComplete().BindLambda([WeakThis](FHttpRequestPtr, FHttpResponsePtr Response, bool bSucceeded)
    {
        if (AQaiConveyorWorld* Runtime = WeakThis.Get())
        {
            Runtime->InferenceRequest.Reset();
            Runtime->HandleModelResponse(Response.IsValid() ? Response->GetContentAsString() : FString(), bSucceeded, Response.IsValid() ? Response->GetResponseCode() : 0);
        }
    });
    bInferenceBusy = true;
    RequestStartSeconds = FPlatformTime::Seconds();
    const bool bRequestStarted = Request->ProcessRequest();
    if (!bRequestStarted)
    {
        InferenceRequest.Reset();
        bInferenceBusy = false;
        BackendStatus = TEXT("request could not start");
        bBackendHealthy = false;
        SimulatorLog(FString::Printf(
            TEXT("inference_request_start_failed backend=%s url=%s"),
            *ActiveBackend,
            *CurrentServerUrl()));
    }
    else
    {
        // Snapshot only after HTTP accepts the request. These are therefore
        // the exact two images fed to Reason2, not a separate preview camera.
        SubmittedEncodedFrames = EncodedFrames;
        SubmittedFrameTextures = EncodedFrameTextures;
        SubmittedFrameTimes = EncodedFrameTimes;
        SubmittedGroundTruthSignal = GroundTruthSignal;
        bSubmittedRedOverlap = SubmittedGroundTruthSignal == TEXT("R");
        bSubmittedForkliftMotion = false;
        if (EncodedForkliftFrameTransforms.Num() >= 2)
        {
            const FTransform& Older = EncodedForkliftFrameTransforms[EncodedForkliftFrameTransforms.Num() - 2];
            const FTransform& Newest = EncodedForkliftFrameTransforms.Last();
            const float TranslationDeltaCm = FVector::Dist(
                Older.GetLocation(),
                Newest.GetLocation());
            const float YawDeltaDegrees = FMath::Abs(FMath::FindDeltaAngleDegrees(
                Older.Rotator().Yaw,
                Newest.Rotator().Yaw));
            // These thresholds reject suspension/solver noise while remaining
            // well below the visible displacement at the one-second cadence.
            bSubmittedForkliftMotion = TranslationDeltaCm >= 1.25f
                || YawDeltaDegrees >= 0.50f;
            SimulatorLog(FString::Printf(
                TEXT("inference_visual_tracker translation_delta_cm=%.2f yaw_delta_deg=%.2f moving=%s red_overlap=%s"),
                TranslationDeltaCm,
                YawDeltaDegrees,
                bSubmittedForkliftMotion ? TEXT("true") : TEXT("false"),
                bSubmittedRedOverlap ? TEXT("true") : TEXT("false")));
        }
        if (!bLoggedFirstInferencePreview)
        {
            bLoggedFirstInferencePreview = true;
            SimulatorLog(FString::Printf(
                TEXT("inference_preview_updated frames=%d oldest=%s newest=%s"),
                SubmittedFrameTextures.Num(),
                SubmittedFrameTimes.IsValidIndex(0) ? *SubmittedFrameTimes[0] : TEXT("-"),
                SubmittedFrameTimes.IsValidIndex(1) ? *SubmittedFrameTimes[1] : TEXT("-")));
        }
    }
}

void AQaiConveyorWorld::HandleModelResponse(const FString& Body, bool bSucceeded, int32 ResponseCode)
{
    bInferenceBusy = false;
    LastInferenceMilliseconds = (FPlatformTime::Seconds() - RequestStartSeconds) * 1000.0;
    if (!bSucceeded || ResponseCode < 200 || ResponseCode >= 300)
    {
        bBackendHealthy = false;
        BackendStatus = FString::Printf(TEXT("HTTP %d"), ResponseCode);
        UE_LOG(LogTemp, Warning, TEXT("Reason2 inference failed: backend=%s status=%d"), *ActiveBackend, ResponseCode);
        SimulatorLog(FString::Printf(TEXT("inference_failed backend=%s http_status=%d"), *ActiveBackend, ResponseCode));
        return;
    }

    TSharedPtr<FJsonObject> Json;
    const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Body);
    FString Content;
    if (FJsonSerializer::Deserialize(Reader, Json) && Json.IsValid())
    {
        const TArray<TSharedPtr<FJsonValue>>* Choices = nullptr;
        if (Json->TryGetArrayField(TEXT("choices"), Choices) && Choices && Choices->Num() > 0)
        {
            const TSharedPtr<FJsonObject>* Choice = nullptr;
            const TSharedPtr<FJsonObject>* Message = nullptr;
            if ((*Choices)[0]->TryGetObject(Choice) && Choice && (*Choice)->TryGetObjectField(TEXT("message"), Message) && Message)
            {
                (*Message)->TryGetStringField(TEXT("content"), Content);
            }
        }
        if (Content.IsEmpty())
        {
            Json->TryGetStringField(TEXT("signal"), Content);
        }
    }

    FString Proposed;
    const FRegexPattern StandaloneSignalPattern(TEXT("(?i)(?:^|[^A-Z])([GAR])(?:[^A-Z]|$)"));
    FRegexMatcher SignalMatcher(StandaloneSignalPattern, Content);
    while (SignalMatcher.FindNext())
    {
        Proposed = SignalMatcher.GetCaptureGroup(1).ToUpper();
    }
    if (Proposed.IsEmpty())
    {
        BackendStatus = TEXT("invalid model response");
        UE_LOG(LogTemp, Warning, TEXT("Reason2 returned no standalone G/A/R signal"));
        SimulatorLog(FString::Printf(TEXT("inference_invalid_response backend=%s"), *ActiveBackend));
        return;
    }
    if (FParse::Param(FCommandLine::Get(), TEXT("QaiSaveInferenceFrames")))
    {
        const FString DiagnosticRoot = FPaths::Combine(
            FPaths::ProjectSavedDir(),
            TEXT("Diagnostics/Reason2Pairs"));
        IFileManager::Get().MakeDirectory(*DiagnosticRoot, true);
        for (int32 FrameIndex = 0; FrameIndex < SubmittedEncodedFrames.Num(); ++FrameIndex)
        {
            TArray<uint8> PngBytes;
            if (FBase64::Decode(SubmittedEncodedFrames[FrameIndex], PngBytes))
            {
                const FString FrameRole = FrameIndex == 0 ? TEXT("older") : TEXT("newest");
                const FString Filename = FString::Printf(
                    TEXT("last-raw-%s-truth-%s-%s.png"),
                    *Proposed,
                    *SubmittedGroundTruthSignal,
                    *FrameRole);
                FFileHelper::SaveArrayToFile(
                    PngBytes,
                    *FPaths::Combine(DiagnosticRoot, Filename));
            }
        }
    }
    RawModelSignal = Proposed;
    // Retain ModelSignal as a compatibility alias for UI/API consumers, but
    // do not apply a physics or temporal consistency gate to model output.
    ModelSignal = Proposed;
    UpdateStackLights();
    BackendStatus = FString::Printf(TEXT("ready, %.0f ms"), LastInferenceMilliseconds);
    bBackendHealthy = true;
    UE_LOG(
        LogTemp,
        Display,
        TEXT("Reason2 inference complete: backend=%s raw=%s ground_truth=%s latency_ms=%.0f"),
        *ActiveBackend,
        *RawModelSignal,
        *SubmittedGroundTruthSignal,
        LastInferenceMilliseconds);
    ++InferenceSuccessCount;
    const double Now = FPlatformTime::Seconds();
    const FString InferenceState = FString::Printf(
        TEXT("%s/%s"),
        *RawModelSignal,
        *SubmittedGroundTruthSignal);
    const bool bStateChanged = InferenceState != LastLoggedInferenceState;
    const bool bPeriodicSummary = Now - LastInferenceSummaryLogSeconds >= 60.0;
    const bool bSlowResponse = LastInferenceMilliseconds >= (ActiveBackend == TEXT("evk") ? 30000.0 : 5000.0);
    if (InferenceSuccessCount == 1 || bStateChanged || bPeriodicSummary || bSlowResponse)
    {
        SimulatorLog(FString::Printf(
            TEXT("inference_complete backend=%s raw=%s ground_truth=%s latency_ms=%.0f successes=%d reason=%s"),
            *ActiveBackend,
            *RawModelSignal,
            *SubmittedGroundTruthSignal,
            LastInferenceMilliseconds,
            InferenceSuccessCount,
            bSlowResponse ? TEXT("slow") : (bStateChanged ? TEXT("state_change") : TEXT("heartbeat"))));
        LastLoggedInferenceState = InferenceState;
        LastInferenceSummaryLogSeconds = Now;
    }
}

void AQaiConveyorWorld::ProbeBackend()
{
    BackendProbeAccumulator = 0.0f;
    if (ProbeRequest)
    {
        ProbeRequest->OnProcessRequestComplete().Unbind();
        ProbeRequest->CancelRequest();
        ProbeRequest.Reset();
    }
    TSharedRef<IHttpRequest> Request = FHttpModule::Get().CreateRequest();
    ProbeRequest = Request;
    Request->SetURL(CurrentServerUrl() + TEXT("/v1/models"));
    Request->SetVerb(TEXT("GET"));
    Request->SetTimeout(3.0f);
    const FString ExpectedModel = CurrentModelName();
    TWeakObjectPtr<AQaiConveyorWorld> WeakThis(this);
    Request->OnProcessRequestComplete().BindLambda([WeakThis, ExpectedModel](FHttpRequestPtr, FHttpResponsePtr Response, bool bSucceeded)
    {
        if (AQaiConveyorWorld* Runtime = WeakThis.Get())
        {
            Runtime->ProbeRequest.Reset();
            const bool bHttpReady = bSucceeded && Response.IsValid() && EHttpResponseCodes::IsOk(Response->GetResponseCode());
            Runtime->bBackendHealthy = bHttpReady && Response->GetContentAsString().Contains(ExpectedModel);
            Runtime->BackendStatus = Runtime->bBackendHealthy
                ? TEXT("ready")
                : (bHttpReady ? TEXT("requested model unavailable") : TEXT("offline; provisioner may still be starting"));
            const double Now = FPlatformTime::Seconds();
            const bool bStateChanged = !Runtime->bHasLoggedBackendProbe
                || bHttpReady != Runtime->bLastLoggedBackendHttpReady
                || Runtime->bBackendHealthy != Runtime->bLastLoggedBackendModelReady;
            if (bStateChanged || Now - Runtime->LastBackendProbeLogSeconds >= 60.0)
            {
                SimulatorLog(FString::Printf(
                    TEXT("backend_probe backend=%s http_ready=%s model_ready=%s status=%d reason=%s"),
                    *Runtime->ActiveBackend,
                    bHttpReady ? TEXT("true") : TEXT("false"),
                    Runtime->bBackendHealthy ? TEXT("true") : TEXT("false"),
                    Response.IsValid() ? Response->GetResponseCode() : 0,
                    bStateChanged ? TEXT("state_change") : TEXT("heartbeat")));
                Runtime->bHasLoggedBackendProbe = true;
                Runtime->bLastLoggedBackendHttpReady = bHttpReady;
                Runtime->bLastLoggedBackendModelReady = Runtime->bBackendHealthy;
                Runtime->LastBackendProbeLogSeconds = Now;
            }
        }
    });
    BackendStatus = TEXT("checking");
    if (!Request->ProcessRequest())
    {
        ProbeRequest.Reset();
        BackendStatus = TEXT("probe could not start");
        bBackendHealthy = false;
        SimulatorLog(FString::Printf(
            TEXT("backend_probe_start_failed backend=%s url=%s"),
            *ActiveBackend,
            *CurrentServerUrl()));
    }
}

FString AQaiConveyorWorld::CurrentServerUrl() const
{
    FString Url = ActiveBackend == TEXT("evk") ? EvkServerUrl : HostServerUrl;
    Url.RemoveFromEnd(TEXT("/"));
    return Url;
}

FString AQaiConveyorWorld::CurrentModelName() const
{
    return ActiveBackend == TEXT("evk") ? EvkModel : HostModel;
}
