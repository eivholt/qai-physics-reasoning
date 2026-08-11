#include "QaiConveyorWorld.h"
#include "QaiConveyorGameMode.h"

#include "Async/Async.h"
#include "Components/PointLightComponent.h"
#include "Components/BoxComponent.h"
#include "Components/CapsuleComponent.h"
#include "Components/SpotLightComponent.h"
#include "Components/MaterialBillboardComponent.h"
#include "Components/PrimitiveComponent.h"
#include "Components/RectLightComponent.h"
#include "Components/LineBatchComponent.h"
#include "Components/LightComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/SceneComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Components/ExponentialHeightFogComponent.h"
#include "Components/LocalFogVolumeComponent.h"
#include "Dom/JsonObject.h"
#include "DrawDebugHelpers.h"
#include "Engine/ExponentialHeightFog.h"
#include "Engine/Engine.h"
#include "Engine/CollisionProfile.h"
#include "Engine/OverlapResult.h"
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
#include "PhysicalMaterials/PhysicalMaterial.h"
#include "PhysicsEngine/PhysicsConstraintComponent.h"
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
    // Loaded industrial lift speed. The previous 90 cm/s outran thin-contact
    // cargo manifolds and could launch marginal pallet loads.
    constexpr float LiftSpeedCm = 55.0f;
    constexpr float MaxLiftCm = 196.0f;
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
    // The west wall and storage rack are translated two metres outward to
    // provide a larger forklift manoeuvring lane. Keep the analytic world
    // guard in step with the physical floor/wall extension.
    constexpr float WorldMinimumX = -850.0f;
    // The broad analytic limits remain a final fallback. The two visually
    // open presentation sides use fitted collision-only walls at the rendered
    // floor edge so a vehicle can never appear to drive into empty space.
    constexpr float WorldMaximumX = 850.0f;
    constexpr float WorldMinimumY = -560.0f;
    constexpr float WorldMaximumY = 700.0f;
    constexpr float RedClearanceCm = 100.0f;
    constexpr float MovingThresholdCm = 5.0f;
    // Keep the conveyor visibly powered, but deliberately slow enough that its
    // parcel motion remains background context instead of competing with the
    // forklift's two-second temporal signal in Reason2 input.
    constexpr float ConveyorSpeedCm = 20.0f;
    // Powered rollers build parcel speed over several contacts.  The old
    // 65-ms velocity servo could apply nearly one g in a single step and turn
    // small contact/impact errors into launches at curves and transitions.
    constexpr float ConveyorTangentialResponseSeconds = 0.65f;
    // The analytic support is geometrically stationary, unlike the skin of a
    // powered roller. This feed-forward term cancels that proxy-only rolling
    // resistance; the velocity-error term above remains the parcel's net drive.
    constexpr float ConveyorProxyResistanceCompensationCm = 135.0f;
    constexpr float ConveyorResistanceFadeSpeedCm = 18.0f;
    constexpr float ConveyorMaximumParcelAccelerationCm = 230.0f;
    // The authoritative Chaos belt is supported by individual cylindrical
    // contacts. Five motor samples across each parcel footprint approximate
    // the powered roller skins and naturally lose support as cargo overhangs.
    constexpr float ConveyorRollerRadiusCm = 4.8f;
    constexpr float ConveyorRollerSpacingCm = 9.25f;
    constexpr float ConveyorRollerHalfLengthCm = 54.0f;
    constexpr int32 ConveyorMotorContactSamples = 5;
    // Curved industrial roller conveyors use tapered or mildly skewed rollers
    // to cancel outward migration. Model that as a small axial surface speed
    // at each real contact patch, not a position or trajectory constraint.
    constexpr float ConveyorAxialTargetSpeedPerOffset = 0.28f;
    constexpr float ConveyorMaximumAxialTargetSpeedCm = 18.0f;
    constexpr float ConveyorAxialResponseSeconds = 0.25f;
    constexpr float ConveyorMaximumAxialAccelerationCm = 80.0f;
    constexpr float ParcelAmberSupportFraction = 0.68f;
    // Red means the parcel has actually lost the conveyor, not merely that a
    // low corner has dipped below the roller crown while the box is tipping.
    constexpr float ParcelRedSupportFraction = 0.05f;
    constexpr float ParcelFallenSupportFraction = 0.25f;
    constexpr float ParcelFallenDropCm = 15.0f;
    constexpr float ParcelDangerPredictionSeconds = 0.65f;
    constexpr float ParcelDangerTiltDegrees = 16.0f;
    // Chaos forklift tuning uses centimetres, kilograms and seconds, matching
    // Unreal's rigid-body force units (kg*cm/s^2).
    constexpr float ForkliftMaximumSpeedCm = 520.0f;
    constexpr float ForkliftMaximumReverseSpeedCm = 360.0f;
    // A loaded industrial truck typically accelerates gently to protect its
    // cargo. Bound tractive effort independently of tire grip so full trigger
    // does not deliver a near-friction-limit launch.
    constexpr float ForkliftMaximumDriveAccelerationCm = 118.0f;
    constexpr float ForkliftMaximumBrakeDecelerationCm = 135.0f;
    constexpr float ForkliftMaximumSteerDegrees = 38.0f;
    // A real counterbalanced forklift's front axle is effectively rigid; most
    // compliance is tire deflection plus limited rear-axle articulation. Keep
    // only enough front movement to read visually as damped compliance.
    constexpr float ForkliftFrontSuspensionRestCm = 2.8f;
    constexpr float ForkliftFrontSuspensionDroopCm = 0.4f;
    constexpr float ForkliftFrontSuspensionMaximumCompressionCm = 1.6f;
    constexpr float ForkliftFrontSuspensionStiffness = 1150000.0f;
    constexpr float ForkliftFrontSuspensionDamping = 115000.0f;
    // The rear axle remains slightly more compliant so one wheel can articulate
    // over a pallet edge without turning the vehicle into a passenger car.
    constexpr float ForkliftRearSuspensionRestCm = 3.6f;
    constexpr float ForkliftRearSuspensionDroopCm = 1.1f;
    constexpr float ForkliftRearSuspensionMaximumCompressionCm = 2.4f;
    constexpr float ForkliftRearSuspensionStiffness = 850000.0f;
    constexpr float ForkliftRearSuspensionDamping = 85000.0f;
    // The imported rendered tire is larger than the finite Chaos wheel disk.
    // This preload accounts for that radius difference and starts both axles
    // near their static-load compression instead of dropping onto the springs.
    constexpr float ForkliftSuspensionPreloadCompressionCm = 4.8f;
    constexpr float ForkliftTireLongitudinalStiffness = 11600.0f;
    constexpr float ForkliftTireLateralStiffness = 18500.0f;
    constexpr float ForkliftTireFriction = 0.92f;
    constexpr float ForkliftRollingResistance = 0.018f;
    // Passive driveline/tire drag when neither drive trigger is held. Raycast
    // wheels do not inherit Chaos's contact-patch static friction, so model a
    // modest static-like hold that prevents a truck straddling the raised mat
    // from creeping downhill without turning it into a position lock.
    constexpr float ForkliftIdleDrivelineDrag = 0.045f;
    constexpr float ForkliftIdleVelocityDampingPerSecond = 3.0f;
    constexpr float ForkliftIdleMaximumDrag = 0.09f;
    // Keep the commanded hydraulic target just inside the physical prismatic
    // stop. Driving a velocity target into the exact Chaos limit turns the
    // end stop into a jack between the carriage and chassis.
    constexpr float ForkliftLiftTravelCm = 200.0f;
    constexpr float ForkliftMaximumCommandedLiftCm = 196.0f;
    constexpr float ForkliftLiftSpeedCm = 72.0f;
    // Spawn/reset the loaded carriage slightly above its mechanical lower
    // stop. This prevents the pallet runners or tine wedges from being caught
    // in the floor while preserving the full downward command range.
    constexpr float ForkliftInitialLiftCm = 10.0f;
    // Real mast carriages run between opposed guide rollers. The prismatic
    // Chaos joint remains authoritative for vertical motion; this bounded
    // equal-and-opposite PD force models the guide-roller preload so a pallet
    // impact can react into and tip the chassis without peeling the carriage
    // horizontally away from the rails.
    constexpr float ForkliftLiftRailStiffness = 85000.0f;
    constexpr float ForkliftLiftRailDamping = 36000.0f;
    constexpr float ForkliftLiftRailMaximumForce = 6000000.0f;
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
    constexpr float WorkerMassKg = 82.0f;
    // A dynamic Chaos capsule resting on the high-friction warehouse floor
    // needs enough tangential force to overcome its own Coulomb contact.  The
    // old 260 cm/s^2 cap was below mu*g, so both workers looked permanently
    // wedged even on an unobstructed route.  This remains a force-limited
    // motor (the forklift can still displace a worker), with a walking-contact
    // feed-forward term rather than a kinematic velocity assignment.
    constexpr float WorkerMotorMaximumAccelerationCm = 1250.0f;
    constexpr float WorkerMotorResponsePerSecond = 8.5f;
    constexpr float WorkerMotorGroundFrictionCompensationCm = 550.0f;
    constexpr float WorkerAvoidanceProbeCm = 32.0f;
    constexpr float WorkerCapsuleHalfHeightCm = 88.0f;
    constexpr int32 LoosePalletStackCount = 3;
    // Detailed pallet runners are narrow contact patches. Extend their Chaos
    // skin only eight millimetres below the render mesh so solver penetration
    // cannot make timber appear to pass through a floor, fork, or lower pallet.
    constexpr float PalletContactSkinCm = 0.8f;
    constexpr float WorkerMaximumVisualTurnRateDegreesPerSecond = 220.0f;
    constexpr float WorkerHeadingResponsePerSecond = 7.0f;
    constexpr float WorkerHeadingCommitSeconds = 0.10f;
    constexpr float WorkerHeadingDeadbandDegrees = 12.0f;
    constexpr float WorkerRouteReversalCooldownSeconds = 1.15f;
    // Keep a small comfort gap outside the 38-cm collision capsules.  Route
    // reservation starts before contact, so pedestrians yield rather than
    // entering an overlap and relying on penetration correction.
    constexpr float WorkerPersonalSpaceCm = 88.0f;
    constexpr float WorkerConflictReleaseDistanceCm = 112.0f;
    constexpr float WorkerConflictLookaheadSeconds = 0.70f;
    constexpr float WorkerCrossingYieldSeconds = 0.65f;
    constexpr float WorkerHeadOnYieldSeconds = 0.16f;
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
    constexpr int32 EvkVideoFrameCount = 2;

    struct FPreparedInferenceMedia
    {
        FString Base64;
        FString MimeType;
        FString Transport;
        FString Error;
        int32 Width = 0;
        int32 Height = 0;
        int64 Bytes = 0;

        bool IsValid() const
        {
            return Error.IsEmpty() && !Base64.IsEmpty();
        }
    };

    FString WindowsPathToWsl(FString Path)
    {
        Path = FPaths::ConvertRelativePathToFull(Path);
        Path.ReplaceInline(TEXT("\\"), TEXT("/"));
        if (Path.Len() >= 3 && Path[1] == TEXT(':'))
        {
            const TCHAR Drive = FChar::ToLower(Path[0]);
            return FString::Printf(TEXT("/mnt/%c/%s"), Drive, *Path.Mid(3));
        }
        return Path;
    }

    bool ResolveLosslessFfmpeg(
        const FString& Override,
        FString& OutExecutable,
        bool& OutUseWsl,
        FString& OutLabel)
    {
        OutUseWsl = false;
        if (!Override.IsEmpty())
        {
            OutExecutable = Override;
            OutLabel = TEXT("command_line_override");
            return true;
        }

#if PLATFORM_WINDOWS
        const TArray<FString> Candidates = {
            FPaths::Combine(FPlatformProcess::BaseDir(), TEXT("ThirdParty/FFmpeg/Win64/ffmpeg.exe")),
            FPaths::Combine(FPaths::ProjectDir(), TEXT("ThirdParty/FFmpeg/Win64/ffmpeg.exe")),
            FPaths::Combine(FPaths::ProjectDir(), TEXT("Binaries/ThirdParty/FFmpeg/Win64/ffmpeg.exe")),
        };
        for (const FString& Candidate : Candidates)
        {
            if (FPaths::FileExists(Candidate))
            {
                OutExecutable = Candidate;
                OutLabel = TEXT("packaged_win64");
                return true;
            }
        }
        const FString Wsl = FPaths::Combine(
            FPlatformMisc::GetEnvironmentVariable(TEXT("WINDIR")),
            TEXT("System32/wsl.exe"));
        if (FPaths::FileExists(Wsl))
        {
            OutExecutable = Wsl;
            OutUseWsl = true;
            OutLabel = TEXT("development_wsl");
            return true;
        }
#elif PLATFORM_MAC
        const TArray<FString> Candidates = {
            FPaths::Combine(FPlatformProcess::BaseDir(), TEXT("ThirdParty/FFmpeg/Mac/ffmpeg")),
            FPaths::Combine(FPaths::ProjectDir(), TEXT("ThirdParty/FFmpeg/Mac/ffmpeg")),
            TEXT("/opt/homebrew/bin/ffmpeg"),
            TEXT("/usr/local/bin/ffmpeg"),
        };
        for (const FString& Candidate : Candidates)
        {
            if (FPaths::FileExists(Candidate))
            {
                OutExecutable = Candidate;
                OutLabel = Candidate.Contains(TEXT("ThirdParty"))
                    ? TEXT("packaged_mac")
                    : TEXT("development_system_mac");
                return true;
            }
        }
#endif
        return false;
    }

    FPreparedInferenceMedia BuildLosslessEvkVideo(
        const TArray<FString>& Base64PngFrames,
        int32 FrameWidth,
        int32 FrameHeight,
        float FramesPerSecond,
        const FString& FfmpegOverride,
        bool bSaveDiagnostics)
    {
        FPreparedInferenceMedia Result;
        Result.MimeType = TEXT("video/mp4");
        Result.Transport = TEXT("pixel_lossless_rgb_h264_mp4");
        Result.Width = FrameWidth;
        Result.Height = FrameHeight;
        if (Base64PngFrames.Num() != EvkVideoFrameCount)
        {
            Result.Error = FString::Printf(
                TEXT("EVK video requires %d frames; received %d"),
                EvkVideoFrameCount,
                Base64PngFrames.Num());
            return Result;
        }

        FString Executable;
        FString EncoderLabel;
        bool bUseWsl = false;
        if (!ResolveLosslessFfmpeg(
                FfmpegOverride, Executable, bUseWsl, EncoderLabel))
        {
            Result.Error = TEXT(
                "no lossless FFmpeg encoder found; package ThirdParty/FFmpeg or set -QaiFfmpeg=<path>");
            return Result;
        }

        const FString RequestId = FString::Printf(
            TEXT("window-%s-%llu"),
            *FDateTime::UtcNow().ToString(TEXT("%Y%m%dT%H%M%S")),
            static_cast<unsigned long long>(FPlatformTime::Cycles64()));
        const FString WorkingDirectory = FPaths::Combine(
            FPaths::ProjectSavedDir(), TEXT("Temp/Reason2Video"), RequestId);
        IFileManager::Get().MakeDirectory(*WorkingDirectory, true);
        for (int32 FrameIndex = 0; FrameIndex < Base64PngFrames.Num(); ++FrameIndex)
        {
            TArray<uint8> Bytes;
            if (!FBase64::Decode(Base64PngFrames[FrameIndex], Bytes)
                || Bytes.IsEmpty()
                || !FFileHelper::SaveArrayToFile(
                    Bytes,
                    *FPaths::Combine(
                        WorkingDirectory,
                        FString::Printf(TEXT("frame-%02d.png"), FrameIndex))))
            {
                Result.Error = FString::Printf(
                    TEXT("could not stage EVK video frame %d"), FrameIndex);
                IFileManager::Get().DeleteDirectory(*WorkingDirectory, false, true);
                return Result;
            }
        }

        const FString InputPattern = FPaths::Combine(WorkingDirectory, TEXT("frame-%02d.png"));
        const FString OutputPath = FPaths::Combine(WorkingDirectory, TEXT("reason2-window.mp4"));
        const FString EncoderArguments = FString::Printf(
            // Match the proven real-video bitstream profile. The deployed
            // GenieX decoder terminates on libx264rgb's ultrafast stream even
            // though it is H.264 compliant. Medium remains pixel-lossless.
            // Do not move the MP4 moov atom to the front: this patched
            // GenieX reader terminates on faststart files but accepts the
            // standard moov-at-end layout used by the validated clips.
            TEXT("-hide_banner -loglevel error -y -framerate %.6g -start_number 0 -i \"%s\" -frames:v %d -vf scale=%d:-2:flags=lanczos -c:v libx264rgb -preset medium -crf 0 -pix_fmt rgb24 \"%s\""),
            FramesPerSecond,
            *(bUseWsl ? WindowsPathToWsl(InputPattern) : InputPattern),
            EvkVideoFrameCount,
            FrameWidth,
            *(bUseWsl ? WindowsPathToWsl(OutputPath) : OutputPath));
        const FString Arguments = bUseWsl
            ? TEXT("-e ffmpeg ") + EncoderArguments
            : EncoderArguments;
        int32 ReturnCode = -1;
        FString StandardOutput;
        FString StandardError;
        const bool bExecuted = FPlatformProcess::ExecProcess(
            *Executable,
            *Arguments,
            &ReturnCode,
            &StandardOutput,
            &StandardError);
        TArray<uint8> Mp4Bytes;
        if (!bExecuted
            || ReturnCode != 0
            || !FFileHelper::LoadFileToArray(Mp4Bytes, *OutputPath)
            || Mp4Bytes.IsEmpty())
        {
            Result.Error = FString::Printf(
                TEXT("lossless FFmpeg encode failed source=%s exit=%d error=%s"),
                *EncoderLabel,
                ReturnCode,
                *StandardError.Left(400).Replace(TEXT("\r"), TEXT(" ")).Replace(TEXT("\n"), TEXT(" ")));
            IFileManager::Get().DeleteDirectory(*WorkingDirectory, false, true);
            return Result;
        }

        if (bSaveDiagnostics)
        {
            const FString DiagnosticRoot = FPaths::Combine(
                FPaths::ProjectSavedDir(), TEXT("Diagnostics/Reason2Windows"));
            IFileManager::Get().MakeDirectory(*DiagnosticRoot, true);
            FFileHelper::SaveArrayToFile(
                Mp4Bytes,
                *FPaths::Combine(DiagnosticRoot, TEXT("last-submitted-evk-lossless.mp4")));
        }
        Result.Bytes = Mp4Bytes.Num();
        Result.Base64 = FBase64::Encode(Mp4Bytes);
        Result.Transport += TEXT("_") + EncoderLabel;
        IFileManager::Get().DeleteDirectory(*WorkingDirectory, false, true);
        return Result;
    }

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
    // Controls and belt forces are authored before Chaos advances its
    // configured fixed substeps. Chaos is the sole rigid-body authority.
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
    bResetLiftTest = FParse::Param(FCommandLine::Get(), TEXT("QaiResetLiftTest"));
    bRuntimeMotionTest = bRuntimeMotionTest || bPhysicsContactTest || bShelfForkTest
        || bEvkPhysicsTest || bForkliftClimbTest || bForkEdgeBalanceTest || bForkWedgeTest
        || bResetLiftTest;
    bHudScreenshotTest = FParse::Param(FCommandLine::Get(), TEXT("QaiHudScreenshotTest"));
    bWorkerSoakTest = FParse::Param(FCommandLine::Get(), TEXT("QaiWorkerSoakTest"));
    bPresentation4K = FParse::Param(FCommandLine::Get(), TEXT("QaiPresentation4K"));
    bResolutionDataset = FParse::Param(FCommandLine::Get(), TEXT("QaiResolutionDataset"));
    bResolutionDatasetWhiteFloor = FParse::Param(FCommandLine::Get(), TEXT("QaiResolutionDatasetWhiteFloor"));
    bResolutionDatasetHideWorkers = FParse::Param(FCommandLine::Get(), TEXT("QaiResolutionDatasetHideWorkers"));
    bResolutionDatasetHideParcels = FParse::Param(FCommandLine::Get(), TEXT("QaiResolutionDatasetHideParcels"));
    bResolutionDatasetAutoExit = FParse::Param(FCommandLine::Get(), TEXT("QaiResolutionDatasetAutoExit"));
    bInteractiveValidationUnloaded =
        FParse::Param(FCommandLine::Get(), TEXT("QaiInteractiveValidationUnloaded"));
    bInteractiveDriveValidation =
        FParse::Param(FCommandLine::Get(), TEXT("QaiInteractiveDriveValidation"));
    FParse::Value(FCommandLine::Get(), TEXT("QaiResolutionDatasetFrames="), ResolutionDatasetFramesPerSignal);
    ResolutionDatasetFramesPerSignal = FMath::Clamp(ResolutionDatasetFramesPerSignal, 8, 40);
    FParse::Value(FCommandLine::Get(), TEXT("QaiForkliftPaint="), ForkliftPaintVariant);
    ForkliftPaintVariant = ForkliftPaintVariant.TrimStartAndEnd().ToLower();
    FParse::Value(FCommandLine::Get(), TEXT("QaiConveyorSpeedScale="), ConveyorSpeedScale);
    ConveyorSpeedScale = FMath::Clamp(ConveyorSpeedScale, 0.0f, 2.0f);
    SimulatorLog(FString::Printf(
        TEXT("conveyor_runtime speed_scale=%.3f target_speed_cm_s=%.2f validation_override=%s"),
        ConveyorSpeedScale,
        ConveyorTuning::ConveyorSpeedCm * ConveyorSpeedScale,
        ConveyorSpeedScale == 1.0f ? TEXT("false") : TEXT("true")));
    FParse::Value(FCommandLine::Get(), TEXT("QaiCaptureInterval="), CaptureIntervalSeconds);
    float CaptureFramesPerSecond = 1.0f / CaptureIntervalSeconds;
    if (FParse::Value(FCommandLine::Get(), TEXT("QaiCaptureFps="), CaptureFramesPerSecond))
    {
        CaptureIntervalSeconds = 1.0f / FMath::Clamp(CaptureFramesPerSecond, 0.5f, 10.0f);
    }
    CaptureIntervalSeconds = FMath::Clamp(CaptureIntervalSeconds, 0.1f, 2.0f);
    // Capture remains at 4 FPS for responsive host inference and a well-spaced
    // source buffer. EVK selects the same two endpoint observations as the host
    // across the two-second temporal baseline instead of a dense storyboard.
    FParse::Value(
        FCommandLine::Get(),
        TEXT("QaiEvkTemporalBaseline="),
        EvkTemporalBaselineSeconds);
    EvkTemporalBaselineSeconds = FMath::Clamp(EvkTemporalBaselineSeconds, 0.75f, 4.0f);
    FParse::Value(
        FCommandLine::Get(),
        TEXT("QaiHostTemporalBaseline="),
        HostTemporalBaselineSeconds);
    HostTemporalBaselineSeconds = FMath::Clamp(HostTemporalBaselineSeconds, 0.5f, 4.0f);
    bSaveInferenceFrames = FParse::Param(FCommandLine::Get(), TEXT("QaiSaveInferenceFrames"));
    FParse::Value(FCommandLine::Get(), TEXT("QaiFfmpeg="), FfmpegExecutableOverride);
    SimulatorLog(FString::Printf(
        TEXT("inference_window host_frames=2 host_span_seconds=%.3f host_transport=two_separate_lossless_png evk_frames=%d capture_fps=%.3f evk_span_seconds=%.3f evk_transport=pixel_lossless_rgb_h264_mp4 save_media=%s"),
        HostTemporalBaselineSeconds,
        EvkTemporalFrameCount,
        1.0f / CaptureIntervalSeconds,
        EvkTemporalBaselineSeconds,
        bSaveInferenceFrames ? TEXT("true") : TEXT("false")));
    FParse::Value(FCommandLine::Get(), TEXT("QaiResolutionDatasetVariant="), ResolutionDatasetVariant);
    FParse::Value(FCommandLine::Get(), TEXT("QaiInferenceAblation="), InferenceAblationVariant);
    InferenceAblationVariant = InferenceAblationVariant.TrimStartAndEnd().ToLower();
    if (InferenceAblationVariant.IsEmpty())
    {
        InferenceAblationVariant = TEXT("full");
    }
    FParse::Value(FCommandLine::Get(), TEXT("QaiInferenceCamera="), InferenceCameraVariant);
    InferenceCameraVariant = InferenceCameraVariant.TrimStartAndEnd().ToLower();
    if (InferenceCameraVariant.IsEmpty())
    {
        InferenceCameraVariant = TEXT("parcel-belt");
    }
    if (bResolutionDataset)
    {
        SimulatorLog(FString::Printf(
            TEXT("resolution_dataset enabled=true variant=%s phases=G,A,R native_frames_per_phase=%d phase_seconds=12 slow_motion_cm_s=7 white_floor=%s hide_workers=%s hide_parcels=%s auto_exit=%s"),
            *ResolutionDatasetVariant,
            ResolutionDatasetFramesPerSignal,
            bResolutionDatasetWhiteFloor ? TEXT("true") : TEXT("false"),
            bResolutionDatasetHideWorkers ? TEXT("true") : TEXT("false"),
            bResolutionDatasetHideParcels ? TEXT("true") : TEXT("false"),
            bResolutionDatasetAutoExit ? TEXT("true") : TEXT("false")));
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
    ApplyForkliftPaintVariant();
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
    EncodedFrameWidths.Reset();
    EncodedFrameHeights.Reset();
    EncodedParcelSafetySignals.Reset();
    EncodedFrameTextures.Reset();
    EncodedFrameTimes.Reset();
    EncodedFrameCaptureSeconds.Reset();
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
        if (bInteractiveValidationUnloaded && !bInteractiveValidationUnloadedApplied)
        {
            bInteractiveValidationUnloadedApplied = true;
            int32 RemovedStartingCargo = 0;
            for (const int32 BodyIndex : {
                     Forklifts[0].PalletDynamicBody,
                     Forklifts[0].CartonDynamicBody})
            {
                if (!DynamicBoxes.IsValidIndex(BodyIndex)
                    || !DynamicBoxes[BodyIndex].Root.IsValid())
                {
                    continue;
                }
                FDynamicBoxRuntime& Cargo = DynamicBoxes[BodyIndex];
                USceneComponent* CargoRoot = Cargo.Root.Get();
                CargoRoot->SetVisibility(false, true);
                CargoRoot->SetHiddenInGame(true, true);
                if (UPrimitiveComponent* CargoBody = Cast<UPrimitiveComponent>(CargoRoot))
                {
                    CargoBody->SetPhysicsLinearVelocity(FVector::ZeroVector);
                    CargoBody->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
                    CargoBody->SetSimulatePhysics(false);
                    CargoBody->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                }
                // Remove the validation-only cargo from the active contact
                // island. Leaving a sleeping proxy threaded around the tines
                // can pin the carriage even after collision is disabled.
                CargoRoot->SetWorldLocation(
                    FVector(0.0f, 0.0f, -10000.0f - 100.0f * RemovedStartingCargo),
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
                Cargo.SupportedForklift = INDEX_NONE;
                Cargo.InitialSupportedForklift = INDEX_NONE;
                Cargo.SupportBodyIndex = INDEX_NONE;
                Cargo.InitialSupportBodyIndex = INDEX_NONE;
                Cargo.bAwake = false;
                ++RemovedStartingCargo;
            }
            SimulatorLog(FString::Printf(
                TEXT("interactive_validation unloaded=true starting_cargo_removed=%d input_path=normal_player_controls"),
                RemovedStartingCargo));
        }
        if (bInteractiveDriveValidation)
        {
            if (InteractiveDriveValidationPhase == INDEX_NONE)
            {
                InteractiveDriveValidationPhase = 0;
                InteractiveDriveValidationPhaseSeconds = 0.0f;
                InteractiveDriveValidationLastTelemetrySecond = INDEX_NONE;
                SimulatorLog(TEXT("interactive_drive_validation started=true motion=chaos_force_input phases=settle,reverse,settle,approach,red_dwell,retreat,green_dwell"));
            }

            InteractiveDriveValidationPhaseSeconds += DeltaSeconds;
            bool bAdvancePhase = false;
            switch (InteractiveDriveValidationPhase)
            {
            case 0: bAdvancePhase = InteractiveDriveValidationPhaseSeconds >= 4.0f; break;
            case 1: bAdvancePhase = InteractiveDriveValidationPhaseSeconds >= 2.5f; break;
            case 2: bAdvancePhase = InteractiveDriveValidationPhaseSeconds >= 2.5f; break;
            case 3:
                bAdvancePhase = GroundTruthSignal == TEXT("R")
                    || InteractiveDriveValidationPhaseSeconds >= 6.0f;
                break;
            case 4: bAdvancePhase = InteractiveDriveValidationPhaseSeconds >= 3.0f; break;
            case 5:
                bAdvancePhase = (InteractiveDriveValidationPhaseSeconds >= 0.75f
                        && GroundTruthSignal != TEXT("R"))
                    || InteractiveDriveValidationPhaseSeconds >= 5.0f;
                break;
            case 6: bAdvancePhase = InteractiveDriveValidationPhaseSeconds >= 3.0f; break;
            default: bAdvancePhase = true; break;
            }
            if (bAdvancePhase)
            {
                InteractiveDriveValidationPhase = InteractiveDriveValidationPhase == 6
                    ? 3
                    : InteractiveDriveValidationPhase + 1;
                InteractiveDriveValidationPhaseSeconds = 0.0f;
                InteractiveDriveValidationLastTelemetrySecond = INDEX_NONE;
                SimulatorLog(FString::Printf(
                    TEXT("interactive_drive_validation phase=%d ground_truth=%s forklift=(%.1f,%.1f,%.1f)"),
                    InteractiveDriveValidationPhase,
                    *GroundTruthSignal,
                    GetActiveForkliftLocation().X,
                    GetActiveForkliftLocation().Y,
                    GetActiveForkliftLocation().Z));
            }

            const float ValidationThrottle = InteractiveDriveValidationPhase == 1
                    || InteractiveDriveValidationPhase == 5
                ? -0.55f
                : (InteractiveDriveValidationPhase == 3 ? 0.55f : 0.0f);
            const bool bValidationBrake = FMath::IsNearlyZero(ValidationThrottle);
            DriveActiveForklift(
                ValidationThrottle,
                0.0f,
                0.0f,
                bValidationBrake,
                DeltaSeconds);

            const int32 TelemetrySecond = FMath::FloorToInt(
                InteractiveDriveValidationPhaseSeconds);
            if (TelemetrySecond != InteractiveDriveValidationLastTelemetrySecond)
            {
                InteractiveDriveValidationLastTelemetrySecond = TelemetrySecond;
                SimulatorLog(FString::Printf(
                    TEXT("interactive_drive_validation telemetry phase=%d phase_seconds=%.2f throttle=%.2f speed_cm_s=%.2f ground_truth=%s forklift=(%.1f,%.1f,%.1f)"),
                    InteractiveDriveValidationPhase,
                    InteractiveDriveValidationPhaseSeconds,
                    ValidationThrottle,
                    Forklifts[0].SpeedCmPerSecond,
                    *GroundTruthSignal,
                    GetActiveForkliftLocation().X,
                    GetActiveForkliftLocation().Y,
                    GetActiveForkliftLocation().Z));
            }
        }
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
                    ParcelCenter.Z = ForkTop + ParcelHalfExtents[0].Z + 1.0f;
                    const FQuat TestParcelRotation = ForkliftQuat
                        * FQuat(FVector::UpVector, PI * 0.5f);
                    Parcels[0]->SetWorldLocationAndRotation(
                        ParcelCenter,
                        TestParcelRotation,
                        false,
                        nullptr,
                        ETeleportType::TeleportPhysics);
                    if (UPrimitiveComponent* PhysicsParcel = Cast<UPrimitiveComponent>(Parcels[0].Get()))
                    {
                        PhysicsParcel->SetPhysicsLinearVelocity(FVector::ZeroVector);
                        PhysicsParcel->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
                        PhysicsParcel->WakeRigidBody();
                    }
                    ParcelLinearVelocities[0] = FVector::ZeroVector;
                    ParcelGrounded[0] = true;
                    ParcelSupportedForklifts[0] = 0;
                    ParcelForkContactsLogged[0] = false;
                    PhysicsContactInitialParcelZ = ParcelCenter.Z;
                    PhysicsContactMaximumParcelLiftCm = 0.0f;
                    bPhysicsContactTestInitialized = true;
                    float HighestChaosTineTop = -TNumericLimits<float>::Max();
                    float ClosestChaosTinePlanarCm = TNumericLimits<float>::Max();
                    int32 ValidChaosTines = 0;
                    for (int32 ShapeIndex = 0;
                         ShapeIndex < Forklift.CollisionBoxes.Num()
                            && ShapeIndex < Forklift.ChaosCollisionComponents.Num();
                         ++ShapeIndex)
                    {
                        const FFittedCollisionBox& Shape = Forklift.CollisionBoxes[ShapeIndex];
                        UBoxComponent* ChaosTine = Forklift.ChaosCollisionComponents[ShapeIndex].Get();
                        if (!ChaosTine
                            || !Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase)
                            || Shape.IsWedge())
                        {
                            continue;
                        }
                        ++ValidChaosTines;
                        HighestChaosTineTop = FMath::Max(
                            HighestChaosTineTop,
                            ChaosTine->Bounds.Origin.Z + ChaosTine->Bounds.BoxExtent.Z);
                        ClosestChaosTinePlanarCm = FMath::Min(
                            ClosestChaosTinePlanarCm,
                            FVector::Dist2D(ChaosTine->Bounds.Origin, ParcelCenter));
                    }
                    SimulatorLog(FString::Printf(
                        TEXT("physics_contact_test_initialized parcel=1 fork_top_z_cm=%.2f chaos_tine_top_z_cm=%.2f chaos_tines=%d closest_tine_planar_cm=%.2f parcel_z_cm=%.2f parcel_bottom_z_cm=%.2f half_extent_cm=(%.2f,%.2f,%.2f) rotated_across_tines=true"),
                        ForkTop,
                        HighestChaosTineTop,
                        ValidChaosTines,
                        ClosestChaosTinePlanarCm,
                        ParcelCenter.Z,
                        ParcelCenter.Z - ParcelHalfExtents[0].Z,
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
                            + 1.0f;
                        Body.Root->SetWorldLocationAndRotation(
                            BodyCenter,
                            TestBodyRotation,
                            false,
                            nullptr,
                            ETeleportType::TeleportPhysics);
                        if (UPrimitiveComponent* PhysicsBody = Cast<UPrimitiveComponent>(Body.Root.Get()))
                        {
                            PhysicsBody->SetPhysicsLinearVelocity(FVector::ZeroVector);
                            PhysicsBody->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
                            PhysicsBody->WakeRigidBody();
                        }
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
                    if (UPrimitiveComponent* PhysicsBody = Cast<UPrimitiveComponent>(EvkRoot))
                    {
                        PhysicsBody->SetPhysicsLinearVelocity(EvkBody.LinearVelocity);
                        PhysicsBody->SetPhysicsAngularVelocityInDegrees(EvkBody.AngularVelocityDegrees);
                        PhysicsBody->WakeRigidBody();
                    }
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
            if (bResetLiftTest)
            {
                if (!bResetLiftTestTriggered && RuntimeMotionTestElapsed < 1.20f)
                {
                    DriveActiveForklift(0.0f, 0.0f, 0.90f, false, DeltaSeconds);
                }
                else if (!bResetLiftTestTriggered)
                {
                    ResetLiftTestLiftBeforeResetCm = Forklifts[0].ActualLiftCm;
                    ResetScene();
                    bResetLiftTestTriggered = true;
                    DriveActiveForklift(0.0f, 0.0f, 0.0f, true, DeltaSeconds);
                    SimulatorLog(FString::Printf(
                        TEXT("reset_lift_test_reset lift_before_cm=%.2f"),
                        ResetLiftTestLiftBeforeResetCm));
                }
                else
                {
                    ResetLiftTestMaximumLiftAfterResetCm = FMath::Max(
                        ResetLiftTestMaximumLiftAfterResetCm,
                        Forklifts[0].ActualLiftCm);
                    const bool bLiftAfterReset = RuntimeMotionTestElapsed < 2.80f;
                    DriveActiveForklift(
                        0.0f,
                        0.0f,
                        bLiftAfterReset ? 0.90f : 0.0f,
                        !bLiftAfterReset,
                        DeltaSeconds);
                }
            }
            else if (bPhysicsContactTest || bShelfForkTest)
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
        // The perception benchmark drives through the same Chaos input path as
        // a keyboard/controller user. Set its command before submitting this
        // frame's forces; captures and ground truth then observe the resulting
        // complete rigid vehicle rather than a teleported visual root.
        TickResolutionDataset(DeltaSeconds);
        FixedAccumulator += FMath::Min(DeltaSeconds, 0.25f);
        int32 StepCount = 0;
        if (bChaosPhysicsActive)
        {
            // Forces are submitted exactly once per rendered frame; Chaos's
            // configured 120 Hz substeps perform the actual integration.
            // Re-submitting them through this actor's legacy fixed loop made
            // force magnitude frame-rate dependent.
            SimulateChaosForklifts(DeltaSeconds);
            SimulateChaosConveyor(DeltaSeconds);
            SyncChaosTelemetry();
            while (FixedAccumulator >= ConveyorTuning::FixedStep
                && StepCount < ConveyorTuning::MaxFixedStepsPerFrame)
            {
                SimulateWorkers(ConveyorTuning::FixedStep);
                UpdateSafetySignal();
                FixedAccumulator -= ConveyorTuning::FixedStep;
                ++StepCount;
            }
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
            if (Workers[0].Root.IsValid() && Workers[1].Root.IsValid())
            {
                WorkerMinimumPairClearanceCm = FMath::Min(
                    WorkerMinimumPairClearanceCm,
                    FVector::Dist2D(
                        Workers[0].Root->GetComponentLocation(),
                        Workers[1].Root->GetComponentLocation())
                        - 2.0f * ConveyorTuning::WorkerRadiusCm);
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
                // The diagnostic signed distance includes the same 2-cm
                // obstacle expansion used by occupancy tests. Allow one
                // centimetre of solver/numeric tolerance inside that margin;
                // the visible capsule still remains clear of authored mesh.
                const bool bPassed = WorkerMinimumStaticClearanceCm[0] >= -1.0f
                && WorkerMinimumStaticClearanceCm[1] >= -1.0f
                && WorkerMinimumPairClearanceCm >= -0.5f
                && Workers[0].MaximumVisualTurnRateDegreesPerSecond
                    <= ConveyorTuning::WorkerMaximumVisualTurnRateDegreesPerSecond + 1.0f
                && Workers[1].MaximumVisualTurnRateDegreesPerSecond
                    <= ConveyorTuning::WorkerMaximumVisualTurnRateDegreesPerSecond + 1.0f;
            const FString Result = FString::Printf(
                TEXT("QAI_WORKER_SOAK_TEST passed=%s duration_seconds=%.1f minimum_static_clearance_cm=(%.1f,%.1f) minimum_pair_clearance_cm=%.1f route_reversals=(%d,%d) right_of_way_yields=(%d,%d) separation_events=(%d,%d) maximum_visual_turn_deg_s=(%.1f,%.1f) worker1=(%.1f,%.1f) worker2=(%.1f,%.1f)"),
                    bPassed ? TEXT("true") : TEXT("false"),
                    WorkerSoakTestElapsed,
                    WorkerMinimumStaticClearanceCm[0],
                    WorkerMinimumStaticClearanceCm[1],
                    WorkerMinimumPairClearanceCm,
                Workers[0].RouteReversalCount,
                Workers[1].RouteReversalCount,
                Workers[0].RightOfWayYieldCount,
                Workers[1].RightOfWayYieldCount,
                Workers[0].SeparationEvents,
                Workers[1].SeparationEvents,
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
            if (bResetLiftTest)
            {
                ResetLiftTestMaximumLiftAfterResetCm = FMath::Max(
                    ResetLiftTestMaximumLiftAfterResetCm,
                    Forklifts[0].ActualLiftCm);
                float LateralRailErrorCm = TNumericLimits<float>::Max();
                if (const UBoxComponent* Chassis = Forklifts[0].ChaosChassis.Get())
                {
                    if (const UBoxComponent* Carriage = Forklifts[0].ChaosCarriage.Get())
                    {
                        const FVector CarriageLocal = Chassis->GetComponentTransform()
                            .InverseTransformPositionNoScale(Carriage->GetComponentLocation());
                        const FVector CarriageBaseLocal = Forklifts[0].ChaosCarriageLocalCenter
                            - Forklifts[0].ChaosChassisLocalCenter;
                        LateralRailErrorCm = FVector2D(
                            CarriageLocal.X - CarriageBaseLocal.X,
                            CarriageLocal.Y - CarriageBaseLocal.Y).Size();
                    }
                }
                const bool bPassed = bResetLiftTestTriggered
                    && ResetLiftTestLiftBeforeResetCm >= 25.0f
                    && ResetLiftTestMaximumLiftAfterResetCm >= 25.0f
                    && LateralRailErrorCm <= 2.0f;
                const FString ResetLiftResult = FString::Printf(
                    TEXT("QAI_RESET_LIFT_TEST passed=%s lift_before_reset_cm=%.2f maximum_lift_after_reset_cm=%.2f lateral_rail_error_cm=%.3f"),
                    bPassed ? TEXT("true") : TEXT("false"),
                    ResetLiftTestLiftBeforeResetCm,
                    ResetLiftTestMaximumLiftAfterResetCm,
                    LateralRailErrorCm);
                UE_LOG(LogTemp, Display, TEXT("%s"), *ResetLiftResult);
                SimulatorLog(ResetLiftResult);
            }
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
            if (bPhysicsContactTest && Forklifts[0].ChaosCollisionComponents.Num() > 0)
            {
                float HighestTineTop = -TNumericLimits<float>::Max();
                for (int32 ShapeIndex = 0;
                     ShapeIndex < Forklifts[0].CollisionBoxes.Num()
                        && ShapeIndex < Forklifts[0].ChaosCollisionComponents.Num();
                     ++ShapeIndex)
                {
                    const FFittedCollisionBox& Shape = Forklifts[0].CollisionBoxes[ShapeIndex];
                    if (const UBoxComponent* Tine = Forklifts[0].ChaosCollisionComponents[ShapeIndex].Get();
                        Tine && Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase)
                            && !Shape.IsWedge())
                    {
                        HighestTineTop = FMath::Max(
                            HighestTineTop,
                            Tine->Bounds.Origin.Z + Tine->Bounds.BoxExtent.Z);
                    }
                }
                SimulatorLog(FString::Printf(
                    TEXT("physics_contact_test_final chaos_tine_top_z_cm=%.2f parcel_z_cm=%.2f parcel_bottom_z_cm=%.2f"),
                    HighestTineTop,
                    Parcels[0]->GetComponentLocation().Z,
                    Parcels[0]->GetComponentLocation().Z - ParcelHalfExtents[0].Z));
            }
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
                    && (bChaosPhysicsActive
                        || DynamicBoxes[ForkEdgeBalanceTestPalletBody].SupportedForklift == INDEX_NONE);
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
                    // Authoritative Chaos deliberately has no synthetic
                    // "supported by forklift" attachment flag. Contact and
                    // friction are proven by actual lift and carry motion.
                    && (bChaosPhysicsActive || ShelfBodySupport == 1);
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
                    ? PhysicsContactMaximumParcelLiftCm
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
            if (const UBoxComponent* Chassis = Forklifts[0].ChaosChassis.Get())
            {
                const FVector P = Chassis->GetComponentLocation();
                const FVector V = Chassis->GetPhysicsLinearVelocity();
                const FVector W = Chassis->GetPhysicsAngularVelocityInDegrees();
                SimulatorLog(FString::Printf(
                    TEXT("QAI_CHAOS_CHASSIS_FINAL location=(%.2f,%.2f,%.2f) velocity_cm_s=(%.2f,%.2f,%.2f) angular_deg_s=(%.2f,%.2f,%.2f) rotation=(%.2f,%.2f,%.2f)"),
                    P.X, P.Y, P.Z,
                    V.X, V.Y, V.Z,
                    W.X, W.Y, W.Z,
                    Chassis->GetComponentRotation().Pitch,
                    Chassis->GetComponentRotation().Yaw,
                    Chassis->GetComponentRotation().Roll));
            }
            if (const UBoxComponent* Carriage = Forklifts[0].ChaosCarriage.Get())
            {
                FVector LinearForce = FVector::ZeroVector;
                FVector AngularForce = FVector::ZeroVector;
                if (UPhysicsConstraintComponent* Constraint = Forklifts[0].ChaosLiftConstraint.Get())
                {
                    Constraint->GetConstraintForce(LinearForce, AngularForce);
                }
                const FVector P = Carriage->GetComponentLocation();
                const FVector V = Carriage->GetPhysicsLinearVelocity();
                SimulatorLog(FString::Printf(
                    TEXT("QAI_CHAOS_CARRIAGE_FINAL target_lift_cm=%.2f location=(%.2f,%.2f,%.2f) velocity_cm_s=(%.2f,%.2f,%.2f) constraint_force=(%.2f,%.2f,%.2f)"),
                    Forklifts[0].LiftCm,
                    P.X, P.Y, P.Z,
                    V.X, V.Y, V.Z,
                    LinearForce.X, LinearForce.Y, LinearForce.Z));
            }
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
        if (UBoxComponent* Chassis = Forklifts[0].ChaosChassis.Get())
        {
            ResolutionDatasetInitialChaosChassis = Chassis->GetComponentTransform();
            Chassis->SetPhysicsLinearVelocity(FVector::ZeroVector);
            Chassis->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
            Chassis->SetSimulatePhysics(false);
            Chassis->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        }
        if (UBoxComponent* Carriage = Forklifts[0].ChaosCarriage.Get())
        {
            ResolutionDatasetInitialChaosCarriage = Carriage->GetComponentTransform();
            Carriage->SetPhysicsLinearVelocity(FVector::ZeroVector);
            Carriage->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
            Carriage->SetSimulatePhysics(false);
            Carriage->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        }
        ResolutionDatasetElapsed = 0.0f;
        ResolutionDatasetPhase = INDEX_NONE;
        // The authored test start has a pallet/carton directly against the
        // lowered tines. It stops a normally driven forklift after only a few
        // centimetres, so it cannot reproduce the user's unobstructed driving
        // classification. Remove only that initial load from this benchmark;
        // belt/shelf parcels remain available to the visual ablation groups.
        int32 RemovedStartingCargo = 0;
        for (const int32 BodyIndex : {
                 Forklifts[0].PalletDynamicBody,
                 Forklifts[0].CartonDynamicBody})
        {
            if (!DynamicBoxes.IsValidIndex(BodyIndex)
                || !DynamicBoxes[BodyIndex].Root.IsValid())
            {
                continue;
            }
            USceneComponent* CargoRoot = DynamicBoxes[BodyIndex].Root.Get();
            CargoRoot->SetVisibility(false, true);
            CargoRoot->SetHiddenInGame(true, true);
            if (UPrimitiveComponent* CargoBody = Cast<UPrimitiveComponent>(CargoRoot))
            {
                CargoBody->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                CargoBody->SetPhysicsLinearVelocity(FVector::ZeroVector);
                CargoBody->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
                CargoBody->PutRigidBodyToSleep();
            }
            ++RemovedStartingCargo;
        }
        SimulatorLog(FString::Printf(
            TEXT("resolution_dataset initialized capture=%dx%d starting_cargo_removed=%d motion=complete_assembly_controlled_replay"),
            HostCaptureWidth,
            HostCaptureHeight,
            RemovedStartingCargo));
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
    // Perception ablations require the exact same visible trajectory in every
    // scene variant. Move both Chaos particles together so the complete truck
    // (body, driver, mast, carriage, forks and wheels) remains assembled. This
    // is a controlled replay, not a physics score; the ordinary interactive
    // demo continues to use tire forces and the prismatic lift joint.
    float ReplayOffsetX = 0.0f;
    float ReplaySpeedCm = 0.0f;
    if (ResolutionDatasetPhase == 1)
    {
        const float PhaseSeconds = ResolutionDatasetElapsed - 12.0f;
        constexpr float AmplitudeCm = 28.0f;
        constexpr float PeriodSeconds = 4.0f;
        const float CycleSeconds = FMath::Fmod(PhaseSeconds, PeriodSeconds);
        if (CycleSeconds < PeriodSeconds * 0.5f)
        {
            ReplayOffsetX = -AmplitudeCm
                + 2.0f * AmplitudeCm * CycleSeconds / (PeriodSeconds * 0.5f);
            ReplaySpeedCm = 2.0f * AmplitudeCm / (PeriodSeconds * 0.5f);
        }
        else
        {
            ReplayOffsetX = AmplitudeCm
                - 2.0f * AmplitudeCm
                    * (CycleSeconds - PeriodSeconds * 0.5f)
                    / (PeriodSeconds * 0.5f);
            ReplaySpeedCm = -2.0f * AmplitudeCm / (PeriodSeconds * 0.5f);
        }
    }
    else if (ResolutionDatasetPhase == 2)
    {
        // The authored truck faces +X; 190 cm places its front axle/body over
        // the physical red mat while retaining the same camera composition.
        ReplayOffsetX = 190.0f;
    }
    const FVector ReplayOffset(ReplayOffsetX, 0.0f, 0.0f);
    if (UBoxComponent* Chassis = Forklift.ChaosChassis.Get())
    {
        FTransform Target = ResolutionDatasetInitialChaosChassis;
        Target.AddToTranslation(ReplayOffset);
        Chassis->SetWorldTransform(Target, false, nullptr, ETeleportType::TeleportPhysics);
    }
    if (UBoxComponent* Carriage = Forklift.ChaosCarriage.Get())
    {
        FTransform Target = ResolutionDatasetInitialChaosCarriage;
        Target.AddToTranslation(ReplayOffset);
        Carriage->SetWorldTransform(Target, false, nullptr, ETeleportType::TeleportPhysics);
    }
    Forklift.SpeedCmPerSecond = ReplaySpeedCm;
    CommandThrottle = 0.0f;
    CommandSteer = 0.0f;
    CommandLift = 0.0f;
    bCommandBrake = true;

    const int32 TelemetrySecond = FMath::FloorToInt(ResolutionDatasetElapsed);
    if (TelemetrySecond != ResolutionDatasetLastTelemetrySecond)
    {
        ResolutionDatasetLastTelemetrySecond = TelemetrySecond;
        const FVector Location = Forklift.Root->GetComponentLocation();
        const UBoxComponent* Chassis = Forklift.ChaosChassis.Get();
        const FVector ChassisLocation = Chassis
            ? Chassis->GetComponentLocation()
            : Location;
        const FVector ChassisVelocity = Chassis
            ? Chassis->GetPhysicsLinearVelocity()
            : FVector::ZeroVector;
        const float SupportedNormalForce =
            Forklift.WheelNormalLoads[0]
            + Forklift.WheelNormalLoads[1]
            + Forklift.WheelNormalLoads[2]
            + Forklift.WheelNormalLoads[3];
        SimulatorLog(FString::Printf(
            TEXT("resolution_dataset telemetry phase=%d elapsed=%.2f location=(%.1f,%.1f,%.1f) chassis=(%.1f,%.1f,%.1f) chassis_velocity=(%.2f,%.2f,%.2f) speed_cm_s=%.2f target_cm_s=%.2f throttle=%.3f steer=%.3f brake=%s wheel_loads=(%.0f,%.0f,%.0f,%.0f) supported_force=%.0f truth=%s"),
            ResolutionDatasetPhase,
            ResolutionDatasetElapsed,
            Location.X,
            Location.Y,
            Location.Z,
            ChassisLocation.X,
            ChassisLocation.Y,
            ChassisLocation.Z,
            ChassisVelocity.X,
            ChassisVelocity.Y,
            ChassisVelocity.Z,
            Forklift.SpeedCmPerSecond,
            ReplaySpeedCm,
            0.0f,
            0.0f,
            TEXT("true"),
            Forklift.WheelNormalLoads[0],
            Forklift.WheelNormalLoads[1],
            Forklift.WheelNormalLoads[2],
            Forklift.WheelNormalLoads[3],
            SupportedNormalForce,
            *GroundTruthSignal));
    }
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
            Json->TryGetStringField(TEXT("evk_media_bridge_url"), EvkMediaBridgeUrl);
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
    const bool bMediaBridgeOverride = FParse::Value(
        FCommandLine::Get(), TEXT("EvkMediaBridge="), EvkMediaBridgeUrl);
    FParse::Value(FCommandLine::Get(), TEXT("EvkModel="), EvkModel);
    FParse::Value(FCommandLine::Get(), TEXT("Backend="), ActiveBackend);
    bInferenceEnabled &= !FParse::Param(FCommandLine::Get(), TEXT("NoInference"));
    ActiveBackend = ActiveBackend.Equals(TEXT("evk"), ESearchCase::IgnoreCase) ? TEXT("evk") : TEXT("host");
    if (EvkMediaBridgeUrl.IsEmpty() || (bMediaBridgeOverride == false && EvkMediaBridgeUrl.Contains(TEXT("127.0.0.1")) && !EvkServerUrl.Contains(TEXT("127.0.0.1"))))
    {
        EvkMediaBridgeUrl = EvkServerUrl.Replace(TEXT(":18181"), TEXT(":18182"));
    }
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
        TEXT("Reason2 runtime configured: backend=%s model=%s host=%s evk=%s evk_media=%s inference=%s"),
        *ActiveBackend,
        *ActiveModel,
        *HostServerUrl,
        *EvkServerUrl,
        *EvkMediaBridgeUrl,
        bInferenceEnabled ? TEXT("on") : TEXT("off"));
    SimulatorLog(FString::Printf(
        TEXT("runtime_configured backend=%s model=%s host=%s evk=%s evk_media=%s inference=%s"),
        *ActiveBackend,
        *ActiveModel,
        *HostServerUrl,
        *EvkServerUrl,
        *EvkMediaBridgeUrl,
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
    UPrimitiveComponent* StructuralFloor = nullptr;
    UPrimitiveComponent* FloorFinish = nullptr;
    FBox NorthWallBounds(EForceInit::ForceInit);
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
            else if (LogicalName == TEXT("Floor"))
            {
                StructuralFloor = Cast<UPrimitiveComponent>(Component);
            }
            else if (LogicalName == TEXT("FloorFinish"))
            {
                FloorFinish = Cast<UPrimitiveComponent>(Component);
            }
            else if (LogicalName.StartsWith(TEXT("NorthWall")))
            {
                if (const UPrimitiveComponent* NorthWall = Cast<UPrimitiveComponent>(Component))
                {
                    NorthWallBounds += NorthWall->Bounds.GetBox();
                }
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

    if (!RackVisual || !WestWall || !StructuralFloor || !FloorFinish || !NorthWallBounds.IsValid)
    {
        return false;
    }

    // Move the rack toward the room-facing surface of the west wall, retaining
    // a small physical clearance. The authored gap is about 43 cm; moving a
    // literal 50 cm would put the rear frame several centimetres into the wall.
    constexpr float DesiredWallClearanceCm = 3.0f;
    constexpr float MaximumCorrectionCm = 50.0f;
    constexpr float RoomExtensionCm = 200.0f;
    const float DirectionToWall = FMath::Sign(WestWall->Bounds.Origin.X - RackVisual->Bounds.Origin.X);
    const float WallInnerFaceX = WestWall->Bounds.Origin.X - DirectionToWall * WestWall->Bounds.BoxExtent.X;
    const float RackWallFaceX = RackVisual->Bounds.Origin.X + DirectionToWall * RackVisual->Bounds.BoxExtent.X;
    const float GapBeforeCm = FMath::Abs(RackWallFaceX - WallInnerFaceX);
    const float CorrectionCm = FMath::Clamp(
        GapBeforeCm - DesiredWallClearanceCm,
        0.0f,
        MaximumCorrectionCm);
    const FVector RoomExtensionDelta(DirectionToWall * RoomExtensionCm, 0.0f, 0.0f);
    const FVector Delta = RoomExtensionDelta
        + FVector(DirectionToWall * CorrectionCm, 0.0f, 0.0f);
    const float RackCenterXAfterPlacement = RackVisual->Bounds.Origin.X + Delta.X;
    constexpr float UprightDepthInsetCm = 0.35f;
    int32 InsetUprightCount = 0;

    // Move the physical west wall outward by exactly two metres. The rack and
    // every independent shelf parcel receive the same room-extension delta
    // below, followed by the existing small wall-clearance correction.
    MakeMovable(WestWall);
    WestWall->AddWorldOffset(
        RoomExtensionDelta,
        false,
        nullptr,
        ETeleportType::TeleportPhysics);
    WestWall->UpdateBounds();

    // Fill the revealed floor and north-wall strips with native runtime
    // geometry. These components block Chaos bodies, so the additional room
    // is genuinely driveable and cannot be crossed through at its new edges.
    UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(
        nullptr,
        TEXT("/Engine/BasicShapes/Cube.Cube"));
    UMaterialInterface* ConcreteMaterial = LoadObject<UMaterialInterface>(
        nullptr,
        TEXT("/Game/ConveyorRuntime/VisualFinish/MI_RedEpoxyFloor.MI_RedEpoxyFloor"));
    UMaterialInterface* WallMaterial = LoadObject<UMaterialInterface>(
        nullptr,
        TEXT("/Game/ConveyorRuntime/VisualFinish/M_UnifiedWall.M_UnifiedWall"));
    if (!CubeMesh || !ConcreteMaterial || !WallMaterial)
    {
        return false;
    }

    const auto AddRoomExtensionBlock = [this, CubeMesh](
        const FName Name,
        const FVector& Center,
        const FVector& Size,
        UMaterialInterface* Material,
        const bool bCollision)
    {
        UStaticMeshComponent* Block = NewObject<UStaticMeshComponent>(this, Name);
        AddInstanceComponent(Block);
        Block->SetStaticMesh(CubeMesh);
        Block->SetMobility(EComponentMobility::Movable);
        Block->SetMaterial(0, Material);
        Block->SetCollisionEnabled(
            bCollision ? ECollisionEnabled::QueryAndPhysics : ECollisionEnabled::NoCollision);
        Block->SetCollisionProfileName(
            bCollision ? UCollisionProfile::BlockAll_ProfileName : UCollisionProfile::NoCollision_ProfileName);
        Block->SetGenerateOverlapEvents(false);
        Block->SetSimulatePhysics(false);
        Block->SetCastShadow(true);
        Block->RegisterComponentWithWorld(GetWorld());
        Block->SetWorldLocation(Center, false, nullptr, ETeleportType::TeleportPhysics);
        Block->SetWorldScale3D(Size / 100.0f);
        Block->UpdateBounds();
    };

    const FBox FloorBounds = StructuralFloor->Bounds.GetBox();
    const FBox MovedWestWallBounds = WestWall->Bounds.GetBox();
    // FloorFinish is a legacy zero-thickness overlay whose Y footprint extends
    // 383 cm beyond the real slab, creating the large tab behind the room.
    // The Chaos floor is independent, so make the actual 30-cm Floor mesh the
    // sole visible floor: extend it to the moved wall and apply the concrete
    // finish directly. This keeps one continuous solid mesh and one footprint.
    FloorFinish->SetVisibility(false, true);
    FloorFinish->SetHiddenInGame(true, true);
    FloorFinish->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    // Carry the single slab through the wall thickness to its exterior face.
    // Stopping at the room-facing surface left the visible wall foot hanging
    // beyond the slab at the open corner.
    const float FloorTargetOuterX = DirectionToWall < 0.0f
        ? MovedWestWallBounds.Min.X
        : MovedWestWallBounds.Max.X;
    const float OriginalFloorSizeX = FloorBounds.GetSize().X;
    const float ExtendedFloorMinX = DirectionToWall < 0.0f
        ? FloorTargetOuterX
        : FloorBounds.Min.X;
    const float ExtendedFloorMaxX = DirectionToWall < 0.0f
        ? FloorBounds.Max.X
        : FloorTargetOuterX;
    const float ExtendedFloorSizeX = ExtendedFloorMaxX - ExtendedFloorMinX;
    if (OriginalFloorSizeX <= UE_KINDA_SMALL_NUMBER || ExtendedFloorSizeX <= OriginalFloorSizeX)
    {
        return false;
    }
    MakeMovable(StructuralFloor);
    const FVector OriginalFloorScale = StructuralFloor->GetComponentScale();
    StructuralFloor->SetWorldScale3D(FVector(
        OriginalFloorScale.X * ExtendedFloorSizeX / OriginalFloorSizeX,
        OriginalFloorScale.Y,
        OriginalFloorScale.Z));
    StructuralFloor->UpdateBounds();
    const FBox ScaledFloorBounds = StructuralFloor->Bounds.GetBox();
    const float FixedEdgeCorrectionX = DirectionToWall < 0.0f
        ? FloorBounds.Max.X - ScaledFloorBounds.Max.X
        : FloorBounds.Min.X - ScaledFloorBounds.Min.X;
    StructuralFloor->AddWorldOffset(
        FVector(FixedEdgeCorrectionX, 0.0f, 0.0f),
        false,
        nullptr,
        ETeleportType::TeleportPhysics);
    StructuralFloor->UpdateBounds();
    if (UStaticMeshComponent* StructuralFloorMesh = Cast<UStaticMeshComponent>(StructuralFloor))
    {
        // MI_RedEpoxyFloor is the owned world-XY projection shader built from
        // the original Omniverse concrete maps. Override only its presentation
        // parameters here: the result is neutral concrete with stable tiling,
        // normal detail and two roughness scales, independent of mesh UVs or
        // the slab's non-uniform room-extension scale.
        UMaterialInstanceDynamic* WorldAlignedConcrete =
            UMaterialInstanceDynamic::Create(
                ConcreteMaterial,
                this,
                TEXT("MI_RuntimeWorldAlignedConcrete"));
        if (!WorldAlignedConcrete)
        {
            return false;
        }
        WorldAlignedConcrete->SetVectorParameterValue(
            TEXT("EpoxyTint"),
            FLinearColor(0.19f, 0.215f, 0.23f, 1.0f));
        WorldAlignedConcrete->SetScalarParameterValue(TEXT("TintBlend"), 0.26f);
        // One source tile spans about 4.55 m; the 5.6-cm micro pass prevents
        // broad areas from reading as a single enlarged bitmap.
        WorldAlignedConcrete->SetScalarParameterValue(TEXT("WorldTextureScale"), 0.0022f);
        WorldAlignedConcrete->SetScalarParameterValue(TEXT("TextureBrightness"), 2.25f);
        WorldAlignedConcrete->SetScalarParameterValue(TEXT("VariationFloor"), 0.54f);
        WorldAlignedConcrete->SetScalarParameterValue(TEXT("RoughnessScale"), 0.88f);
        WorldAlignedConcrete->SetScalarParameterValue(TEXT("MicroWorldScale"), 0.018f);
        WorldAlignedConcrete->SetScalarParameterValue(TEXT("MicroRoughnessStrength"), 0.14f);
        for (int32 Slot = 0; Slot < StructuralFloorMesh->GetNumMaterials(); ++Slot)
        {
            StructuralFloorMesh->SetMaterial(Slot, WorldAlignedConcrete);
        }
    }

    const float NorthWallExtensionOuterX = DirectionToWall < 0.0f
        ? MovedWestWallBounds.Min.X
        : MovedWestWallBounds.Max.X;
    const float NorthWallExtensionInnerX = DirectionToWall < 0.0f
        ? NorthWallBounds.Min.X
        : NorthWallBounds.Max.X;
    const float NorthWallExtensionMinX = FMath::Min(
        NorthWallExtensionOuterX,
        NorthWallExtensionInnerX);
    const float NorthWallExtensionMaxX = FMath::Max(
        NorthWallExtensionOuterX,
        NorthWallExtensionInnerX);
    AddRoomExtensionBlock(
        TEXT("NorthWallWestExtension"),
        FVector(
            0.5f * (NorthWallExtensionMinX + NorthWallExtensionMaxX),
            NorthWallBounds.GetCenter().Y,
            NorthWallBounds.GetCenter().Z),
        FVector(
            NorthWallExtensionMaxX - NorthWallExtensionMinX,
            NorthWallBounds.GetSize().Y,
            NorthWallBounds.GetSize().Z),
        WallMaterial,
        true);
    SimulatorLog(FString::Printf(
        TEXT("room_extension_geometry unified_structural_floor_x=(%.1f,%.1f) floor_y=(%.1f,%.1f) scale_x=%.4f legacy_finish=hidden north_wall_x=(%.1f,%.1f) moved_west_wall_x=(%.1f,%.1f)"),
        ExtendedFloorMinX,
        ExtendedFloorMaxX,
        FloorBounds.Min.Y,
        FloorBounds.Max.Y,
        StructuralFloor->GetComponentScale().X,
        NorthWallExtensionMinX,
        NorthWallExtensionMaxX,
        MovedWestWallBounds.Min.X,
        MovedWestWallBounds.Max.X));

    for (USceneComponent* Root : ShelfRoots)
    {
        MakeMovable(Root);
        Root->AddWorldOffset(Delta, false, nullptr, ETeleportType::TeleportPhysics);
        const FString LogicalName = LogicalComponentName(Root);
        if (LogicalName.Contains(TEXT("Upright")))
        {
            // The visible upright proxies met the orange horizontal members
            // on a coplanar face, producing intermittent depth fighting. Move
            // each post only 3.5 mm toward the rack depth centre. Keeping the
            // render and collision component together preserves F8 accuracy.
            const float InwardSign = FMath::Sign(
                RackCenterXAfterPlacement - Root->GetComponentLocation().X);
            Root->AddWorldOffset(
                FVector(InwardSign * UprightDepthInsetCm, 0.0f, 0.0f),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            ++InsetUprightCount;
        }
    }

    bWestShelfPlacementApplied = true;
    SimulatorLog(FString::Printf(
        TEXT("west_shelf_placement gap_before_cm=%.1f clearance_after_cm=%.1f room_extension_cm=%.1f wall_delta_x_cm=%.1f rack_delta_x_cm=%.1f roots=%d shelf_colliders=%d shelf_parcels=%d rack_stack_light=%s inset_uprights=%d upright_inset_cm=%.2f"),
        GapBeforeCm,
        FMath::Max(0.0f, GapBeforeCm - CorrectionCm),
        RoomExtensionCm,
        RoomExtensionDelta.X,
        Delta.X,
        ShelfRoots.Num(),
        ShelfColliderCount,
        ShelfParcelCount,
        bMovedRackStackLight ? TEXT("moved") : TEXT("missing"),
        InsetUprightCount,
        UprightDepthInsetCm));
    return true;
}

void AQaiConveyorWorld::ConfigureForkliftMastMaterials()
{
    if (bForkliftMastMaterialsConfigured || !GetWorld())
    {
        return;
    }

    UMaterialInterface* CoatedSteelBase = LoadObject<UMaterialInterface>(
        nullptr,
        TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
    int32 RoughenedComponents = 0;
    int32 HiddenMastDecals = 0;
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
            // Decals02 is a separate, tall mesh laid over the mast. It contains
            // the imported logistics wordmark and uses a reflective decal
            // atlas, so changing the underlying mast material cannot remove
            // the mirror-like patch. Keep Decals01 (body/safety labels), but
            // suppress this mast-only overlay on both forklifts.
            if (Name.Contains(TEXT("SM_Forklift_C01_Decals02_01"), ESearchCase::IgnoreCase))
            {
                MeshComponent->SetVisibility(false, true);
                MeshComponent->SetHiddenInGame(true, true);
                MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                ++HiddenMastDecals;
                continue;
            }
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
                // The imported atlas bakes a bright metallic patch into the
                // mast side. Scalar overrides cannot remove that reflection,
                // so use a neutral coated-steel material for the complete
                // mast/carriage assembly instead of inheriting the atlas.
                UMaterialInstanceDynamic* MastMaterial = CoatedSteelBase
                    ? UMaterialInstanceDynamic::Create(CoatedSteelBase, this)
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
                MastMaterial->SetVectorParameterValue(
                    TEXT("Color"),
                    FLinearColor(0.055f, 0.062f, 0.070f, 1.0f));
                MeshComponent->SetMaterial(MaterialIndex, MastMaterial);
                ++RoughenedComponents;
            }
        }
    }

    bForkliftMastMaterialsConfigured = RoughenedComponents > 0 && HiddenMastDecals > 0;
    SimulatorLog(FString::Printf(
        TEXT("forklift_mast_material components=%d mast_decals_hidden=%d finish=dark_coated_steel imported_metallic_atlas=false"),
        RoughenedComponents,
        HiddenMastDecals));
}

void AQaiConveyorWorld::ApplyForkliftPaintVariant()
{
    const bool bUseIsaacYellow = ForkliftPaintVariant == TEXT("yellow")
        || ForkliftPaintVariant == TEXT("orange")
        || ForkliftPaintVariant == TEXT("isaac-yellow")
        || ForkliftPaintVariant == TEXT("isaac_yellow");
    const bool bUseDragonwingPurple = ForkliftPaintVariant.IsEmpty()
        || ForkliftPaintVariant == TEXT("purple")
        || ForkliftPaintVariant == TEXT("dragonwing-purple")
        || ForkliftPaintVariant == TEXT("dragonwing_purple");

    if (!bUseIsaacYellow && !bUseDragonwingPurple)
    {
        SimulatorLog(FString::Printf(
            TEXT("forklift_paint warning=unknown_variant requested=%s fallback=dragonwing-purple"),
            *SanitizeLogField(ForkliftPaintVariant)));
        ForkliftPaintVariant = TEXT("dragonwing-purple");
    }

    const TCHAR* MaterialPath = bUseIsaacYellow
        ? TEXT("/Game/ConveyorRuntime/VisualFinish/MI_ForkliftClearCoat_IsaacYellow.MI_ForkliftClearCoat_IsaacYellow")
        : TEXT("/Game/ConveyorRuntime/VisualFinish/MI_ForkliftClearCoat.MI_ForkliftClearCoat");
    UMaterialInterface* PaintMaterial = LoadObject<UMaterialInterface>(nullptr, MaterialPath);
    if (!PaintMaterial)
    {
        SimulatorLog(FString::Printf(
            TEXT("forklift_paint error=material_load_failed variant=%s path=%s"),
            *SanitizeLogField(ForkliftPaintVariant),
            MaterialPath));
        return;
    }

    UMaterialInterface* AppliedPaintMaterial = PaintMaterial;
    const FLinearColor StrikingSafetyYellow(1.0f, 0.658375f, 0.0f, 1.0f);
    if (bUseIsaacYellow)
    {
        if (UMaterialInstanceDynamic* YellowMaterial =
                UMaterialInstanceDynamic::Create(PaintMaterial, this))
        {
            // The original NVIDIA diffuse tint is accurate but becomes ochre
            // after the forklift atlas and fixed warehouse grade are applied.
            // Preserve the authored material response while lifting the tint
            // to display sRGB #FFD400, a clearer industrial safety yellow.
            YellowMaterial->SetVectorParameterValue(
                TEXT("DragonwingPurple"),
                StrikingSafetyYellow);
            AppliedPaintMaterial = YellowMaterial;
        }
    }
    RuntimeForkliftPaintMaterials.Add(AppliedPaintMaterial);
    int32 PaintedComponents = 0;
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        TInlineComponentArray<UStaticMeshComponent*> MeshComponents(*It);
        for (UStaticMeshComponent* MeshComponent : MeshComponents)
        {
            if (!MeshComponent || !MeshComponent->GetStaticMesh())
            {
                continue;
            }
            const FString ComponentName = MeshComponent->GetName();
            const FString MeshName = MeshComponent->GetStaticMesh()->GetName();
            if (!ComponentName.Contains(TEXT("SM_Forklift_C01_Body02_01"), ESearchCase::IgnoreCase)
                && !MeshName.Contains(TEXT("SM_Forklift_C01_Body02_01"), ESearchCase::IgnoreCase))
            {
                continue;
            }
            MeshComponent->SetMaterial(0, AppliedPaintMaterial);
            ++PaintedComponents;
        }
    }

    const FString EffectiveVariant = bUseIsaacYellow
        ? TEXT("isaac-yellow")
        : TEXT("dragonwing-purple");
    SimulatorLog(FString::Printf(
        TEXT("forklift_paint variant=%s material=%s components=%d authored_linear_rgb=%s"),
        *EffectiveVariant,
        MaterialPath,
        PaintedComponents,
        bUseIsaacYellow ? TEXT("1.0000000,0.6583750,0.0000000") : TEXT("0.0318960,0.0003035,0.2086369")));
    if (PaintedComponents != 2)
    {
        SimulatorLog(FString::Printf(
            TEXT("forklift_paint warning=unexpected_component_count expected=2 actual=%d"),
            PaintedComponents));
    }
}

bool AQaiConveyorWorld::ConfigureInferenceScene()
{
    if (bInferenceSceneConfigured)
    {
        return true;
    }
    if (!GetWorld() || !DetectorCamera.IsValid())
    {
        return false;
    }

    UStaticMeshComponent* RedZone = nullptr;
    for (TActorIterator<AActor> It(GetWorld()); It && !RedZone; ++It)
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
            if (ComponentName.Contains(TEXT("ClearanceZone"), ESearchCase::IgnoreCase)
                || MeshName.Contains(TEXT("ClearanceZone"), ESearchCase::IgnoreCase))
            {
                RedZone = Mesh;
                break;
            }
        }
    }
    if (!RedZone)
    {
        SimulatorLog(TEXT("inference_scene status=missing_red_zone"));
        return false;
    }

    RedZone->UpdateBounds();
    const FBox RedBounds = RedZone->Bounds.GetBox();
    if (!RedBounds.IsValid || RedBounds.GetSize().X < 100.0f || RedBounds.GetSize().Y < 100.0f)
    {
        SimulatorLog(TEXT("inference_scene status=invalid_red_zone_bounds"));
        return false;
    }
    InferenceRedZoneBounds = RedBounds;

    // Keep the safety surface visually stable across captures: retain a small
    // amount of the authored concrete variation and normal detail, but remove
    // the glossy epoxy response that changed with camera angle and local
    // colored lights.
    UMaterialInterface* RedFloorBase = LoadObject<UMaterialInterface>(
        nullptr,
        TEXT("/Game/ConveyorRuntime/VisualFinish/MI_RedEpoxyFloor.MI_RedEpoxyFloor"));
    UMaterialInstanceDynamic* MatteRed = RedFloorBase
        ? UMaterialInstanceDynamic::Create(RedFloorBase, this)
        : nullptr;
    if (MatteRed)
    {
        const bool bParcelSafetyExperiment =
            InferenceCameraVariant.Equals(TEXT("parcel-belt"), ESearchCase::IgnoreCase);
        if (bParcelSafetyExperiment)
        {
            // The previous red safety mat was the label for the forklift-zone
            // task. In the parcel experiment it is irrelevant and strongly
            // biases a vision-language model toward the literal answer "R".
            // Keep the same physical textured surface but render it as neutral
            // charcoal, so the task is decided by parcel support geometry.
            MatteRed->SetVectorParameterValue(
                TEXT("EpoxyTint"),
                FLinearColor(0.085f, 0.105f, 0.13f, 1.0f));
        }
        MatteRed->SetScalarParameterValue(
            TEXT("TintBlend"),
            bParcelSafetyExperiment ? 0.48f : 0.68f);
        MatteRed->SetScalarParameterValue(TEXT("TextureBrightness"), 2.15f);
        MatteRed->SetScalarParameterValue(TEXT("VariationFloor"), 0.70f);
        MatteRed->SetScalarParameterValue(TEXT("RoughnessScale"), 1.20f);
        MatteRed->SetScalarParameterValue(TEXT("MicroRoughnessStrength"), 0.08f);
        RedZone->SetMaterial(0, MatteRed);
        RuntimeInferenceSceneMaterials.Add(MatteRed);
    }

    // The border is physical scene geometry, not a detector-only overlay. At
    // 512x288 its 10-cm width remains legible while the short alternating
    // segments provide strong boundaries under both bright and dark objects.
    UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(
        nullptr,
        TEXT("/Engine/BasicShapes/Cube.Cube"));
    UMaterialInterface* BasicMaterial = LoadObject<UMaterialInterface>(
        nullptr,
        TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
    UMaterialInstanceDynamic* WhiteMaterial = BasicMaterial
        ? UMaterialInstanceDynamic::Create(BasicMaterial, this)
        : nullptr;
    UMaterialInstanceDynamic* BlackMaterial = BasicMaterial
        ? UMaterialInstanceDynamic::Create(BasicMaterial, this)
        : nullptr;
    if (!CubeMesh || !WhiteMaterial || !BlackMaterial)
    {
        SimulatorLog(FString::Printf(
            TEXT("inference_scene status=missing_border_assets cube=%s material=%s"),
            CubeMesh ? TEXT("true") : TEXT("false"),
            BasicMaterial ? TEXT("true") : TEXT("false")));
        return false;
    }
    WhiteMaterial->SetVectorParameterValue(
        TEXT("Color"), FLinearColor(0.82f, 0.82f, 0.78f, 1.0f));
    BlackMaterial->SetVectorParameterValue(
        TEXT("Color"), FLinearColor(0.008f, 0.009f, 0.011f, 1.0f));
    RuntimeInferenceSceneMaterials.Add(WhiteMaterial);
    RuntimeInferenceSceneMaterials.Add(BlackMaterial);

    constexpr float StripeLengthCm = 42.0f;
    constexpr float BorderWidthCm = 10.0f;
    constexpr float BorderThicknessCm = 0.45f;
    const float BorderZ = RedBounds.Max.Z + BorderThicknessCm * 0.5f + 0.20f;
    int32 StripeCounter = 0;
    const auto AddStrip = [
        this,
        CubeMesh,
        WhiteMaterial,
        BlackMaterial,
        BorderZ,
        StripeLengthCm,
        BorderWidthCm,
        BorderThicknessCm,
        &StripeCounter](
        const FVector2D& Start,
        const FVector2D& End)
    {
        const FVector2D Delta = End - Start;
        const float Length = Delta.Size();
        if (Length < 1.0f)
        {
            return;
        }
        const FVector2D Direction = Delta / Length;
        const int32 SegmentCount = FMath::Max(1, FMath::CeilToInt(Length / StripeLengthCm));
        const float SegmentLength = Length / static_cast<float>(SegmentCount);
        const float Yaw = FMath::RadiansToDegrees(FMath::Atan2(Direction.Y, Direction.X));
        for (int32 SegmentIndex = 0; SegmentIndex < SegmentCount; ++SegmentIndex)
        {
            const FVector2D Center = Start
                + Direction * ((static_cast<float>(SegmentIndex) + 0.5f) * SegmentLength);
            UStaticMeshComponent* Stripe = NewObject<UStaticMeshComponent>(
                this,
                FName(*FString::Printf(TEXT("InferenceSafetyStripe_%03d"), ++StripeCounter)));
            AddInstanceComponent(Stripe);
            Stripe->SetStaticMesh(CubeMesh);
            Stripe->SetMobility(EComponentMobility::Movable);
            Stripe->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            Stripe->SetGenerateOverlapEvents(false);
            Stripe->SetCastShadow(false);
            Stripe->SetAffectDynamicIndirectLighting(false);
            Stripe->SetAffectDistanceFieldLighting(false);
            Stripe->SetReceivesDecals(false);
            Stripe->SetMaterial(
                0,
                ((StripeCounter - 1) & 1) == 0 ? WhiteMaterial : BlackMaterial);
            Stripe->RegisterComponentWithWorld(GetWorld());
            Stripe->SetWorldLocationAndRotation(
                FVector(Center.X, Center.Y, BorderZ),
                FRotator(0.0f, Yaw, 0.0f));
            Stripe->SetWorldScale3D(FVector(
                SegmentLength / 100.0f,
                BorderWidthCm / 100.0f,
                BorderThicknessCm / 100.0f));
            RuntimeSafetyBorderComponents.Add(Stripe);
        }
    };

    const float HalfBorder = BorderWidthCm * 0.5f;
    AddStrip(
        FVector2D(RedBounds.Min.X, RedBounds.Min.Y - HalfBorder),
        FVector2D(RedBounds.Max.X, RedBounds.Min.Y - HalfBorder));
    AddStrip(
        FVector2D(RedBounds.Max.X + HalfBorder, RedBounds.Min.Y),
        FVector2D(RedBounds.Max.X + HalfBorder, RedBounds.Max.Y));
    AddStrip(
        FVector2D(RedBounds.Max.X, RedBounds.Max.Y + HalfBorder),
        FVector2D(RedBounds.Min.X, RedBounds.Max.Y + HalfBorder));
    AddStrip(
        FVector2D(RedBounds.Min.X - HalfBorder, RedBounds.Max.Y),
        FVector2D(RedBounds.Min.X - HalfBorder, RedBounds.Min.Y));

    // Two legacy point fills were authored for the presentation camera. The
    // far-bay source is nearly co-located with the removed rear-wall stack and
    // leaves its own white hotspot behind; the conveyor source adds another
    // angle-dependent gradient across the detector view. Retire both for the
    // inference-stable scene and retain an explicit opt-in for lighting A/Bs.
    int32 DisabledLegacyPointFills = 0;
    if (!FParse::Param(FCommandLine::Get(), TEXT("QaiEnableLocalPointFills")))
    {
        const FVector LegacyPointFillLocations[] = {
            FVector(420.0f, 110.0f, 215.0f),
            FVector(410.0f, -330.0f, 185.0f),
        };
        constexpr float LegacyPointFillMatchRadiusCm = 45.0f;
        for (TActorIterator<AActor> It(GetWorld()); It; ++It)
        {
            AActor* Candidate = *It;
            UPointLightComponent* PointLight = Candidate
                ? Candidate->FindComponentByClass<UPointLightComponent>()
                : nullptr;
            if (!PointLight)
            {
                continue;
            }
            bool bSignalEmitter = false;
            for (const FName& Tag : Candidate->Tags)
            {
                const FString TagString = Tag.ToString();
                if (TagString.StartsWith(TEXT("Qai.Green."))
                    || TagString.StartsWith(TEXT("Qai.Amber."))
                    || TagString.StartsWith(TEXT("Qai.Red.")))
                {
                    bSignalEmitter = true;
                    break;
                }
            }
            if (bSignalEmitter)
            {
                continue;
            }
            bool bMatchesLegacyFill = false;
            for (const FVector& LegacyLocation : LegacyPointFillLocations)
            {
                if (FVector::DistSquared(Candidate->GetActorLocation(), LegacyLocation)
                    <= FMath::Square(LegacyPointFillMatchRadiusCm))
                {
                    bMatchesLegacyFill = true;
                    break;
                }
            }
            if (!bMatchesLegacyFill)
            {
                continue;
            }
            PointLight->SetIntensity(0.0f);
            PointLight->SetVisibility(false);
            PointLight->Deactivate();
            ++DisabledLegacyPointFills;
        }
    }

    // A broad neutral workcell fixture gives the physical scene (and therefore
    // the sensor camera) a stable warehouse illumination reference. It is much
    // dimmer and larger than the old inference fill, so Lumen supplies natural
    // bounce without flattening white parcels or tinting the safety surface.
    const FVector RedCenter = RedBounds.GetCenter();
    const bool bNeutralWorkcellFillEnabled =
        !FParse::Param(FCommandLine::Get(), TEXT("QaiDisableNeutralWorkcellFill"));
    constexpr float NeutralWorkcellFillLumens = 1200.0f;
    const float WorkcellFillCenterX = RedCenter.X - 145.0f;
    RuntimeInferenceZoneFillActor = bNeutralWorkcellFillEnabled
        ? GetWorld()->SpawnActor<ARectLight>(
            FVector(WorkcellFillCenterX, RedCenter.Y, RedBounds.Max.Z + 430.0f),
            FRotationMatrix::MakeFromX(FVector(0.0f, 0.0f, -1.0f)).Rotator())
        : nullptr;
    if (RuntimeInferenceZoneFillActor && bNeutralWorkcellFillEnabled)
    {
        RuntimeInferenceZoneFillActor->Tags.Add(TEXT("Qai.NeutralWorkcellFixture"));
        if (URectLightComponent* Fill = RuntimeInferenceZoneFillActor->FindComponentByClass<URectLightComponent>())
        {
            Fill->SetMobility(EComponentMobility::Movable);
            Fill->SetIntensityUnits(ELightUnits::Lumens);
            Fill->SetIntensity(NeutralWorkcellFillLumens);
            Fill->SetLightColor(FLinearColor::White, false);
            Fill->SetUseTemperature(true);
            Fill->SetTemperature(4700.0f);
            Fill->SetSourceWidth(RedBounds.GetSize().X + 300.0f);
            Fill->SetSourceHeight(RedBounds.GetSize().Y * 0.90f);
            Fill->SetAttenuationRadius(800.0f);
            Fill->SetCastShadows(false);
            Fill->SetIndirectLightingIntensity(0.65f);
            Fill->SetVolumetricScatteringIntensity(0.0f);
        }
    }

    // Give the warehouse a restrained mixture of real fixture types while
    // retaining neutral, stable illumination for the inference camera. The
    // four imported ceiling rectangles are identified by their high mounting
    // position, which remains reliable in packaged builds where editor labels
    // are unavailable. Temperatures stay close enough to avoid theatrical
    // color casts: three cool/neutral fluorescent panels and one slightly
    // warmer high-bay lamp. Two low practical fills add subtle local warmth.
    int32 RetunedCeilingPanels = 0;
    int32 RetunedPracticalFills = 0;
    if (GetWorld())
    {
        for (TActorIterator<AActor> It(GetWorld()); It; ++It)
        {
            AActor* LightActor = *It;
            if (!LightActor)
            {
                continue;
            }
            if (URectLightComponent* Panel = LightActor->FindComponentByClass<URectLightComponent>())
            {
                const FVector Location = Panel->GetComponentLocation();
                if (Location.Z >= 550.0f)
                {
                    const float TemperatureKelvin = Location.Y >= 0.0f
                        ? (Location.X < 0.0f ? 4300.0f : 5000.0f)
                        : (Location.X < 0.0f ? 4600.0f : 3900.0f);
                    Panel->SetLightColor(FLinearColor::White, false);
                    Panel->SetUseTemperature(true);
                    Panel->SetTemperature(TemperatureKelvin);
                    ++RetunedCeilingPanels;
                }
            }

            UPointLightComponent* Practical = LightActor->FindComponentByClass<UPointLightComponent>();
            if (!Practical)
            {
                continue;
            }
            const FVector Location = Practical->GetComponentLocation();
            float TemperatureKelvin = 0.0f;
            if (FVector::DistSquared(Location, FVector(-120.0f, 325.0f, 245.0f)) <= FMath::Square(60.0f))
            {
                TemperatureKelvin = 4400.0f;
            }
            else if (FVector::DistSquared(Location, FVector(-330.0f, -120.0f, 190.0f)) <= FMath::Square(60.0f))
            {
                TemperatureKelvin = 3600.0f;
            }
            if (TemperatureKelvin > 0.0f)
            {
                Practical->SetLightColor(FLinearColor::White, false);
                Practical->SetUseTemperature(true);
                Practical->SetTemperature(TemperatureKelvin);
                ++RetunedPracticalFills;
            }
        }
    }
    SimulatorLog(FString::Printf(
        TEXT("warehouse_light_mix ceiling_panels=%d panel_kelvin=4300,5000,4600,3900 practical_fills=%d fill_kelvin=4400,3600 workcell_kelvin=4700 evk_led_colors_unchanged=true"),
        RetunedCeilingPanels,
        RetunedPracticalFills));

    // The parcel-safety experiment uses a fixed industrial camera centred on
    // the roller loop. It retains enough of the west approach for the model to
    // see a forklift make contact, but spends most image tokens on parcel/edge
    // relationships instead of the former red-zone boundary.
    const FTransform AuthoredDetectorTransform = DetectorCamera->GetComponentTransform();
    const bool bUseAuthoredWideCamera =
        InferenceCameraVariant.Equals(TEXT("authored-wide"), ESearchCase::IgnoreCase);
    const bool bUseParcelBeltCamera =
        InferenceCameraVariant.Equals(TEXT("parcel-belt"), ESearchCase::IgnoreCase);
    const float ApproachMinimumX = FMath::Min(
        RedBounds.Min.X - 80.0f,
        Forklifts[0].Root.IsValid()
            ? Forklifts[0].Root->GetComponentLocation().X - 260.0f
            : RedBounds.Min.X - 440.0f);
    const float ViewMaximumX = RedBounds.Max.X + 55.0f;
    const float ApproachAndZoneCenterX = 0.5f * (ApproachMinimumX + ViewMaximumX);
    // Bias toward the safety surface while retaining the complete active
    // forklift and its approach. This removes the irrelevant far-west aisle
    // that previously consumed roughly a quarter of the detector image.
    const float ViewCenterX = FMath::Lerp(
        ApproachAndZoneCenterX,
        RedBounds.GetCenter().X,
        0.35f);
    // Use a low-mounted fixed safety camera aimed through the approach lane,
    // with the red floor occupying most of the frame. This reduces the upper
    // wall and makes fork/red-boundary relationships substantially larger.
    // Translate the entire sensor rig left and down, including its aim point,
    // so the optical axis and perspective remain unchanged. The additional
    // one-metre left/down shift prioritizes the forklift approach and safety
    // boundary while removing more of the conveyor from the detector image.
    constexpr float SensorCameraLeftOffsetCm = -215.0f;
    constexpr float SensorCameraDownOffsetCm = -100.0f;
    // Keep the validated 105-degree lens, but return to the authored camera
    // distance. The previous one-metre retreat made the forklift too small for
    // reliable temporal-motion classification and increased false red results
    // near (but outside) the safety boundary.
    constexpr float AuthoredCameraBackwardOffsetCm = 0.0f;
    const float ShiftedViewCenterX = ViewCenterX + SensorCameraLeftOffsetCm;
    const FVector CameraLocation(
        ShiftedViewCenterX,
        RedBounds.Max.Y + 330.0f,
        RedBounds.Max.Z + 245.0f + SensorCameraDownOffsetCm);
    const FVector CameraTarget(
        ShiftedViewCenterX,
        RedBounds.GetCenter().Y - RedBounds.GetSize().Y * 0.04f,
        RedBounds.Max.Z + 48.0f + SensorCameraDownOffsetCm);
    const FRotator CameraRotation = FRotationMatrix::MakeFromX(
        CameraTarget - CameraLocation).Rotator();
    const FVector ParcelCameraTarget(
        ConveyorTuning::BeltCenterX,
        ConveyorTuning::BeltCenterY,
        ConveyorSurfaceZCm + 24.0f);
    const FVector ParcelCameraLocation(
        ConveyorTuning::BeltCenterX,
        ConveyorTuning::BeltCenterY + 460.0f,
        ConveyorSurfaceZCm + 780.0f);
    const FRotator ParcelCameraRotation = FRotationMatrix::MakeFromX(
        ParcelCameraTarget - ParcelCameraLocation).Rotator();
    const FVector AuthoredCameraForward =
        AuthoredDetectorTransform.GetRotation().GetForwardVector();
    const FVector BackedAuthoredCameraLocation =
        AuthoredDetectorTransform.GetLocation()
        - AuthoredCameraForward * AuthoredCameraBackwardOffsetCm;
    MakeMovable(DetectorCamera.Get());
    if (bUseParcelBeltCamera)
    {
        DetectorCamera->SetWorldLocationAndRotation(
            ParcelCameraLocation,
            ParcelCameraRotation,
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
    }
    else if (bUseAuthoredWideCamera)
    {
        // Keep the original optical axis but gain scene coverage by moving the
        // camera back, rather than by stretching the edges with a shorter lens.
        DetectorCamera->SetWorldLocationAndRotation(
            BackedAuthoredCameraLocation,
            AuthoredDetectorTransform.GetRotation(),
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
    }
    else
    {
        DetectorCamera->SetWorldLocationAndRotation(
            CameraLocation,
            CameraRotation,
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
    }

    const FVector EffectiveCameraLocation = bUseParcelBeltCamera
        ? ParcelCameraLocation
        : (bUseAuthoredWideCamera ? BackedAuthoredCameraLocation : CameraLocation);
    const FVector EffectiveCameraTarget = bUseParcelBeltCamera
        ? ParcelCameraTarget
        : (bUseAuthoredWideCamera
            ? BackedAuthoredCameraLocation + AuthoredCameraForward * 1000.0f
            : CameraTarget);

    bInferenceSceneConfigured = true;
    SimulatorLog(FString::Printf(
        TEXT("inference_scene status=ready camera=%s focus=conveyor_parcels location=(%.1f,%.1f,%.1f) target=(%.1f,%.1f,%.1f) sensor_left_offset_cm=%.1f sensor_down_offset_cm=%.1f sensor_backward_offset_cm=%.1f red_bounds_min=(%.1f,%.1f,%.1f) red_bounds_max=(%.1f,%.1f,%.1f) border=physical_white_black stripe_cm=%.1f width_cm=%.1f stripes=%d floor_finish=%s neutral_workcell_fill_lumens=%.0f legacy_point_fills_disabled=%d billboard_excluded_by_composition=%s"),
        bUseParcelBeltCamera
            ? TEXT("fixed_oblique_parcel_belt")
            : (bUseAuthoredWideCamera ? TEXT("authored_wide") : TEXT("fixed_low_oblique_red_focus")),
        EffectiveCameraLocation.X,
        EffectiveCameraLocation.Y,
        EffectiveCameraLocation.Z,
        EffectiveCameraTarget.X,
        EffectiveCameraTarget.Y,
        EffectiveCameraTarget.Z,
        SensorCameraLeftOffsetCm,
        SensorCameraDownOffsetCm,
        bUseAuthoredWideCamera && !bUseParcelBeltCamera ? AuthoredCameraBackwardOffsetCm : 0.0f,
        RedBounds.Min.X,
        RedBounds.Min.Y,
        RedBounds.Min.Z,
        RedBounds.Max.X,
        RedBounds.Max.Y,
        RedBounds.Max.Z,
        StripeLengthCm,
        BorderWidthCm,
        RuntimeSafetyBorderComponents.Num(),
        bUseParcelBeltCamera ? TEXT("neutral_charcoal") : TEXT("matte_red"),
        bNeutralWorkcellFillEnabled ? NeutralWorkcellFillLumens : 0.0f,
        DisabledLegacyPointFills,
        bUseAuthoredWideCamera && !bUseParcelBeltCamera ? TEXT("false") : TEXT("true")));
    return true;
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
        // The enclosure has rubber feet; it should remain planted on the
        // stainless worktop until it receives a meaningful contact impulse.
        Profile.StaticFriction = 0.82f;
        Profile.DynamicFriction = 0.66f;
        Profile.Restitution = 0.03f;
        Profile.AirLinearDamping = 0.85f;
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
        Profile.StaticFriction = 0.70f;
        Profile.DynamicFriction = 0.54f;
        Profile.Restitution = 0.04f;
        Profile.AirLinearDamping = 0.65f;
        Profile.GroundAngularDamping = 3.2f;
        Profile.SleepLinearSpeedCm = 0.72f;
        Profile.SleepAngularSpeedDegrees = 0.66f;
    }
    else
    {
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
            -HalfExtent.Z * 0.10f);
        Profile.StaticFriction = 0.66f;
        Profile.DynamicFriction = 0.50f;
        Profile.Restitution = 0.025f;
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
    WarehouseFloorSupportBounds = FBox(ForceInit);

    FBox ConveyorBounds(ForceInit);
    FBox VisibleFloorBounds(ForceInit);
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
    auto AddObstacleBox = [this](
        const FString& Name,
        const FVector& Center,
        const FVector& HalfExtent)
    {
        if (HalfExtent.X <= UE_KINDA_SMALL_NUMBER
            || HalfExtent.Y <= UE_KINDA_SMALL_NUMBER
            || HalfExtent.Z <= UE_KINDA_SMALL_NUMBER)
        {
            return;
        }
        FCollisionObstacle& Obstacle = CollisionObstacles.AddDefaulted_GetRef();
        Obstacle.Name = Name;
        Obstacle.Center = Center;
        Obstacle.HalfExtent = HalfExtent;
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
                else if (LogicalName == TEXT("Floor"))
                {
                    // ApplyWestShelfPlacement has already extended the real
                    // structural/visible floor. Use those final bounds as the
                    // reference for the open-side safety apron.
                    VisibleFloorBounds += Primitive->Bounds.GetBox();
                }
                else if (LogicalName.StartsWith(TEXT("ShelfPlane")))
                {
                    // Match the visible rack as two discrete deck bays rather
                    // than one broad slab. Preserve the authored top and
                    // underside exactly, then add the two load-bearing edge
                    // beams beneath it. This leaves the upright gaps open and
                    // makes the F8 view describe the structure a fork meets.
                    const FVector Center = Primitive->Bounds.Origin;
                    const FVector Extent = Primitive->Bounds.BoxExtent;
                    const float BayHalfY = Extent.Y * 0.5f;
                    for (int32 Bay = 0; Bay < 2; ++Bay)
                    {
                        const float BaySign = Bay == 0 ? -1.0f : 1.0f;
                        const FVector BayCenter(
                            Center.X,
                            Center.Y + BaySign * BayHalfY,
                            Center.Z);
                        const FVector BayExtent(Extent.X, BayHalfY, Extent.Z);
                        AddObstacleBox(TEXT("shelf plane"), BayCenter, BayExtent);

                        constexpr float BeamHalfDepthCm = 4.0f;
                        constexpr float BeamHalfHeightCm = 4.5f;
                        for (int32 Edge = 0; Edge < 2; ++Edge)
                        {
                            const float EdgeSign = Edge == 0 ? -1.0f : 1.0f;
                            AddObstacleBox(
                                TEXT("shelf beam"),
                                FVector(
                                    Center.X + EdgeSign * (Extent.X - BeamHalfDepthCm),
                                    BayCenter.Y,
                                    Center.Z + Extent.Z - BeamHalfHeightCm),
                                FVector(BeamHalfDepthCm, BayHalfY, BeamHalfHeightCm));
                        }
                    }
                }
                else if (LogicalName.Contains(TEXT("Upright")))
                {
                    // Each authored upright is already a close individual
                    // proxy. Keep those six pillars; the former broad rear-
                    // guard slab falsely filled all the open rack bays.
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
            }
        }
    }

    if (VisibleFloorBounds.IsValid)
    {
        // Keep the barrier face flush with the visible slab. A previous
        // interpretation put it two metres outside the room and supported the
        // gap with an invisible apron; the truck therefore appeared to drive
        // or fall over the rendered edge before reaching the catch wall.
        constexpr float OpenApronDepthCm = 0.0f;
        constexpr float BarrierHalfThicknessCm = 10.0f;
        constexpr float BarrierHeightCm = 400.0f;
        const float FloorTopZ = VisibleFloorBounds.Max.Z;
        const float BarrierCenterZ = FloorTopZ + BarrierHeightCm * 0.5f;
        const float EastInnerFaceX = VisibleFloorBounds.Max.X + OpenApronDepthCm;
        const float FrontInnerFaceY = VisibleFloorBounds.Min.Y - OpenApronDepthCm;

        // The hidden support now matches the visible slab exactly. The fitted
        // perimeter below prevents vehicles and loose rigid bodies from ever
        // entering unsupported, invisible space.
        constexpr float FloorSupportThicknessCm = 10.0f;
        WarehouseFloorSupportBounds = FBox(
            FVector(
                VisibleFloorBounds.Min.X,
                FrontInnerFaceY,
                FloorTopZ - FloorSupportThicknessCm),
            FVector(
                EastInnerFaceX,
                VisibleFloorBounds.Max.Y,
                FloorTopZ));

        // East/open side. Extend to the front catch wall so no diagonal gap is
        // left at the invisible outer corner.
        const float EastMinY = FrontInnerFaceY;
        const float EastMaxY = VisibleFloorBounds.Max.Y;
        AddObstacleBox(
            TEXT("invisible perimeter"),
            FVector(
                EastInnerFaceX + BarrierHalfThicknessCm,
                0.5f * (EastMinY + EastMaxY),
                BarrierCenterZ),
            FVector(
                BarrierHalfThicknessCm,
                0.5f * (EastMaxY - EastMinY),
                BarrierHeightCm * 0.5f));

        // Front/presentation side. Its east end meets the barrier above and
        // its west end meets the existing visible west wall.
        const float FrontMinX = VisibleFloorBounds.Min.X;
        const float FrontMaxX = EastInnerFaceX;
        AddObstacleBox(
            TEXT("invisible perimeter"),
            FVector(
                0.5f * (FrontMinX + FrontMaxX),
                FrontInnerFaceY - BarrierHalfThicknessCm,
                BarrierCenterZ),
            FVector(
                0.5f * (FrontMaxX - FrontMinX),
                BarrierHalfThicknessCm,
                BarrierHeightCm * 0.5f));

        SimulatorLog(FString::Printf(
            TEXT("open_side_perimeter apron_cm=%.1f east_inner_x=%.1f front_inner_y=%.1f height_cm=%.1f floor_support=(%.1f,%.1f)-(%.1f,%.1f) debug_visible=false"),
            OpenApronDepthCm,
            EastInnerFaceX,
            FrontInnerFaceY,
            BarrierHeightCm,
            WarehouseFloorSupportBounds.Min.X,
            WarehouseFloorSupportBounds.Min.Y,
            WarehouseFloorSupportBounds.Max.X,
            WarehouseFloorSupportBounds.Max.Y));
    }

    if (ConveyorBounds.IsValid)
    {
        // Follow the analytic belt centreline with tangent-aligned boxes. This
        // is close to the retained roller mesh, rotates correctly through both
        // curves, and remains far cheaper than per-triangle Chaos collision.
        constexpr int32 ConveyorSegments = 96;
        constexpr float StraightHalf = 235.1374f;
        constexpr float Radius = 150.0f;
        constexpr float Perimeter = 4.0f * StraightHalf + 2.0f * PI * Radius;
        const FVector Extent3 = ConveyorBounds.GetExtent();
        const float TrackHalfWidth = FMath::Min(55.0f, Extent3.X * 0.25f);
        // Only a two-percent seam overlap is needed. The former 40-box path
        // overlapped by sixteen percent; a carton spanning several differently
        // rotated boxes on a curve received conflicting contact normals.
        const float SegmentHalfLength = Perimeter / static_cast<float>(ConveyorSegments) * 0.51f;
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
    if (bWorkerSoakTest)
    {
        for (const FCollisionObstacle& Obstacle : CollisionObstacles)
        {
            if (Obstacle.Name == TEXT("packing table")
                || Obstacle.Name == TEXT("factory wall")
                || Obstacle.Name == TEXT("shelf frame"))
            {
                SimulatorLog(FString::Printf(
                    TEXT("worker_static_obstacle name=%s center=(%.1f,%.1f) half_extent=(%.1f,%.1f)"),
                    *Obstacle.Name,
                    Obstacle.Center.X,
                    Obstacle.Center.Y,
                    Obstacle.HalfExtent.X,
                    Obstacle.HalfExtent.Y));
            }
        }
    }
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
            const USceneComponent* Worker = Workers[WorkerIndex].ChaosBody.IsValid()
                ? static_cast<const USceneComponent*>(Workers[WorkerIndex].ChaosBody.Get())
                : Workers[WorkerIndex].Root.Get();
            if (Worker)
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
        const USceneComponent* OtherWorker = Workers[WorkerIndex].ChaosBody.IsValid()
            ? static_cast<const USceneComponent*>(Workers[WorkerIndex].ChaosBody.Get())
            : Workers[WorkerIndex].Root.Get();
        if (OtherWorker)
        {
            const FVector Other = OtherWorker->GetComponentLocation();
            if (FVector2D::DistSquared(Center, FVector2D(Other.X, Other.Y))
                < FMath::Square(ConveyorTuning::WorkerPersonalSpaceCm))
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
                    Forklift.WheelPivotInitialRelativeLocation[WheelIndex] =
                        Pivot->GetRelativeLocation();
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
        if (ForkliftIndex == 0)
        {
            // Give the dedicated safety camera a clearer initial view of the
            // active vehicle. Component binding may be retried while the
            // streamed stage settles, so tag the root and apply this authored
            // start offset at most once per world instance.
            static const FName SensorStartAdjustedTag(TEXT("Qai.SensorStartAdjusted"));
            if (!Forklift.Root->ComponentHasTag(SensorStartAdjustedTag))
            {
                const FVector StartOffset(0.0f, 60.0f, 0.0f);
                Forklift.Root->AddWorldOffset(
                    StartOffset,
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
                // The imported pallet and carton are siblings of the vehicle,
                // not descendants of its root. Move the complete authored
                // load with the truck or it remains 60 cm behind and balances
                // on only one tine after the start-layout adjustment.
                for (USceneComponent* CargoRoot : {
                    Forklift.Pallet.Get(),
                    Forklift.Carton.Get()})
                {
                    if (CargoRoot)
                    {
                        CargoRoot->AddWorldOffset(
                            StartOffset,
                            false,
                            nullptr,
                            ETeleportType::TeleportPhysics);
                    }
                }
                Forklift.Root->ComponentTags.Add(SensorStartAdjustedTag);
                SimulatorLog(TEXT("forklift_sensor_start_offset forklift=1 toward_sensor_cm=60 cargo_roots_moved=2"));
            }

            // Start with the active truck facing the west storage rack. Rotate
            // the complete authored assembly about the chassis origin so the
            // sibling pallet/carton roots remain on the fork side. The later
            // Chaos setup refines their contact pose and records this as the
            // authoritative reset transform.
            static const FName ShelfFacingStartTag(TEXT("Qai.ShelfFacingStart"));
            if (!Forklift.Root->ComponentHasTag(ShelfFacingStartTag))
            {
                const FVector AssemblyOrigin = Forklift.Root->GetComponentLocation();
                const FQuat HalfTurn(FVector::UpVector, PI);
                Forklift.Root->SetWorldLocationAndRotation(
                    AssemblyOrigin,
                    HalfTurn * Forklift.Root->GetComponentQuat(),
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
                for (USceneComponent* CargoRoot : {
                    Forklift.Pallet.Get(),
                    Forklift.Carton.Get()})
                {
                    if (!CargoRoot)
                    {
                        continue;
                    }
                    const FVector RelativeLocation =
                        CargoRoot->GetComponentLocation() - AssemblyOrigin;
                    CargoRoot->SetWorldLocationAndRotation(
                        AssemblyOrigin + HalfTurn.RotateVector(RelativeLocation),
                        HalfTurn * CargoRoot->GetComponentQuat(),
                        false,
                        nullptr,
                        ETeleportType::TeleportPhysics);
                }
                Forklift.Root->ComponentTags.Add(ShelfFacingStartTag);
                SimulatorLog(TEXT("forklift_start_orientation forklift=1 yaw_delta_deg=180 facing=west_shelves cargo_roots_rotated=2"));
            }
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
            // The imported body AABB includes trim close to the mast. Using
            // its absolute front edge for the entire lower chassis made the
            // vehicle stop about 20 cm before the visible mast reached an
            // obstacle whenever the raised forks were clear. End the chassis
            // hull just behind the mast and let the independently fitted mast
            // collider represent the true leading structure.
            constexpr float LowerChassisFrontInsetCm = 20.0f;
            const float LowerChassisFrontX = FMath::Max(
                XAt(0.29f) + 5.0f,
                Maximum.X - LowerChassisFrontInsetCm);

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
                    FVector(LowerChassisFrontX, Center.Y + InnerHalfWidth, ZAt(0.43f))),
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

        const FName BodyName(*FString::Printf(TEXT("QaiChaosBody_%d"), DynamicBoxes.Num() + 1));
        const FTransform BodyTransform(VisualRoot->GetComponentQuat(), WorldBounds.GetCenter());
        FActorSpawnParameters SpawnParameters;
        SpawnParameters.Name = BodyName;
        SpawnParameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AActor* PhysicsActor = GetWorld()->SpawnActor<AActor>(
            AActor::StaticClass(), BodyTransform, SpawnParameters);
        if (!PhysicsActor)
        {
            return INDEX_NONE;
        }
        RuntimeChaosActors.Add(PhysicsActor);
        UBoxComponent* Pivot = NewObject<UBoxComponent>(PhysicsActor, BodyName);
        PhysicsActor->AddInstanceComponent(Pivot);
        PhysicsActor->SetRootComponent(Pivot);
        Pivot->SetMobility(EComponentMobility::Movable);
        Pivot->RegisterComponentWithWorld(GetWorld());
        Pivot->SetWorldTransform(BodyTransform);
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
        Body.bPallet = Name.Contains(TEXT("pallet"), ESearchCase::IgnoreCase);
        Body.bAwake = false;
        Pivot->SetBoxExtent(Body.HalfExtent, false);
        Pivot->SetHiddenInGame(true);
        Pivot->SetVisibility(false);
        Pivot->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        RuntimeChaosBodies.Add(Pivot);
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

    USceneComponent* PalletVisualTemplate = Forklifts[0].Pallet.Get();
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

    int32 BoundLoosePallets = 0;
    if (PalletVisualTemplate
        && PackingTableDesktopBounds.IsValid
        && DynamicBoxes.IsValidIndex(Forklifts[0].PalletDynamicBody))
    {
        FBox TemplateBounds(ForceInit);
        TArray<UStaticMeshComponent*> TemplateMeshes;
        TInlineComponentArray<UStaticMeshComponent*> CandidateMeshes(
            PalletVisualTemplate->GetOwner());
        for (UStaticMeshComponent* Mesh : CandidateMeshes)
        {
            if (!Mesh || !Mesh->GetStaticMesh()
                || (Mesh != PalletVisualTemplate && !Mesh->IsAttachedTo(PalletVisualTemplate)))
            {
                continue;
            }
            Mesh->UpdateBounds();
            TemplateBounds += Mesh->Bounds.GetBox();
            TemplateMeshes.Add(Mesh);
        }

        const FVector PalletHalfExtent =
            DynamicBoxes[Forklifts[0].PalletDynamicBody].HalfExtent;
        if (TemplateBounds.IsValid && TemplateMeshes.Num() > 0)
        {
            // Place the stack in the clear bay immediately west of the
            // sorting station, between its left edge and the shelving. The
            // pallets are rotated below, so local Y becomes their world-X
            // half width. Retain enough clearance that neither the table nor
            // rack begins in contact with a pallet.
            constexpr float PalletStackTableClearanceCm = 38.0f;
            const float RotatedPalletHalfWidthX = PalletHalfExtent.Y;
            const FVector StackBaseCenter(
                PackingTableDesktopBounds.Min.X
                    - RotatedPalletHalfWidthX
                    - PalletStackTableClearanceCm,
                PackingTableDesktopBounds.GetCenter().Y,
                PalletHalfExtent.Z + ConveyorTuning::PalletContactSkinCm + 0.35f);
            const FTransform TemplateTransform = PalletVisualTemplate->GetComponentTransform();
            // Turn the loose stack across the aisle so its fork pockets face
            // the forklift approach instead of running parallel to the tines.
            const FQuat StackRotation =
                FQuat(FVector::UpVector, HALF_PI) * TemplateTransform.GetRotation();
            const FVector TemplateCenterOffset =
                TemplateTransform.InverseTransformPositionNoScale(TemplateBounds.GetCenter());

            for (int32 PalletIndex = 0;
                 PalletIndex < ConveyorTuning::LoosePalletStackCount;
                 ++PalletIndex)
            {
                const FVector DesiredCenter = StackBaseCenter + FVector(
                    0.0f,
                    0.0f,
                    PalletIndex * (
                        PalletHalfExtent.Z * 2.0f
                        + ConveyorTuning::PalletContactSkinCm
                        + 0.35f));
                const FVector VisualRootLocation = DesiredCenter
                    - StackRotation.RotateVector(TemplateCenterOffset);
                const FTransform VisualTransform(
                    StackRotation,
                    VisualRootLocation,
                    TemplateTransform.GetScale3D());
                FActorSpawnParameters VisualParams;
                VisualParams.Name = FName(*FString::Printf(
                    TEXT("QaiLoosePalletVisual%d"), PalletIndex + 1));
                VisualParams.SpawnCollisionHandlingOverride =
                    ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
                AActor* VisualActor = GetWorld()->SpawnActor<AActor>(
                    AActor::StaticClass(), VisualTransform, VisualParams);
                if (!VisualActor)
                {
                    continue;
                }
                RuntimeChaosActors.Add(VisualActor);
                USceneComponent* VisualRoot = NewObject<USceneComponent>(
                    VisualActor,
                    FName(*FString::Printf(TEXT("QaiLoosePallet%dVisualRoot"), PalletIndex + 1)));
                VisualActor->AddInstanceComponent(VisualRoot);
                VisualActor->SetRootComponent(VisualRoot);
                VisualRoot->SetMobility(EComponentMobility::Movable);
                VisualRoot->RegisterComponentWithWorld(GetWorld());
                VisualRoot->SetWorldTransform(VisualTransform);

                int32 MeshIndex = 0;
                for (UStaticMeshComponent* TemplateMesh : TemplateMeshes)
                {
                    UStaticMeshComponent* Clone = NewObject<UStaticMeshComponent>(
                        VisualActor,
                        FName(*FString::Printf(
                            TEXT("QaiLoosePallet%dMesh%d"),
                            PalletIndex + 1,
                            ++MeshIndex)));
                    VisualActor->AddInstanceComponent(Clone);
                    Clone->SetMobility(EComponentMobility::Movable);
                    Clone->SetupAttachment(VisualRoot);
                    Clone->SetStaticMesh(TemplateMesh->GetStaticMesh());
                    for (int32 MaterialIndex = 0;
                         MaterialIndex < TemplateMesh->GetNumMaterials();
                         ++MaterialIndex)
                    {
                        Clone->SetMaterial(MaterialIndex, TemplateMesh->GetMaterial(MaterialIndex));
                    }
                    Clone->SetRelativeTransform(
                        TemplateMesh->GetComponentTransform().GetRelativeTransform(
                            TemplateTransform));
                    Clone->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                    Clone->SetGenerateOverlapEvents(false);
                    Clone->RegisterComponentWithWorld(GetWorld());
                }

                const int32 BodyIndex = AddDynamicBox(
                    VisualRoot,
                    FString::Printf(TEXT("loose pallet %d"), PalletIndex + 1),
                    INDEX_NONE,
                    false,
                    nullptr);
                BoundLoosePallets += BodyIndex != INDEX_NONE ? 1 : 0;
            }
            SimulatorLog(FString::Printf(
                TEXT("loose_pallet_stack_bound count=%d requested=%d center=(%.1f,%.1f) base_z=%.1f spacing_z=%.1f yaw_offset_deg=90 placement=between_west_shelf_and_sorting_table individual_chaos=true"),
                BoundLoosePallets,
                ConveyorTuning::LoosePalletStackCount,
                StackBaseCenter.X,
                StackBaseCenter.Y,
                StackBaseCenter.Z,
                PalletHalfExtent.Z * 2.0f
                    + ConveyorTuning::PalletContactSkinCm
                    + 0.35f));
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
            PackingTableDesktopBounds.Max.Z + ConveyorTuning::IQ9EvkHalfExtentCm.Z + 1.0f);
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
            TEXT("iq9_evk_placement location=(%.2f,%.2f,%.2f) desktop_min=(%.2f,%.2f,%.2f) desktop_max=(%.2f,%.2f,%.2f) tray_min=(%.2f,%.2f,%.2f) tray_max=(%.2f,%.2f,%.2f) horizontal=midpoint_left_edge_to_tray front_edge_inset_cm=10.00 chaos_contact_gap_cm=1.00"),
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
        TEXT("dynamic_boxes_bound total=%d forklift_cargo=%d shelf_parcels=%d loose_pallets=%d dynamic_props=%d"),
        DynamicBoxes.Num(),
        ConveyorTuning::EnabledForkliftCount * 2,
        BoundShelfParcels,
        BoundLoosePallets,
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
            if (ParcelOrientedBounds.IsValid)
            {
                const FName PivotName(*FString::Printf(TEXT("QaiParcelPivot_%d"), Index));
                const FTransform PivotTransform(
                    ParcelRootRotation,
                    ParcelRootOrigin + ParcelRootRotation.RotateVector(ParcelOrientedBounds.GetCenter()));
                FActorSpawnParameters SpawnParameters;
                SpawnParameters.Name = PivotName;
                SpawnParameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
                AActor* PhysicsActor = GetWorld()->SpawnActor<AActor>(
                    AActor::StaticClass(), PivotTransform, SpawnParameters);
                if (!PhysicsActor)
                {
                    continue;
                }
                RuntimeChaosActors.Add(PhysicsActor);
                UBoxComponent* Pivot = NewObject<UBoxComponent>(PhysicsActor, PivotName);
                PhysicsActor->AddInstanceComponent(Pivot);
                PhysicsActor->SetRootComponent(Pivot);
                Pivot->SetMobility(EComponentMobility::Movable);
                Pivot->RegisterComponentWithWorld(GetWorld());
                Pivot->SetWorldTransform(PivotTransform);
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
            if (UBoxComponent* ChaosParcel = Cast<UBoxComponent>(Parcel))
            {
                ChaosParcel->SetBoxExtent(ParcelHalfExtent, false);
                ChaosParcel->SetHiddenInGame(true);
                ChaosParcel->SetVisibility(false);
                ChaosParcel->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                RuntimeChaosConveyorBodies.Add(ChaosParcel);
            }
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
            // Begin with the carton's long local axis aligned to belt travel.
            // The authored USD yaw scatter made otherwise healthy parcels
            // scrub diagonally across successive roller contacts. Chaos is
            // still free to rotate a parcel after a forklift or another box
            // applies an off-centre impulse.
            ParcelHeadingOffsets.Add(FQuat::Identity);
            AuthoredPosition.Z = ConveyorSurfaceZCm + HalfHeight;
            Parcel->SetWorldLocationAndRotation(
                AuthoredPosition,
                Heading * ParcelHeadingOffsets.Last(),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            ParcelInitialTransforms.Last() = Parcel->GetComponentTransform();
            ParcelLinearVelocities.Add(
                AuthoredTangent * ConveyorTuning::ConveyorSpeedCm * ConveyorSpeedScale);
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
    SimulatorLog(TEXT("conveyor_contact_model support=individual_cylindrical_rollers drive=distributed_contact_patch_forces centering=weak_skewed_roller_axial_traction edge_overhang=physical fixed_step_hz=120"));

    Workers[0].Root = FindTaggedComponent(TEXT("Qai.Worker1"));
    // Keep both workers in separated lanes at the far end of the room. This
    // leaves them physically present while keeping the forklift approach and
    // red-zone sensor view clear.
    Workers[0].Waypoints = {
        FVector(-285.0f, -210.0f, 0.0f), FVector(-105.0f, -210.0f, 0.0f)};
    Workers[0].DwellSeconds = {0.55f, 0.85f};
    Workers[1].Root = FindTaggedComponent(TEXT("Qai.Worker2"));
    Workers[1].Waypoints = {
        FVector(-285.0f, -115.0f, 0.0f), FVector(-105.0f, -115.0f, 0.0f)};
    Workers[1].DwellSeconds = {1.20f, 0.40f};
    int32 BoundWorkers = 0;
    for (int32 WorkerIndex = 0; WorkerIndex < UE_ARRAY_COUNT(Workers); ++WorkerIndex)
    {
        FWorkerRuntime& Worker = Workers[WorkerIndex];
        if (!Worker.Root.IsValid() || Worker.Waypoints.Num() < 2)
        {
            continue;
        }
        MakeMovable(Worker.Root.Get());
        // Start the second worker from the opposite end and send it in the
        // opposite direction. Distinct dwell schedules prevent the two lanes
        // from becoming a visually coupled, perfectly parallel procession.
        const int32 InitialWaypointIndex = WorkerIndex == 0
            ? 0
            : Worker.Waypoints.Num() - 1;
        const int32 InitialDestinationIndex = WorkerIndex == 0
            ? 1
            : FMath::Max(0, Worker.Waypoints.Num() - 2);
        FVector WorkerSpawnLocation = Worker.Waypoints[InitialWaypointIndex];
        WorkerSpawnLocation.Z = Worker.Root->GetComponentLocation().Z;
        Worker.Root->SetWorldLocation(
            WorkerSpawnLocation,
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
        const FVector InitialDirection =
            Worker.Waypoints[InitialDestinationIndex] - Worker.Waypoints[InitialWaypointIndex];
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
        // The imported character hierarchies do not share a reliable forward
        // axis. Their final visual correction is calibrated from foot-to-toe
        // bones once the skeletal component is identified for Chaos binding.
        Worker.DestinationIndex = InitialDestinationIndex;
        Worker.RouteDirection = WorkerIndex == 0 ? 1 : -1;
        Worker.RouteReversalCount = 0;
        Worker.AvoidanceTurnSign = WorkerIndex == 0 ? 1 : -1;
        Worker.RouteReversalCooldownSeconds = 0.0f;
        Worker.WorkerYieldRemainingSeconds = 0.0f;
        Worker.RightOfWayYieldCount = 0;
        Worker.BlockedSeconds = 0.0f;
        Worker.InitialTransform = Worker.Root->GetComponentTransform();
        ++BoundWorkers;
    }
    SimulatorLog(FString::Printf(
        TEXT("worker_motion_ready workers=%d visual_turn_rate_deg_s=%.1f heading_commit_ms=%.0f reversal_cooldown_ms=%.0f personal_space_cm=%.1f reserved_lanes=true deterministic_right_of_way=true"),
        BoundWorkers,
        ConveyorTuning::WorkerMaximumVisualTurnRateDegreesPerSecond,
        ConveyorTuning::WorkerHeadingCommitSeconds * 1000.0f,
        ConveyorTuning::WorkerRouteReversalCooldownSeconds * 1000.0f,
        ConveyorTuning::WorkerPersonalSpaceCm));

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
    ConfigureInferenceScene();
    const int32 BoundColliders = BuildCollisionGuard();
    const bool bCollisionReady = ValidateCollisionGuard();
    ConfigureAuthoritativeChaosPhysics();
    const bool bRequiredComponentsReady = Parcels.Num() == 8
        && Forklifts[0].LiftAssembly.IsValid() && Forklifts[1].LiftAssembly.IsValid()
        && Forklifts[0].Pallet.IsValid() && Forklifts[0].Carton.IsValid()
        && Forklifts[1].Pallet.IsValid() && Forklifts[1].Carton.IsValid()
        && DetectorCamera.IsValid() && BoundWheels == 8 && BoundWorkers == 2
        && BoundShelfParcels == 24
        && BoundLoosePallets == ConveyorTuning::LoosePalletStackCount
        && DynamicBoxes.Num() == 26 + ConveyorTuning::EnabledForkliftCount * 2
            + ConveyorTuning::LoosePalletStackCount
        && EvkVisualRoot.IsValid() && DynamicBoxes.IsValidIndex(EvkDynamicBody)
        && PackingTableTrayVisualRoot.IsValid()
        && DynamicBoxes.IsValidIndex(PackingTableTrayDynamicBody)
        && BoundColliders >= 16 && bCollisionReady
        && bChaosPhysicsActive;
    if (!bRequiredComponentsReady)
    {
        StageStatus = bChaosPhysicsActive
            ? TEXT("NATIVE ASSET IMPORT INCOMPLETE")
            : TEXT("CHAOS PHYSICS INITIALIZATION FAILED");
        StageError = FString::Printf(
            TEXT("Expected authoritative Chaos plus 8 belt parcels, 24 dynamic shelf parcels, one movable IQ9 EVK, one movable desktop tray, 8 authored wheels, 2 authored lift assemblies, 2 workers, %d active cargo components, DetectorEndline, and 16 collision components; found chaos=%s, %d belt parcels, %d shelf parcels, %d wheels, %d workers, %d dynamic bodies, and %d collision components."),
            ConveyorTuning::EnabledForkliftCount * 2,
            bChaosPhysicsActive ? TEXT("ready") : TEXT("failed"),
            Parcels.Num(), BoundShelfParcels, BoundWheels, BoundWorkers, DynamicBoxes.Num(), BoundColliders);
        LogStageBindingFailure(FString::Printf(
            TEXT("reason=incomplete_scene chaos=%s belt_parcels=%d shelf_parcels=%d wheels=%d workers=%d dynamic_bodies=%d colliders=%d"),
            bChaosPhysicsActive ? TEXT("ready") : TEXT("failed"),
            Parcels.Num(),
            BoundShelfParcels,
            BoundWheels,
            BoundWorkers,
            DynamicBoxes.Num(),
            BoundColliders));
        return;
    }

    bStageReady = true;
    // Construction leaves dozens of Chaos bodies and the lift constraint
    // becoming live in one frame. Apply the same atomic pose restoration used
    // by X immediately after binding, so first presentation and reset begin at
    // an identical yaw rather than allowing a one-frame pallet impulse.
    ResetScene();
    SimulatorLog(TEXT("initial_scene_stabilized pose=reset_equivalent forklift_yaw_preserved=true"));
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
        TEXT("physics_solver_ready mode=chaos_rigid_body fixed_hz=120 max_substeps=8 dynamic_props=%d conveyor_props=%d ccd=true sleeping=enabled solver_iterations=12/4 gpu_cost=none"),
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

    // The current inference-stability ablation deliberately removes every
    // stack-light pixel and emitter from the workcell. The model result is
    // still reported verbatim in the HUD, but it can no longer recolor its
    // own next input through an emissive lens, direct light, Lumen bounce, or
    // volumetric fog. Pass -QaiEnableStackLights to restore the presentation
    // rig for an explicit A/B run without rebuilding the client.
    const bool bStackLightsEnabled = FParse::Param(FCommandLine::Get(), TEXT("QaiEnableStackLights"));
    if (!bStackLightsEnabled)
    {
        int32 HiddenComponents = 0;
        int32 DisabledLights = 0;
        int32 HiddenTaggedActors = 0;
        if (GetWorld())
        {
            for (TActorIterator<AActor> It(GetWorld()); It; ++It)
            {
                AActor* Actor = *It;
                bool bTaggedStackActor = false;
                for (const FName& Tag : Actor->Tags)
                {
                    const FString TagString = Tag.ToString();
                    const bool bTaggedSignalEmitter =
                        (TagString.StartsWith(TEXT("Qai.Green."))
                            || TagString.StartsWith(TEXT("Qai.Amber."))
                            || TagString.StartsWith(TEXT("Qai.Red.")))
                        && TagString.EndsWith(TEXT(".Glow"));
                    if (TagString.StartsWith(TEXT("Qai.Stack")) || bTaggedSignalEmitter)
                    {
                        bTaggedStackActor = true;
                        break;
                    }
                }
                if (bTaggedStackActor)
                {
                    Actor->SetActorHiddenInGame(true);
                    Actor->SetActorEnableCollision(false);
                    ++HiddenTaggedActors;
                }

                TInlineComponentArray<USceneComponent*> Components(Actor);
                for (USceneComponent* Component : Components)
                {
                    if (!Component)
                    {
                        continue;
                    }
                    bool bBelongsToStack = bTaggedStackActor;
                    for (USceneComponent* Ancestor = Component; Ancestor && !bBelongsToStack;
                        Ancestor = Ancestor->GetAttachParent())
                    {
                        FString LogicalName = Ancestor->GetName();
                        int32 LastUnderscore = INDEX_NONE;
                        if (LogicalName.FindLastChar(TEXT('_'), LastUnderscore)
                            && LogicalName.Mid(LastUnderscore + 1).IsNumeric())
                        {
                            LogicalName.LeftInline(LastUnderscore, EAllowShrinking::No);
                        }
                        bBelongsToStack = LogicalName.StartsWith(TEXT("StackLight"));
                    }
                    if (!bBelongsToStack)
                    {
                        continue;
                    }

                    Component->SetVisibility(false, true);
                    Component->SetHiddenInGame(true, true);
                    Component->Deactivate();
                    if (UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Component))
                    {
                        Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                        Primitive->SetGenerateOverlapEvents(false);
                    }
                    if (ULightComponent* Light = Cast<ULightComponent>(Component))
                    {
                        Light->SetIntensity(0.0f);
                        Light->SetLightColor(FLinearColor::Black, false);
                        ++DisabledLights;
                    }
                    ++HiddenComponents;
                }
            }
        }

        for (ALocalFogVolume* Actor : RuntimeStackHaloActors)
        {
            if (Actor)
            {
                Actor->Destroy();
            }
        }
        RuntimeStackHaloActors.Reset();
        for (ASpotLight* Actor : RuntimeStackWallSpotActors)
        {
            if (Actor)
            {
                Actor->Destroy();
            }
        }
        RuntimeStackWallSpotActors.Reset();
        for (ASpotLight* Actor : RuntimeStackRoomSpotActors)
        {
            if (Actor)
            {
                Actor->Destroy();
            }
        }
        RuntimeStackRoomSpotActors.Reset();
        for (ARectLight* Actor : RuntimeStackWashActors)
        {
            if (Actor)
            {
                Actor->Destroy();
            }
        }
        RuntimeStackWashActors.Reset();
        for (UMaterialBillboardComponent* Billboard : RuntimeStackBillboards)
        {
            if (Billboard)
            {
                Billboard->DestroyComponent();
            }
        }
        RuntimeStackBillboards.Reset();
        if (RuntimeSignalFogActor)
        {
            RuntimeSignalFogActor->Destroy();
            RuntimeSignalFogActor = nullptr;
        }

        StackWashLights.Reset();
        StackWashMultipliers.Reset();
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
            StackSignalWeights[ColorIndex] = 0.0f;
            StackTargetWeights[ColorIndex] = 0.0f;
        }
        bStackLightRigInitialized = false;
        SimulatorLog(FString::Printf(
            TEXT("stack_light_ablation enabled=true imported_components_hidden=%d lights_disabled=%d tagged_actors_hidden=%d hud_signal_retained=true restore_flag=-QaiEnableStackLights"),
            HiddenComponents,
            DisabledLights,
            HiddenTaggedActors));
        return;
    }

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
        // Match the photographed board: a close amber/green pair immediately
        // inside the expansion-connector edge, then a second green indicator
        // at the adjacent Ethernet edge. The five-view impostor is centred at
        // its physics pivot and its exposed PCB surface is at about Z=2.18 cm.
        // The top impostor is oriented opposite to the source photograph, so
        // mirror the complete layout through the PCB centre.
        FVector(3.25f, 0.35f, 2.36f), // blinking amber, outer edge
        FVector(2.60f, 0.35f, 2.36f), // steady green beside amber
        FVector(1.35f, 4.65f, 2.36f), // steady Ethernet-side green, 1 cm inboard of port edge
    };
    static const FLinearColor LedColors[] = {
        FLinearColor(1.0f, 0.24f, 0.005f),
        FLinearColor(0.01f, 1.0f, 0.05f),
        FLinearColor(0.01f, 1.0f, 0.05f),
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
        // The former 8-mm spheres read as oversized indicator caps. Four-mm
        // meshes preserve visibility while matching the actual surface LEDs.
        LedMesh->SetRelativeScale3D(FVector(0.004f));
        UMaterialInstanceDynamic* LedMID = UMaterialInstanceDynamic::Create(LedMaterial, this);
        if (LedMID)
        {
            LedMID->SetVectorParameterValue(TEXT("LedColor"), LedColors[LedIndex]);
            LedMID->SetScalarParameterValue(TEXT("LedStrength"), LedIndex == 0 ? 0.0f : 180.0f);
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
        LedLight->SetIntensity(LedIndex == 0 ? 0.0f : 28.0f);
        LedLight->SetLightColor(LedColors[LedIndex]);
        LedLight->SetAttenuationRadius(24.0f);
        LedLight->SetSourceRadius(0.175f);
        LedLight->SetSoftSourceRadius(0.5f);
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
                Fog->SetFogEmissive(LedIndex == 0 ? FLinearColor::Black : LedColors[LedIndex] * 0.48f);
                Fog->SetFogStartDistance(0.0f);
                EvkLedFogComponents.Add(Fog);
            }
            RuntimeEvkLedFogActors.Add(FogActor);
        }
        EvkLedTimers.Add(LedIndex == 0 ? 0.32f : 0.0f);
        EvkLedStates.Add(LedIndex != 0);
    }
    SimulatorLog(FString::Printf(
        TEXT("iq9_evk_led_rig status=ready leds=%d mode=two_steady_green+random_blink_amber mesh_diameter_cm=0.4 local_fog_volumes=%d point_radius_cm=24 random_seed=0x19E9"),
        RuntimeEvkLedMeshes.Num(),
        RuntimeEvkLedFogActors.Num()));
}

void AQaiConveyorWorld::TickEvkLeds(float DeltaSeconds)
{
    static const FLinearColor LedColors[] = {
        FLinearColor(1.0f, 0.24f, 0.005f),
        FLinearColor(0.01f, 1.0f, 0.05f),
        FLinearColor(0.01f, 1.0f, 0.05f),
    };
    for (int32 LedIndex = 0; LedIndex < EvkLedTimers.Num(); ++LedIndex)
    {
        const bool bBlinkingAmber = LedIndex == 0;
        bool bEnabled = !bBlinkingAmber;
        if (bBlinkingAmber)
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

    if (bChaosPhysicsActive)
    {
        // Reset must be atomic from Chaos's point of view.  Re-enabling the
        // chassis or carriage while cargo still occupies its pre-reset pose
        // creates a contact impulse through the live lift joint and can wedge
        // the carriage against its locked lateral axes.
        for (FForkliftRuntime& Forklift : Forklifts)
        {
            if (UPhysicsConstraintComponent* Constraint = Forklift.ChaosLiftConstraint.Get())
            {
                Constraint->BreakConstraint();
            }
            for (UBoxComponent* Body : {
                    Forklift.ChaosChassis.Get(),
                    Forklift.ChaosCarriage.Get()})
            {
                if (!Body)
                {
                    continue;
                }
                Body->SetPhysicsLinearVelocity(FVector::ZeroVector);
                Body->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
                Body->SetSimulatePhysics(false);
                Body->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            }
            for (const TWeakObjectPtr<UBoxComponent>& Shape : Forklift.ChaosCollisionComponents)
            {
                if (UBoxComponent* Collider = Shape.Get())
                {
                    Collider->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                }
            }
        }
        for (UBoxComponent* Body : RuntimeChaosBodies)
        {
            if (Body)
            {
                Body->SetPhysicsLinearVelocity(FVector::ZeroVector);
                Body->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
                Body->SetSimulatePhysics(false);
                Body->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            }
        }
        for (FDynamicBoxRuntime& Body : DynamicBoxes)
        {
            for (const TWeakObjectPtr<UBoxComponent>& ShapeWeak :
                 Body.CompoundCollisionComponents)
            {
                if (UBoxComponent* Shape = ShapeWeak.Get())
                {
                    Shape->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                }
            }
        }
        for (UCapsuleComponent* WorkerBody : RuntimeChaosWorkerBodies)
        {
            if (WorkerBody)
            {
                WorkerBody->SetPhysicsLinearVelocity(FVector::ZeroVector);
                WorkerBody->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
                WorkerBody->SetSimulatePhysics(false);
                WorkerBody->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            }
        }
        for (UBoxComponent* Body : RuntimeChaosConveyorBodies)
        {
            if (Body)
            {
                Body->SetPhysicsLinearVelocity(FVector::ZeroVector);
                Body->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
                Body->SetSimulatePhysics(false);
                Body->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            }
        }
    }

    for (int32 ForkliftIndex = 0; ForkliftIndex < UE_ARRAY_COUNT(Forklifts); ++ForkliftIndex)
    {
        FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        if (!Forklift.Root.IsValid())
        {
            continue;
        }
        if (bChaosPhysicsActive && Forklift.ChaosChassis.IsValid())
        {
            UBoxComponent* Chassis = Forklift.ChaosChassis.Get();
            UBoxComponent* Carriage = Forklift.ChaosCarriage.Get();
            Chassis->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            Chassis->SetWorldTransform(
                Forklift.InitialChaosChassis,
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            Chassis->SetPhysicsLinearVelocity(FVector::ZeroVector);
            Chassis->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
            if (Carriage)
            {
                Carriage->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                Carriage->SetWorldTransform(
                    Forklift.InitialChaosCarriage,
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
                Carriage->SetPhysicsLinearVelocity(FVector::ZeroVector);
                Carriage->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
            }
            if (UPhysicsConstraintComponent* Constraint = Forklift.ChaosLiftConstraint.Get())
            {
                Constraint->SetLinearPositionTarget(FVector(
                    0.0f,
                    0.0f,
                    -ConveyorTuning::ForkliftInitialLiftCm));
                Constraint->SetLinearVelocityTarget(FVector::ZeroVector);
            }
        }
        else
        {
            Forklift.Root->SetWorldTransform(
                Forklift.InitialRoot, false, nullptr, ETeleportType::TeleportPhysics);
        }
        Forklift.SpeedCmPerSecond = 0.0f;
        Forklift.SteeringInput = 0.0f;
        Forklift.SurfaceLinearVelocityCmPerSecond = FVector::ZeroVector;
        Forklift.SurfaceYawVelocityDegreesPerSecond = 0.0f;
        Forklift.PreviousSpeedCmPerSecond = 0.0f;
        Forklift.CombinedMassKg = Forklift.ChassisMassKg;
        Forklift.CombinedCenterOfMassLocalCm = Forklift.BaseCenterOfMassLocalCm;
        Forklift.LiftCm = ConveyorTuning::ForkliftInitialLiftCm;
        Forklift.ActualLiftCm = ConveyorTuning::ForkliftInitialLiftCm;
        Forklift.LiftDeltaCm = 0.0f;
        Forklift.bLiftUpperStopLatched = false;
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
                WheelPivot->SetRelativeLocation(
                    Forklift.WheelPivotInitialRelativeLocation[WheelIndex]);
                WheelPivot->SetRelativeRotation(Forklift.WheelPivotInitialRelative[WheelIndex]);
            }
        }
        if (!bChaosPhysicsActive)
        {
            UpdateCargo(Forklift);
        }
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
            if (UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Root); Primitive && bChaosPhysicsActive)
            {
                Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                Primitive->SetWorldTransform(
                    Body.InitialTransform,
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
                Primitive->SetPhysicsLinearVelocity(FVector::ZeroVector);
                Primitive->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
            }
            else
            {
                Root->SetWorldTransform(Body.InitialTransform, false, nullptr, ETeleportType::TeleportPhysics);
            }
        }
    }

    constexpr float ConveyorPerimeter = 2.0f * (2.0f * 235.1374f) + 2.0f * PI * 150.0f;
    for (int32 Index = 0; Index < Parcels.Num(); ++Index)
    {
        if (USceneComponent* Parcel = Parcels[Index].Get(); Parcel && ParcelInitialTransforms.IsValidIndex(Index))
        {
            if (UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Parcel); Primitive && bChaosPhysicsActive)
            {
                Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                Primitive->SetWorldTransform(
                    ParcelInitialTransforms[Index],
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
                Primitive->SetPhysicsLinearVelocity(FVector::ZeroVector);
                Primitive->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
            }
            else
            {
                Parcel->SetWorldTransform(
                    ParcelInitialTransforms[Index], false, nullptr, ETeleportType::TeleportPhysics);
            }
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
            ParcelLinearVelocities[Index] =
                Tangent * ConveyorTuning::ConveyorSpeedCm * ConveyorSpeedScale;
        }
    }

    if (bChaosPhysicsActive)
    {
        // Recreate each prismatic joint only after both particles are back at
        // their authored transforms. SetConstrainedComponents rebuilds the
        // Chaos connector handles and local frames invalidated by BreakConstraint.
        for (FForkliftRuntime& Forklift : Forklifts)
        {
            UBoxComponent* Chassis = Forklift.ChaosChassis.Get();
            UBoxComponent* Carriage = Forklift.ChaosCarriage.Get();
            if (!Chassis || !Carriage)
            {
                continue;
            }
            for (const TWeakObjectPtr<UBoxComponent>& Shape : Forklift.ChaosCollisionComponents)
            {
                if (UBoxComponent* Collider = Shape.Get())
                {
                    Collider->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
                }
            }
            Chassis->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
            Carriage->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
            Chassis->SetSimulatePhysics(true);
            Carriage->SetSimulatePhysics(true);
            Chassis->SetMassOverrideInKg(NAME_None, Forklift.ChassisMassKg, true);
            Chassis->SetCenterOfMass(
                Forklift.BaseCenterOfMassLocalCm - Forklift.ChaosChassisLocalCenter);
            Carriage->SetMassOverrideInKg(NAME_None, 420.0f, true);
            Carriage->SetCenterOfMass(FVector(-4.0f, 0.0f, -18.0f));
            Chassis->SetPhysicsLinearVelocity(FVector::ZeroVector);
            Chassis->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
            Carriage->SetPhysicsLinearVelocity(FVector::ZeroVector);
            Carriage->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
            if (UPhysicsConstraintComponent* Constraint = Forklift.ChaosLiftConstraint.Get())
            {
                // Rebuild the joint at its true zero-height reference, exactly
                // as startup does, then restore the raised initial carriage.
                // Binding directly at the 10-cm pose would redefine that pose
                // as zero and make the next -10-cm drive lift it twice.
                FTransform MechanicalMinimumCarriage = Forklift.InitialChaosCarriage;
                MechanicalMinimumCarriage.AddToTranslation(
                    -Forklift.InitialChaosChassis.GetUnitAxis(EAxis::Z)
                    * ConveyorTuning::ForkliftInitialLiftCm);
                Carriage->SetWorldTransform(
                    MechanicalMinimumCarriage,
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
                Constraint->SetWorldLocationAndRotation(
                    MechanicalMinimumCarriage.GetLocation(),
                    Forklift.InitialChaosChassis.GetRotation());
                Constraint->SetConstrainedComponents(
                    Chassis, NAME_None, Carriage, NAME_None);
                Constraint->SetLinearPositionDrive(false, false, true);
                Constraint->SetLinearVelocityDrive(false, false, true);
                Constraint->SetLinearDriveParams(620000.0f, 95000.0f, 12000000.0f);
                Constraint->SetLinearPositionTarget(FVector(
                    0.0f,
                    0.0f,
                    -ConveyorTuning::ForkliftInitialLiftCm));
                Constraint->SetLinearVelocityTarget(FVector::ZeroVector);
                Constraint->SetProjectionEnabled(true);
                Constraint->SetProjectionParams(0.10f, 0.25f, 1.0f, 1.0f);
                Carriage->SetWorldTransform(
                    Forklift.InitialChaosCarriage,
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
                Carriage->SetPhysicsLinearVelocity(FVector::ZeroVector);
                Carriage->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
            }
            Chassis->WakeRigidBody();
            Carriage->WakeRigidBody();
        }
        for (FDynamicBoxRuntime& BodyRuntime : DynamicBoxes)
        {
            UPrimitiveComponent* Body = Cast<UPrimitiveComponent>(BodyRuntime.Root.Get());
            if (!Body)
            {
                continue;
            }
            Body->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
            Body->SetSimulatePhysics(true);
            Body->SetMassOverrideInKg(NAME_None, BodyRuntime.Physics.MassKg, true);
            Body->SetCenterOfMass(BodyRuntime.Physics.CenterOfMassLocalOffset);
            Body->SetLinearDamping(BodyRuntime.Physics.AirLinearDamping);
            Body->SetAngularDamping(BodyRuntime.Physics.AirAngularDamping);
            Body->SetPhysicsLinearVelocity(FVector::ZeroVector);
            Body->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
            Body->PutRigidBodyToSleep();
            for (const TWeakObjectPtr<UBoxComponent>& ShapeWeak :
                 BodyRuntime.CompoundCollisionComponents)
            {
                if (UBoxComponent* Shape = ShapeWeak.Get())
                {
                    Shape->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
                }
            }
        }
        // Match startup sequencing for belt parcels: hold them collisionless
        // until the broadphase and lift joint have settled, then activate all
        // eight together in SimulateChaosConveyor.
        for (UBoxComponent* Body : RuntimeChaosConveyorBodies)
        {
            if (!Body)
            {
                continue;
            }
            Body->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            Body->SetSimulatePhysics(false);
        }
        bChaosConveyorBodiesActivated = false;
        ChaosConveyorStartupGraceSeconds = 0.75f;
        ChaosConveyorMotorRampSeconds = 0.0f;
        SimulatorLog(TEXT("chaos_scene_reset phase=atomic joint=reinitialized dynamic_bodies=settled conveyor_activation=deferred"));
    }
    ParcelImpactCount = 0;
    ParcelForkContactCount = 0;
    bLoggedFirstParcelContact = false;

    for (int32 WorkerIndex = 0; WorkerIndex < UE_ARRAY_COUNT(Workers); ++WorkerIndex)
    {
        FWorkerRuntime& Worker = Workers[WorkerIndex];
        if (UCapsuleComponent* WorkerBody = Worker.ChaosBody.Get())
        {
            WorkerBody->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            WorkerBody->SetWorldTransform(
                Worker.InitialChaosTransform,
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
            WorkerBody->SetPhysicsLinearVelocity(FVector::ZeroVector);
            WorkerBody->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
            if (USceneComponent* VisualPivot = Worker.VisualPivot.Get())
            {
                VisualPivot->SetRelativeTransform(FTransform::Identity);
            }
            if (USceneComponent* VisualRoot = Worker.Root.Get())
            {
                VisualRoot->SetRelativeTransform(Worker.InitialVisualRelativeTransform);
            }
            WorkerBody->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
            WorkerBody->SetSimulatePhysics(true);
            WorkerBody->SetMassOverrideInKg(
                NAME_None,
                ConveyorTuning::WorkerMassKg,
                true);
            WorkerBody->WakeRigidBody();
            Worker.bLoggedFootGroundingRuntime = false;
        }
        else if (USceneComponent* Root = Worker.Root.Get())
        {
            Root->SetWorldTransform(Worker.InitialTransform, false, nullptr, ETeleportType::TeleportPhysics);
        }
        Worker.DestinationIndex = WorkerIndex == 0 ? 1 : 0;
        Worker.RouteDirection = WorkerIndex == 0 ? 1 : -1;
        Worker.RouteReversalCount = 0;
        Worker.AvoidanceTurnSign = WorkerIndex == 0 ? 1 : -1;
        Worker.DwellRemaining = WorkerIndex == 0 ? 0.0f : 0.35f;
        Worker.BlockedSeconds = 0.0f;
        Worker.RouteReversalCooldownSeconds = 0.0f;
        Worker.WorkerYieldRemainingSeconds = 0.0f;
        Worker.TargetHeadingYawDegrees = Worker.InitialHeadingYawDegrees;
        Worker.PendingHeadingYawDegrees = Worker.InitialHeadingYawDegrees;
        Worker.PendingHeadingSeconds = 0.0f;
        Worker.VisualHeadingYawDegrees = Worker.InitialHeadingYawDegrees;
        Worker.MaximumVisualTurnRateDegreesPerSecond = 0.0f;
        Worker.SeparationEvents = 0;
        Worker.RightOfWayYieldCount = 0;
    }
    WorkerMinimumPairClearanceCm = TNumericLimits<float>::Max();
    WorkerConflictElapsedSeconds = 0.0f;
    WorkerRightOfWayIndex = INDEX_NONE;
    WorkerNextRightOfWayIndex = 0;
    bWorkerConflictHeadOn = false;

    EncodedFrames.Reset();
    EncodedFrameWidths.Reset();
    EncodedFrameHeights.Reset();
    EncodedParcelSafetySignals.Reset();
    EncodedFrameTextures.Reset();
    EncodedFrameTimes.Reset();
    EncodedFrameCaptureSeconds.Reset();
    SubmittedEncodedFrames.Reset();
    SubmittedFrameTextures.Reset();
    SubmittedFrameTimes.Reset();
    SubmittedGroundTruthSignal = TEXT("G");
    RawModelSignal = TEXT("-");
    ModelSignal = TEXT("-");
    GroundTruthSignal = TEXT("G");
    ParcelSafetyCandidateSignal = TEXT("G");
    ParcelSafetyCandidateSeconds = 0.0f;
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
        // These catch walls deliberately sit outside the presented room. F8
        // describes collision fitted to visible scene geometry, so omit the
        // distant safety perimeter from that local diagnostic.
        if (Obstacle.Name == TEXT("invisible perimeter"))
        {
            continue;
        }
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
            int32 DebugWheelIndex = 0;
            for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
            {
                FVector LocalCenter = Shape.LocalCenter;
                LocalCenter.Z += Shape.bLiftDriven
                    ? (bChaosPhysicsActive ? Forklift.ActualLiftCm : Forklift.LiftCm)
                    : 0.0f;
                const FVector ShapeCenter = Root->GetComponentLocation()
                    + RootQuat.RotateVector(LocalCenter);
                if (Shape.IsCylinder())
                {
                    FVector WheelCenter = ShapeCenter;
                    FVector AxleDirection = RootQuat.GetRightVector();
                    if (DebugWheelIndex < 4)
                    {
                        if (const USceneComponent* WheelPivot =
                                Forklift.WheelPivots[DebugWheelIndex].Get())
                        {
                            // Raycast tires are not separate Chaos rigid
                            // bodies. Draw their resolved physical hub pose,
                            // which is also used by the rendered wheel, rather
                            // than the chassis-authored reference cylinder.
                            WheelCenter = WheelPivot->GetComponentLocation();
                            AxleDirection = WheelPivot->GetRightVector();
                        }
                    }
                    ++DebugWheelIndex;
                    const FVector Axle = AxleDirection * Shape.CylinderHalfLength;
                    DrawDebugCylinder(
                        GetWorld(),
                        WheelCenter - Axle,
                        WheelCenter + Axle,
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
        const USceneComponent* Root = Worker.ChaosBody.IsValid()
            ? static_cast<const USceneComponent*>(Worker.ChaosBody.Get())
            : Worker.Root.Get();
        if (Root)
        {
            Lines->DrawCapsule(
                Root->GetComponentLocation()
                    + (Worker.ChaosBody.IsValid()
                        ? FVector::ZeroVector
                        : FVector(0.0f, 0.0f, ConveyorTuning::WorkerCapsuleHalfHeightCm)),
                ConveyorTuning::WorkerCapsuleHalfHeightCm,
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
            if (Body.bPallet && Body.CompoundCollisionComponents.Num() > 0)
            {
                if (const UBoxComponent* RootBox = Cast<UBoxComponent>(Root))
                {
                    Lines->DrawBox(
                        RootBox->GetComponentLocation(),
                        RootBox->GetUnscaledBoxExtent(),
                        RootBox->GetComponentQuat(),
                        FLinearColor(1.0f, 0.72f, 0.0f),
                        LifeTime,
                        DepthPriority,
                        2.5f);
                }
                for (const TWeakObjectPtr<UBoxComponent>& ShapeWeak :
                     Body.CompoundCollisionComponents)
                {
                    if (const UBoxComponent* Shape = ShapeWeak.Get())
                    {
                        Lines->DrawBox(
                            Shape->GetComponentLocation(),
                            Shape->GetUnscaledBoxExtent(),
                            Shape->GetComponentQuat(),
                            FLinearColor(1.0f, 0.72f, 0.0f),
                            LifeTime,
                            DepthPriority,
                            2.5f);
                    }
                }
            }
            else
            {
                Lines->DrawBox(
                    Root->GetComponentLocation(),
                    Body.HalfExtent,
                    Root->GetComponentQuat(),
                    FLinearColor(1.0f, 0.72f, 0.0f),
                    LifeTime,
                    DepthPriority,
                    2.5f);
            }
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

    // Conveyor and shelf cartons share the same loaded-cardboard Chaos
    // profile, so use the same yellow debug language for both populations.
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
            FLinearColor(1.0f, 0.72f, 0.0f),
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
                FColor(255, 245, 120),
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
    return (bChaosPhysicsActive
        ? Forklifts[ActiveForklift].ActualLiftCm
        : Forklifts[ActiveForklift].LiftCm) / 100.0f;
}

void AQaiConveyorWorld::ConfigureAuthoritativeChaosPhysics()
{
    if (bChaosPhysicsActive || !GetWorld())
    {
        return;
    }

    RuntimeChaosMaterials.Reset();
    RuntimeChaosStaticColliders.Reset();
    RuntimeChaosRollerColliders.Reset();
    RuntimeChaosForkliftColliders.Reset();
    RuntimeChaosConstraints.Reset();
    RuntimeChaosWorkerBodies.Reset();

    // The USD collision nodes are useful authoring references, but they are
    // not allowed to participate beside the fitted Chaos bodies. Every solid
    // in this mode is installed exactly once below.
    for (const TWeakObjectPtr<UPrimitiveComponent>& Authored : AuthoredCollisionComponents)
    {
        if (UPrimitiveComponent* Primitive = Authored.Get())
        {
            Primitive->SetSimulatePhysics(false);
            Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            Primitive->SetGenerateOverlapEvents(false);
        }
    }
    // The optimized USD stage is a single imported actor and retains a broad
    // generated StaticMeshComponent collision hull on its actor root. It is
    // not one of the tagged authoring colliders and overlaps nearly the entire
    // room. Leaving it active beside the fitted proxies makes Chaos eject
    // every rigid body from the scene. Rendering stays on; only all imported
    // physics participation is removed here.
    if (Forklifts[0].Root.IsValid() && Forklifts[0].Root->GetOwner())
    {
        TInlineComponentArray<UPrimitiveComponent*> ImportedStagePrimitives(
            Forklifts[0].Root->GetOwner());
        for (UPrimitiveComponent* Primitive : ImportedStagePrimitives)
        {
            Primitive->SetSimulatePhysics(false);
            Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            Primitive->SetGenerateOverlapEvents(false);
        }
    }
    // The neutral presentation backdrop is a 300 m visual cube. It must
    // never be a physical room: its inward-facing render surface encloses all
    // gameplay bodies and Chaos interprets that as one enormous penetration.
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        TInlineComponentArray<UPrimitiveComponent*> Primitives(*It);
        for (UPrimitiveComponent* Primitive : Primitives)
        {
            const FVector Extent = Primitive->Bounds.BoxExtent;
            if (Primitive->GetCollisionObjectType() == ECC_WorldStatic
                && Extent.GetMax() >= 5000.0f)
            {
                Primitive->SetSimulatePhysics(false);
                Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                Primitive->SetGenerateOverlapEvents(false);
                SimulatorLog(FString::Printf(
                    TEXT("chaos_visual_backdrop_collision_disabled actor=%s component=%s extent=(%.0f,%.0f,%.0f)"),
                    *It->GetName(), *Primitive->GetName(), Extent.X, Extent.Y, Extent.Z));
            }
        }
    }
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        if (!It->ActorHasTag(TEXT("Qai.DetailedPackingTable")))
        {
            continue;
        }
        TInlineComponentArray<UPrimitiveComponent*> TableVisuals(*It);
        for (UPrimitiveComponent* Primitive : TableVisuals)
        {
            Primitive->SetSimulatePhysics(false);
            Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            Primitive->SetGenerateOverlapEvents(false);
        }
    }

    const auto MakePhysicalMaterial = [this](
        const FString& Name,
        float StaticFriction,
        float DynamicFriction,
        float Restitution,
        EFrictionCombineMode::Type FrictionCombineMode = EFrictionCombineMode::Average)
    {
        UPhysicalMaterial* Material = NewObject<UPhysicalMaterial>(
            this,
            FName(*FString::Printf(TEXT("PM_Authoritative_%s_%d"),
                *Name.Replace(TEXT(" "), TEXT("_")), RuntimeChaosMaterials.Num())));
        Material->StaticFriction = StaticFriction;
        Material->Friction = DynamicFriction;
        Material->Restitution = Restitution;
        Material->bOverrideFrictionCombineMode = true;
        Material->FrictionCombineMode = FrictionCombineMode;
        Material->bOverrideRestitutionCombineMode = true;
        Material->RestitutionCombineMode = EFrictionCombineMode::Min;
        Material->SleepLinearVelocityThreshold = 0.8f;
        Material->SleepAngularVelocityThreshold = FMath::DegreesToRadians(0.8f);
        Material->SleepCounterThreshold = 10;
        RuntimeChaosMaterials.Add(Material);
        return Material;
    };

    UPhysicalMaterial* WarehouseMaterial = MakePhysicalMaterial(
        TEXT("warehouse"), 0.84f, 0.67f, 0.025f);
    UPhysicalMaterial* ForkliftMaterial = MakePhysicalMaterial(
        TEXT("forklift"), 0.76f, 0.60f, 0.02f);
    UPhysicalMaterial* PoweredRollerMaterial = MakePhysicalMaterial(
        TEXT("powered_roller"),
        // The support proxy itself is stationary, while the real roller skin
        // moves. Keep proxy friction low so it does not oppose the explicit,
        // speed-controlled roller traction applied below.
        0.04f,
        0.025f,
        0.015f,
        EFrictionCombineMode::Min);

    const auto ConfigureCollision = [](UPrimitiveComponent* Component, ECollisionChannel ObjectType)
    {
        Component->SetCollisionObjectType(ObjectType);
        Component->SetCollisionResponseToAllChannels(ECR_Block);
        Component->SetCollisionResponseToChannel(ECC_Camera, ECR_Ignore);
        Component->SetGenerateOverlapEvents(false);
        Component->SetNotifyRigidBodyCollision(false);
    };

    // Install the complete static warehouse first. This guarantees that no
    // dynamic body is ever activated before its resting surface exists.
    for (int32 Index = 0; Index < CollisionObstacles.Num(); ++Index)
    {
        const FCollisionObstacle& Obstacle = CollisionObstacles[Index];
        // The old belt support was 96 overlapping tangent-aligned boxes. Their
        // seam normals nudged parcels from side to side. Individual cylinders
        // below are now the authoritative roller contact geometry.
        if (Obstacle.Name == TEXT("conveyor"))
        {
            continue;
        }
        FActorSpawnParameters Params;
        Params.Name = FName(*FString::Printf(TEXT("QaiChaosStaticActor_%d"), Index + 1));
        Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AActor* StaticActor = GetWorld()->SpawnActor<AActor>(
            AActor::StaticClass(), FTransform::Identity, Params);
        if (!StaticActor)
        {
            continue;
        }
        RuntimeChaosActors.Add(StaticActor);
        UBoxComponent* Collider = NewObject<UBoxComponent>(
            StaticActor,
            FName(*FString::Printf(TEXT("QaiChaosStatic_%d"), Index + 1)));
        StaticActor->AddInstanceComponent(Collider);
        StaticActor->SetRootComponent(Collider);
        // AActor has no native scene root. Build and position its runtime root
        // while movable, then freeze it; setting a world transform after
        // declaring Static is rejected by Unreal and previously left every
        // proxy stacked at the origin.
        Collider->SetMobility(EComponentMobility::Movable);
        Collider->SetBoxExtent(Obstacle.HalfExtent, false);
        Collider->SetHiddenInGame(true);
        Collider->SetVisibility(false);
        Collider->SetPhysMaterialOverride(
            Obstacle.Name == TEXT("conveyor")
                ? PoweredRollerMaterial
                : WarehouseMaterial);
        ConfigureCollision(Collider, ECC_WorldStatic);
        Collider->RegisterComponentWithWorld(GetWorld());
        Collider->SetWorldLocationAndRotation(Obstacle.Center, Obstacle.Rotation);
        Collider->SetMobility(EComponentMobility::Static);
        Collider->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        RuntimeChaosStaticColliders.Add(Collider);
    }

    // Build a closely spaced bank of real cylindrical Chaos contacts around
    // the complete analytic loop. The cylinders remain fixed bearings; motor
    // torque is applied at each parcel/roller contact patch in
    // SimulateChaosConveyor. This retains rigid-body overhang, tipping and
    // falling while eliminating the lateral seam impulses from box proxies.
    UStaticMesh* RollerCylinderMesh = LoadObject<UStaticMesh>(
        nullptr,
        TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
    constexpr float StraightHalf = 235.1374f;
    constexpr float Radius = 150.0f;
    constexpr float ConveyorPerimeter = 4.0f * StraightHalf + 2.0f * PI * Radius;
    const int32 RollerCount = FMath::CeilToInt(
        ConveyorPerimeter / ConveyorTuning::ConveyorRollerSpacingCm);
    if (RollerCylinderMesh)
    {
        for (int32 RollerIndex = 0; RollerIndex < RollerCount; ++RollerIndex)
        {
            FVector RollerPosition;
            FVector RollerTangent;
            ConveyorTuning::EvaluateConveyor(
                (static_cast<float>(RollerIndex) + 0.5f)
                    * ConveyorPerimeter / static_cast<float>(RollerCount),
                RollerPosition,
                RollerTangent);
            const FVector RollerAxis(
                -RollerTangent.Y,
                RollerTangent.X,
                0.0f);
            RollerPosition.Z = ConveyorSurfaceZCm
                - ConveyorTuning::ConveyorRollerRadiusCm;
            const FQuat RollerRotation = FQuat::FindBetweenNormals(
                FVector::UpVector,
                RollerAxis.GetSafeNormal());
            FActorSpawnParameters RollerParams;
            RollerParams.Name = FName(*FString::Printf(
                TEXT("QaiPoweredRollerActor_%03d"), RollerIndex + 1));
            RollerParams.SpawnCollisionHandlingOverride =
                ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
            AActor* RollerActor = GetWorld()->SpawnActor<AActor>(
                AActor::StaticClass(),
                FTransform(RollerRotation, RollerPosition),
                RollerParams);
            if (!RollerActor)
            {
                continue;
            }
            RuntimeChaosActors.Add(RollerActor);
            UStaticMeshComponent* Roller = NewObject<UStaticMeshComponent>(
                RollerActor,
                FName(*FString::Printf(TEXT("QaiPoweredRoller_%03d"), RollerIndex + 1)));
            RollerActor->AddInstanceComponent(Roller);
            RollerActor->SetRootComponent(Roller);
            Roller->SetMobility(EComponentMobility::Movable);
            Roller->SetStaticMesh(RollerCylinderMesh);
            Roller->SetWorldScale3D(FVector(
                ConveyorTuning::ConveyorRollerRadiusCm / 50.0f,
                ConveyorTuning::ConveyorRollerRadiusCm / 50.0f,
                ConveyorTuning::ConveyorRollerHalfLengthCm / 50.0f));
            Roller->SetHiddenInGame(true);
            Roller->SetVisibility(false);
            Roller->SetPhysMaterialOverride(PoweredRollerMaterial);
            ConfigureCollision(Roller, ECC_WorldStatic);
            Roller->RegisterComponentWithWorld(GetWorld());
            Roller->SetWorldLocationAndRotation(RollerPosition, RollerRotation);
            Roller->SetMobility(EComponentMobility::Static);
            Roller->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
            RuntimeChaosRollerColliders.Add(Roller);
        }
    }
    SimulatorLog(FString::Printf(
        TEXT("chaos_powered_rollers geometry=cylinders count=%d radius_cm=%.2f spacing_cm=%.2f length_cm=%.2f motor_samples=%d box_support_removed=true"),
        RuntimeChaosRollerColliders.Num(),
        ConveyorTuning::ConveyorRollerRadiusCm,
        ConveyorPerimeter / FMath::Max(1, RuntimeChaosRollerColliders.Num()),
        ConveyorTuning::ConveyorRollerHalfLengthCm * 2.0f,
        ConveyorTuning::ConveyorMotorContactSamples));

    // The front of the room is intentionally open visually, so add a floor
    // slab independently of the authored wall/furniture proxies.
    FActorSpawnParameters FloorParams;
    FloorParams.Name = TEXT("QaiChaosRoomFloorActor");
    FloorParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    AActor* FloorActor = GetWorld()->SpawnActor<AActor>(
        AActor::StaticClass(), FTransform::Identity, FloorParams);
    if (FloorActor)
    {
        RuntimeChaosActors.Add(FloorActor);
        UBoxComponent* Floor = NewObject<UBoxComponent>(FloorActor, TEXT("QaiChaosRoomFloor"));
        FloorActor->AddInstanceComponent(Floor);
        FloorActor->SetRootComponent(Floor);
        Floor->SetMobility(EComponentMobility::Movable);
        const FBox FloorSupport = WarehouseFloorSupportBounds.IsValid
            ? WarehouseFloorSupportBounds
            : FBox(FVector(-750.0f, -700.0f, -10.0f), FVector(850.0f, 750.0f, 0.0f));
        Floor->SetBoxExtent(FloorSupport.GetExtent().ComponentMax(FVector(1.0f)), false);
        Floor->SetHiddenInGame(true);
        Floor->SetVisibility(false);
        Floor->SetPhysMaterialOverride(WarehouseMaterial);
        ConfigureCollision(Floor, ECC_WorldStatic);
        Floor->RegisterComponentWithWorld(GetWorld());
        Floor->SetWorldLocation(FloorSupport.GetCenter());
        Floor->SetMobility(EComponentMobility::Static);
        Floor->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        RuntimeChaosStaticColliders.Add(Floor);
    }

    // The rendered safety mat stands several centimetres above the warehouse
    // floor, but the former physics scene contained only the room-floor slab.
    // Install one thin static support matching the actual mesh bounds. This
    // keeps tires and loose props on its visible top instead of embedding into
    // it, while remaining a low step that the wheel suspension can climb.
    if (InferenceRedZoneBounds.IsValid)
    {
        FActorSpawnParameters MatParams;
        MatParams.Name = TEXT("QaiChaosRedSafetyMatActor");
        MatParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AActor* MatActor = GetWorld()->SpawnActor<AActor>(
            AActor::StaticClass(), FTransform::Identity, MatParams);
        if (MatActor)
        {
            RuntimeChaosActors.Add(MatActor);
            UBoxComponent* MatCollider = NewObject<UBoxComponent>(
                MatActor, TEXT("QaiChaosRedSafetyMat"));
            MatActor->AddInstanceComponent(MatCollider);
            MatActor->SetRootComponent(MatCollider);
            MatCollider->SetMobility(EComponentMobility::Movable);
            MatCollider->SetBoxExtent(
                InferenceRedZoneBounds.GetExtent().ComponentMax(FVector(1.0f)),
                false);
            MatCollider->SetHiddenInGame(true);
            MatCollider->SetVisibility(false);
            MatCollider->SetPhysMaterialOverride(WarehouseMaterial);
            ConfigureCollision(MatCollider, ECC_WorldStatic);
            MatCollider->RegisterComponentWithWorld(GetWorld());
            MatCollider->SetWorldLocation(InferenceRedZoneBounds.GetCenter());
            MatCollider->SetMobility(EComponentMobility::Static);
            MatCollider->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
            RuntimeChaosStaticColliders.Add(MatCollider);
            SimulatorLog(FString::Printf(
                TEXT("red_safety_mat_collider center=(%.1f,%.1f,%.1f) extent=(%.1f,%.1f,%.1f) top_z_cm=%.1f friction=warehouse"),
                InferenceRedZoneBounds.GetCenter().X,
                InferenceRedZoneBounds.GetCenter().Y,
                InferenceRedZoneBounds.GetCenter().Z,
                InferenceRedZoneBounds.GetExtent().X,
                InferenceRedZoneBounds.GetExtent().Y,
                InferenceRedZoneBounds.GetExtent().Z,
                InferenceRedZoneBounds.Max.Z));
        }
    }

    const auto DisableVisualCollision = [](USceneComponent* Root)
    {
        if (!Root)
        {
            return;
        }
        TArray<USceneComponent*> Components;
        Root->GetChildrenComponents(true, Components);
        Components.Add(Root);
        for (USceneComponent* Component : Components)
        {
            if (UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Component))
            {
                Primitive->SetSimulatePhysics(false);
                Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                Primitive->SetGenerateOverlapEvents(false);
            }
        }
    };

    for (int32 ForkliftIndex = 0;
         ForkliftIndex < ConveyorTuning::EnabledForkliftCount;
         ++ForkliftIndex)
    {
        FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        USceneComponent* VisualRoot = Forklift.Root.Get();
        if (!VisualRoot)
        {
            continue;
        }
        const FTransform VisualWorld = VisualRoot->GetComponentTransform();
        const FFittedCollisionBox* ChassisShape = Forklift.CollisionBoxes.FindByPredicate(
            [](const FFittedCollisionBox& Shape)
            {
                return Shape.Name == TEXT("lower chassis");
            });
        const FFittedCollisionBox* CarriageShape = Forklift.CollisionBoxes.FindByPredicate(
            [](const FFittedCollisionBox& Shape)
            {
                return Shape.Name == TEXT("mast");
            });
        if (!ChassisShape || !CarriageShape)
        {
            SimulatorLog(FString::Printf(
                TEXT("chaos_forklift_failed forklift=%d reason=missing_fitted_chassis_or_mast"),
                ForkliftIndex + 1));
            continue;
        }

        Forklift.ChaosChassisLocalCenter = ChassisShape->LocalCenter;
        Forklift.ChaosCarriageLocalCenter = CarriageShape->LocalCenter;

        const auto SpawnBody = [this, &ConfigureCollision, ForkliftMaterial](
            const FName ActorName,
            const FName ComponentName,
            const FTransform& Transform,
            const FVector& Extent,
            float MassKg)
        {
            FActorSpawnParameters Params;
            Params.Name = ActorName;
            Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
            AActor* BodyActor = GetWorld()->SpawnActor<AActor>(AActor::StaticClass(), Transform, Params);
            if (!BodyActor)
            {
                return static_cast<UBoxComponent*>(nullptr);
            }
            RuntimeChaosActors.Add(BodyActor);
            UBoxComponent* Body = NewObject<UBoxComponent>(BodyActor, ComponentName);
            BodyActor->AddInstanceComponent(Body);
            BodyActor->SetRootComponent(Body);
            Body->SetMobility(EComponentMobility::Movable);
            Body->SetBoxExtent(Extent, false);
            Body->SetHiddenInGame(true);
            Body->SetVisibility(false);
            Body->SetPhysMaterialOverride(ForkliftMaterial);
            ConfigureCollision(Body, ECC_PhysicsBody);
            Body->RegisterComponentWithWorld(GetWorld());
            Body->SetWorldTransform(Transform);
            Body->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            Body->SetLinearDamping(0.10f);
            Body->SetAngularDamping(0.34f);
            Body->SetMassOverrideInKg(NAME_None, MassKg, true);
            Body->SetUseCCD(true);
            Body->SetMaxDepenetrationVelocity(NAME_None, 80.0f);
            Body->BodyInstance.SetPositionSolverIterationCount(16);
            Body->BodyInstance.SetVelocitySolverIterationCount(8);
            Body->BodyInstance.SetInertiaConditioningEnabled(true);
            return Body;
        };

        const FTransform ChassisWorld(
            VisualWorld.GetRotation(),
            VisualWorld.TransformPositionNoScale(ChassisShape->LocalCenter));
        UBoxComponent* Chassis = SpawnBody(
            FName(*FString::Printf(TEXT("QaiChaosForklift%dChassisActor"), ForkliftIndex + 1)),
            FName(*FString::Printf(TEXT("QaiChaosForklift%dChassis"), ForkliftIndex + 1)),
            ChassisWorld,
            ChassisShape->HalfExtent,
            Forklift.ChassisMassKg);
        const FTransform CarriageWorld(
            VisualWorld.GetRotation(),
            VisualWorld.TransformPositionNoScale(CarriageShape->LocalCenter));
        UBoxComponent* Carriage = SpawnBody(
            FName(*FString::Printf(TEXT("QaiChaosForklift%dCarriageActor"), ForkliftIndex + 1)),
            FName(*FString::Printf(TEXT("QaiChaosForklift%dCarriage"), ForkliftIndex + 1)),
            CarriageWorld,
            CarriageShape->HalfExtent,
            420.0f);
        if (!Chassis || !Carriage)
        {
            continue;
        }
        Forklift.ChaosChassis = Chassis;
        Forklift.ChaosCarriage = Carriage;
        Forklift.InitialChaosChassis = ChassisWorld;
        Forklift.InitialChaosCarriage = CarriageWorld;
        Chassis->SetCenterOfMass(
            Forklift.BaseCenterOfMassLocalCm - Forklift.ChaosChassisLocalCenter);
        Carriage->SetCenterOfMass(FVector(-4.0f, 0.0f, -18.0f));

        DisableVisualCollision(VisualRoot);
        VisualRoot->AttachToComponent(
            Chassis,
            FAttachmentTransformRules(EAttachmentRule::KeepWorld, true));
        if (USceneComponent* LiftVisualRoot = Forklift.LiftAssembly.Get())
        {
            LiftVisualRoot->AttachToComponent(
                Carriage,
                FAttachmentTransformRules(EAttachmentRule::KeepWorld, true));
        }

        const auto AddCompoundBox = [this, &ConfigureCollision, ForkliftMaterial](
            UBoxComponent* Parent,
            const FName Name,
            const FVector& RelativeLocation,
            const FVector& Extent,
            const FRotator& RelativeRotation)
        {
            UBoxComponent* Shape = NewObject<UBoxComponent>(Parent->GetOwner(), Name);
            Parent->GetOwner()->AddInstanceComponent(Shape);
            Shape->SetMobility(EComponentMobility::Movable);
            Shape->SetupAttachment(Parent);
            Shape->SetBoxExtent(Extent, false);
            Shape->SetRelativeLocationAndRotation(RelativeLocation, RelativeRotation);
            Shape->SetHiddenInGame(true);
            Shape->SetVisibility(false);
            Shape->SetPhysMaterialOverride(ForkliftMaterial);
            ConfigureCollision(Shape, ECC_PhysicsBody);
            Shape->RegisterComponentWithWorld(GetWorld());
            Shape->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
            // Attachment alone creates another body instance. Welding makes
            // this an actual compound member whose contacts feed the parent
            // chassis/carriage mass and inertia tensor. The parent is kept
            // kinematic during construction so no body can fall before its
            // constraint and complete aggregate are ready; explicitly allow
            // that construction-time kinematic weld.
            Shape->WeldTo(Parent, NAME_None, true);
            RuntimeChaosForkliftColliders.Add(Shape);
            return Shape;
        };

        Forklift.ChaosCollisionComponents.Reset();
        Forklift.ChaosCollisionComponents.SetNum(Forklift.CollisionBoxes.Num());
        for (int32 ShapeIndex = 0; ShapeIndex < Forklift.CollisionBoxes.Num(); ++ShapeIndex)
        {
            const FFittedCollisionBox& Shape = Forklift.CollisionBoxes[ShapeIndex];
            if (Shape.IsCylinder()
                || Shape.Name == TEXT("lower chassis")
                || Shape.Name == TEXT("mast"))
            {
                continue;
            }
            UBoxComponent* Parent = Shape.bLiftDriven ? Carriage : Chassis;
            const FVector ParentCenter = Shape.bLiftDriven
                ? Forklift.ChaosCarriageLocalCenter
                : Forklift.ChaosChassisLocalCenter;
            FVector RelativeLocation = Shape.LocalCenter - ParentCenter;
            FVector Extent = Shape.GetCollisionHalfExtent();
            FRotator RelativeRotation = FRotator::ZeroRotator;
            if (Shape.IsWedge())
            {
                const float FullThickness = Shape.HalfExtent.Z * 2.0f;
                const float HeightDelta = FMath::Max(
                    0.0f,
                    FullThickness - Shape.WedgeTipThicknessCm);
                const float SlopeDegrees = FMath::RadiansToDegrees(FMath::Atan2(
                    HeightDelta,
                    Shape.HalfExtent.X * 2.0f));
                RelativeRotation.Pitch = Shape.bWedgeTipAtPositiveX
                    ? -SlopeDegrees
                    : SlopeDegrees;
                Extent.Z = FMath::Max(0.55f, Shape.WedgeTipThicknessCm * 0.55f);
                RelativeLocation.Z += 0.5f * (
                    Shape.WedgeTipThicknessCm - FullThickness) * 0.25f;
            }
            UBoxComponent* Compound = AddCompoundBox(
                Parent,
                FName(*FString::Printf(TEXT("QaiChaosForklift%dShape%d"),
                    ForkliftIndex + 1, ShapeIndex + 1)),
                RelativeLocation,
                Extent,
                RelativeRotation);
            Forklift.ChaosCollisionComponents[ShapeIndex] = Compound;
        }

        // Build the constraint against live dynamic particles. Creating a
        // drive while both bodies are kinematic leaves Chaos with kinematic
        // connector handles; later toggling simulation does not reliably
        // recreate the drive and the carriage simply sags through the mast.
        Chassis->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        Carriage->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        Chassis->SetSimulatePhysics(true);
        Carriage->SetSimulatePhysics(true);
        Chassis->SetMassOverrideInKg(NAME_None, Forklift.ChassisMassKg, true);
        Chassis->SetCenterOfMass(
            Forklift.BaseCenterOfMassLocalCm - Forklift.ChaosChassisLocalCenter);
        Carriage->SetMassOverrideInKg(NAME_None, 420.0f, true);
        Carriage->SetCenterOfMass(FVector(-4.0f, 0.0f, -18.0f));

        UPhysicsConstraintComponent* LiftConstraint = NewObject<UPhysicsConstraintComponent>(
            Chassis->GetOwner(),
            FName(*FString::Printf(TEXT("QaiChaosForklift%dLiftConstraint"), ForkliftIndex + 1)));
        Chassis->GetOwner()->AddInstanceComponent(LiftConstraint);
        LiftConstraint->SetMobility(EComponentMobility::Movable);
        LiftConstraint->SetupAttachment(Chassis);
        LiftConstraint->RegisterComponentWithWorld(GetWorld());
        const FVector ConstraintWorldLocation = CarriageWorld.GetLocation();
        LiftConstraint->SetWorldLocationAndRotation(
            ConstraintWorldLocation,
            VisualWorld.GetRotation());
        LiftConstraint->SetConstrainedComponents(Chassis, NAME_None, Carriage, NAME_None);
        LiftConstraint->SetDisableCollision(true);
        LiftConstraint->SetLinearXLimit(LCM_Locked, 0.0f);
        LiftConstraint->SetLinearYLimit(LCM_Locked, 0.0f);
        LiftConstraint->SetLinearZLimit(
            LCM_Limited,
            ConveyorTuning::ForkliftLiftTravelCm);
        LiftConstraint->SetAngularSwing1Limit(ACM_Locked, 0.0f);
        LiftConstraint->SetAngularSwing2Limit(ACM_Locked, 0.0f);
        LiftConstraint->SetAngularTwistLimit(ACM_Locked, 0.0f);
        // Chaos's prismatic drive owns both mast alignment and hydraulic
        // travel. The old free-force actuator could build several centimetres
        // of lateral/angular constraint error under cargo and visually pull
        // the fork carriage away from its rails.
        LiftConstraint->SetLinearPositionDrive(false, false, true);
        LiftConstraint->SetLinearVelocityDrive(false, false, true);
        LiftConstraint->SetLinearDriveParams(620000.0f, 95000.0f, 12000000.0f);
        LiftConstraint->SetLinearPositionTarget(FVector(
            0.0f,
            0.0f,
            -ConveyorTuning::ForkliftInitialLiftCm));
        LiftConstraint->SetLinearVelocityTarget(FVector::ZeroVector);
        LiftConstraint->SetProjectionEnabled(true);
        LiftConstraint->SetProjectionParams(0.10f, 0.25f, 1.0f, 1.0f);
        Chassis->BodyInstance.SetPositionSolverIterationCount(32);
        Chassis->BodyInstance.SetVelocitySolverIterationCount(16);
        Carriage->BodyInstance.SetPositionSolverIterationCount(32);
        Carriage->BodyInstance.SetVelocitySolverIterationCount(16);
        Forklift.ChaosLiftConstraint = LiftConstraint;
        RuntimeChaosConstraints.Add(LiftConstraint);

        // The constraint reference is deliberately authored at the true
        // mechanical minimum. Once that reference exists, move the complete
        // physical carriage (visual lift assembly and compound fork shapes)
        // to the desired 10-cm start height and record that pose for reset.
        const FVector InitialLiftWorldOffset =
            VisualWorld.GetUnitAxis(EAxis::Z) * ConveyorTuning::ForkliftInitialLiftCm;
        Carriage->SetWorldLocationAndRotation(
            CarriageWorld.GetLocation() + InitialLiftWorldOffset,
            CarriageWorld.GetRotation(),
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
        Forklift.InitialChaosCarriage = Carriage->GetComponentTransform();
        Forklift.LiftCm = ConveyorTuning::ForkliftInitialLiftCm;
        Forklift.ActualLiftCm = ConveyorTuning::ForkliftInitialLiftCm;

        Chassis->SetPhysicsLinearVelocity(FVector::ZeroVector);
        Chassis->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
        Carriage->SetPhysicsLinearVelocity(FVector::ZeroVector);
        Carriage->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
        Chassis->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        Carriage->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        Chassis->WakeRigidBody();
        Carriage->WakeRigidBody();

        int32 WeldedShapeCount = 0;
        for (const TWeakObjectPtr<UBoxComponent>& Shape : Forklift.ChaosCollisionComponents)
        {
            WeldedShapeCount += Shape.IsValid() && Shape->IsWelded() ? 1 : 0;
        }
        int32 InitialStaticOverlapCount = 0;
        FCollisionObjectQueryParams StaticObjects;
        StaticObjects.AddObjectTypesToQuery(ECC_WorldStatic);
        FCollisionQueryParams InitialOverlapParams(
            FName(*FString::Printf(TEXT("QaiForkliftInitialOverlap%d"), ForkliftIndex + 1)),
            false,
            Chassis->GetOwner());
        InitialOverlapParams.AddIgnoredActor(Carriage->GetOwner());
        const auto CountInitialOverlaps = [this, &StaticObjects, &InitialOverlapParams,
            &InitialStaticOverlapCount](const UBoxComponent* Shape)
        {
            if (!Shape)
            {
                return;
            }
            TArray<FOverlapResult> Overlaps;
            GetWorld()->OverlapMultiByObjectType(
                Overlaps,
                Shape->GetComponentLocation(),
                Shape->GetComponentQuat(),
                StaticObjects,
                FCollisionShape::MakeBox(Shape->GetUnscaledBoxExtent() * 0.98f),
                InitialOverlapParams);
            InitialStaticOverlapCount += Overlaps.Num();
            for (const FOverlapResult& Overlap : Overlaps)
            {
                if (const UPrimitiveComponent* HitComponent = Overlap.GetComponent())
                {
                    const FVector HitCenter = HitComponent->Bounds.Origin;
                    const FVector HitExtent = HitComponent->Bounds.BoxExtent;
                    SimulatorLog(FString::Printf(
                        TEXT("chaos_forklift_initial_overlap shape=%s actor=%s static=%s center=(%.1f,%.1f,%.1f) extent=(%.1f,%.1f,%.1f)"),
                        *Shape->GetName(),
                        HitComponent->GetOwner()
                            ? *HitComponent->GetOwner()->GetName()
                            : TEXT("none"),
                        *HitComponent->GetName(),
                        HitCenter.X, HitCenter.Y, HitCenter.Z,
                        HitExtent.X, HitExtent.Y, HitExtent.Z));
                }
            }
        };
        CountInitialOverlaps(Chassis);
        for (const TWeakObjectPtr<UBoxComponent>& Shape : Forklift.ChaosCollisionComponents)
        {
            CountInitialOverlaps(Shape.Get());
        }
        SimulatorLog(FString::Printf(
            TEXT("chaos_forklift_ready forklift=%d requested_mass_kg=(%.0f,420) actual_mass_kg=(%.1f,%.1f) chassis=(%.1f,%.1f,%.1f) suspension=raycast4 rear_steer=true driven_axle=front max_acceleration_cm_s2=%.1f max_braking_cm_s2=%.1f lift_constraint=prismatic compound_shapes=%d welded_shapes=%d initial_static_overlaps=%d"),
            ForkliftIndex + 1,
            Forklift.ChassisMassKg,
            Chassis->GetMass(),
            Carriage->GetMass(),
            Chassis->GetComponentLocation().X,
            Chassis->GetComponentLocation().Y,
            Chassis->GetComponentLocation().Z,
            ConveyorTuning::ForkliftMaximumDriveAccelerationCm,
            ConveyorTuning::ForkliftMaximumBrakeDecelerationCm,
            RuntimeChaosForkliftColliders.Num(),
            WeldedShapeCount,
            InitialStaticOverlapCount));
    }

    UPhysicalMaterial* WorkerMaterial = MakePhysicalMaterial(
        TEXT("worker"), 0.72f, 0.56f, 0.0f);
    for (int32 WorkerIndex = 0; WorkerIndex < UE_ARRAY_COUNT(Workers); ++WorkerIndex)
    {
        FWorkerRuntime& Worker = Workers[WorkerIndex];
        USceneComponent* VisualRoot = Worker.Root.Get();
        if (!VisualRoot)
        {
            continue;
        }

        DisableVisualCollision(VisualRoot);
        const FVector VisualLocation = VisualRoot->GetComponentLocation();
        // Authored worker scene roots are not consistently located at their
        // feet. Build the Chaos capsule from the rendered hierarchy's actual
        // lower bound so its bottom and the visible soles share the same plane.
        FBox VisualBounds(EForceInit::ForceInit);
        if (const UPrimitiveComponent* RootPrimitive = Cast<UPrimitiveComponent>(VisualRoot))
        {
            VisualBounds += RootPrimitive->Bounds.GetBox();
        }
        TArray<USceneComponent*> VisualChildren;
        VisualRoot->GetChildrenComponents(true, VisualChildren);
        for (const USceneComponent* Child : VisualChildren)
        {
            if (const UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Child))
            {
                const FBox PrimitiveBounds = Primitive->Bounds.GetBox();
                if (Primitive->IsVisible())
                {
                    VisualBounds += PrimitiveBounds;
                }
            }
        }
        const FVector VisualCenter = VisualBounds.IsValid
            ? VisualBounds.GetCenter()
            : VisualLocation;
        const float VisualFeetZ = VisualBounds.IsValid
            ? VisualBounds.Min.Z
            : VisualLocation.Z;

        // Component bounds imported from USD describe the reference pose and
        // do not reliably follow the walking animation. Calibrate the rendered
        // sole against actual foot/toe bones now, then use those animated bones
        // to keep the visible worker grounded at runtime.
        USkeletalMeshComponent* WorkerSkeletalMesh = nullptr;
        TArray<FName> WorkerFootBones;
        for (USceneComponent* Child : VisualChildren)
        {
            USkeletalMeshComponent* Candidate = Cast<USkeletalMeshComponent>(Child);
            if (!Candidate || !Candidate->IsVisible())
            {
                continue;
            }
            TArray<FName> BoneNames;
            Candidate->GetBoneNames(BoneNames);
            TArray<FName> CandidateFootBones;
            for (const FName BoneName : BoneNames)
            {
                const FString LowerBoneName = BoneName.ToString().ToLower();
                if (LowerBoneName.Contains(TEXT("foot"))
                    || LowerBoneName.Contains(TEXT("ankle"))
                    || LowerBoneName.Contains(TEXT("toe"))
                    || LowerBoneName.Contains(TEXT("ball")))
                {
                    CandidateFootBones.Add(BoneName);
                }
            }
            if (CandidateFootBones.Num() > WorkerFootBones.Num())
            {
                WorkerSkeletalMesh = Candidate;
                WorkerFootBones = MoveTemp(CandidateFootBones);
            }
        }
        float SoleBelowLowestFootBoneCm = 0.0f;
        float InitialLowestFootBoneZ = TNumericLimits<float>::Max();
        if (WorkerSkeletalMesh && WorkerFootBones.Num() > 0)
        {
            for (const FName FootBone : WorkerFootBones)
            {
                InitialLowestFootBoneZ = FMath::Min(
                    InitialLowestFootBoneZ,
                    WorkerSkeletalMesh->GetBoneLocation(FootBone, EBoneSpaces::WorldSpace).Z);
            }
            if (InitialLowestFootBoneZ < TNumericLimits<float>::Max())
            {
                SoleBelowLowestFootBoneCm = FMath::Clamp(
                    InitialLowestFootBoneZ - VisualFeetZ,
                    -10.0f,
                    60.0f);
            }

            // Foot-to-toe vectors reveal the character's actual rendered
            // forward axis regardless of USD scene-root or skeletal import
            // rotations. Average both feet so a single animated stride does
            // not bias the calibration.
            FVector VisualForward = FVector::ZeroVector;
            int32 ForwardSamples = 0;
            const auto AddFootForward = [&VisualForward, &ForwardSamples, WorkerSkeletalMesh](
                const FName FootBone,
                const FName ToeBone)
            {
                if (WorkerSkeletalMesh->GetBoneIndex(FootBone) == INDEX_NONE
                    || WorkerSkeletalMesh->GetBoneIndex(ToeBone) == INDEX_NONE)
                {
                    return;
                }
                FVector Direction =
                    WorkerSkeletalMesh->GetBoneLocation(ToeBone, EBoneSpaces::WorldSpace)
                    - WorkerSkeletalMesh->GetBoneLocation(FootBone, EBoneSpaces::WorldSpace);
                Direction.Z = 0.0f;
                if (Direction.SizeSquared2D() > 1.0f)
                {
                    VisualForward += Direction.GetSafeNormal2D();
                    ++ForwardSamples;
                }
            };
            AddFootForward(TEXT("L_Foot"), TEXT("L_ToeBase"));
            AddFootForward(TEXT("R_Foot"), TEXT("R_ToeBase"));
            if (ForwardSamples > 0 && !VisualForward.IsNearlyZero())
            {
                const float VisualForwardYaw = VisualForward.Rotation().Yaw;
                const float CorrectionYaw = FMath::FindDeltaAngleDegrees(
                    VisualForwardYaw,
                    Worker.InitialHeadingYawDegrees);
                const FQuat CorrectedVisualRootRotation =
                    FQuat(FVector::UpVector, FMath::DegreesToRadians(CorrectionYaw))
                    * VisualRoot->GetComponentQuat();
                const FQuat InitialHeading = FRotator(
                    0.0f,
                    Worker.InitialHeadingYawDegrees,
                    0.0f).Quaternion();
                Worker.HeadingOffset =
                    InitialHeading.Inverse() * CorrectedVisualRootRotation;
                SimulatorLog(FString::Printf(
                    TEXT("worker_heading_calibrated worker=%d source=foot_to_toe samples=%d visual_forward_yaw_deg=%.1f route_forward_yaw_deg=%.1f correction_yaw_deg=%.1f"),
                    WorkerIndex + 1,
                    ForwardSamples,
                    VisualForwardYaw,
                    Worker.InitialHeadingYawDegrees,
                    CorrectionYaw));
            }
        }
        const FTransform CapsuleTransform(
            FQuat::Identity,
            FVector(
                VisualCenter.X,
                VisualCenter.Y,
                VisualFeetZ + ConveyorTuning::WorkerCapsuleHalfHeightCm));
        FActorSpawnParameters WorkerParams;
        WorkerParams.Name = FName(*FString::Printf(
            TEXT("QaiChaosWorker%dActor"), WorkerIndex + 1));
        WorkerParams.SpawnCollisionHandlingOverride =
            ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AActor* WorkerActor = GetWorld()->SpawnActor<AActor>(
            AActor::StaticClass(), CapsuleTransform, WorkerParams);
        if (!WorkerActor)
        {
            continue;
        }
        RuntimeChaosActors.Add(WorkerActor);
        UCapsuleComponent* Capsule = NewObject<UCapsuleComponent>(
            WorkerActor,
            FName(*FString::Printf(TEXT("QaiChaosWorker%d"), WorkerIndex + 1)));
        WorkerActor->AddInstanceComponent(Capsule);
        WorkerActor->SetRootComponent(Capsule);
        Capsule->SetMobility(EComponentMobility::Movable);
        Capsule->SetCapsuleSize(
            ConveyorTuning::WorkerRadiusCm,
            ConveyorTuning::WorkerCapsuleHalfHeightCm,
            false);
        Capsule->SetHiddenInGame(true);
        Capsule->SetVisibility(false);
        Capsule->SetPhysMaterialOverride(WorkerMaterial);
        // Configure the walking plane before registration creates the Chaos
        // body. Applying these flags afterward left the live body unconstrained
        // and it settled on a hidden floor proxy 52 cm below the rendered slab.
        Capsule->BodyInstance.DOFMode = EDOFMode::SixDOF;
        Capsule->BodyInstance.bLockXRotation = true;
        Capsule->BodyInstance.bLockYRotation = true;
        Capsule->BodyInstance.bLockZRotation = true;
        Capsule->BodyInstance.bLockZTranslation = true;
        ConfigureCollision(Capsule, ECC_PhysicsBody);
        Capsule->RegisterComponentWithWorld(GetWorld());
        Capsule->SetWorldTransform(CapsuleTransform);
        Capsule->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        Capsule->SetLinearDamping(2.8f);
        Capsule->SetAngularDamping(10.0f);
        Capsule->SetMassOverrideInKg(
            NAME_None,
            ConveyorTuning::WorkerMassKg,
            true);
        Capsule->SetUseCCD(true);
        Capsule->SetMaxDepenetrationVelocity(NAME_None, 120.0f);
        Capsule->BodyInstance.SetPositionSolverIterationCount(12);
        Capsule->BodyInstance.SetVelocitySolverIterationCount(8);
        // A pedestrian route motor is constrained to the warehouse walking
        // plane; vertical gravity would only preload the DOF joint. Horizontal
        // motion and forklift impulses remain fully dynamic Chaos interactions.
        Capsule->SetEnableGravity(false);
        Capsule->SetSimulatePhysics(true);
        Capsule->SetConstraintMode(EDOFMode::SixDOF);
        Capsule->SetMassOverrideInKg(
            NAME_None,
            ConveyorTuning::WorkerMassKg,
            true);
        Capsule->SetPhysicsLinearVelocity(FVector::ZeroVector);
        Capsule->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);

        USceneComponent* VisualPivot = NewObject<USceneComponent>(
            WorkerActor,
            FName(*FString::Printf(TEXT("QaiWorker%dVisualPivot"), WorkerIndex + 1)));
        WorkerActor->AddInstanceComponent(VisualPivot);
        VisualPivot->SetMobility(EComponentMobility::Movable);
        VisualPivot->SetupAttachment(Capsule);
        VisualPivot->RegisterComponentWithWorld(GetWorld());
        VisualPivot->SetRelativeTransform(FTransform::Identity);
        VisualRoot->AttachToComponent(
            VisualPivot,
            FAttachmentTransformRules(EAttachmentRule::KeepWorld, true));
        Worker.ChaosBody = Capsule;
        Worker.VisualPivot = VisualPivot;
        Worker.SkeletalMesh = WorkerSkeletalMesh;
        Worker.FootBones = MoveTemp(WorkerFootBones);
        Worker.SoleBelowLowestFootBoneCm = SoleBelowLowestFootBoneCm;
        SimulatorLog(FString::Printf(
            TEXT("worker_foot_bones worker=%d names=%s"),
            WorkerIndex + 1,
            *FString::JoinBy(Worker.FootBones, TEXT(","), [](const FName BoneName)
            {
                return BoneName.ToString();
            })));
        Worker.InitialChaosTransform = CapsuleTransform;
        Worker.InitialVisualRelativeTransform = VisualRoot->GetRelativeTransform();
        RuntimeChaosWorkerBodies.Add(Capsule);
        SimulatorLog(FString::Printf(
            TEXT("chaos_worker_ready worker=%d mass_kg=%.1f capsule_radius_cm=%.1f capsule_half_height_cm=%.1f hierarchy_root=(%.1f,%.1f) visual_center=(%.1f,%.1f) visual_feet_z_cm=%.1f capsule_bottom_z_cm=%.1f foot_bones=%d initial_lowest_foot_bone_z_cm=%.1f sole_below_bone_cm=%.1f visual_pivot=body_centered animated_feet_alignment=foot_bones locomotion=force_limited_route_motor forklift_contact=dynamic_push constraints=upright_and_floor_plane"),
            WorkerIndex + 1,
            ConveyorTuning::WorkerMassKg,
            ConveyorTuning::WorkerRadiusCm,
            ConveyorTuning::WorkerCapsuleHalfHeightCm,
            VisualLocation.X,
            VisualLocation.Y,
            VisualCenter.X,
            VisualCenter.Y,
            VisualFeetZ,
            CapsuleTransform.GetLocation().Z - ConveyorTuning::WorkerCapsuleHalfHeightCm,
            Worker.FootBones.Num(),
            InitialLowestFootBoneZ < TNumericLimits<float>::Max()
                ? InitialLowestFootBoneZ
                : VisualFeetZ,
            Worker.SoleBelowLowestFootBoneCm));
    }

    const auto ActivateProp = [this, &MakePhysicalMaterial, &ConfigureCollision](
        UBoxComponent* Body,
        const FPropPhysicsProfile& Profile,
        const FString& Name)
    {
        if (!Body)
        {
            return;
        }
        Body->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        Body->SetMobility(EComponentMobility::Movable);
        ConfigureCollision(Body, ECC_PhysicsBody);
        const bool bConveyorCarton = Name.StartsWith(TEXT("conveyor_carton_"));
        Body->SetPhysMaterialOverride(MakePhysicalMaterial(
            Name,
            Profile.StaticFriction,
            Profile.DynamicFriction,
            Profile.Restitution,
            bConveyorCarton
                ? EFrictionCombineMode::Min
                : EFrictionCombineMode::Average));
        Body->SetLinearDamping(Profile.AirLinearDamping);
        Body->SetAngularDamping(Profile.AirAngularDamping);
        Body->SetMassOverrideInKg(NAME_None, Profile.MassKg, true);
        Body->SetCenterOfMass(Profile.CenterOfMassLocalOffset);
        Body->SetUseCCD(true);
        Body->SetMaxDepenetrationVelocity(NAME_None, 45.0f);
        Body->BodyInstance.SetPositionSolverIterationCount(12);
        Body->BodyInstance.SetVelocitySolverIterationCount(6);
        Body->BodyInstance.SetInertiaConditioningEnabled(true);
        Body->SetEnableGravity(true);
        // A body must have physics collision when its rigid particle is
        // created. Enabling simulation while still NoCollision emits an
        // invalid-simulate warning and can leave a prop without a live Chaos
        // particle until some later state change.
        Body->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        Body->SetSimulatePhysics(true);
        // Reapply properties to the newly-created rigid particle; box volume
        // mass calculation can otherwise replace the authored carton mass.
        Body->SetMassOverrideInKg(NAME_None, Profile.MassKg, true);
        Body->SetCenterOfMass(Profile.CenterOfMassLocalOffset);
        Body->SetLinearDamping(Profile.AirLinearDamping);
        Body->SetAngularDamping(Profile.AirAngularDamping);
        Body->SetPhysicsLinearVelocity(FVector::ZeroVector);
        Body->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
        Body->PutRigidBodyToSleep();
    };

    UPhysicalMaterial* PalletCompoundMaterial = MakePhysicalMaterial(
        TEXT("wood_pallet_compound"), 0.74f, 0.58f, 0.035f);
    const auto BuildPalletCompound = [this, &ConfigureCollision, PalletCompoundMaterial](
        FDynamicBoxRuntime& Runtime,
        UBoxComponent* Parent)
    {
        if (!Parent || !Runtime.bPallet)
        {
            return;
        }

        Runtime.CompoundCollisionComponents.Reset();
        const FVector H = Runtime.HalfExtent;
        const float BoardHalfThickness = FMath::Clamp(H.Z * 0.17f, 0.9f, 1.5f);
        const float RunnerHalfWidth = FMath::Clamp(H.Y * 0.055f, 1.8f, 2.5f);
        const float BlockHalfX = FMath::Clamp(H.X * 0.10f, 4.5f, 7.0f);
        const float BlockHalfY = FMath::Clamp(H.Y * 0.10f, 3.5f, 5.2f);
        const float OuterBlockHalfY = FMath::Min(BlockHalfY, 2.6f);
        const float BlockHalfZ = FMath::Max(2.0f, H.Z - BoardHalfThickness * 2.0f);
        const float OuterRunnerY = H.Y - RunnerHalfWidth - 0.25f;
        const float OuterBlockX = H.X * 0.80f;
        const float RunnerCollisionHalfThickness = BoardHalfThickness
            + ConveyorTuning::PalletContactSkinCm * 0.5f;
        const float RunnerZ = -H.Z + BoardHalfThickness
            - ConveyorTuning::PalletContactSkinCm * 0.5f;
        const float TopBoardZ = H.Z - BoardHalfThickness;

        // The root particle contributes only the central support block. A
        // full-bounds root box would close both fork pockets even if detailed
        // child shapes were added around it.
        Parent->SetBoxExtent(
            FVector(BlockHalfX, BlockHalfY, BlockHalfZ),
            false);

        const auto AddShape = [this, &ConfigureCollision, PalletCompoundMaterial,
            &Runtime, Parent](
            const FString& Suffix,
            const FVector& RelativeLocation,
            const FVector& Extent)
        {
            UBoxComponent* Shape = NewObject<UBoxComponent>(
                Parent->GetOwner(),
                FName(*FString::Printf(
                    TEXT("%s_%s"),
                    *Parent->GetName(),
                    *Suffix)));
            Parent->GetOwner()->AddInstanceComponent(Shape);
            Shape->SetMobility(EComponentMobility::Movable);
            Shape->SetupAttachment(Parent);
            Shape->SetBoxExtent(Extent, false);
            Shape->SetRelativeLocation(RelativeLocation);
            Shape->SetHiddenInGame(true);
            Shape->SetVisibility(false);
            Shape->SetPhysMaterialOverride(PalletCompoundMaterial);
            ConfigureCollision(Shape, ECC_PhysicsBody);
            Shape->RegisterComponentWithWorld(GetWorld());
            Shape->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
            Shape->WeldTo(Parent, NAME_None, true);
            Runtime.CompoundCollisionComponents.Add(Shape);
        };

        // Five independent upper deck boards provide contact across the load
        // surface while retaining the visible gaps between timber slats.
        constexpr int32 TopBoardCount = 5;
        for (int32 BoardIndex = 0; BoardIndex < TopBoardCount; ++BoardIndex)
        {
            const float Alpha = TopBoardCount > 1
                ? static_cast<float>(BoardIndex) / static_cast<float>(TopBoardCount - 1)
                : 0.5f;
            const float BoardY = FMath::Lerp(-H.Y * 0.88f, H.Y * 0.88f, Alpha);
            const float BoardHalfWidth = BoardIndex == 0 || BoardIndex == TopBoardCount - 1
                ? FMath::Clamp(H.Y * 0.11f, 4.0f, 5.5f)
                : FMath::Clamp(H.Y * 0.075f, 2.7f, 4.0f);
            AddShape(
                FString::Printf(TEXT("top_board_%d"), BoardIndex + 1),
                FVector(0.0f, BoardY, TopBoardZ),
                FVector(H.X * 0.99f, BoardHalfWidth, BoardHalfThickness));
        }

        // Three bottom runners and nine spacer blocks form two unobstructed
        // longitudinal pockets. The truck's tines at local Y +/-29 cm fit
        // between the central and outer runners instead of striking a coarse
        // pallet AABB.
        const float RunnerYs[] = {-OuterRunnerY, 0.0f, OuterRunnerY};
        for (int32 RunnerIndex = 0; RunnerIndex < UE_ARRAY_COUNT(RunnerYs); ++RunnerIndex)
        {
            AddShape(
                FString::Printf(TEXT("bottom_runner_%d"), RunnerIndex + 1),
                FVector(0.0f, RunnerYs[RunnerIndex], RunnerZ),
                FVector(H.X * 0.99f, RunnerHalfWidth, RunnerCollisionHalfThickness));
        }
        const float BlockXs[] = {-OuterBlockX, 0.0f, OuterBlockX};
        for (int32 XIndex = 0; XIndex < UE_ARRAY_COUNT(BlockXs); ++XIndex)
        {
            for (int32 YIndex = 0; YIndex < UE_ARRAY_COUNT(RunnerYs); ++YIndex)
            {
                if (XIndex == 1 && YIndex == 1)
                {
                    continue; // supplied by the parent/root particle
                }
                AddShape(
                    FString::Printf(TEXT("block_%d_%d"), XIndex + 1, YIndex + 1),
                    FVector(BlockXs[XIndex], RunnerYs[YIndex], 0.0f),
                    FVector(
                        BlockHalfX,
                        YIndex == 1 ? BlockHalfY : OuterBlockHalfY,
                        BlockHalfZ));
            }
        }
        SimulatorLog(FString::Printf(
            TEXT("chaos_pallet_compound name=%s shapes=%d root=central_block top_boards=%d bottom_runners=3 spacer_blocks=9 fork_pockets=2 pocket_centers_y_cm=+/-%.1f floor_contact_skin_cm=%.1f"),
            *Runtime.Name,
            Runtime.CompoundCollisionComponents.Num() + 1,
            TopBoardCount,
            0.5f * (OuterRunnerY - OuterBlockHalfY + BlockHalfY),
            ConveyorTuning::PalletContactSkinCm));
    };

    // Remove initial authoring penetrations before activation. No runtime
    // support attachment is created: after this placement all loads are free
    // rigid bodies resting on actual Chaos contact manifolds.
    TMap<int32, int32> ShelfSupportTierCounts;
    for (FDynamicBoxRuntime& Body : DynamicBoxes)
    {
        UBoxComponent* Box = Cast<UBoxComponent>(Body.Root.Get());
        if (!Box)
        {
            continue;
        }
        DisableVisualCollision(Box);
        if (Body.bPallet)
        {
            BuildPalletCompound(Body, Box);
        }
        else
        {
            Box->SetBoxExtent(Body.HalfExtent, false);
        }
        FVector Location = Box->GetComponentLocation();
        FQuat Rotation = Box->GetComponentQuat();
        if (Body.bShelfParcel)
        {
            Rotation = FRotator(0.0f, Rotation.Rotator().Yaw, 0.0f).Quaternion();
            float BestShelfTop = -TNumericLimits<float>::Max();
            for (const FCollisionObstacle& Obstacle : CollisionObstacles)
            {
                if (Obstacle.Name != TEXT("shelf plane"))
                {
                    continue;
                }
                const FVector Local = Obstacle.Rotation.UnrotateVector(Location - Obstacle.Center);
                const float SupportTop = Obstacle.Center.Z + Obstacle.HalfExtent.Z;
                if (FMath::Abs(Local.X) <= Obstacle.HalfExtent.X + 2.0f
                    && FMath::Abs(Local.Y) <= Obstacle.HalfExtent.Y + 2.0f
                    // Select the highest deck below this carton's authored
                    // centre. Without this test every X/Y-overlapping carton
                    // selected the top deck and all 24 piled onto level three.
                    && SupportTop <= Location.Z)
                {
                    BestShelfTop = FMath::Max(BestShelfTop, SupportTop);
                }
            }
            if (BestShelfTop > -100000.0f)
            {
                Location.Z = BestShelfTop
                    + ConveyorTuning::ProjectedVerticalHalfExtent(Rotation, Body.HalfExtent)
                    + 0.35f;
                ShelfSupportTierCounts.FindOrAdd(FMath::RoundToInt(BestShelfTop))++;
            }
        }
        else if (Body.InitialSupportedForklift != INDEX_NONE
            && Forklifts[Body.InitialSupportedForklift].ChaosCarriage.IsValid())
        {
            const FForkliftRuntime& Forklift = Forklifts[Body.InitialSupportedForklift];
            float ForkTop = -TNumericLimits<float>::Max();
            for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
            {
                if (!Shape.IsWedge()
                    && Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase))
                {
                    ForkTop = FMath::Max(
                        ForkTop,
                        Forklift.InitialRoot.TransformPositionNoScale(Shape.LocalCenter).Z
                            + Shape.HalfExtent.Z
                            + ConveyorTuning::ForkliftInitialLiftCm);
                }
            }
            if (ForkTop > -100000.0f)
            {
                if (Body.bPallet)
                {
                    // Carry the 120x80 pallet with its long side across the
                    // truck. This uses the pallet's alternate four-way entry:
                    // both tines pass between the centre and outer spacer-block
                    // rows. Seat its rear edge close to the fork heels while
                    // leaving a centimetre of real collision clearance.
                    float TineCenterY = 0.0f;
                    int32 TineCount = 0;
                    float ForkHeelFrontX = -TNumericLimits<float>::Max();
                    for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
                    {
                        if (!Shape.IsWedge()
                            && Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase))
                        {
                            TineCenterY += Shape.LocalCenter.Y;
                            ++TineCount;
                        }
                        if (Shape.Name.Contains(TEXT("fork heel"), ESearchCase::IgnoreCase))
                        {
                            ForkHeelFrontX = FMath::Max(
                                ForkHeelFrontX,
                                Shape.LocalCenter.X + Shape.GetCollisionHalfExtent().X);
                        }
                    }
                    if (TineCount > 0)
                    {
                        TineCenterY /= static_cast<float>(TineCount);
                        FVector PalletLocal = Forklift.InitialRoot
                            .InverseTransformPositionNoScale(Location);
                        PalletLocal.Y = TineCenterY;
                        if (ForkHeelFrontX > -100000.0f)
                        {
                            constexpr float ForkHeelClearanceCm = 1.0f;
                            const float RotatedForwardHalfExtentCm = Body.HalfExtent.Y;
                            PalletLocal.X = ForkHeelFrontX
                                + ForkHeelClearanceCm
                                + RotatedForwardHalfExtentCm;
                        }
                        Location = Forklift.InitialRoot.TransformPositionNoScale(PalletLocal);
                        Rotation = Forklift.InitialRoot.GetRotation()
                            * FQuat(FVector::UpVector, HALF_PI);
                    }

                    const float BlockHalfX = FMath::Clamp(
                        Body.HalfExtent.X * 0.10f,
                        4.5f,
                        7.0f);
                    const float OuterBlockX = Body.HalfExtent.X * 0.80f;
                    const float PocketInnerEdge = BlockHalfX;
                    const float PocketOuterEdge = OuterBlockX - BlockHalfX;
                    float MinimumPocketClearanceCm = TNumericLimits<float>::Max();
                    for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
                    {
                        if (!Shape.IsWedge()
                            && Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase))
                        {
                            const float RelativeTineY = FMath::Abs(
                                Shape.LocalCenter.Y - TineCenterY);
                            const float TineHalfWidth = Shape.GetCollisionHalfExtent().Y;
                            MinimumPocketClearanceCm = FMath::Min(
                                MinimumPocketClearanceCm,
                                FMath::Min(
                                    RelativeTineY - TineHalfWidth - PocketInnerEdge,
                                    PocketOuterEdge - RelativeTineY - TineHalfWidth));
                        }
                    }
                    const bool bTinesFitPockets = TineCount == 2
                        && MinimumPocketClearanceCm >= 0.0f;

                    const float BoardHalfThickness = FMath::Clamp(
                        Body.HalfExtent.Z * 0.17f,
                        0.9f,
                        1.5f);
                    const float UpperDeckUndersideLocalZ =
                        Body.HalfExtent.Z - 2.0f * BoardHalfThickness;
                    constexpr float InitialThreadingClearanceCm = 0.20f;
                    Location.Z = ForkTop
                        - UpperDeckUndersideLocalZ
                        + InitialThreadingClearanceCm;
                    SimulatorLog(FString::Printf(
                        TEXT("initial_pallet_threading name=%s orientation_yaw_relative_deg=90 entry=four_way_side tine_count=%d pallet_center_local=(%.2f,%.2f) fork_heel_front_x_cm=%.2f fork_top_z_cm=%.2f upper_deck_underside_local_z_cm=%.2f vertical_clearance_cm=%.2f pocket_inner_cm=%.2f pocket_outer_cm=%.2f minimum_lateral_clearance_cm=%.2f tines_fit_pockets=%s"),
                        *Body.Name,
                        TineCount,
                        Forklift.InitialRoot.InverseTransformPositionNoScale(Location).X,
                        TineCenterY,
                        ForkHeelFrontX,
                        ForkTop,
                        UpperDeckUndersideLocalZ,
                        InitialThreadingClearanceCm,
                        PocketInnerEdge,
                        PocketOuterEdge,
                        MinimumPocketClearanceCm,
                        bTinesFitPockets ? TEXT("true") : TEXT("false")));
                }
                else
                {
                    Location.Z = ForkTop
                        + ConveyorTuning::ProjectedVerticalHalfExtent(Rotation, Body.HalfExtent)
                        + 0.35f;
                }
            }
        }
        if (Body.SupportBodyIndex != INDEX_NONE
            && DynamicBoxes.IsValidIndex(Body.SupportBodyIndex)
            && DynamicBoxes[Body.SupportBodyIndex].Root.IsValid())
        {
            const FDynamicBoxRuntime& Support = DynamicBoxes[Body.SupportBodyIndex];
            if (Support.bPallet)
            {
                // Keep the starting carton centred on, and yaw-aligned with,
                // the newly side-loaded pallet. It remains an independent
                // Chaos body after activation.
                Location.X = Support.Root->GetComponentLocation().X;
                Location.Y = Support.Root->GetComponentLocation().Y;
                Rotation = Support.Root->GetComponentQuat();
            }
            Location.Z = Support.Root->GetComponentLocation().Z
                + Support.HalfExtent.Z
                + Body.HalfExtent.Z
                + 0.35f;
        }
        Box->SetWorldLocationAndRotation(Location, Rotation, false, nullptr, ETeleportType::TeleportPhysics);
        Body.InitialTransform = Box->GetComponentTransform();
        Body.SupportedForklift = INDEX_NONE;
        Body.InitialSupportedForklift = INDEX_NONE;
        Body.SupportBodyIndex = INDEX_NONE;
        Body.InitialSupportBodyIndex = INDEX_NONE;
        ActivateProp(Box, Body.Physics, Body.Name);
    }
    {
        TArray<int32> TierHeights;
        ShelfSupportTierCounts.GetKeys(TierHeights);
        TierHeights.Sort();
        FString TierSummary;
        int32 TotalShelfCartons = 0;
        for (const int32 Height : TierHeights)
        {
            const int32 TierCount = ShelfSupportTierCounts.FindRef(Height);
            TotalShelfCartons += TierCount;
            TierSummary += FString::Printf(
                TEXT("%s%dcm:%d"),
                TierSummary.IsEmpty() ? TEXT("") : TEXT(","),
                Height,
                TierCount);
        }
        SimulatorLog(FString::Printf(
            TEXT("chaos_shelf_carton_placement tiers=%s total=%d support=authored_deck_below_carton"),
            TierSummary.IsEmpty() ? TEXT("none") : *TierSummary,
            TotalShelfCartons));
    }

    for (int32 ParcelIndex = 0; ParcelIndex < RuntimeChaosConveyorBodies.Num(); ++ParcelIndex)
    {
        UBoxComponent* Box = RuntimeChaosConveyorBodies[ParcelIndex];
        if (!Box || !ParcelPhysicsProfiles.IsValidIndex(ParcelIndex))
        {
            continue;
        }
        DisableVisualCollision(Box);
        FVector Location = Box->GetComponentLocation();
        Location.Z = ConveyorSurfaceZCm
            + (ParcelHalfExtents.IsValidIndex(ParcelIndex)
                ? ParcelHalfExtents[ParcelIndex].Z
                : 15.0f)
            + 0.35f;
        Box->SetWorldLocationAndRotation(
            Location,
            FRotator(0.0f, Box->GetComponentRotation().Yaw, 0.0f),
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
        ParcelInitialTransforms[ParcelIndex] = Box->GetComponentTransform();
        ActivateProp(
            Box,
            ParcelPhysicsProfiles[ParcelIndex],
            FString::Printf(TEXT("conveyor_carton_%d"), ParcelIndex + 1));
    }

    bChaosPhysicsActive = true;
    bChaosConveyorBodiesActivated = true;
    bLoggedChaosConveyorDrive = false;
    ChaosConveyorStartupGraceSeconds = 0.35f;
    ChaosConveyorMotorRampSeconds = 0.0f;
    SimulatorLog(FString::Printf(
        TEXT("physics_backend mode=authoritative_chaos dynamic_props=%d conveyor_props=%d static_shapes=%d roller_shapes=%d forklifts=%d custom_gravity=false transform_support=false cargo_attachment=false"),
        RuntimeChaosBodies.Num(),
        RuntimeChaosConveyorBodies.Num(),
        RuntimeChaosStaticColliders.Num(),
        RuntimeChaosRollerColliders.Num(),
        ConveyorTuning::EnabledForkliftCount));
}

void AQaiConveyorWorld::ConfigureChaosPhysics()
{
    if (bChaosPhysicsActive || !GetWorld())
    {
        return;
    }

    RuntimeChaosMaterials.Reset();
    RuntimeChaosStaticColliders.Reset();
    RuntimeChaosForkliftColliders.Reset();

    // The imported USD collision meshes fed the former software OBB solver.
    // Disable their engine collision before installing fitted Chaos proxies;
    // otherwise identical surfaces produce duplicate contact manifolds and
    // excessive depenetration impulses on lightweight props.
    for (const TWeakObjectPtr<UPrimitiveComponent>& AuthoredCollider : AuthoredCollisionComponents)
    {
        if (UPrimitiveComponent* Primitive = AuthoredCollider.Get())
        {
            Primitive->SetSimulatePhysics(false);
            Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            Primitive->SetGenerateOverlapEvents(false);
        }
    }
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        if (!It->ActorHasTag(TEXT("Qai.DetailedPackingTable")))
        {
            continue;
        }
        TInlineComponentArray<UPrimitiveComponent*> TableVisuals(*It);
        for (UPrimitiveComponent* Primitive : TableVisuals)
        {
            if (Primitive)
            {
                Primitive->SetSimulatePhysics(false);
                Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                Primitive->SetGenerateOverlapEvents(false);
            }
        }
    }

    const auto CreateMaterial = [this](const FPropPhysicsProfile& Profile, const FString& Name)
    {
        UPhysicalMaterial* Material = NewObject<UPhysicalMaterial>(
            this,
            FName(*FString::Printf(TEXT("PM_Chaos_%s_%d"), *Name.Replace(TEXT(" "), TEXT("_")), RuntimeChaosMaterials.Num())));
        Material->Friction = Profile.DynamicFriction;
        Material->StaticFriction = Profile.StaticFriction;
        Material->Restitution = Profile.Restitution;
        Material->bOverrideFrictionCombineMode = true;
        // Use the grippier contacting surface. Min made rubber feet and
        // textured plastic behave like polished metal on the sorting table.
        Material->FrictionCombineMode = EFrictionCombineMode::Max;
        Material->bOverrideRestitutionCombineMode = true;
        Material->RestitutionCombineMode = EFrictionCombineMode::Min;
        Material->SleepLinearVelocityThreshold = Profile.SleepLinearSpeedCm;
        Material->SleepAngularVelocityThreshold = FMath::DegreesToRadians(Profile.SleepAngularSpeedDegrees);
        Material->SleepCounterThreshold = 8;
        RuntimeChaosMaterials.Add(Material);
        return Material;
    };

    const auto DisableVisualCollision = [](USceneComponent* Root, UPrimitiveComponent* Keep)
    {
        if (!Root)
        {
            return;
        }
        TArray<USceneComponent*> Descendants;
        Root->GetChildrenComponents(true, Descendants);
        for (USceneComponent* Child : Descendants)
        {
            if (UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Child); Primitive && Primitive != Keep)
            {
                Primitive->SetSimulatePhysics(false);
                Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                Primitive->SetGenerateOverlapEvents(false);
            }
        }
    };

    for (FDynamicBoxRuntime& Body : DynamicBoxes)
    {
        UBoxComponent* ChaosBody = Cast<UBoxComponent>(Body.Root.Get());
        if (!ChaosBody)
        {
            continue;
        }
        if (Body.bShelfParcel)
        {
            FVector SettledLocation = ChaosBody->GetComponentLocation();
            float BestSupportTop = -TNumericLimits<float>::Max();
            for (const FCollisionObstacle& Obstacle : CollisionObstacles)
            {
                if (Obstacle.Name != TEXT("shelf plane"))
                {
                    continue;
                }
                const FVector Local = Obstacle.Rotation.UnrotateVector(
                    SettledLocation - Obstacle.Center);
                const bool bInsideShelf = FMath::Abs(Local.X) <= Obstacle.HalfExtent.X + 2.0f
                    && FMath::Abs(Local.Y) <= Obstacle.HalfExtent.Y + 2.0f;
                const float SupportTop = Obstacle.Center.Z + Obstacle.HalfExtent.Z;
                if (bInsideShelf
                    && SupportTop <= SettledLocation.Z + 2.0f
                    && SupportTop > BestSupportTop)
                {
                    BestSupportTop = SupportTop;
                }
            }
            if (BestSupportTop > -TNumericLimits<float>::Max() * 0.5f)
            {
                SettledLocation.Z = BestSupportTop + Body.HalfExtent.Z + 0.25f;
                const float Yaw = ChaosBody->GetComponentRotation().Yaw;
                ChaosBody->SetWorldLocationAndRotation(
                    SettledLocation,
                    FRotator(0.0f, Yaw, 0.0f),
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
            }
        }
        DisableVisualCollision(ChaosBody, ChaosBody);
        ChaosBody->SetBoxExtent(Body.HalfExtent, true);
        // Keep the body collisionless until every static support proxy has
        // been registered. Registering shelves/table/conveyor underneath an
        // already-simulating body can create a new penetrating manifold and
        // turn the first depenetration solve into a launch impulse.
        ChaosBody->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        ChaosBody->SetCollisionObjectType(ECC_PhysicsBody);
        ChaosBody->SetCollisionResponseToAllChannels(ECR_Block);
        ChaosBody->SetCollisionResponseToChannel(ECC_Camera, ECR_Ignore);
        ChaosBody->SetGenerateOverlapEvents(false);
        ChaosBody->SetPhysMaterialOverride(CreateMaterial(Body.Physics, Body.Name));
        ChaosBody->SetLinearDamping(Body.Physics.AirLinearDamping);
        ChaosBody->SetAngularDamping(Body.Physics.AirAngularDamping);
        ChaosBody->SetMassOverrideInKg(NAME_None, Body.Physics.MassKg, true);
        ChaosBody->SetCenterOfMass(Body.Physics.CenterOfMassLocalOffset);
        ChaosBody->SetUseCCD(true);
        // Imported USD supports can begin a few millimetres inside their
        // fitted cargo boxes. Bound startup separation so Chaos cannot turn
        // that harmless authored overlap into an explosive impulse.
        ChaosBody->SetMaxDepenetrationVelocity(NAME_None, 8.0f);
        ChaosBody->BodyInstance.SetPositionSolverIterationCount(12);
        ChaosBody->BodyInstance.SetVelocitySolverIterationCount(4);
        ChaosBody->SetEnableGravity(true);
        ChaosBody->SetSimulatePhysics(false);
        ChaosBody->SetPhysicsLinearVelocity(FVector::ZeroVector);
        ChaosBody->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
        ChaosBody->PutRigidBodyToSleep();
        Body.bAwake = false;
    }

    for (int32 ParcelIndex = 0; ParcelIndex < RuntimeChaosConveyorBodies.Num(); ++ParcelIndex)
    {
        UBoxComponent* ChaosBody = RuntimeChaosConveyorBodies[ParcelIndex];
        if (!ChaosBody || !ParcelPhysicsProfiles.IsValidIndex(ParcelIndex))
        {
            continue;
        }
        const FPropPhysicsProfile& Profile = ParcelPhysicsProfiles[ParcelIndex];
        FVector SettledLocation = ChaosBody->GetComponentLocation();
        const float ParcelHalfHeight = ParcelHalfExtents.IsValidIndex(ParcelIndex)
            ? ParcelHalfExtents[ParcelIndex].Z
            : 15.0f;
        SettledLocation.Z = ConveyorSurfaceZCm + ParcelHalfHeight + 0.35f;
        const float Yaw = ChaosBody->GetComponentRotation().Yaw;
        ChaosBody->SetWorldLocationAndRotation(
            SettledLocation,
            FRotator(0.0f, Yaw, 0.0f),
            false,
            nullptr,
            ETeleportType::TeleportPhysics);
        DisableVisualCollision(ChaosBody, ChaosBody);
        ChaosBody->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        ChaosBody->SetCollisionObjectType(ECC_PhysicsBody);
        ChaosBody->SetCollisionResponseToAllChannels(ECR_Block);
        ChaosBody->SetCollisionResponseToChannel(ECC_Camera, ECR_Ignore);
        ChaosBody->SetPhysMaterialOverride(CreateMaterial(Profile, FString::Printf(TEXT("belt_%d"), ParcelIndex + 1)));
        ChaosBody->SetLinearDamping(Profile.AirLinearDamping);
        ChaosBody->SetAngularDamping(Profile.AirAngularDamping);
        ChaosBody->SetMassOverrideInKg(NAME_None, Profile.MassKg, true);
        ChaosBody->SetCenterOfMass(Profile.CenterOfMassLocalOffset);
        ChaosBody->SetUseCCD(true);
        ChaosBody->SetMaxDepenetrationVelocity(NAME_None, 8.0f);
        ChaosBody->BodyInstance.SetPositionSolverIterationCount(12);
        ChaosBody->BodyInstance.SetVelocitySolverIterationCount(4);
        ChaosBody->SetEnableGravity(true);
        ChaosBody->SetSimulatePhysics(false);
        ChaosBody->SetPhysicsLinearVelocity(FVector::ZeroVector);
        ChaosBody->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
        ChaosBody->PutRigidBodyToSleep();
    }

    FPropPhysicsProfile StaticProfile;
    StaticProfile.StaticFriction = 0.82f;
    StaticProfile.DynamicFriction = 0.64f;
    StaticProfile.Restitution = 0.03f;
    UPhysicalMaterial* StaticMaterial = CreateMaterial(StaticProfile, TEXT("warehouse_static"));
    for (const TWeakObjectPtr<UPrimitiveComponent>& AuthoredCollider : AuthoredCollisionComponents)
    {
        if (UPrimitiveComponent* Primitive = AuthoredCollider.Get())
        {
            FString LogicalName = Primitive->GetName();
            int32 LastUnderscore = INDEX_NONE;
            if (LogicalName.FindLastChar(TEXT('_'), LastUnderscore)
                && LogicalName.Mid(LastUnderscore + 1).IsNumeric())
            {
                LogicalName.LeftInline(LastUnderscore, EAllowShrinking::No);
            }
            // Imported roller-module surfaces overlap at their joins and
            // produce multiple Chaos manifolds. The smooth analytic support
            // boxes installed below follow the same belt centreline without
            // launching cartons from those seams.
            if (LogicalName == TEXT("Surface")
                || LogicalName.StartsWith(TEXT("ShelfPlane"))
                || LogicalName == TEXT("PackingTable"))
            {
                Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                continue;
            }
            Primitive->SetMobility(EComponentMobility::Static);
            Primitive->SetSimulatePhysics(false);
            Primitive->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
            Primitive->SetCollisionObjectType(ECC_WorldStatic);
            Primitive->SetCollisionResponseToAllChannels(ECR_Block);
            Primitive->SetCollisionResponseToChannel(ECC_Camera, ECR_Ignore);
            Primitive->SetGenerateOverlapEvents(false);
            Primitive->SetPhysMaterialOverride(StaticMaterial);
        }
    }
    for (int32 Index = 0; Index < CollisionObstacles.Num(); ++Index)
    {
        const FCollisionObstacle& Obstacle = CollisionObstacles[Index];
        // The conveyor uses tangent-aligned smooth support boxes because the
        // imported module surfaces overlap at segment joins. Signal poles
        // likewise have no suitable authored physical component.
        if (Obstacle.Name != TEXT("stack light")
            && Obstacle.Name != TEXT("conveyor")
            && Obstacle.Name != TEXT("shelf plane")
            && Obstacle.Name != TEXT("packing table")
            && Obstacle.Name != TEXT("invisible perimeter"))
        {
            continue;
        }
        UBoxComponent* Collider = NewObject<UBoxComponent>(
            this,
            FName(*FString::Printf(TEXT("QaiChaosStatic_%d"), Index + 1)));
        AddInstanceComponent(Collider);
        Collider->SetMobility(EComponentMobility::Movable);
        Collider->RegisterComponentWithWorld(GetWorld());
        Collider->SetWorldLocationAndRotation(Obstacle.Center, Obstacle.Rotation);
        Collider->SetBoxExtent(Obstacle.HalfExtent, false);
        Collider->SetHiddenInGame(true);
        Collider->SetVisibility(false);
        Collider->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        Collider->SetCollisionObjectType(ECC_WorldStatic);
        Collider->SetCollisionResponseToAllChannels(ECR_Block);
        Collider->SetCollisionResponseToChannel(ECC_Camera, ECR_Ignore);
        Collider->SetPhysMaterialOverride(StaticMaterial);
        RuntimeChaosStaticColliders.Add(Collider);
    }

    UBoxComponent* FloorCollider = NewObject<UBoxComponent>(this, TEXT("QaiChaosRoomFloor"));
    AddInstanceComponent(FloorCollider);
    FloorCollider->SetMobility(EComponentMobility::Movable);
    FloorCollider->RegisterComponentWithWorld(GetWorld());
    const FBox FloorSupport = WarehouseFloorSupportBounds.IsValid
        ? WarehouseFloorSupportBounds
        : FBox(FVector(-750.0f, -700.0f, -10.0f), FVector(850.0f, 750.0f, 0.0f));
    FloorCollider->SetWorldLocation(FloorSupport.GetCenter());
    FloorCollider->SetBoxExtent(FloorSupport.GetExtent().ComponentMax(FVector(1.0f)), false);
    FloorCollider->SetHiddenInGame(true);
    FloorCollider->SetVisibility(false);
    FloorCollider->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    FloorCollider->SetCollisionObjectType(ECC_WorldStatic);
    FloorCollider->SetCollisionResponseToAllChannels(ECR_Block);
    FloorCollider->SetCollisionResponseToChannel(ECC_Camera, ECR_Ignore);
    FloorCollider->SetPhysMaterialOverride(StaticMaterial);
    RuntimeChaosStaticColliders.Add(FloorCollider);

    for (int32 ForkliftIndex = 0; ForkliftIndex < ConveyorTuning::EnabledForkliftCount; ++ForkliftIndex)
    {
        FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        USceneComponent* ForkliftRoot = Forklift.Root.Get();
        if (!ForkliftRoot)
        {
            continue;
        }
        TArray<USceneComponent*> ForkliftVisuals;
        ForkliftRoot->GetChildrenComponents(true, ForkliftVisuals);
        for (USceneComponent* Visual : ForkliftVisuals)
        {
            if (UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Visual))
            {
                Primitive->SetSimulatePhysics(false);
                Primitive->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                Primitive->SetGenerateOverlapEvents(false);
            }
        }
        Forklift.ChaosCollisionComponents.Reset();
        for (int32 ShapeIndex = 0; ShapeIndex < Forklift.CollisionBoxes.Num(); ++ShapeIndex)
        {
            const FFittedCollisionBox& Shape = Forklift.CollisionBoxes[ShapeIndex];
            const FName ColliderName(*FString::Printf(
                TEXT("QaiChaosForklift%d_%d"), ForkliftIndex + 1, ShapeIndex + 1));
            const FTransform ForkliftTransform = ForkliftRoot->GetComponentTransform();
            const FTransform ColliderTransform(
                ForkliftTransform.GetRotation(),
                ForkliftTransform.TransformPositionNoScale(Shape.LocalCenter));
            FActorSpawnParameters SpawnParameters;
            SpawnParameters.Name = ColliderName;
            SpawnParameters.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
            AActor* ColliderActor = GetWorld()->SpawnActor<AActor>(
                AActor::StaticClass(), ColliderTransform, SpawnParameters);
            if (!ColliderActor)
            {
                continue;
            }
            RuntimeChaosActors.Add(ColliderActor);
            UBoxComponent* Collider = NewObject<UBoxComponent>(ColliderActor, ColliderName);
            ColliderActor->AddInstanceComponent(Collider);
            ColliderActor->SetRootComponent(Collider);
            Collider->SetMobility(EComponentMobility::Movable);
            Collider->RegisterComponentWithWorld(GetWorld());
            Collider->SetWorldTransform(ColliderTransform);
            FVector ChaosExtent = Shape.GetCollisionHalfExtent();
            if (Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase))
            {
                // Chaos contact skin for the thin fork blades. The rendered
                // tines are only about 7 cm thick and can advance by several
                // millimetres per substep. A modest skin establishes a stable
                // manifold before a resting carton/pallet can be crossed by
                // the moving kinematic surface. It does not bridge the gap
                // between the two independent tines.
                ChaosExtent.Y += 1.0f;
                ChaosExtent.Z += 1.5f;
            }
            Collider->SetBoxExtent(ChaosExtent, false);
            Collider->SetHiddenInGame(true);
            Collider->SetVisibility(false);
            Collider->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
            // Contact transfer is handled by the bounded Chaos fork-contact
            // solve below. Keeping an additional set of independently
            // simulated 5-tonne proxy blocks here caused initial pallet and
            // shelf penetrations to wake the entire scene explosively.
            Collider->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            Collider->SetPhysMaterialOverride(StaticMaterial);
            // Runtime-created non-simulating shapes updated their query pose
            // but did not produce solver contacts on this path. Make each
            // compound part a heavy, gravity-free Chaos body and servo it by
            // velocity. This preserves the manifold and transfers real force
            // to cargo without attaching cargo to the fork.
            Collider->SetEnableGravity(false);
            Collider->SetSimulatePhysics(false);
            RuntimeChaosForkliftColliders.Add(Collider);
            Forklift.ChaosCollisionComponents.Add(Collider);
        }
    }

    // Activate movable bodies only after the complete static scene exists.
    // Construct the Chaos body while collisionless, clear any inherited
    // velocity, sleep it at its authored resting pose, and enable contacts as
    // the final operation. This avoids the startup parcel/EVK explosion that
    // occurred when support colliders were registered beneath live bodies.
    const auto ActivateSettledBody = [](UBoxComponent* Body)
    {
        if (!Body)
        {
            return;
        }
        Body->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        Body->SetSimulatePhysics(true);
        Body->SetPhysicsLinearVelocity(FVector::ZeroVector);
        Body->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
        Body->PutRigidBodyToSleep();
        Body->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    };
    for (UBoxComponent* Body : RuntimeChaosBodies)
    {
        ActivateSettledBody(Body);
    }
    // Conveyor parcels are activated after a short render/physics warm-up in
    // SimulateChaosConveyor. Keeping them collisionless while the first scene
    // frames and broadphase settle prevents the initial belt explosion.
    bChaosConveyorBodiesActivated = false;

    bChaosPhysicsActive = true;
    ChaosConveyorStartupGraceSeconds = 0.75f;
    ChaosConveyorMotorRampSeconds = 0.0f;
    SimulatorLog(FString::Printf(
        TEXT("chaos_physics_ready dynamic_bodies=%d conveyor_bodies=%d static_colliders=%d forklift_shapes=%d activation=static_first_settled ccd=true solver_iterations=12/4 substep_hz=120 custom_integrator=false"),
        RuntimeChaosBodies.Num(),
        RuntimeChaosConveyorBodies.Num(),
        RuntimeChaosStaticColliders.Num(),
        RuntimeChaosForkliftColliders.Num()));
}

void AQaiConveyorWorld::SimulateChaosForklifts(float DeltaSeconds)
{
    if (!bChaosPhysicsActive || !GetWorld() || DeltaSeconds <= UE_SMALL_NUMBER)
    {
        return;
    }

    const float SafeDeltaSeconds = FMath::Min(DeltaSeconds, 1.0f / 20.0f);
    for (int32 ForkliftIndex = 0;
         ForkliftIndex < ConveyorTuning::EnabledForkliftCount;
         ++ForkliftIndex)
    {
        FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
        UBoxComponent* Chassis = Forklift.ChaosChassis.Get();
        UBoxComponent* Carriage = Forklift.ChaosCarriage.Get();
        if (!Chassis || !Carriage || !Chassis->IsSimulatingPhysics())
        {
            continue;
        }

        const bool bControlled = ForkliftIndex == ActiveForklift;
        const float Throttle = bControlled ? CommandThrottle : 0.0f;
        const float RequestedSteer = bControlled ? CommandSteer : 0.0f;
        const float LiftInput = bControlled ? CommandLift : 0.0f;
        const bool bBrake = bControlled ? bCommandBrake : true;

        // A physical steering rack cannot jump from centre to full lock. This
        // rate limit applies equally to keyboard and controller requests.
        Forklift.SteeringInput = FMath::FInterpConstantTo(
            Forklift.SteeringInput,
            RequestedSteer,
            SafeDeltaSeconds,
            1.85f);
        const float RearSteerRadians = FMath::DegreesToRadians(
            -Forklift.SteeringInput * ConveyorTuning::ForkliftMaximumSteerDegrees);

        const FTransform ChassisTransform = Chassis->GetComponentTransform();
        const FVector ChassisUp = ChassisTransform.GetUnitAxis(EAxis::Z);
        const FVector ChassisForward = ChassisTransform.GetUnitAxis(EAxis::X);
        const FVector ChassisVelocity = Chassis->GetPhysicsLinearVelocity();

        // Supplement the hard prismatic constraint with the finite stiffness
        // of the mast's opposed guide rollers. Only error perpendicular to the
        // lift axis is corrected: vertical travel and cargo reaction remain
        // fully physical, and the opposite force is applied to the chassis.
        const FVector CarriageBaseLocal = Forklift.ChaosCarriageLocalCenter
            - Forklift.ChaosChassisLocalCenter;
        const FVector CarriageLocal = ChassisTransform.InverseTransformPositionNoScale(
            Carriage->GetComponentLocation());
        const FVector RailErrorLocal(
            CarriageLocal.X - CarriageBaseLocal.X,
            CarriageLocal.Y - CarriageBaseLocal.Y,
            0.0f);
        const FVector RailErrorWorld = ChassisTransform.TransformVectorNoScale(
            RailErrorLocal);
        const FVector ChassisVelocityAtCarriage = Chassis->GetPhysicsLinearVelocityAtPoint(
            Carriage->GetComponentLocation());
        const FVector RelativeCarriageVelocity =
            Carriage->GetPhysicsLinearVelocity() - ChassisVelocityAtCarriage;
        const FVector RailRelativeVelocity = RelativeCarriageVelocity
            - ChassisUp * FVector::DotProduct(RelativeCarriageVelocity, ChassisUp);
        const FVector RailGuideForce = (
            -RailErrorWorld * ConveyorTuning::ForkliftLiftRailStiffness
            -RailRelativeVelocity * ConveyorTuning::ForkliftLiftRailDamping)
            .GetClampedToMaxSize(ConveyorTuning::ForkliftLiftRailMaximumForce);
        if (!RailGuideForce.IsNearlyZero(1.0f))
        {
            Carriage->AddForce(RailGuideForce);
            Chassis->AddForceAtLocation(
                -RailGuideForce,
                Carriage->GetComponentLocation());
        }
        const float LongitudinalSpeed = FVector::DotProduct(ChassisVelocity, ChassisForward);
        const float TargetSpeed = Throttle >= 0.0f
            ? Throttle * ConveyorTuning::ForkliftMaximumSpeedCm
            : Throttle * ConveyorTuning::ForkliftMaximumReverseSpeedCm;

        FCollisionQueryParams QueryParams(
            FName(*FString::Printf(TEXT("QaiForkliftSuspension%d"), ForkliftIndex + 1)),
            false,
            Chassis->GetOwner());
        QueryParams.AddIgnoredActor(Carriage->GetOwner());
        FCollisionObjectQueryParams ObjectTypes;
        ObjectTypes.AddObjectTypesToQuery(ECC_WorldStatic);
        ObjectTypes.AddObjectTypesToQuery(ECC_PhysicsBody);

        int32 WheelIndex = 0;
        float SupportedNormalForce = 0.0f;
        FVector TotalPlanarTireForce = FVector::ZeroVector;
        for (const FFittedCollisionBox& Wheel : Forklift.CollisionBoxes)
        {
            if (!Wheel.IsCylinder() || WheelIndex >= 4)
            {
                continue;
            }
            const FVector WheelLocalInChassis = Wheel.LocalCenter
                - Forklift.ChaosChassisLocalCenter;
            const FVector NominalHubWorld = ChassisTransform.TransformPositionNoScale(
                WheelLocalInChassis);
            const bool bRearWheel = WheelIndex >= 2;
            const float SuspensionRestCm = bRearWheel
                ? ConveyorTuning::ForkliftRearSuspensionRestCm
                : ConveyorTuning::ForkliftFrontSuspensionRestCm;
            const float SuspensionDroopCm = bRearWheel
                ? ConveyorTuning::ForkliftRearSuspensionDroopCm
                : ConveyorTuning::ForkliftFrontSuspensionDroopCm;
            const float SuspensionMaximumCompressionCm = bRearWheel
                ? ConveyorTuning::ForkliftRearSuspensionMaximumCompressionCm
                : ConveyorTuning::ForkliftFrontSuspensionMaximumCompressionCm;
            const float SuspensionStiffness = bRearWheel
                ? ConveyorTuning::ForkliftRearSuspensionStiffness
                : ConveyorTuning::ForkliftFrontSuspensionStiffness;
            const float SuspensionDamping = bRearWheel
                ? ConveyorTuning::ForkliftRearSuspensionDamping
                : ConveyorTuning::ForkliftFrontSuspensionDamping;
            const FVector SuspensionMount = NominalHubWorld
                + ChassisUp * (
                    SuspensionRestCm
                    - ConveyorTuning::ForkliftSuspensionPreloadCompressionCm);
            const float MaximumTraceLength = Wheel.Radius
                + SuspensionRestCm
                + SuspensionDroopCm;
            const FVector TraceEnd = SuspensionMount - ChassisUp * MaximumTraceLength;
            FHitResult Hit;
            const bool bHit = GetWorld()->LineTraceSingleByObjectType(
                Hit,
                SuspensionMount,
                TraceEnd,
                ObjectTypes,
                QueryParams);
            Forklift.WheelNormalLoads[WheelIndex] = 0.0f;
            FVector ResolvedHubWorld = SuspensionMount - ChassisUp * (
                SuspensionRestCm + SuspensionDroopCm);
            if (bHit && Hit.bBlockingHit)
            {
                const float SpringLength = FMath::Max(0.0f, Hit.Distance - Wheel.Radius);
                const float Compression = FMath::Clamp(
                    SuspensionRestCm - SpringLength,
                    0.0f,
                    SuspensionMaximumCompressionCm);
                ResolvedHubWorld = Hit.ImpactPoint + ChassisUp * Wheel.Radius;
                const FVector PointVelocity = Chassis->GetPhysicsLinearVelocityAtPoint(
                    Hit.ImpactPoint);
                const float CompressionVelocity = -FVector::DotProduct(PointVelocity, ChassisUp);
                const float NormalForce = FMath::Clamp(
                    Compression * SuspensionStiffness
                        + CompressionVelocity * SuspensionDamping,
                    0.0f,
                    Forklift.ChassisMassKg
                        * ConveyorTuning::GravityCmPerSecondSquared * 0.72f);
                Forklift.SuspensionCompressionCm[WheelIndex] = Compression;
                Forklift.WheelNormalLoads[WheelIndex] = NormalForce;
                SupportedNormalForce += NormalForce;

                const FVector SuspensionForce = ChassisUp * NormalForce;
                Chassis->AddForceAtLocation(SuspensionForce, Hit.ImpactPoint);
                if (UPrimitiveComponent* SupportBody = Hit.GetComponent();
                    SupportBody && SupportBody->IsSimulatingPhysics())
                {
                    SupportBody->AddForceAtLocation(-SuspensionForce, Hit.ImpactPoint);
                    SupportBody->WakeRigidBody();
                }

                const FQuat SteeringRotation(ChassisUp, bRearWheel ? RearSteerRadians : 0.0f);
                const FVector TireForward = SteeringRotation.RotateVector(ChassisForward).GetSafeNormal();
                const FVector TireRight = FVector::CrossProduct(ChassisUp, TireForward).GetSafeNormal();
                const float TireLongitudinalSpeed = FVector::DotProduct(PointVelocity, TireForward);
                const float TireLateralSpeed = FVector::DotProduct(PointVelocity, TireRight);
                const float FrictionLimit = NormalForce * ConveyorTuning::ForkliftTireFriction;

                float LongitudinalForce = 0.0f;
                if (bBrake)
                {
                    const float MaximumBrakeForcePerWheel =
                        Chassis->GetMass()
                        * ConveyorTuning::ForkliftMaximumBrakeDecelerationCm
                        / 4.0f;
                    LongitudinalForce = FMath::Clamp(
                        -TireLongitudinalSpeed
                            * ConveyorTuning::ForkliftTireLongitudinalStiffness * 2.0f,
                        -MaximumBrakeForcePerWheel,
                        MaximumBrakeForcePerWheel);
                }
                else if (!bRearWheel && FMath::Abs(Throttle) > 0.01f)
                {
                    const float MaximumDriveForcePerWheel =
                        Chassis->GetMass()
                        * ConveyorTuning::ForkliftMaximumDriveAccelerationCm
                        * FMath::Abs(Throttle)
                        / 2.0f;
                    LongitudinalForce = FMath::Clamp(
                        (TargetSpeed - TireLongitudinalSpeed)
                            * ConveyorTuning::ForkliftTireLongitudinalStiffness,
                        -MaximumDriveForcePerWheel,
                        MaximumDriveForcePerWheel);
                }
                else if (FMath::Abs(TireLongitudinalSpeed) > 0.2f)
                {
                    LongitudinalForce = -FMath::Sign(TireLongitudinalSpeed)
                        * NormalForce * ConveyorTuning::ForkliftRollingResistance;
                }
                const float LateralForce = FMath::Clamp(
                    -TireLateralSpeed * ConveyorTuning::ForkliftTireLateralStiffness,
                    -FrictionLimit,
                    FrictionLimit);
                const FVector PlanarTireForce = (
                    TireForward * LongitudinalForce + TireRight * LateralForce)
                    .GetClampedToMaxSize(FrictionLimit);
                TotalPlanarTireForce += PlanarTireForce;
                Chassis->AddForceAtLocation(PlanarTireForce, Hit.ImpactPoint);
                if (UPrimitiveComponent* SupportBody = Hit.GetComponent();
                    SupportBody && SupportBody->IsSimulatingPhysics())
                {
                    SupportBody->AddForceAtLocation(-PlanarTireForce, Hit.ImpactPoint);
                }
            }
            else
            {
                Forklift.SuspensionCompressionCm[WheelIndex] = FMath::FInterpTo(
                    Forklift.SuspensionCompressionCm[WheelIndex],
                    0.0f,
                    SafeDeltaSeconds,
                    8.0f);
            }
            if (USceneComponent* WheelPivot = Forklift.WheelPivots[WheelIndex].Get())
            {
                // The imported wheel used to remain rigidly attached to the
                // body while the raycast tire moved invisibly underneath it.
                // Resolve both presentation and F8 bounds to the same hub.
                WheelPivot->SetWorldLocation(
                    ResolvedHubWorld,
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
            }
            ++WheelIndex;
        }

        // Passive driveline drag supplies the static-like tire resistance that
        // raycast wheels otherwise lack. This is force-limited: throttle,
        // braking, a forklift impact, or a sufficiently large external load
        // still moves the truck normally.
        if (SupportedNormalForce > 0.0f && FMath::Abs(Throttle) <= 0.01f && !bBrake)
        {
            FVector PlanarVelocity = ChassisVelocity;
            PlanarVelocity.Z = 0.0f;
            if (!PlanarVelocity.IsNearlyZero(0.03f))
            {
                const FVector IdleDragForce = (
                    -PlanarVelocity.GetSafeNormal()
                        * SupportedNormalForce * ConveyorTuning::ForkliftIdleDrivelineDrag
                    -PlanarVelocity
                        * Chassis->GetMass()
                        * ConveyorTuning::ForkliftIdleVelocityDampingPerSecond)
                    .GetClampedToMaxSize(
                        SupportedNormalForce
                        * ConveyorTuning::ForkliftIdleMaximumDrag);
                Chassis->AddForce(
                    IdleDragForce,
                    NAME_None,
                    false);
            }
        }

        if (bResolutionDataset && ForkliftIndex == 0)
        {
            const int32 TelemetrySecond = FMath::FloorToInt(ResolutionDatasetElapsed);
            if (TelemetrySecond != ResolutionDatasetLastForceTelemetrySecond)
            {
                ResolutionDatasetLastForceTelemetrySecond = TelemetrySecond;
                SimulatorLog(FString::Printf(
                    TEXT("resolution_dataset tire_force elapsed=%.2f throttle=%.3f forward=(%.3f,%.3f,%.3f) planar_force=(%.0f,%.0f,%.0f) magnitude=%.0f supported_force=%.0f mass=%.1f"),
                    ResolutionDatasetElapsed,
                    Throttle,
                    ChassisForward.X,
                    ChassisForward.Y,
                    ChassisForward.Z,
                    TotalPlanarTireForce.X,
                    TotalPlanarTireForce.Y,
                    TotalPlanarTireForce.Z,
                    TotalPlanarTireForce.Size(),
                    SupportedNormalForce,
                    Chassis->GetMass()));
            }
        }

        const float PreviousActualLift = Forklift.ActualLiftCm;
        const float PreviousLiftTarget = Forklift.LiftCm;
        Forklift.LiftCm = FMath::Clamp(
            Forklift.LiftCm + LiftInput * ConveyorTuning::ForkliftLiftSpeedCm * SafeDeltaSeconds,
            0.0f,
            ConveyorTuning::ForkliftMaximumCommandedLiftCm);
        // Derive velocity from the clamped target displacement, not directly
        // from the held key. At either end stop this becomes exactly zero, so
        // holding E/Q cannot continuously inject force into the chassis.
        const float ResolvedLiftVelocityCmPerSecond =
            (Forklift.LiftCm - PreviousLiftTarget) / SafeDeltaSeconds;
        if (UPhysicsConstraintComponent* LiftConstraint = Forklift.ChaosLiftConstraint.Get())
        {
            // The constrained carriage is body two; Chaos's relative Z drive
            // uses the opposite sign from the forklift-local visual lift axis.
            // A positive target previously drove the carriage downward while
            // F8 correctly drew the intended upward fork position.
            LiftConstraint->SetLinearPositionTarget(FVector(0.0f, 0.0f, -Forklift.LiftCm));
            LiftConstraint->SetLinearVelocityTarget(FVector(
                0.0f,
                0.0f,
                -ResolvedLiftVelocityCmPerSecond));
        }
        const bool bHoldingUpperStop =
            LiftInput > 0.01f
            && Forklift.LiftCm >= ConveyorTuning::ForkliftMaximumCommandedLiftCm - 0.01f;
        if (bHoldingUpperStop && !Forklift.bLiftUpperStopLatched)
        {
            Forklift.bLiftUpperStopLatched = true;
            SimulatorLog(FString::Printf(
                TEXT("forklift_lift_upper_stop forklift=%d target_cm=%.1f actual_cm=%.1f velocity_target_cm_s=0.0"),
                ForkliftIndex + 1,
                Forklift.LiftCm,
                Forklift.ActualLiftCm));
        }
        else if (Forklift.LiftCm < ConveyorTuning::ForkliftMaximumCommandedLiftCm - 2.0f)
        {
            Forklift.bLiftUpperStopLatched = false;
        }
        const FVector ExpectedLoweredCarriageWorld =
            ChassisTransform.TransformPositionNoScale(CarriageBaseLocal);
        Forklift.ActualLiftCm = FMath::Clamp(
            FVector::DotProduct(
                Carriage->GetComponentLocation() - ExpectedLoweredCarriageWorld,
                ChassisUp),
            0.0f,
            ConveyorTuning::ForkliftLiftTravelCm);
        Forklift.LiftDeltaCm = Forklift.ActualLiftCm - PreviousActualLift;
        if (FMath::Abs(LiftInput) > 0.01f)
        {
            Carriage->WakeRigidBody();
            Chassis->WakeRigidBody();
        }

        Forklift.SpeedCmPerSecond = LongitudinalSpeed;
        Forklift.SurfaceLinearVelocityCmPerSecond = ChassisVelocity;
        Forklift.SurfaceYawVelocityDegreesPerSecond =
            Chassis->GetPhysicsAngularVelocityInDegrees().Z;
        Forklift.BodyPitchDegrees = Chassis->GetComponentRotation().Pitch;
        Forklift.BodyRollDegrees = Chassis->GetComponentRotation().Roll;
        Forklift.MaximumAbsoluteTipDegrees = FMath::Max(
            Forklift.MaximumAbsoluteTipDegrees,
            FMath::Max(FMath::Abs(Forklift.BodyPitchDegrees), FMath::Abs(Forklift.BodyRollDegrees)));
        Forklift.WheelAngleDegrees = FMath::Fmod(
            Forklift.WheelAngleDegrees
                + FMath::RadiansToDegrees(
                    LongitudinalSpeed * SafeDeltaSeconds
                    / ConveyorTuning::ForkliftWheelRadiusCm),
            360.0f);
        for (int32 VisualWheelIndex = 0; VisualWheelIndex < 4; ++VisualWheelIndex)
        {
            if (USceneComponent* WheelPivot = Forklift.WheelPivots[VisualWheelIndex].Get())
            {
                const float VisualSteer = VisualWheelIndex >= 2 ? RearSteerRadians : 0.0f;
                const FQuat Steering(FVector::UpVector, VisualSteer);
                const FQuat Rolling(
                    FVector::RightVector,
                    FMath::DegreesToRadians(Forklift.WheelAngleDegrees));
                WheelPivot->SetRelativeRotation(
                    Forklift.WheelPivotInitialRelative[VisualWheelIndex] * Steering * Rolling);
            }
        }
    }
}

void AQaiConveyorWorld::SimulateChaosConveyor(float StepSeconds)
{
    if (!bChaosPhysicsActive)
    {
        return;
    }
    if (ChaosConveyorStartupGraceSeconds > 0.0f)
    {
        ChaosConveyorStartupGraceSeconds = FMath::Max(
            0.0f,
            ChaosConveyorStartupGraceSeconds - StepSeconds);
        if (ChaosConveyorStartupGraceSeconds > 0.0f)
        {
            return;
        }
    }
    if (!bChaosConveyorBodiesActivated)
    {
        for (int32 ParcelIndex = 0; ParcelIndex < RuntimeChaosConveyorBodies.Num(); ++ParcelIndex)
        {
            UBoxComponent* Body = RuntimeChaosConveyorBodies[ParcelIndex];
            if (!Body)
            {
                continue;
            }
            FVector Location = Body->GetComponentLocation();
            const float HalfHeight = ParcelHalfExtents.IsValidIndex(ParcelIndex)
                ? ParcelHalfExtents[ParcelIndex].Z
                : 15.0f;
            Location.Z = ConveyorSurfaceZCm + HalfHeight + 1.0f;
            Body->SetCollisionEnabled(ECollisionEnabled::NoCollision);
            Body->SetSimulatePhysics(false);
            Body->SetWorldLocation(Location, false, nullptr, ETeleportType::TeleportPhysics);
            Body->SetSimulatePhysics(true);
            // Creating the Chaos body can recalculate box mass from its volume.
            // Reapply the authored carton properties after body creation so
            // roller force, inertia and contact response all use the same mass.
            if (ParcelPhysicsProfiles.IsValidIndex(ParcelIndex))
            {
                const FPropPhysicsProfile& Profile = ParcelPhysicsProfiles[ParcelIndex];
                Body->SetMassOverrideInKg(NAME_None, Profile.MassKg, true);
                Body->SetCenterOfMass(Profile.CenterOfMassLocalOffset);
                Body->SetLinearDamping(Profile.AirLinearDamping);
                Body->SetAngularDamping(Profile.AirAngularDamping);
            }
            Body->SetPhysicsLinearVelocity(FVector::ZeroVector);
            Body->SetPhysicsAngularVelocityInDegrees(FVector::ZeroVector);
            Body->PutRigidBodyToSleep();
            Body->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        }
        bChaosConveyorBodiesActivated = true;
        SimulatorLog(FString::Printf(
            TEXT("chaos_conveyor_activation bodies=%d clearance_cm=1.00 state=settled"),
            RuntimeChaosConveyorBodies.Num()));
        return;
    }
    ChaosConveyorMotorRampSeconds = FMath::Min(
        1.5f,
        ChaosConveyorMotorRampSeconds + StepSeconds);
    const float MotorRamp = FMath::SmoothStep(
        0.0f,
        1.5f,
        ChaosConveyorMotorRampSeconds);
    int32 DrivenBodyCount = 0;
    int32 DrivenContactCount = 0;
    float TotalTangentialSpeedCmPerSecond = 0.0f;
    float TotalTangentialAccelerationCmPerSecondSquared = 0.0f;
    float TotalAxialAccelerationCmPerSecondSquared = 0.0f;
    const auto ApplyBeltTraction = [this, MotorRamp, &DrivenBodyCount,
        &DrivenContactCount,
        &TotalTangentialSpeedCmPerSecond,
        &TotalTangentialAccelerationCmPerSecondSquared,
        &TotalAxialAccelerationCmPerSecondSquared](
        UPrimitiveComponent* Body,
        const FVector& HalfExtent,
        const FPropPhysicsProfile& Physics)
    {
        if (!Body || !Body->IsSimulatingPhysics())
        {
            return;
        }
        const FVector Location = Body->GetComponentLocation();
        float BeltDistance = 0.0f;
        float LateralDistance = 0.0f;
        FVector BeltPoint;
        FVector BeltTangent;
        ConveyorTuning::ProjectToConveyor(
            Location,
            BeltDistance,
            BeltPoint,
            BeltTangent,
            LateralDistance);
        const float VerticalExtent = ConveyorTuning::ProjectedVerticalHalfExtent(
            Body->GetComponentQuat(),
            HalfExtent);
        const float Bottom = Location.Z - VerticalExtent;
        const bool bOnRollers = LateralDistance <= ConveyorTuning::BeltHalfWidth + 3.0f
            && Bottom >= ConveyorSurfaceZCm - 4.0f
            && Bottom <= ConveyorSurfaceZCm + 5.0f;
        if (!bOnRollers)
        {
            return;
        }
        const FVector Tangent = BeltTangent.GetSafeNormal2D();
        const FVector BeltNormal(-Tangent.Y, Tangent.X, 0.0f);
        const FQuat BodyRotation = Body->GetComponentQuat();
        const float HalfAcrossRollers =
            FMath::Abs(FVector::DotProduct(BodyRotation.GetForwardVector(), BeltNormal))
                * HalfExtent.X
            + FMath::Abs(FVector::DotProduct(BodyRotation.GetRightVector(), BeltNormal))
                * HalfExtent.Y;
        FVector ContactPoints[ConveyorTuning::ConveyorMotorContactSamples];
        FVector ContactTangents[ConveyorTuning::ConveyorMotorContactSamples];
        FVector ContactNormals[ConveyorTuning::ConveyorMotorContactSamples];
        float ContactLateralOffsets[ConveyorTuning::ConveyorMotorContactSamples];
        int32 ContactCount = 0;
        for (int32 SampleIndex = 0;
             SampleIndex < ConveyorTuning::ConveyorMotorContactSamples;
             ++SampleIndex)
        {
            const float Alpha = ConveyorTuning::ConveyorMotorContactSamples > 1
                ? -0.8f + 1.6f * static_cast<float>(SampleIndex)
                    / static_cast<float>(ConveyorTuning::ConveyorMotorContactSamples - 1)
                : 0.0f;
            const FVector SamplePosition = Location
                + BeltNormal * (Alpha * HalfAcrossRollers);
            float SampleDistance = 0.0f;
            float SampleLateralDistance = 0.0f;
            FVector SampleBeltPoint;
            FVector SampleTangent;
            ConveyorTuning::ProjectToConveyor(
                SamplePosition,
                SampleDistance,
                SampleBeltPoint,
                SampleTangent,
                SampleLateralDistance);
            if (SampleLateralDistance > ConveyorTuning::BeltHalfWidth - 1.0f)
            {
                continue;
            }
            ContactPoints[ContactCount] = FVector(
                SamplePosition.X,
                SamplePosition.Y,
                ConveyorSurfaceZCm);
            ContactTangents[ContactCount] = SampleTangent.GetSafeNormal2D();
            ContactNormals[ContactCount] = FVector(
                -ContactTangents[ContactCount].Y,
                ContactTangents[ContactCount].X,
                0.0f);
            ContactLateralOffsets[ContactCount] = FVector::DotProduct(
                SamplePosition - SampleBeltPoint,
                ContactNormals[ContactCount]);
            ++ContactCount;
        }
        if (ContactCount == 0)
        {
            return;
        }

        const float BodyMassKg = FMath::Max(0.05f, Body->GetMass());
        const float ConveyorTargetSpeedCmPerSecond =
            ConveyorTuning::ConveyorSpeedCm * ConveyorSpeedScale;
        float BodyTangentialSpeed = 0.0f;
        float BodyTangentialAcceleration = 0.0f;
        float BodyAxialAcceleration = 0.0f;
        for (int32 ContactIndex = 0; ContactIndex < ContactCount; ++ContactIndex)
        {
            FVector PointVelocity = Body->GetPhysicsLinearVelocityAtPoint(
                ContactPoints[ContactIndex]);
            PointVelocity.Z = 0.0f;
            const float TangentialSpeed = FVector::DotProduct(
                PointVelocity,
                ContactTangents[ContactIndex]);
            const float TangentialSpeedError =
                ConveyorTargetSpeedCmPerSecond - TangentialSpeed;
            const float ResistanceCompensation =
                ConveyorTuning::ConveyorProxyResistanceCompensationCm
                * FMath::Clamp(
                    FMath::Abs(TangentialSpeedError)
                        / ConveyorTuning::ConveyorResistanceFadeSpeedCm,
                    0.0f,
                    1.0f)
                * FMath::Sign(TangentialSpeedError);
            float TangentialAcceleration = FMath::Clamp(
                TangentialSpeedError
                    / ConveyorTuning::ConveyorTangentialResponseSeconds
                    + ResistanceCompensation,
                -ConveyorTuning::ConveyorMaximumParcelAccelerationCm,
                ConveyorTuning::ConveyorMaximumParcelAccelerationCm);
            const float MaximumFrictionAcceleration = FMath::Max(
                0.85f,
                Physics.StaticFriction * 1.25f)
                * ConveyorTuning::GravityCmPerSecondSquared;
            TangentialAcceleration = FMath::Clamp(
                TangentialAcceleration,
                -MaximumFrictionAcceleration,
                MaximumFrictionAcceleration) * MotorRamp;
            const float AxialSpeed = FVector::DotProduct(
                PointVelocity,
                ContactNormals[ContactIndex]);
            const float TargetAxialSpeed = FMath::Clamp(
                -ContactLateralOffsets[ContactIndex]
                    * ConveyorTuning::ConveyorAxialTargetSpeedPerOffset,
                -ConveyorTuning::ConveyorMaximumAxialTargetSpeedCm,
                ConveyorTuning::ConveyorMaximumAxialTargetSpeedCm);
            const float AxialAcceleration = FMath::Clamp(
                (TargetAxialSpeed - AxialSpeed)
                    / ConveyorTuning::ConveyorAxialResponseSeconds,
                -ConveyorTuning::ConveyorMaximumAxialAccelerationCm,
                ConveyorTuning::ConveyorMaximumAxialAccelerationCm) * MotorRamp;
            const FVector ContactForce =
                (ContactTangents[ContactIndex] * TangentialAcceleration
                    + ContactNormals[ContactIndex] * AxialAcceleration)
                * (BodyMassKg / static_cast<float>(ContactCount));
            Body->AddForceAtLocation(
                ContactForce,
                ContactPoints[ContactIndex],
                NAME_None);
            BodyTangentialSpeed += TangentialSpeed;
            BodyTangentialAcceleration += TangentialAcceleration;
            BodyAxialAcceleration += FMath::Abs(AxialAcceleration);
        }
        ++DrivenBodyCount;
        DrivenContactCount += ContactCount;
        TotalTangentialSpeedCmPerSecond +=
            BodyTangentialSpeed / static_cast<float>(ContactCount);
        TotalTangentialAccelerationCmPerSecondSquared +=
            BodyTangentialAcceleration / static_cast<float>(ContactCount);
        TotalAxialAccelerationCmPerSecondSquared +=
            BodyAxialAcceleration / static_cast<float>(ContactCount);
        Body->WakeRigidBody();
    };

    for (int32 ParcelIndex = 0; ParcelIndex < RuntimeChaosConveyorBodies.Num(); ++ParcelIndex)
    {
        if (ParcelHalfExtents.IsValidIndex(ParcelIndex)
            && ParcelPhysicsProfiles.IsValidIndex(ParcelIndex))
        {
            ApplyBeltTraction(
                RuntimeChaosConveyorBodies[ParcelIndex],
                ParcelHalfExtents[ParcelIndex],
                ParcelPhysicsProfiles[ParcelIndex]);
        }
    }
    for (FDynamicBoxRuntime& Prop : DynamicBoxes)
    {
        ApplyBeltTraction(
            Cast<UPrimitiveComponent>(Prop.Root.Get()),
            Prop.HalfExtent,
            Prop.Physics);
    }
    if (!bLoggedChaosConveyorDrive && MotorRamp >= 0.999f)
    {
        bLoggedChaosConveyorDrive = true;
        SimulatorLog(FString::Printf(
            TEXT("chaos_conveyor_drive active_bodies=%d active_contacts=%d average_tangent_speed_cm_s=%.2f target_cm_s=%.2f average_command_acceleration_cm_s2=%.2f average_axial_acceleration_cm_s2=%.2f startup_ramp=complete powered_roller_geometry=cylinders centering=skewed_roller_contact"),
            DrivenBodyCount,
            DrivenContactCount,
            DrivenBodyCount > 0
                ? TotalTangentialSpeedCmPerSecond / static_cast<float>(DrivenBodyCount)
                : 0.0f,
            ConveyorTuning::ConveyorSpeedCm * ConveyorSpeedScale,
            DrivenBodyCount > 0
                ? TotalTangentialAccelerationCmPerSecondSquared / static_cast<float>(DrivenBodyCount)
                : 0.0f,
            DrivenBodyCount > 0
                ? TotalAxialAccelerationCmPerSecondSquared / static_cast<float>(DrivenBodyCount)
                : 0.0f));
    }
}

void AQaiConveyorWorld::SimulateChaosForkContacts(float StepSeconds)
{
    if (!bChaosPhysicsActive || StepSeconds <= UE_SMALL_NUMBER)
    {
        return;
    }

    const auto ApplyForkForces = [this, StepSeconds](
        UPrimitiveComponent* Body,
        const FVector& HalfExtent,
        const FPropPhysicsProfile& Profile,
        int32* SupportedForklift)
    {
        if (!Body || !Body->IsSimulatingPhysics())
        {
            return;
        }
        const FVector BodyLocation = Body->GetComponentLocation();
        const FQuat BodyRotation = Body->GetComponentQuat();
        const float BodyBottom = BodyLocation.Z
            - ConveyorTuning::ProjectedVerticalHalfExtent(BodyRotation, HalfExtent);

        for (int32 ForkliftIndex = 0; ForkliftIndex < ConveyorTuning::EnabledForkliftCount; ++ForkliftIndex)
        {
            if (SupportedForklift && *SupportedForklift != INDEX_NONE
                && *SupportedForklift != ForkliftIndex)
            {
                continue;
            }
            const FForkliftRuntime& Forklift = Forklifts[ForkliftIndex];
            const USceneComponent* ForkliftRoot = Forklift.Root.Get();
            if (!ForkliftRoot)
            {
                continue;
            }
            const FQuat ForkliftRotation = ForkliftRoot->GetComponentQuat();
            const FVector ForkForward = ForkliftRotation.GetForwardVector();
            const FVector ForkRight = ForkliftRotation.GetRightVector();
            const FVector BodyForward = BodyRotation.GetForwardVector();
            const FVector BodyRight = BodyRotation.GetRightVector();
            const float BodyProjectionX = FMath::Abs(FVector::DotProduct(BodyForward, ForkForward)) * HalfExtent.X
                + FMath::Abs(FVector::DotProduct(BodyRight, ForkForward)) * HalfExtent.Y;
            const float BodyProjectionY = FMath::Abs(FVector::DotProduct(BodyForward, ForkRight)) * HalfExtent.X
                + FMath::Abs(FVector::DotProduct(BodyRight, ForkRight)) * HalfExtent.Y;
            const FVector ForkLocalBody = ForkliftRoot->GetComponentTransform()
                .InverseTransformPositionNoScale(BodyLocation);
            const bool bCachedSupport = SupportedForklift && *SupportedForklift == ForkliftIndex;
            const bool bInsideForkEnvelope = ForkLocalBody.X + BodyProjectionX >= -50.0f
                && ForkLocalBody.X - BodyProjectionX <= 250.0f
                && FMath::Abs(ForkLocalBody.Y) <= 80.0f + BodyProjectionY;
            if (bCachedSupport && !bInsideForkEnvelope)
            {
                *SupportedForklift = INDEX_NONE;
                continue;
            }
            float HighestSupportTop = -TNumericLimits<float>::Max();
            int32 ContactTines = 0;
            for (const FFittedCollisionBox& Shape : Forklift.CollisionBoxes)
            {
                if (Shape.IsWedge()
                    || !Shape.Name.Contains(TEXT("fork tine"), ESearchCase::IgnoreCase))
                {
                    continue;
                }
                FVector LocalCenter = Shape.LocalCenter;
                LocalCenter.Z += Forklift.LiftCm;
                FVector TineCenter = ForkliftRoot->GetComponentLocation()
                    + ForkliftRotation.RotateVector(LocalCenter);
                const float TineTop = TineCenter.Z + Shape.HalfExtent.Z + 1.5f;
                // Project the rotated cargo footprint onto forklift-local X/Y
                // and require overlap with an individual tine. This avoids a
                // 3-D SAT rejection caused solely by the intentionally small
                // vertical contact gap.
                if (!bCachedSupport
                    && (FMath::Abs(ForkLocalBody.X - LocalCenter.X)
                        > BodyProjectionX + Shape.HalfExtent.X
                    || FMath::Abs(ForkLocalBody.Y - LocalCenter.Y)
                        > BodyProjectionY + Shape.HalfExtent.Y + 1.0f))
                {
                    continue;
                }
                HighestSupportTop = FMath::Max(HighestSupportTop, TineTop);
                ++ContactTines;
            }
            if (ContactTines == 0)
            {
                continue;
            }
            const float GapCm = BodyBottom - HighestSupportTop;
            const float VerticalHalfExtent = ConveyorTuning::ProjectedVerticalHalfExtent(
                BodyRotation,
                HalfExtent);
            // Recover a missed thin-surface manifold only while the tine top
            // is still inside the supported object's vertical silhouette.
            // Once it exits the object's top this is no longer support.
            if (!bCachedSupport && (GapCm < -3.0f * VerticalHalfExtent || GapCm > 8.0f))
            {
                continue;
            }
            if (SupportedForklift)
            {
                *SupportedForklift = ForkliftIndex;
            }

            const float MassKg = FMath::Max(0.1f, Profile.MassKg);
            const FVector Velocity = Body->GetPhysicsLinearVelocity();
            const FVector CarrierVelocity = Forklift.SurfaceLinearVelocityCmPerSecond;
            const float PenetrationCm = FMath::Max(0.0f, -GapCm);
            const float NormalAcceleration = ConveyorTuning::GravityCmPerSecondSquared
                + (CarrierVelocity.Z - Velocity.Z) * 24.0f
                + PenetrationCm * 120.0f;
            const float NormalForce = FMath::Clamp(
                MassKg * NormalAcceleration,
                0.0f,
                MassKg * ConveyorTuning::GravityCmPerSecondSquared * 8.0f);
            FVector PlanarError = CarrierVelocity - Velocity;
            PlanarError.Z = 0.0f;
            FVector FrictionForce = PlanarError * (MassKg / 0.10f);
            const float MaximumFriction = FMath::Max(0.0f, Profile.DynamicFriction * NormalForce);
            FrictionForce = FrictionForce.GetClampedToMaxSize(MaximumFriction);
            FVector ResolvedVelocity = Velocity;
            ResolvedVelocity.Z += NormalForce * StepSeconds / MassKg;
            ResolvedVelocity.Z = FMath::Max(
                ResolvedVelocity.Z,
                FMath::Min(
                    180.0f,
                    CarrierVelocity.Z + PenetrationCm * 20.0f));
            ResolvedVelocity.Z = FMath::Min(
                ResolvedVelocity.Z,
                FMath::Max(0.0f, CarrierVelocity.Z) + 10.0f);
            ResolvedVelocity.X += FrictionForce.X * StepSeconds / MassKg;
            ResolvedVelocity.Y += FrictionForce.Y * StepSeconds / MassKg;
            const FVector PlanarSlipCorrection = FVector(
                (CarrierVelocity.X - Velocity.X) * StepSeconds,
                (CarrierVelocity.Y - Velocity.Y) * StepSeconds,
                0.0f).GetClampedToMaxSize(
                    FMath::Clamp(Profile.StaticFriction * 4.0f, 0.5f, 3.0f));
            if (GapCm < 0.0f || !PlanarSlipCorrection.IsNearlyZero())
            {
                // Position-level non-penetration projection, analogous to
                // the positional solve in a rigid-body contact constraint.
                // Correct only the measured vertical overlap; never pull a
                // body toward a fork across an air gap.
                Body->SetWorldLocation(
                    BodyLocation
                        + PlanarSlipCorrection
                        + FVector(0.0f, 0.0f, GapCm < 0.0f ? -GapCm + 0.05f : 0.0f),
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
            }
            Body->SetPhysicsLinearVelocity(ResolvedVelocity);
            Body->WakeRigidBody();
            break;
        }
    };

    for (FDynamicBoxRuntime& DynamicBody : DynamicBoxes)
    {
        ApplyForkForces(
            Cast<UPrimitiveComponent>(DynamicBody.Root.Get()),
            DynamicBody.HalfExtent,
            DynamicBody.Physics,
            &DynamicBody.SupportedForklift);
    }
    for (int32 ParcelIndex = 0; ParcelIndex < RuntimeChaosConveyorBodies.Num(); ++ParcelIndex)
    {
        if (ParcelHalfExtents.IsValidIndex(ParcelIndex)
            && ParcelPhysicsProfiles.IsValidIndex(ParcelIndex))
        {
            ApplyForkForces(
                RuntimeChaosConveyorBodies[ParcelIndex],
                ParcelHalfExtents[ParcelIndex],
                ParcelPhysicsProfiles[ParcelIndex],
                ParcelSupportedForklifts.IsValidIndex(ParcelIndex)
                    ? &ParcelSupportedForklifts[ParcelIndex]
                    : nullptr);
        }
    }
}

void AQaiConveyorWorld::SyncChaosTelemetry()
{
    for (FDynamicBoxRuntime& Body : DynamicBoxes)
    {
        if (UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Body.Root.Get()))
        {
            Body.LinearVelocity = Primitive->GetPhysicsLinearVelocity();
            Body.AngularVelocityDegrees = Primitive->GetPhysicsAngularVelocityInDegrees();
            Body.bAwake = Primitive->IsAnyRigidBodyAwake();
            Body.SupportedForklift = INDEX_NONE;
            Body.SupportBodyIndex = INDEX_NONE;
        }
    }
    for (int32 Index = 0; Index < Parcels.Num(); ++Index)
    {
        if (UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Parcels[Index].Get()))
        {
            if (ParcelLinearVelocities.IsValidIndex(Index))
            {
                ParcelLinearVelocities[Index] = Primitive->GetPhysicsLinearVelocity();
            }
            if (ParcelGrounded.IsValidIndex(Index))
            {
                const FVector HalfExtent = ParcelHalfExtents.IsValidIndex(Index)
                    ? ParcelHalfExtents[Index]
                    : FVector::ZeroVector;
                const float VerticalHalfExtent = ConveyorTuning::ProjectedVerticalHalfExtent(
                    Primitive->GetComponentQuat(), HalfExtent);
                ParcelGrounded[Index] = Primitive->GetComponentLocation().Z
                    - VerticalHalfExtent <= ConveyorSurfaceZCm + 3.0f;
            }
            if (ParcelSupportedForklifts.IsValidIndex(Index))
            {
                ParcelSupportedForklifts[Index] = INDEX_NONE;
            }
        }
    }
    if (bPhysicsContactTestInitialized && Parcels.IsValidIndex(0) && Parcels[0].IsValid())
    {
        PhysicsContactMaximumParcelLiftCm = FMath::Max(
            PhysicsContactMaximumParcelLiftCm,
            Parcels[0]->GetComponentLocation().Z - PhysicsContactInitialParcelZ);
    }
}

void AQaiConveyorWorld::MaintainChaosGravity()
{
    if (!bChaosPhysicsActive)
    {
        return;
    }

    const auto HasStaticSupport = [this](
        const FVector& Location,
        const FQuat& Rotation,
        const FVector& HalfExtent)
    {
        const float Bottom = Location.Z
            - ConveyorTuning::ProjectedVerticalHalfExtent(Rotation, HalfExtent);
        if (Bottom <= 1.5f)
        {
            return true;
        }
        for (const FCollisionObstacle& Obstacle : CollisionObstacles)
        {
            if (Obstacle.Name != TEXT("shelf plane")
                && Obstacle.Name != TEXT("packing table")
                && Obstacle.Name != TEXT("conveyor"))
            {
                continue;
            }
            const FVector Local = Obstacle.Rotation.UnrotateVector(Location - Obstacle.Center);
            const float BodyProjectionX = FMath::Abs(FVector::DotProduct(
                Rotation.GetForwardVector(), Obstacle.Rotation.GetForwardVector())) * HalfExtent.X
                + FMath::Abs(FVector::DotProduct(
                    Rotation.GetRightVector(), Obstacle.Rotation.GetForwardVector())) * HalfExtent.Y;
            const float BodyProjectionY = FMath::Abs(FVector::DotProduct(
                Rotation.GetForwardVector(), Obstacle.Rotation.GetRightVector())) * HalfExtent.X
                + FMath::Abs(FVector::DotProduct(
                    Rotation.GetRightVector(), Obstacle.Rotation.GetRightVector())) * HalfExtent.Y;
            const bool bOverSupport = FMath::Abs(Local.X) <= Obstacle.HalfExtent.X + BodyProjectionX
                && FMath::Abs(Local.Y) <= Obstacle.HalfExtent.Y + BodyProjectionY;
            const float SupportTop = Obstacle.Center.Z + Obstacle.HalfExtent.Z;
            if (bOverSupport && FMath::Abs(Bottom - SupportTop) <= 2.0f)
            {
                return true;
            }
        }
        return false;
    };

    for (FDynamicBoxRuntime& Body : DynamicBoxes)
    {
        UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Body.Root.Get());
        if (!Primitive || !Primitive->IsSimulatingPhysics())
        {
            continue;
        }
        const bool bCarried = Body.SupportedForklift != INDEX_NONE;
        const bool bSupported = bCarried || HasStaticSupport(
            Primitive->GetComponentLocation(),
            Primitive->GetComponentQuat(),
            Body.HalfExtent);
        if (!bSupported && !Primitive->IsAnyRigidBodyAwake())
        {
            // Chaos may put a slowly rotating body to sleep after resolving a
            // bad imported contact. An unsupported rigid body must never be
            // allowed to remain suspended; waking it restores gravity without
            // adding any artificial trajectory or downward impulse.
            Primitive->WakeRigidBody();
        }
    }
}

void AQaiConveyorWorld::FixedSimulationStep(float StepSeconds)
{
    SimulateForklifts(StepSeconds);
    if (bChaosPhysicsActive)
    {
        SimulateChaosForkContacts(StepSeconds);
        SimulateChaosConveyor(StepSeconds);
        MaintainChaosGravity();
        SyncChaosTelemetry();
        if (bPhysicsContactTestInitialized && Parcels.IsValidIndex(0) && Parcels[0].IsValid())
        {
            PhysicsContactMaximumParcelLiftCm = FMath::Max(
                PhysicsContactMaximumParcelLiftCm,
                Parcels[0]->GetComponentLocation().Z - PhysicsContactInitialParcelZ);
        }
    }
    else
    {
        SimulateDynamicBoxes(StepSeconds);
        SimulateConveyor(StepSeconds);
    }
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

            if (InferenceRedZoneBounds.IsValid)
            {
                ConsiderSupport(
                    TEXT("red safety mat"),
                    InferenceRedZoneBounds.GetCenter(),
                    FQuat::Identity,
                    InferenceRedZoneBounds.GetExtent());
            }

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
        for (int32 ShapeIndex = 0;
             ShapeIndex < Forklift.ChaosCollisionComponents.Num()
                && ShapeIndex < Forklift.CollisionBoxes.Num();
             ++ShapeIndex)
        {
            if (UBoxComponent* Collider = Forklift.ChaosCollisionComponents[ShapeIndex].Get())
            {
                FVector LocalCenter = Forklift.CollisionBoxes[ShapeIndex].LocalCenter;
                LocalCenter.Z += Forklift.CollisionBoxes[ShapeIndex].bLiftDriven
                    ? Forklift.LiftCm
                    : 0.0f;
                const FTransform ChaosForkliftTransform = Forklift.Root->GetComponentTransform();
                const FVector TargetLocation = ChaosForkliftTransform.TransformPositionNoScale(LocalCenter);
                const FQuat TargetRotation = ChaosForkliftTransform.GetRotation();
                Collider->SetWorldLocationAndRotation(
                    TargetLocation,
                    TargetRotation,
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
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
            const FVector TargetVelocity = CurrentTangent
                * ConveyorTuning::ConveyorSpeedCm * ConveyorSpeedScale;
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
            SupportVelocity = NextTangent
                * ConveyorTuning::ConveyorSpeedCm * ConveyorSpeedScale;
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
            Velocity = ResetTangent
                * ConveyorTuning::ConveyorSpeedCm * ConveyorSpeedScale;
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
    // USD-imported skeletal bounds remain tied to the reference pose, so they
    // can claim that the worker is grounded while the evaluated walking pose
    // is visibly below the floor. Track actual animated foot/toe bones instead.
    // The sole-to-bone distance was calibrated from the intact imported mesh at
    // startup, preserving each character's different shoe/rig proportions.
    const auto AlignAnimatedFeetToCapsule = [this](FWorkerRuntime& Worker)
    {
        UCapsuleComponent* Body = Worker.ChaosBody.Get();
        USceneComponent* VisualRoot = Worker.Root.Get();
        USkeletalMeshComponent* SkeletalMesh = Worker.SkeletalMesh.Get();
        if (!Body || !VisualRoot || !SkeletalMesh || Worker.FootBones.IsEmpty())
        {
            return;
        }

        float LowestAnimatedFootBoneZ = TNumericLimits<float>::Max();
        for (const FName FootBone : Worker.FootBones)
        {
            if (SkeletalMesh->GetBoneIndex(FootBone) != INDEX_NONE)
            {
                LowestAnimatedFootBoneZ = FMath::Min(
                    LowestAnimatedFootBoneZ,
                    SkeletalMesh->GetBoneLocation(FootBone, EBoneSpaces::WorldSpace).Z);
            }
        }
        if (LowestAnimatedFootBoneZ == TNumericLimits<float>::Max())
        {
            return;
        }

        const float CapsuleBottomZ = Body->GetComponentLocation().Z
            - ConveyorTuning::WorkerCapsuleHalfHeightCm;
        const float EstimatedAnimatedSoleZ = LowestAnimatedFootBoneZ
            - Worker.SoleBelowLowestFootBoneCm;
        const float HeightErrorCm = CapsuleBottomZ - EstimatedAnimatedSoleZ;
        const float RootZBefore = VisualRoot->GetComponentLocation().Z;
        if (FMath::Abs(HeightErrorCm) > 0.05f)
        {
            // One frame of animation can briefly report the previous evaluated
            // pose. Limit that correction, but converge large import offsets in
            // a few fixed steps instead of allowing a visible floor crossing.
            const float CorrectionCm = FMath::Clamp(HeightErrorCm, -12.0f, 12.0f);
            VisualRoot->AddWorldOffset(
                FVector(0.0f, 0.0f, CorrectionCm),
                false,
                nullptr,
                ETeleportType::TeleportPhysics);
        }
        if (!Worker.bLoggedFootGroundingRuntime
            && GetWorld()
            && GetWorld()->GetTimeSeconds() >= 3.0f)
        {
            Worker.bLoggedFootGroundingRuntime = true;
            SimulatorLog(FString::Printf(
                TEXT("worker_grounding_runtime body=%s capsule_bottom_z_cm=%.2f lowest_foot_bone_z_cm=%.2f estimated_sole_z_cm=%.2f correction_cm=%.2f visual_root_z_before_cm=%.2f visual_root_z_after_cm=%.2f"),
                *Body->GetName(),
                CapsuleBottomZ,
                LowestAnimatedFootBoneZ,
                EstimatedAnimatedSoleZ,
                HeightErrorCm,
                RootZBefore,
                VisualRoot->GetComponentLocation().Z));
        }
    };
    for (FWorkerRuntime& Worker : Workers)
    {
        if (UCapsuleComponent* Body = Worker.ChaosBody.Get())
        {
            const float WalkingPlaneZ = Worker.InitialChaosTransform.GetLocation().Z;
            FVector BodyLocation = Body->GetComponentLocation();
            if (FMath::Abs(BodyLocation.Z - WalkingPlaneZ) > 0.10f)
            {
                BodyLocation.Z = WalkingPlaneZ;
                Body->SetWorldLocation(
                    BodyLocation,
                    false,
                    nullptr,
                    ETeleportType::TeleportPhysics);
            }
            FVector BodyVelocity = Body->GetPhysicsLinearVelocity();
            if (FMath::Abs(BodyVelocity.Z) > 0.01f)
            {
                BodyVelocity.Z = 0.0f;
                Body->SetPhysicsLinearVelocity(BodyVelocity);
            }
        }
        AlignAnimatedFeetToCapsule(Worker);
    }

    const auto MotionRoot = [](FWorkerRuntime& Worker) -> USceneComponent*
    {
        return Worker.ChaosBody.IsValid()
            ? static_cast<USceneComponent*>(Worker.ChaosBody.Get())
            : Worker.Root.Get();
    };
    const auto ApplyRouteMotor = [](
        FWorkerRuntime& Worker,
        const FVector& DesiredPlanarVelocity)
    {
        UCapsuleComponent* Body = Worker.ChaosBody.Get();
        if (!Body || !Body->IsSimulatingPhysics())
        {
            return;
        }
        const FVector CurrentVelocity = Body->GetPhysicsLinearVelocity();
        const FVector CurrentPlanar(CurrentVelocity.X, CurrentVelocity.Y, 0.0f);
        FVector DesiredPlanar(
            DesiredPlanarVelocity.X,
            DesiredPlanarVelocity.Y,
            0.0f);
        DesiredPlanar = DesiredPlanar.GetClampedToMaxSize(
            ConveyorTuning::WorkerSpeedCm);
        FVector RequestedAcceleration =
            (DesiredPlanar - CurrentPlanar)
            * ConveyorTuning::WorkerMotorResponsePerSecond;
        if (!DesiredPlanar.IsNearlyZero(0.5f))
        {
            // Model the horizontal ground reaction supplied by a walking foot.
            // Without it, the motor merely fights static floor friction and a
            // simulated worker cannot start walking despite a clear route.
            RequestedAcceleration += DesiredPlanar.GetSafeNormal2D()
                * ConveyorTuning::WorkerMotorGroundFrictionCompensationCm;
        }
        RequestedAcceleration = RequestedAcceleration.GetClampedToMaxSize(
            ConveyorTuning::WorkerMotorMaximumAccelerationCm);
        Body->AddForce(
            RequestedAcceleration * ConveyorTuning::WorkerMassKg,
            NAME_None,
            false);
        if (!DesiredPlanar.IsNearlyZero(0.5f)
            || CurrentPlanar.SizeSquared() > FMath::Square(2.0f))
        {
            Body->WakeRigidBody();
        }
    };

    // Prediction prevents new pedestrian overlaps, but an authored/reset pose
    // or a moving prop can still leave two capsules intersecting. Separate an
    // existing overlap before route planning so neither worker deadlocks while
    // waiting for the other to leave the same occupied space.
    for (int32 FirstIndex = 0; FirstIndex < UE_ARRAY_COUNT(Workers); ++FirstIndex)
    {
        USceneComponent* FirstRoot = MotionRoot(Workers[FirstIndex]);
        if (!FirstRoot)
        {
            continue;
        }
        for (int32 SecondIndex = FirstIndex + 1; SecondIndex < UE_ARRAY_COUNT(Workers); ++SecondIndex)
        {
            USceneComponent* SecondRoot = MotionRoot(Workers[SecondIndex]);
            if (!SecondRoot)
            {
                continue;
            }
            FVector Delta = SecondRoot->GetComponentLocation() - FirstRoot->GetComponentLocation();
            Delta.Z = 0.0f;
            const float Distance = Delta.Size2D();
            const float RequiredDistance = ConveyorTuning::WorkerPersonalSpaceCm;
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

    // Reserve the shared crossing before either kinematic capsule commits its
    // next step.  Previously each pedestrian independently picked a steering
    // arc, which allowed both to choose incompatible corrections and oscillate
    // nose-to-nose.  A stable token gives one worker right of way until the
    // pair has separated; the other yields or reverses out of a head-on lane.
    USceneComponent* ConflictRoots[2] = {
        MotionRoot(Workers[0]),
        MotionRoot(Workers[1])};
    if (ConflictRoots[0] && ConflictRoots[1])
    {
        const FVector ConflictLocations[2] = {
            ConflictRoots[0]->GetComponentLocation(),
            ConflictRoots[1]->GetComponentLocation()};
        FVector DesiredDirections[2] = {FVector::ZeroVector, FVector::ZeroVector};
        FVector PredictedLocations[2] = {ConflictLocations[0], ConflictLocations[1]};
        for (int32 WorkerIndex = 0; WorkerIndex < 2; ++WorkerIndex)
        {
            const FWorkerRuntime& Worker = Workers[WorkerIndex];
            if (Worker.Waypoints.IsValidIndex(Worker.DestinationIndex)
                && Worker.DwellRemaining <= 0.0f
                && Worker.WorkerYieldRemainingSeconds <= 0.0f)
            {
                FVector Destination = Worker.Waypoints[Worker.DestinationIndex];
                Destination.Z = ConflictLocations[WorkerIndex].Z;
                DesiredDirections[WorkerIndex] =
                    (Destination - ConflictLocations[WorkerIndex]).GetSafeNormal2D();
                PredictedLocations[WorkerIndex] += DesiredDirections[WorkerIndex]
                    * ConveyorTuning::WorkerSpeedCm
                    * ConveyorTuning::WorkerConflictLookaheadSeconds;
            }
        }

        const float CurrentDistance = FVector::Dist2D(ConflictLocations[0], ConflictLocations[1]);
        const float PredictedDistance = FVector::Dist2D(PredictedLocations[0], PredictedLocations[1]);
        if (WorkerRightOfWayIndex != INDEX_NONE)
        {
            WorkerConflictElapsedSeconds += StepSeconds;
            if (CurrentDistance >= ConveyorTuning::WorkerConflictReleaseDistanceCm)
            {
                SimulatorLog(FString::Printf(
                    TEXT("worker_right_of_way_released priority=%d separation_cm=%.1f elapsed_ms=%.0f"),
                    WorkerRightOfWayIndex + 1,
                    CurrentDistance,
                    WorkerConflictElapsedSeconds * 1000.0f));
                WorkerRightOfWayIndex = INDEX_NONE;
                WorkerConflictElapsedSeconds = 0.0f;
                bWorkerConflictHeadOn = false;
            }
        }

        const bool bClosing = PredictedDistance + 0.5f < CurrentDistance;
        if (WorkerRightOfWayIndex == INDEX_NONE
            && bClosing
            && PredictedDistance < ConveyorTuning::WorkerPersonalSpaceCm)
        {
            WorkerRightOfWayIndex = WorkerNextRightOfWayIndex;
            WorkerNextRightOfWayIndex = 1 - WorkerNextRightOfWayIndex;
            const int32 YieldingIndex = 1 - WorkerRightOfWayIndex;
            const float DirectionDot = FVector::DotProduct(
                DesiredDirections[0],
                DesiredDirections[1]);
            bWorkerConflictHeadOn = DirectionDot < -0.35f;
            FWorkerRuntime& YieldingWorker = Workers[YieldingIndex];
            YieldingWorker.AvoidanceTurnSign = YieldingIndex == 0 ? 1 : -1;
            YieldingWorker.WorkerYieldRemainingSeconds = bWorkerConflictHeadOn
                ? ConveyorTuning::WorkerHeadOnYieldSeconds
                : ConveyorTuning::WorkerCrossingYieldSeconds;
            ++YieldingWorker.RightOfWayYieldCount;

            if (bWorkerConflictHeadOn
                && YieldingWorker.RouteReversalCooldownSeconds <= 0.0f
                && YieldingWorker.Waypoints.Num() > 1)
            {
                YieldingWorker.RouteDirection *= -1;
                ++YieldingWorker.RouteReversalCount;
                YieldingWorker.RouteReversalCooldownSeconds =
                    ConveyorTuning::WorkerRouteReversalCooldownSeconds;
                const int32 LastWaypoint = YieldingWorker.Waypoints.Num() - 1;
                int32 NextDestination =
                    YieldingWorker.DestinationIndex + YieldingWorker.RouteDirection;
                if (NextDestination > LastWaypoint)
                {
                    YieldingWorker.RouteDirection = -1;
                    NextDestination = FMath::Max(0, LastWaypoint - 1);
                }
                else if (NextDestination < 0)
                {
                    YieldingWorker.RouteDirection = 1;
                    NextDestination = FMath::Min(LastWaypoint, 1);
                }
                YieldingWorker.DestinationIndex = NextDestination;
            }

            SimulatorLog(FString::Printf(
                TEXT("worker_right_of_way_reserved priority=%d yielding=%d conflict=%s current_cm=%.1f predicted_cm=%.1f yield_ms=%.0f"),
                WorkerRightOfWayIndex + 1,
                YieldingIndex + 1,
                bWorkerConflictHeadOn ? TEXT("head_on") : TEXT("crossing"),
                CurrentDistance,
                PredictedDistance,
                YieldingWorker.WorkerYieldRemainingSeconds * 1000.0f));
        }
    }

    for (int32 WorkerIndex = 0; WorkerIndex < UE_ARRAY_COUNT(Workers); ++WorkerIndex)
    {
        FWorkerRuntime& Worker = Workers[WorkerIndex];
        USceneComponent* WorkerRoot = MotionRoot(Worker);
        if (!WorkerRoot || !Worker.Waypoints.IsValidIndex(Worker.DestinationIndex))
        {
            continue;
        }
        Worker.RouteReversalCooldownSeconds = FMath::Max(
            0.0f,
            Worker.RouteReversalCooldownSeconds - StepSeconds);
        Worker.WorkerYieldRemainingSeconds = FMath::Max(
            0.0f,
            Worker.WorkerYieldRemainingSeconds - StepSeconds);
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
        if (Worker.WorkerYieldRemainingSeconds > 0.0f)
        {
            Worker.BlockedSeconds = 0.0f;
            ApplyRouteMotor(Worker, FVector::ZeroVector);
            continue;
        }
        if (Worker.DwellRemaining > 0.0f)
        {
            Worker.DwellRemaining = FMath::Max(0.0f, Worker.DwellRemaining - StepSeconds);
            ApplyRouteMotor(Worker, FVector::ZeroVector);
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
            ApplyRouteMotor(Worker, FVector::ZeroVector);
            continue;
        }
        const float StepCm = ConveyorTuning::WorkerSpeedCm * StepSeconds;
        const FVector Direction = Delta.GetSafeNormal2D();
        const bool bWouldArrive = Distance <= StepCm + 0.1f;
        FVector Next = bWouldArrive ? Destination : Current + Direction * StepCm;
        // Look farther ahead than one 120-Hz integration step. Once these are
        // real dynamic bodies, checking only the sub-centimetre next pose is
        // too late to brake a walking capsule before a wall or rack contact.
        const float ProbeDistance = FMath::Min(
            Distance,
            FMath::Max(StepCm, ConveyorTuning::WorkerAvoidanceProbeCm));
        const FVector ForwardProbe = Current + Direction * ProbeDistance;
        if (!CanWorkerOccupy(WorkerIndex, ForwardProbe))
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
                const FVector ProbeCandidate = Current + AvoidanceDirection * ProbeDistance;
                if (CanWorkerOccupy(WorkerIndex, ProbeCandidate))
                {
                    Next = Current + AvoidanceDirection * StepCm;
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
                ApplyRouteMotor(Worker, FVector::ZeroVector);
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
            ApplyRouteMotor(Worker, FVector::ZeroVector);
            continue;
        }
        FVector ResolvedHeadingDirection = ActualDelta;
        if (UCapsuleComponent* Body = Worker.ChaosBody.Get())
        {
            FVector PhysicalVelocity = Body->GetPhysicsLinearVelocity();
            PhysicalVelocity.Z = 0.0f;
            // At normal walking speed, face the motion Chaos actually resolved,
            // including collision deflection, rather than the requested route
            // step. Retain the intended direction near rest to avoid noise.
            if (PhysicalVelocity.SizeSquared2D() >= FMath::Square(12.0f))
            {
                ResolvedHeadingDirection = PhysicalVelocity;
            }
        }
        const float InstantHeadingYaw = ResolvedHeadingDirection.Rotation().Yaw;
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
        if (Worker.ChaosBody.IsValid())
        {
            const float RequestedSpeed = FMath::Min(
                ConveyorTuning::WorkerSpeedCm,
                ActualDelta.Size2D() / FMath::Max(StepSeconds, UE_SMALL_NUMBER));
            ApplyRouteMotor(
                Worker,
                ActualDelta.GetSafeNormal2D() * RequestedSpeed);
            const FQuat DesiredVisualWorldRotation = VisualHeading * Worker.HeadingOffset;
            if (USceneComponent* VisualPivot = Worker.VisualPivot.Get())
            {
                // Rotate the imported hierarchy about the rendered character's
                // body centre. Rotating its remote USD root directly makes the
                // mesh orbit away from its Chaos capsule.
                VisualPivot->SetWorldRotation(
                    DesiredVisualWorldRotation
                    * Worker.InitialVisualRelativeTransform.GetRotation().Inverse());
            }
            else if (USceneComponent* VisualRoot = Worker.Root.Get())
            {
                VisualRoot->SetWorldRotation(DesiredVisualWorldRotation);
            }
        }
        else
        {
            WorkerRoot->SetWorldLocationAndRotation(
                Next,
                VisualHeading * Worker.HeadingOffset);
        }
        if (bWouldArrive && FVector::DistSquared2D(Next, Destination) <= 1.0f)
        {
            Worker.DwellRemaining = Worker.DwellSeconds.IsValidIndex(Worker.DestinationIndex)
                ? Worker.DwellSeconds[Worker.DestinationIndex]
                : 0.0f;
            AdvanceRoute();
        }
    }
}

FString AQaiConveyorWorld::EvaluateParcelSafetySignal(FString* OutDetails) const
{
    FString WorstSignal = TEXT("G");
    FString WorstDetails = TEXT("all conveyor parcels fully supported");
    int32 WorstSeverity = 0;
    float WorstSupportFraction = 1.0f;

    for (int32 ParcelIndex = 0; ParcelIndex < Parcels.Num(); ++ParcelIndex)
    {
        USceneComponent* Parcel = Parcels[ParcelIndex].Get();
        if (!Parcel || !ParcelHalfExtents.IsValidIndex(ParcelIndex))
        {
            continue;
        }
        // A parcel intentionally lifted by a forklift has not fallen from the
        // conveyor. The experiment is about loss of roller support.
        if (ParcelSupportedForklifts.IsValidIndex(ParcelIndex)
            && ParcelSupportedForklifts[ParcelIndex] != INDEX_NONE)
        {
            continue;
        }

        const FVector Location = Parcel->GetComponentLocation();
        const FQuat Rotation = Parcel->GetComponentQuat();
        const FVector HalfExtent = ParcelHalfExtents[ParcelIndex];
        float BeltDistance = 0.0f;
        float LateralDistance = 0.0f;
        FVector BeltPoint;
        FVector BeltTangent;
        ConveyorTuning::ProjectToConveyor(
            Location,
            BeltDistance,
            BeltPoint,
            BeltTangent,
            LateralDistance);
        const FVector Tangent = BeltTangent.GetSafeNormal2D();
        const FVector BeltNormal(-Tangent.Y, Tangent.X, 0.0f);
        const float SignedLateralOffset = FVector::DotProduct(
            Location - BeltPoint,
            BeltNormal);
        const float HalfAcrossRollers = FMath::Max(
            1.0f,
            FMath::Abs(FVector::DotProduct(Rotation.GetForwardVector(), BeltNormal))
                    * HalfExtent.X
                + FMath::Abs(FVector::DotProduct(Rotation.GetRightVector(), BeltNormal))
                    * HalfExtent.Y);
        const float ParcelMinimum = SignedLateralOffset - HalfAcrossRollers;
        const float ParcelMaximum = SignedLateralOffset + HalfAcrossRollers;
        const float SupportedWidth = FMath::Max(
            0.0f,
            FMath::Min(ParcelMaximum, ConveyorTuning::BeltHalfWidth)
                - FMath::Max(ParcelMinimum, -ConveyorTuning::BeltHalfWidth));
        const float SupportFraction = FMath::Clamp(
            SupportedWidth / (2.0f * HalfAcrossRollers),
            0.0f,
            1.0f);
        const float VerticalHalfExtent = ConveyorTuning::ProjectedVerticalHalfExtent(
            Rotation,
            HalfExtent);
        const float Bottom = Location.Z - VerticalHalfExtent;
        FVector Velocity = ParcelLinearVelocities.IsValidIndex(ParcelIndex)
            ? ParcelLinearVelocities[ParcelIndex]
            : FVector::ZeroVector;
        if (const UPrimitiveComponent* Primitive = Cast<UPrimitiveComponent>(Parcel))
        {
            Velocity = Primitive->GetPhysicsLinearVelocity();
        }
        const float OutwardSpeed = SignedLateralOffset >= 0.0f
            ? FVector::DotProduct(Velocity, BeltNormal)
            : -FVector::DotProduct(Velocity, BeltNormal);
        const float PredictedOffset = FMath::Abs(SignedLateralOffset)
            + FMath::Max(0.0f, OutwardSpeed)
                * ConveyorTuning::ParcelDangerPredictionSeconds;
        const float PredictedMinimum = PredictedOffset - HalfAcrossRollers;
        const float PredictedMaximum = PredictedOffset + HalfAcrossRollers;
        const float PredictedSupportedWidth = FMath::Max(
            0.0f,
            FMath::Min(PredictedMaximum, ConveyorTuning::BeltHalfWidth)
                - FMath::Max(PredictedMinimum, -ConveyorTuning::BeltHalfWidth));
        const float PredictedSupportFraction = FMath::Clamp(
            PredictedSupportedWidth / (2.0f * HalfAcrossRollers),
            0.0f,
            1.0f);
        const FRotator ParcelRotation = Rotation.Rotator();
        const float TiltDegrees = FMath::Max(
            FMath::Abs(ParcelRotation.Pitch),
            FMath::Abs(ParcelRotation.Roll));

        const bool bFallen = SupportFraction <= ConveyorTuning::ParcelRedSupportFraction
            || (Bottom < ConveyorSurfaceZCm - ConveyorTuning::ParcelFallenDropCm
                && SupportFraction < ConveyorTuning::ParcelFallenSupportFraction);
        const bool bDanger = !bFallen
            && (SupportFraction < ConveyorTuning::ParcelAmberSupportFraction
                || (OutwardSpeed > 4.0f && PredictedSupportFraction < 0.55f)
                || (TiltDegrees >= ConveyorTuning::ParcelDangerTiltDegrees
                    && SupportFraction < 0.88f));
        const int32 Severity = bFallen ? 2 : (bDanger ? 1 : 0);
        if (Severity > WorstSeverity
            || (Severity == WorstSeverity && SupportFraction < WorstSupportFraction))
        {
            WorstSeverity = Severity;
            WorstSupportFraction = SupportFraction;
            WorstSignal = bFallen ? TEXT("R") : (bDanger ? TEXT("A") : TEXT("G"));
            WorstDetails = FString::Printf(
                TEXT("parcel=%d support=%.1f%% predicted_support=%.1f%% lateral_cm=%.1f outward_cm_s=%.1f bottom_delta_cm=%.1f tilt_deg=%.1f"),
                ParcelIndex + 1,
                SupportFraction * 100.0f,
                PredictedSupportFraction * 100.0f,
                FMath::Abs(SignedLateralOffset),
                OutwardSpeed,
                Bottom - ConveyorSurfaceZCm,
                TiltDegrees);
        }
    }

    if (OutDetails)
    {
        *OutDetails = WorstDetails;
    }
    return WorstSignal;
}

void AQaiConveyorWorld::UpdateSafetySignal()
{
    FString Details;
    const FString InstantSignal = EvaluateParcelSafetySignal(&Details);
    if (InstantSignal != ParcelSafetyCandidateSignal)
    {
        ParcelSafetyCandidateSignal = InstantSignal;
        ParcelSafetyCandidateSeconds = 0.0f;
    }
    else
    {
        ParcelSafetyCandidateSeconds += GetWorld()
            ? FMath::Clamp(GetWorld()->GetDeltaSeconds(), 0.0f, 0.05f)
            : 0.0f;
    }

    const float RequiredEvidenceSeconds = InstantSignal == TEXT("R")
        ? 0.06f
        : (InstantSignal == TEXT("A") ? 0.12f : 0.35f);
    if (InstantSignal != GroundTruthSignal
        && ParcelSafetyCandidateSeconds >= RequiredEvidenceSeconds)
    {
        SimulatorLog(FString::Printf(
            TEXT("ground_truth_transition from=%s to=%s focus=conveyor_parcel_support %s"),
            *GroundTruthSignal,
            *InstantSignal,
            *Details));
        GroundTruthSignal = InstantSignal;
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
        // F8 is a local diagnostic only. Scene captures otherwise render the
        // world's foreground line batcher just like the player viewport, which
        // leaked collision boxes into Reason2 input. Hide every debug batcher
        // from this capture without changing its visibility in the game view.
        constexpr UWorld::ELineBatcherType DebugBatcherTypes[] = {
            UWorld::ELineBatcherType::World,
            UWorld::ELineBatcherType::WorldPersistent,
            UWorld::ELineBatcherType::Foreground,
            UWorld::ELineBatcherType::ForegroundPersistent,
        };
        int32 HiddenDebugBatchers = 0;
        for (const UWorld::ELineBatcherType BatcherType : DebugBatcherTypes)
        {
            if (ULineBatchComponent* DebugBatcher = GetWorld()->GetLineBatcher(BatcherType))
            {
                Capture->HiddenComponents.Add(DebugBatcher);
                ++HiddenDebugBatchers;
            }
        }
        // Lumen's physically bounced radiance requires a very different fixed
        // exposure from the temporary raster baseline. A 512x288 bracket on
        // the RTX Ultra path selected -8 EV: mean sRGB 124, p95 195 and zero
        // clipped neutral-white pixels, compared with 94% clipping at +11 EV.
        float CaptureExposureBias = -8.0f;
        FParse::Value(FCommandLine::Get(), TEXT("QaiCaptureExposureBias="), CaptureExposureBias);
        Capture->TextureTarget = CaptureTarget;
        Capture->bCaptureEveryFrame = false;
        Capture->bCaptureOnMovement = false;
        // A manually sampled SceneCapture otherwise discards its view state.
        // Keep it for Lumen's surface-cache history, but do not use temporal
        // anti-aliasing here: this camera is sampled at only 4 FPS. TSR/TAA then
        // interprets large object displacement as short-frame motion and leaves
        // black, stippled disocclusion trails in the lossless PNGs themselves.
        Capture->bAlwaysPersistRenderingState = true;
        Capture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
        const bool bUseAuthoredWideCamera =
            InferenceCameraVariant.Equals(TEXT("authored-wide"), ESearchCase::IgnoreCase);
        const bool bUseParcelBeltCamera =
            InferenceCameraVariant.Equals(TEXT("parcel-belt"), ESearchCase::IgnoreCase);
        if (bUseAuthoredWideCamera)
        {
            // The controlled classification run favoured the earlier
            // 105-degree view over the natural 95.5-degree DetectorEndline
            // optics. The camera is also backed away so important geometry is
            // less concentrated at the more distorted outer edge.
            constexpr float DetectorHorizontalFovDegrees = 105.0f;
            constexpr float DetectorHorizontalApertureMm = 20.955f;
            constexpr float DetectorHorizontalApertureOffsetMm = -3.0f;
            const float DetectorHalfFovRadians = FMath::DegreesToRadians(
                DetectorHorizontalFovDegrees * 0.5f);
            Capture->FOVAngle = DetectorHorizontalFovDegrees;
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
        }
        else if (bUseParcelBeltCamera)
        {
            // A natural moderately wide lens keeps the complete roller loop
            // large in frame without stretching parcels at the image edges.
            constexpr float DetectorHorizontalFovDegrees = 78.0f;
            Capture->FOVAngle = DetectorHorizontalFovDegrees;
            Capture->bUseCustomProjectionMatrix = false;
        }
        else
        {
            // The tighter centred lens deliberately excludes more of the upper
            // wall while retaining the safety boundary and approach lane.
            constexpr float DetectorHorizontalFovDegrees = 56.0f;
            Capture->FOVAngle = DetectorHorizontalFovDegrees;
            Capture->bUseCustomProjectionMatrix = false;
        }
        Capture->ShowFlags.SetMotionBlur(false);
        Capture->ShowFlags.SetTemporalAA(false);
        Capture->ShowFlags.SetGlobalIllumination(true);
        Capture->ShowFlags.SetLumenGlobalIllumination(true);
        Capture->ShowFlags.SetLumenReflections(true);
        // Do not locally relight dark and bright image regions independently.
        // One global manual exposure is closer to a calibrated industrial
        // camera and preserves relative luminance cues for Reason2.
        Capture->ShowFlags.SetLocalExposure(false);
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
        Capture->PostProcessSettings.ColorContrast = FVector4(1.0f, 1.0f, 1.0f, 1.0f);
        Capture->PostProcessSettings.bOverride_ColorGain = true;
        Capture->PostProcessSettings.ColorGain = FVector4(1.0f, 1.0f, 1.0f, 1.0f);
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

        // Controlled vision ablations affect this sensor only. The physical
        // scene, collision, illumination, player view, and ground-truth tracker
        // remain unchanged, so every run differs only in which visual primitive
        // groups Reason2 can see.
        int32 AblationVisiblePrimitives = 0;
        int32 AblationForkliftPrimitives = 0;
        int32 AblationArchitecturePrimitives = 0;
        int32 AblationConveyorPrimitives = 0;
        int32 AblationWorkerPrimitives = 0;
        int32 AblationParcelPrimitives = 0;
        int32 AblationStoragePrimitives = 0;
        int32 AblationTablePrimitives = 0;
        int32 AblationBillboardPrimitives = 0;
        int32 AblationPortalPrimitives = 0;
        int32 AblationRestPrimitives = 0;
        if (!InferenceAblationVariant.Equals(TEXT("full"), ESearchCase::IgnoreCase))
        {
            TArray<FString> AblationGroups;
            // FParse::Value terminates an unquoted command-line value at a
            // comma on Windows. Accept '+' as the CLI-safe group separator so
            // automated add-back runs do not silently collapse to "minimal".
            FString NormalizedAblationGroups = InferenceAblationVariant;
            NormalizedAblationGroups.ReplaceInline(TEXT("+"), TEXT(","));
            NormalizedAblationGroups.ParseIntoArray(AblationGroups, TEXT(","), true);
            for (FString& Group : AblationGroups)
            {
                Group = Group.TrimStartAndEnd().ToLower();
            }
            const auto IncludesGroup = [&AblationGroups](const TCHAR* GroupName)
            {
                return AblationGroups.ContainsByPredicate(
                    [GroupName](const FString& Value)
                    {
                        return Value.Equals(GroupName, ESearchCase::IgnoreCase);
                    });
            };
            const bool bRestoreWorkers = IncludesGroup(TEXT("workers"));
            const bool bRestoreParcels = IncludesGroup(TEXT("parcels"));
            const bool bRestoreBeltParcels = bRestoreParcels
                || IncludesGroup(TEXT("belt-parcels"));
            const bool bRestoreShelfParcels = bRestoreParcels
                || IncludesGroup(TEXT("shelf-parcels"));
            const bool bRestorePallets = bRestoreParcels
                || IncludesGroup(TEXT("pallets"));
            const bool bRestoreStorage = IncludesGroup(TEXT("storage"));
            const bool bRestoreTable = IncludesGroup(TEXT("table"));
            const bool bRestoreBillboard = IncludesGroup(TEXT("billboard"));
            const bool bRestorePortals = IncludesGroup(TEXT("portals"));
            const bool bRestoreRest = IncludesGroup(TEXT("rest"));

            Capture->PrimitiveRenderMode =
                ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
            Capture->ClearShowOnlyComponents();
            for (TActorIterator<AActor> It(GetWorld()); It; ++It)
            {
                AActor* Actor = *It;
                if (!Actor || Actor == CaptureActor)
                {
                    continue;
                }
                const FString OwnerName = Actor->GetName().ToLower();
                const FString OwnerLabel = Actor->GetActorNameOrLabel().ToLower();
                TInlineComponentArray<UPrimitiveComponent*> Primitives(Actor);
                for (UPrimitiveComponent* Primitive : Primitives)
                {
                    if (!Primitive
                        || (!Primitive->IsA<UStaticMeshComponent>()
                            && !Primitive->IsA<USkeletalMeshComponent>()))
                    {
                        continue;
                    }

                    FString MeshPath;
                    if (const UStaticMeshComponent* StaticMesh =
                            Cast<UStaticMeshComponent>(Primitive))
                    {
                        if (StaticMesh->GetStaticMesh())
                        {
                            MeshPath = StaticMesh->GetStaticMesh()->GetPathName().ToLower();
                        }
                    }
                    else if (const USkeletalMeshComponent* SkeletalMesh =
                                 Cast<USkeletalMeshComponent>(Primitive))
                    {
                        if (SkeletalMesh->GetSkeletalMeshAsset())
                        {
                            MeshPath = SkeletalMesh->GetSkeletalMeshAsset()->GetPathName().ToLower();
                        }
                    }
                    FString Identity = Primitive->GetName().ToLower() + TEXT("|") + MeshPath;
                    for (const FName& Tag : Primitive->ComponentTags)
                    {
                        Identity += TEXT("|") + Tag.ToString().ToLower();
                    }
                    for (const USceneComponent* Parent = Primitive->GetAttachParent();
                         Parent;
                         Parent = Parent->GetAttachParent())
                    {
                        Identity += TEXT("|") + Parent->GetName().ToLower();
                    }
                    const FString ActorIdentity = OwnerName + TEXT("|") + OwnerLabel;

                    const bool bBillboard =
                        ActorIdentity.Contains(TEXT("qualcommwallbillboard"))
                        || ActorIdentity.Contains(TEXT("dragonwingbillboard"))
                        || Identity.Contains(TEXT("qualcommwallbillboard"));
                    const bool bWorker = Identity.Contains(TEXT("qai.worker"))
                        || Identity.Contains(TEXT("/workers/worker"))
                        || ActorIdentity.Contains(TEXT("worker1"))
                        || ActorIdentity.Contains(TEXT("worker2"));
                    const bool bPallet = Identity.Contains(TEXT("pallet"))
                        || ActorIdentity.Contains(TEXT("palletstack"));
                    // Shelf cartons deliberately reuse the same authored box
                    // meshes as conveyor parcels. Runtime semantic tags must
                    // therefore win over asset-path classification.
                    const bool bExplicitShelfParcel =
                        Identity.Contains(TEXT("qai.shelfparcel"))
                        || ActorIdentity.Contains(TEXT("shelfparcel"));
                    const bool bBeltParcel = !bPallet && !bExplicitShelfParcel
                        && (Identity.Contains(TEXT("qai.parcel"))
                            || Identity.Contains(TEXT("/conveyor/parcels/")));
                    const bool bShelfParcel = !bPallet && !bBeltParcel
                        && (bExplicitShelfParcel
                            || Identity.Contains(TEXT("/dynamiccargo/"))
                            || Identity.Contains(TEXT("parcel"))
                            || Identity.Contains(TEXT("carton"))
                            );
                    const bool bParcel = bBeltParcel || bShelfParcel || bPallet;
                    const bool bForklift = !bParcel
                        && (Identity.Contains(TEXT("qai.forklift1"))
                            || Identity.Contains(TEXT("/forklifts/forklift1/"))
                            || ActorIdentity.Equals(TEXT("forklift1")));
                    const bool bStorage = !bParcel
                        && (Identity.Contains(TEXT("/storage/"))
                            || Identity.Contains(TEXT("west rack"))
                            || Identity.Contains(TEXT("westrack"))
                            || ActorIdentity.Contains(TEXT("shelf"))
                            || ActorIdentity.Contains(TEXT("rack")));
                    const bool bTable = Identity.Contains(TEXT("packing_table"))
                        || Identity.Contains(TEXT("packingtable"))
                        || Identity.Contains(TEXT("sortingtable"))
                        || Identity.Contains(TEXT("iq9"))
                        || ActorIdentity.Contains(TEXT("packingtable"))
                        || ActorIdentity.Contains(TEXT("sortingtable"))
                        || ActorIdentity.Contains(TEXT("iq9"));
                    const bool bPortal = ActorIdentity.Contains(TEXT("sortingportal"))
                        || Identity.Contains(TEXT("sortingportal"));
                    const bool bArchitecture = !bBillboard
                        && (Identity.Contains(TEXT("/floor/"))
                            || Identity.Contains(TEXT("/floorfinish/"))
                            || Identity.Contains(TEXT("floorfinish"))
                            || Identity.Contains(TEXT("sm_floor"))
                            || Identity.Contains(TEXT("behindwallfloor"))
                            || Identity.Contains(TEXT("/shell/"))
                            || Identity.Contains(TEXT("clearancezone"))
                            || ActorIdentity.Contains(TEXT("safetyborder"))
                            || ActorIdentity.Contains(TEXT("safetystripe"))
                            || ActorIdentity.Contains(TEXT("northwall"))
                            || ActorIdentity.Contains(TEXT("westwall"))
                            || ActorIdentity.Contains(TEXT("solidwall")));
                    const bool bConveyor = !bParcel && !bPortal
                        && (Identity.Contains(TEXT("/conveyor/modules/"))
                            || Identity.Contains(TEXT("conveyormodule"))
                            || Identity.Contains(TEXT("conveyorvisual")));
                    const bool bKnownGroup = bForklift || bArchitecture || bConveyor
                        || bWorker || bParcel || bStorage || bTable || bBillboard || bPortal;
                    const bool bShow = bForklift || bArchitecture || bConveyor
                        || (bWorker && bRestoreWorkers)
                        || (bBeltParcel && bRestoreBeltParcels)
                        || (bShelfParcel && bRestoreShelfParcels)
                        || (bPallet && bRestorePallets)
                        || (bStorage && bRestoreStorage)
                        || (bTable && bRestoreTable)
                        || (bBillboard && bRestoreBillboard)
                        || (bPortal && bRestorePortals)
                        || (!bKnownGroup && bRestoreRest);
                    if (!bShow)
                    {
                        continue;
                    }
                    Capture->ShowOnlyComponent(Primitive);
                    ++AblationVisiblePrimitives;
                    AblationForkliftPrimitives += bForklift ? 1 : 0;
                    AblationArchitecturePrimitives += bArchitecture ? 1 : 0;
                    AblationConveyorPrimitives += bConveyor ? 1 : 0;
                    AblationWorkerPrimitives += bWorker ? 1 : 0;
                    AblationParcelPrimitives += bParcel ? 1 : 0;
                    AblationStoragePrimitives += bStorage ? 1 : 0;
                    AblationTablePrimitives += bTable ? 1 : 0;
                    AblationBillboardPrimitives += bBillboard ? 1 : 0;
                    AblationPortalPrimitives += bPortal ? 1 : 0;
                    AblationRestPrimitives += !bKnownGroup ? 1 : 0;
                }
            }
            SimulatorLog(FString::Printf(
                TEXT("inference_ablation variant=%s visible=%d forklift=%d architecture=%d conveyor=%d workers=%d parcels=%d storage=%d table=%d billboard=%d portals=%d rest=%d"),
                *InferenceAblationVariant,
                AblationVisiblePrimitives,
                AblationForkliftPrimitives,
                AblationArchitecturePrimitives,
                AblationConveyorPrimitives,
                AblationWorkerPrimitives,
                AblationParcelPrimitives,
                AblationStoragePrimitives,
                AblationTablePrimitives,
                AblationBillboardPrimitives,
                AblationPortalPrimitives,
                AblationRestPrimitives));
        }

        const bool bEvkCapture = ActiveBackend == TEXT("evk");
        SimulatorLog(FString::Printf(
            TEXT("capture_initialized render_size=%dx%d output_size=%dx%d backend=%s exposure_mode=manual_global physical_camera=false exposure_compensation=%.2f local_exposure=false highlight_scale=0.62 bloom=0.12 lumen_view_state=persistent horizontal_fov_deg=%.2f projection=%s composition=%s debug_line_batchers_hidden=%d"),
            CaptureRenderWidth,
            CaptureRenderHeight,
            bEvkCapture ? ConveyorTuning::EvkCaptureWidth : HostCaptureWidth,
            bEvkCapture ? ConveyorTuning::EvkCaptureHeight : HostCaptureHeight,
            *ActiveBackend,
            CaptureExposureBias,
            Capture->FOVAngle,
            bUseAuthoredWideCamera ? TEXT("omniverse_off_axis") : TEXT("centered"),
            bUseAuthoredWideCamera ? TEXT("authored_detector_endline") : TEXT("fixed_red_zone_and_approach"),
            HiddenDebugBatchers));
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
    CaptureAccumulator = 0.0f;
    EncodedFrames.Reset();
    EncodedFrameWidths.Reset();
    EncodedFrameHeights.Reset();
    EncodedParcelSafetySignals.Reset();
    EncodedFrameTextures.Reset();
    EncodedFrameTimes.Reset();
    EncodedFrameCaptureSeconds.Reset();
    SubmittedEncodedFrames.Reset();
    SubmittedFrameTextures.Reset();
    SubmittedFrameTimes.Reset();
    SubmittedGroundTruthSignal = TEXT("G");
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
    if (CaptureAccumulator >= CaptureIntervalSeconds)
    {
        // Retain fractional cadence error without issuing catch-up bursts.
        // The ring buffer therefore stays close to a true fixed 4 FPS clock.
        CaptureAccumulator = FMath::Fmod(CaptureAccumulator, CaptureIntervalSeconds);
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
                Pixels.SetNumUninitialized(RenderWidth * RenderHeight);
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

                // Scene-capture alpha is renderer/RHI dependent and can be
                // zero even though RGB is valid. These frames are deliberately
                // opaque RGB observations; normalise alpha before both PNG
                // encoding and HUD texture creation so a translucent canvas
                // path cannot turn valid captures into black/empty previews.
                for (FColor& Pixel : OutputPixels)
                {
                    Pixel.A = 255;
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
    UTexture2D* Texture = UTexture2D::CreateTransient(
        FrameWidth,
        FrameHeight,
        PF_B8G8R8A8,
        NAME_None);
    if (!Texture || !Texture->GetPlatformData() || Texture->GetPlatformData()->Mips.IsEmpty())
    {
        return nullptr;
    }
    Texture->SRGB = true;
    Texture->NeverStream = true;
    Texture->Filter = TF_Bilinear;
    FTexture2DMipMap& Mip = Texture->GetPlatformData()->Mips[0];
    void* Destination = Mip.BulkData.Lock(LOCK_READ_WRITE);
    if (!Destination)
    {
        Mip.BulkData.Unlock();
        return nullptr;
    }
    FMemory::Memcpy(
        Destination,
        Pixels.GetData(),
        static_cast<SIZE_T>(Pixels.Num()) * sizeof(FColor));
    Mip.BulkData.Unlock();
    // Upload the explicitly populated CPU mip. The UE 5.8 ImageData overload
    // could create the RHI resource before its initialization buffer survived
    // the rolling-history handoff, leaving a valid but black HUD texture.
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
        if (SignalIndex != INDEX_NONE
            && ResolutionDatasetFrameCounts[SignalIndex] < ResolutionDatasetFramesPerSignal)
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
                    if (bResolutionDatasetAutoExit
                        && ResolutionDatasetFrameCounts[0] >= ResolutionDatasetFramesPerSignal
                        && ResolutionDatasetFrameCounts[1] >= ResolutionDatasetFramesPerSignal
                        && ResolutionDatasetFrameCounts[2] >= ResolutionDatasetFramesPerSignal)
                    {
                        SimulatorLog(FString::Printf(
                            TEXT("resolution_dataset complete frames_per_signal=%d action=request_exit"),
                            ResolutionDatasetFramesPerSignal));
                        FPlatformMisc::RequestExit(true);
                    }
                }
            }
        }
    }
    if (!bInferenceEnabled)
    {
        EncodedFrames.Reset();
        EncodedFrameWidths.Reset();
        EncodedFrameHeights.Reset();
        EncodedParcelSafetySignals.Reset();
        EncodedFrameTextures.Reset();
        EncodedFrameTimes.Reset();
        EncodedFrameCaptureSeconds.Reset();
        return;
    }
    EncodedFrames.Add(MoveTemp(EncodedPng));
    EncodedFrameTextures.Add(CreateInferencePreviewTexture(Pixels, FrameWidth, FrameHeight));
    EncodedFrameWidths.Add(FrameWidth);
    EncodedFrameHeights.Add(FrameHeight);
    EncodedParcelSafetySignals.Add(EvaluateParcelSafetySignal());
    EncodedFrameTimes.Add(FDateTime::Now().ToString(TEXT("%H:%M:%S")));
    EncodedFrameCaptureSeconds.Add(FPlatformTime::Seconds());
    while (EncodedFrames.Num() > 32
        || (EncodedFrameCaptureSeconds.Num() > 2
            && EncodedFrameCaptureSeconds.Last() - EncodedFrameCaptureSeconds[0] > 6.0))
    {
        EncodedFrames.RemoveAt(0);
        EncodedFrameWidths.RemoveAt(0);
        EncodedFrameHeights.RemoveAt(0);
        EncodedParcelSafetySignals.RemoveAt(0);
        EncodedFrameTextures.RemoveAt(0);
        EncodedFrameTimes.RemoveAt(0);
        EncodedFrameCaptureSeconds.RemoveAt(0);
    }
    const bool bEvkRequest = ActiveBackend == TEXT("evk");
    const int32 RequiredFrames = bEvkRequest ? EvkTemporalFrameCount : 2;
    if (EncodedFrames.Num() < RequiredFrames)
    {
        return;
    }
    const int32 WindowStartIndex = 0;
    const double RequiredSpanSeconds = bEvkRequest
        ? EvkTemporalBaselineSeconds
        : HostTemporalBaselineSeconds;
    if (EncodedFrameCaptureSeconds.Last() - EncodedFrameCaptureSeconds[WindowStartIndex]
        < RequiredSpanSeconds - CaptureIntervalSeconds * 0.55f)
    {
        // Never duplicate a warm-up frame. Both transports receive genuinely
        // distinct chronological observations spanning their requested time.
        return;
    }
    SubmitInference();
}

void AQaiConveyorWorld::SubmitInference()
{
    const bool bEvkRequest = ActiveBackend == TEXT("evk");
    const int32 RequiredFrames = bEvkRequest ? EvkTemporalFrameCount : 2;
    if (!bInferenceEnabled
        || !bBackendHealthy
        || bInferenceBusy
        || EncodedFrames.Num() < RequiredFrames)
    {
        return;
    }
    const int32 NewestIndex = EncodedFrames.Num() - 1;
    int32 OldestIndex = INDEX_NONE;
    {
        // Both transports preserve their configured temporal baseline. Host
        // sends the endpoints; EVK additionally selects two evenly distributed
        // interior observations from the same buffered time range.
        const double TemporalBaselineSeconds = bEvkRequest
            ? EvkTemporalBaselineSeconds
            : HostTemporalBaselineSeconds;
        const double TargetSeconds = EncodedFrameCaptureSeconds[NewestIndex]
            - TemporalBaselineSeconds;
        double BestErrorSeconds = TNumericLimits<double>::Max();
        for (int32 FrameIndex = 0; FrameIndex < NewestIndex; ++FrameIndex)
        {
            const double ErrorSeconds = FMath::Abs(
                EncodedFrameCaptureSeconds[FrameIndex] - TargetSeconds);
            if (ErrorSeconds < BestErrorSeconds)
            {
                BestErrorSeconds = ErrorSeconds;
                OldestIndex = FrameIndex;
            }
        }
    }
    if (OldestIndex < 0
        || !EncodedFrameWidths.IsValidIndex(NewestIndex)
        || !EncodedFrameHeights.IsValidIndex(NewestIndex))
    {
        return;
    }
    const double ActualBaselineSeconds = EncodedFrameCaptureSeconds[NewestIndex]
        - EncodedFrameCaptureSeconds[OldestIndex];
    const int32 FrameWidth = EncodedFrameWidths[NewestIndex];
    const int32 FrameHeight = EncodedFrameHeights[NewestIndex];

    TArray<int32> SelectedIndices;
    TArray<FString> SelectedEncodedFrames;
    SelectedIndices.Reserve(RequiredFrames);
    SelectedEncodedFrames.Reserve(RequiredFrames);
    if (bEvkRequest)
    {
        // Select genuinely distinct observations nearest to evenly spaced
        // target times. Keeping the endpoints exact preserves the red test in
        // the newest frame and the full motion baseline.
        SelectedIndices.Add(OldestIndex);
        for (int32 SampleIndex = 1; SampleIndex < EvkTemporalFrameCount - 1; ++SampleIndex)
        {
            const double Alpha = static_cast<double>(SampleIndex)
                / static_cast<double>(EvkTemporalFrameCount - 1);
            const double TargetSeconds = FMath::Lerp(
                EncodedFrameCaptureSeconds[OldestIndex],
                EncodedFrameCaptureSeconds[NewestIndex],
                Alpha);
            const int32 MinimumIndex = SelectedIndices.Last() + 1;
            const int32 RemainingInteriorFrames = EvkTemporalFrameCount - SampleIndex - 1;
            const int32 MaximumIndex = NewestIndex - RemainingInteriorFrames;
            int32 BestIndex = MinimumIndex;
            double BestErrorSeconds = TNumericLimits<double>::Max();
            for (int32 FrameIndex = MinimumIndex; FrameIndex <= MaximumIndex; ++FrameIndex)
            {
                const double ErrorSeconds = FMath::Abs(
                    EncodedFrameCaptureSeconds[FrameIndex] - TargetSeconds);
                if (ErrorSeconds < BestErrorSeconds)
                {
                    BestErrorSeconds = ErrorSeconds;
                    BestIndex = FrameIndex;
                }
            }
            SelectedIndices.Add(BestIndex);
        }
        SelectedIndices.Add(NewestIndex);
    }
    else
    {
        SelectedIndices.Add(OldestIndex);
        SelectedIndices.Add(NewestIndex);
    }
    for (const int32 FrameIndex : SelectedIndices)
    {
        if (!EncodedFrames.IsValidIndex(FrameIndex)
            || !EncodedFrameWidths.IsValidIndex(FrameIndex)
            || !EncodedFrameHeights.IsValidIndex(FrameIndex)
            || EncodedFrameWidths[FrameIndex] != FrameWidth
            || EncodedFrameHeights[FrameIndex] != FrameHeight)
        {
            BackendStatus = TEXT("temporal frame dimensions changed");
            SimulatorLog(TEXT("inference_window_rejected reason=inconsistent_frame_dimensions"));
            return;
        }
        SelectedEncodedFrames.Add(EncodedFrames[FrameIndex]);
    }

    const float FramesPerSecond = bEvkRequest && ActualBaselineSeconds > UE_SMALL_NUMBER
        ? static_cast<float>(SelectedIndices.Num() - 1) / static_cast<float>(ActualBaselineSeconds)
        : 1.0f / CaptureIntervalSeconds;
    // Keep the physical-analysis instructions byte-for-byte equivalent across
    // host and EVK. Reason2 is substantially more reliable when it describes
    // parcel support naturally than when a constrained one-letter grammar
    // forces it to guess a code before it has stated what it sees.
    const FString Prompt = TEXT(
        "These are two chronological views from one fixed camera above a powered roller conveyor; the newest image is last.\n"
        "Analyze the newest image and consider only cardboard parcels that are on, falling from, or have fallen from the silver rollers.\n"
        "Describe which parcels are fully supported, which are partly supported and in danger of falling, and which have lost support and fallen or are visibly falling.\n"
        "Ordinary parcel rotation while following the U-shaped curve is supported motion, not danger.\n"
        "Use the older image only to track which parcels came from the conveyor. Ignore the forklift, people, shelves, pallets, floor cargo, billboard, and floor colors.\n"
        "End after a concise physical description; do not use one-letter codes.\n");

    // Pin the exact chronological source frames for HUD and diagnostics before
    // media preparation leaves the game thread. Capture continues at 4 FPS
    // while FFmpeg and inference work on this immutable window.
    SubmittedEncodedFrames = SelectedEncodedFrames;
    SubmittedFrameTextures.Reset(SelectedIndices.Num());
    SubmittedFrameTimes.Reset(SelectedIndices.Num());
    for (const int32 FrameIndex : SelectedIndices)
    {
        SubmittedFrameTextures.Add(EncodedFrameTextures[FrameIndex]);
        SubmittedFrameTimes.Add(EncodedFrameTimes[FrameIndex]);
    }
    // Pin deterministic support truth to the newest submitted RGB frame. It
    // remains diagnostic metadata only and is never part of the model input.
    SubmittedGroundTruthSignal = EncodedParcelSafetySignals.IsValidIndex(NewestIndex)
        ? EncodedParcelSafetySignals[NewestIndex]
        : GroundTruthSignal;
    FString SubmittedParcelDetails;
    EvaluateParcelSafetySignal(&SubmittedParcelDetails);
    SimulatorLog(FString::Printf(
        TEXT("inference_parcel_tracker frames=%d span_seconds=%.3f newest_truth=%s %s"),
        SelectedIndices.Num(),
        ActualBaselineSeconds,
        *SubmittedGroundTruthSignal,
        *SubmittedParcelDetails));

    bInferenceBusy = true;
    BackendStatus = bEvkRequest
        ? TEXT("preparing 2-frame video")
        : TEXT("submitting lossless image pair");
    RequestStartSeconds = FPlatformTime::Seconds();
    const uint32 RequestGeneration = InferenceCaptureGeneration;
    if (!bEvkRequest)
    {
        TArray<FString> MediaMimeTypes;
        MediaMimeTypes.Init(TEXT("image/png"), SelectedEncodedFrames.Num());
        int64 MediaBytes = 0;
        for (const FString& EncodedFrame : SelectedEncodedFrames)
        {
            MediaBytes += static_cast<int64>(FBase64::GetDecodedDataSize(EncodedFrame));
        }
        DispatchPreparedInference(
            MoveTemp(SelectedEncodedFrames),
            MoveTemp(MediaMimeTypes),
            Prompt,
            TEXT("two_separate_lossless_png_approximately_2s"),
            MoveTemp(SelectedIndices),
            ActualBaselineSeconds,
            FrameWidth,
            FrameHeight,
            MediaBytes,
            RequestGeneration);
        return;
    }

    const FString FfmpegOverride = FfmpegExecutableOverride;
    const bool bSaveDiagnostics = bSaveInferenceFrames;
    TWeakObjectPtr<AQaiConveyorWorld> WeakThis(this);
    Async(EAsyncExecution::ThreadPool,
        [WeakThis,
         SelectedEncodedFrames = MoveTemp(SelectedEncodedFrames),
         SelectedIndices = MoveTemp(SelectedIndices),
         Prompt,
         FrameWidth,
         FrameHeight,
         FramesPerSecond,
         ActualBaselineSeconds,
         RequestGeneration,
         FfmpegOverride,
         bSaveDiagnostics]() mutable
    {
        FPreparedInferenceMedia Media = BuildLosslessEvkVideo(
            SelectedEncodedFrames,
            FrameWidth,
            FrameHeight,
            FramesPerSecond,
            FfmpegOverride,
            bSaveDiagnostics);
        AsyncTask(ENamedThreads::GameThread,
            [WeakThis,
             Media = MoveTemp(Media),
             Prompt,
             SelectedIndices = MoveTemp(SelectedIndices),
             ActualBaselineSeconds,
             RequestGeneration]() mutable
        {
            AQaiConveyorWorld* Runtime = WeakThis.Get();
            if (!Runtime
                || Runtime->bIsShuttingDown
                || RequestGeneration != Runtime->InferenceCaptureGeneration)
            {
                return;
            }
            if (!Media.IsValid())
            {
                Runtime->bInferenceBusy = false;
                Runtime->BackendStatus = TEXT("media preparation failed");
                SimulatorLog(FString::Printf(
                    TEXT("inference_media_failed backend=%s error=\"%s\""),
                    *Runtime->ActiveBackend,
                    *SanitizeLogField(Media.Error)));
                return;
            }
            TArray<FString> MediaItems;
            MediaItems.Add(MoveTemp(Media.Base64));
            TArray<FString> MediaMimeTypes;
            MediaMimeTypes.Add(MoveTemp(Media.MimeType));
            Runtime->DispatchPreparedInference(
                MoveTemp(MediaItems),
                MoveTemp(MediaMimeTypes),
                Prompt,
                MoveTemp(Media.Transport),
                MoveTemp(SelectedIndices),
                ActualBaselineSeconds,
                Media.Width,
                Media.Height,
                Media.Bytes,
                RequestGeneration);
        });
    });
}

void AQaiConveyorWorld::DispatchPreparedInference(
    TArray<FString> EncodedMediaItems,
    TArray<FString> MediaMimeTypes,
    FString Prompt,
    FString TransportName,
    TArray<int32> SelectedIndices,
    double ActualBaselineSeconds,
    int32 MediaWidth,
    int32 MediaHeight,
    int64 MediaBytes,
    uint32 RequestGeneration)
{
    if (bIsShuttingDown
        || !bInferenceEnabled
        || !bBackendHealthy
        || RequestGeneration != InferenceCaptureGeneration)
    {
        bInferenceBusy = false;
        return;
    }
    if (EncodedMediaItems.IsEmpty()
        || EncodedMediaItems.Num() != MediaMimeTypes.Num())
    {
        bInferenceBusy = false;
        BackendStatus = TEXT("invalid inference media items");
        SimulatorLog(FString::Printf(
            TEXT("inference_media_failed backend=%s reason=invalid_item_count media=%d mime=%d"),
            *ActiveBackend,
            EncodedMediaItems.Num(),
            MediaMimeTypes.Num()));
        return;
    }

    // GenieX's native-video path expects a path/URL and does not safely
    // accept an embedded MP4 data URL. Upload the bounded clip to the EVK
    // bridge first; it returns an EVK-local file URI for the model request.
    if (ActiveBackend == TEXT("evk")
        && !EncodedMediaItems[0].StartsWith(TEXT("file://")))
    {
        UploadEvkMediaAndDispatch(
            MoveTemp(EncodedMediaItems[0]),
            MoveTemp(MediaMimeTypes[0]),
            MoveTemp(Prompt),
            MoveTemp(TransportName),
            MoveTemp(SelectedIndices),
            ActualBaselineSeconds,
            MediaWidth,
            MediaHeight,
            MediaBytes,
            RequestGeneration);
        return;
    }

    TArray<TSharedPtr<FJsonValue>> Content;
    for (int32 MediaIndex = 0; MediaIndex < EncodedMediaItems.Num(); ++MediaIndex)
    {
        const FString& EncodedMedia = EncodedMediaItems[MediaIndex];
        TSharedPtr<FJsonObject> Url = MakeShared<FJsonObject>();
        Url->SetStringField(
            TEXT("url"),
            EncodedMedia.StartsWith(TEXT("file://"))
                ? EncodedMedia
                : FString::Printf(
                    TEXT("data:%s;base64,%s"),
                    *MediaMimeTypes[MediaIndex],
                    *EncodedMedia));
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
    Body->SetNumberField(TEXT("max_completion_tokens"), 256);
    Body->SetBoolField(TEXT("enable_think"), false);
    Body->SetArrayField(TEXT("messages"), {MakeShared<FJsonValueObject>(Message)});
    FString BodyText;
    const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&BodyText);
    FJsonSerializer::Serialize(Body.ToSharedRef(), Writer);

    TSharedRef<IHttpRequest> Request = FHttpModule::Get().CreateRequest();
    InferenceRequest = Request;
    Request->SetURL(CurrentServerUrl() + TEXT("/v1/chat/completions"));
    Request->SetVerb(TEXT("POST"));
    Request->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    if (ActiveBackend == TEXT("evk"))
    {
        // The patched persistent GenieX service handles independent native
        // video requests most reliably on independent HTTP connections.
        Request->SetHeader(TEXT("Connection"), TEXT("close"));
    }
    Request->SetTimeout(ActiveBackend == TEXT("evk") ? 120.0f : 60.0f);
    Request->SetContentAsString(BodyText);
    if (!bLoggedFirstInferenceSubmission)
    {
        bLoggedFirstInferenceSubmission = true;
        SimulatorLog(FString::Printf(
            TEXT("inference_submitted backend=%s request_bytes=%d media_bytes=%lld frames=%d source_size=%dx%d media_size=%dx%d span_seconds=%.3f transport=%s prompt_profile=%s detector_projection=authored_endline preparation_ms=%.0f"),
            *ActiveBackend,
            BodyText.Len(),
            MediaBytes,
            SubmittedEncodedFrames.Num(),
            EncodedFrameWidths.IsEmpty() ? 0 : EncodedFrameWidths.Last(),
            EncodedFrameHeights.IsEmpty() ? 0 : EncodedFrameHeights.Last(),
            MediaWidth,
            MediaHeight,
            ActualBaselineSeconds,
            *TransportName,
            TEXT("omniverse_exact_two_image"),
            (FPlatformTime::Seconds() - RequestStartSeconds) * 1000.0));
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
        if (!bLoggedFirstInferencePreview)
        {
            bLoggedFirstInferencePreview = true;
            SimulatorLog(FString::Printf(
                TEXT("inference_preview_updated frames=%d oldest=%s newest=%s"),
                SubmittedFrameTextures.Num(),
                SubmittedFrameTimes.IsValidIndex(0) ? *SubmittedFrameTimes[0] : TEXT("-"),
                SubmittedFrameTimes.IsValidIndex(SubmittedFrameTimes.Num() - 1)
                    ? *SubmittedFrameTimes.Last()
                    : TEXT("-")));
        }
    }
}

void AQaiConveyorWorld::UploadEvkMediaAndDispatch(
    FString EncodedMedia,
    FString MediaMimeType,
    FString Prompt,
    FString TransportName,
    TArray<int32> SelectedIndices,
    double ActualWindowSeconds,
    int32 MediaWidth,
    int32 MediaHeight,
    int64 MediaBytes,
    uint32 RequestGeneration)
{
    TArray<uint8> VideoBytes;
    if (MediaMimeType != TEXT("video/mp4")
        || !FBase64::Decode(EncodedMedia, VideoBytes)
        || VideoBytes.IsEmpty())
    {
        bInferenceBusy = false;
        BackendStatus = TEXT("invalid EVK video payload");
        SimulatorLog(TEXT("evk_media_upload_failed reason=invalid_local_mp4"));
        return;
    }

    TSharedRef<IHttpRequest> Upload = FHttpModule::Get().CreateRequest();
    InferenceRequest = Upload;
    Upload->SetURL(EvkMediaBridgeUrl + TEXT("/v1/media"));
    Upload->SetVerb(TEXT("POST"));
    Upload->SetHeader(TEXT("Content-Type"), TEXT("video/mp4"));
    Upload->SetHeader(TEXT("Connection"), TEXT("close"));
    Upload->SetHeader(
        TEXT("X-Qai-Media-Id"),
        FString::Printf(
            TEXT("window-%s-%llu"),
            *FDateTime::UtcNow().ToString(TEXT("%Y%m%dT%H%M%S")),
            static_cast<unsigned long long>(FPlatformTime::Cycles64())));
    Upload->SetTimeout(20.0f);
    Upload->SetContent(VideoBytes);
    TWeakObjectPtr<AQaiConveyorWorld> WeakThis(this);
    Upload->OnProcessRequestComplete().BindLambda(
        [WeakThis,
         MediaMimeType = MoveTemp(MediaMimeType),
         Prompt = MoveTemp(Prompt),
         TransportName = MoveTemp(TransportName),
         SelectedIndices = MoveTemp(SelectedIndices),
         ActualWindowSeconds,
         MediaWidth,
         MediaHeight,
         MediaBytes,
         RequestGeneration](FHttpRequestPtr, FHttpResponsePtr Response, bool bSucceeded) mutable
        {
            AQaiConveyorWorld* Runtime = WeakThis.Get();
            if (!Runtime)
            {
                return;
            }
            Runtime->InferenceRequest.Reset();
            if (Runtime->bIsShuttingDown
                || RequestGeneration != Runtime->InferenceCaptureGeneration)
            {
                Runtime->bInferenceBusy = false;
                return;
            }
            const int32 ResponseCode = Response.IsValid() ? Response->GetResponseCode() : 0;
            FString MediaUrl;
            TSharedPtr<FJsonObject> Json;
            if (bSucceeded && Response.IsValid() && EHttpResponseCodes::IsOk(ResponseCode))
            {
                const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(
                    Response->GetContentAsString());
                if (FJsonSerializer::Deserialize(Reader, Json) && Json.IsValid())
                {
                    Json->TryGetStringField(TEXT("url"), MediaUrl);
                }
            }
            if (!MediaUrl.StartsWith(TEXT("file:///home/ubuntu/qai-conveyor/run/media/")))
            {
                Runtime->bInferenceBusy = false;
                Runtime->BackendStatus = FString::Printf(
                    TEXT("EVK media upload HTTP %d"), ResponseCode);
                SimulatorLog(FString::Printf(
                    TEXT("evk_media_upload_failed http_status=%d bridge=%s"),
                    ResponseCode,
                    *Runtime->EvkMediaBridgeUrl));
                return;
            }
            SimulatorLog(FString::Printf(
                TEXT("evk_media_uploaded bytes=%lld bridge=%s"),
                MediaBytes,
                *Runtime->EvkMediaBridgeUrl));
            TArray<FString> MediaItems;
            MediaItems.Add(MoveTemp(MediaUrl));
            TArray<FString> MediaMimeTypes;
            MediaMimeTypes.Add(MoveTemp(MediaMimeType));
            Runtime->DispatchPreparedInference(
                MoveTemp(MediaItems),
                MoveTemp(MediaMimeTypes),
                MoveTemp(Prompt),
                MoveTemp(TransportName) + TEXT("_evk_file_bridge"),
                MoveTemp(SelectedIndices),
                ActualWindowSeconds,
                MediaWidth,
                MediaHeight,
                MediaBytes,
                RequestGeneration);
        });
    if (!Upload->ProcessRequest())
    {
        InferenceRequest.Reset();
        bInferenceBusy = false;
        BackendStatus = TEXT("EVK media upload could not start");
        SimulatorLog(FString::Printf(
            TEXT("evk_media_upload_failed reason=request_start bridge=%s"),
            *EvkMediaBridgeUrl));
    }
    else
    {
        BackendStatus = TEXT("uploading lossless EVK video");
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
    const FString LowerContent = Content.ToLower();
    const auto ContainsAny = [&LowerContent](std::initializer_list<const TCHAR*> Phrases)
    {
        for (const TCHAR* Phrase : Phrases)
        {
            if (LowerContent.Contains(Phrase))
            {
                return true;
            }
        }
        return false;
    };

    // Parse the model's own natural-language physical assessment. Red is
    // checked first because a genuinely falling parcel may also be described
    // as tilted; amber is checked before the generic supported vocabulary.
    if (ContainsAny({
            TEXT("falling off"),
            TEXT("has fallen"),
            TEXT("have fallen"),
            TEXT("fallen off"),
            TEXT("lost support"),
            TEXT("no longer supported"),
            TEXT("below the conveyor"),
            TEXT("below the rollers"),
            TEXT("beside the conveyor"),
            TEXT("process of falling")}))
    {
        Proposed = TEXT("R");
    }
    else if (ContainsAny({
                 TEXT("in danger"),
                 TEXT("partly supported"),
                 TEXT("partially supported"),
                 TEXT("about to fall"),
                 TEXT("closer to the edge"),
                 TEXT("close to the edge"),
                 TEXT("near the edge"),
                 TEXT("overhang"),
                 TEXT("tilted"),
                 TEXT("tipping")}))
    {
        Proposed = TEXT("A");
    }
    else if (ContainsAny({
                 TEXT("fully supported"),
                 TEXT("supported by"),
                 TEXT("securely placed"),
                 TEXT("placed on the rollers"),
                 TEXT("positioned on the rollers"),
                 TEXT("resting on"),
                 TEXT("resting securely"),
                 TEXT("stable"),
                 TEXT("supported")}))
    {
        Proposed = TEXT("G");
    }
    else
    {
        // Compatibility fallback for a backend that still returns an explicit
        // uppercase code. Deliberately case-sensitive so the article "a" in a
        // narrative response cannot be mistaken for amber.
        const FRegexPattern StandaloneSignalPattern(TEXT("(?:^|[^A-Z])([GAR])(?:[^A-Z]|$)"));
        FRegexMatcher SignalMatcher(StandaloneSignalPattern, Content);
        while (SignalMatcher.FindNext())
        {
            Proposed = SignalMatcher.GetCaptureGroup(1);
        }
    }

    FString LoggedDescription = Content;
    LoggedDescription.ReplaceInline(TEXT("\r"), TEXT(" "));
    LoggedDescription.ReplaceInline(TEXT("\n"), TEXT(" "));
    LoggedDescription.ReplaceInline(TEXT("\""), TEXT("'"));
    LoggedDescription.TrimStartAndEndInline();
    if (LoggedDescription.Len() > 1000)
    {
        LoggedDescription = LoggedDescription.Left(997) + TEXT("...");
    }
    SimulatorLog(FString::Printf(
        TEXT("inference_model_description backend=%s parsed=%s text=\"%s\""),
        *ActiveBackend,
        Proposed.IsEmpty() ? TEXT("invalid") : *Proposed,
        *LoggedDescription));
    if (Proposed.IsEmpty())
    {
        BackendStatus = TEXT("invalid model response");
        UE_LOG(LogTemp, Warning, TEXT("Reason2 returned no recognizable parcel-support assessment"));
        SimulatorLog(FString::Printf(TEXT("inference_invalid_response backend=%s response_length=%d"), *ActiveBackend, Content.Len()));
        return;
    }
    if (bSaveInferenceFrames)
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
                const FString FrameRole = FString::Printf(
                    TEXT("frame-%02d-%s"),
                    FrameIndex + 1,
                    FrameIndex == 0
                        ? TEXT("oldest")
                        : (FrameIndex == SubmittedEncodedFrames.Num() - 1
                            ? TEXT("newest")
                            : TEXT("middle")));
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
