#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "QaiConveyorWorld.generated.h"

class ASceneCapture2D;
class ARectLight;
class ASpotLight;
class AExponentialHeightFog;
class ALocalFogVolume;
class FRHIGPUTextureReadback;
class IHttpRequest;
class UMaterialInstanceDynamic;
class UMaterialInterface;
class UMaterialBillboardComponent;
class ULocalFogVolumeComponent;
class UPointLightComponent;
class UStaticMeshComponent;
class USpotLightComponent;
class UPrimitiveComponent;
class URectLightComponent;
class USceneComponent;
class UTexture2D;
class UTextureRenderTarget2D;
class UBoxComponent;
class UCapsuleComponent;
class USkeletalMeshComponent;
class UPhysicalMaterial;
class UPhysicsConstraintComponent;

UCLASS()
class QAICONVEYOR_API AQaiConveyorWorld : public AActor
{
    GENERATED_BODY()

public:
    AQaiConveyorWorld();

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void Tick(float DeltaSeconds) override;

    void DriveActiveForklift(float Throttle, float Steer, float Lift, bool bBrake, float DeltaSeconds);
    void CycleForklift(int32 Direction = 1);
    void SelectForklift(int32 Index);
    void ResetScene();
    void ToggleBackend();
    void ToggleInference();
    void ToggleCollisionDebug();

    FVector GetActiveForkliftLocation() const;
    FVector GetEvkLocation() const;
    int32 GetActiveForkliftIndex() const { return ActiveForklift; }
    float GetActiveSpeedMetersPerSecond() const;
    float GetActiveLiftMeters() const;
    const FString& GetStageStatus() const { return StageStatus; }
    const FString& GetStageError() const { return StageError; }
    const FString& GetGroundTruthSignal() const { return GroundTruthSignal; }
    const FString& GetModelSignal() const { return ModelSignal; }
    const FString& GetRawModelSignal() const { return RawModelSignal; }
    const FString& GetBackendName() const { return ActiveBackend; }
    const FString& GetBackendStatus() const { return BackendStatus; }
    const FString& GetModelName() const { return ActiveModel; }
    bool IsStageReady() const { return bStageReady; }
    bool IsInferenceEnabled() const { return bInferenceEnabled; }
    bool IsInferenceBusy() const { return bInferenceBusy; }
    double GetLastInferenceMilliseconds() const { return LastInferenceMilliseconds; }
    const FString& GetCollisionStatus() const { return CollisionStatus; }
    int32 GetCollisionBlockCount() const { return CollisionBlockCount; }
    int32 GetCollisionProxyCount() const { return CollisionObstacles.Num(); }
    bool IsCollisionDebugEnabled() const { return bDrawCollisionDebug; }
    int32 GetSubmittedInferenceFrameCount() const { return SubmittedFrameTextures.Num(); }
    UTexture2D* GetSubmittedInferenceFrame(int32 Index) const;
    FString GetSubmittedInferenceFrameTime(int32 Index) const;
    void RecordControllerInput(const FString& Details);

private:
    struct FFittedCollisionBox
    {
        enum class EShape : uint8
        {
            Box,
            Cylinder,
            Wedge,
        };

        FString Name;
        FVector LocalCenter = FVector::ZeroVector;
        FVector HalfExtent = FVector::ZeroVector;
        float Radius = 0.0f;
        float CylinderHalfLength = 0.0f;
        float WedgeTipThicknessCm = 0.5f;
        EShape Shape = EShape::Box;
        bool bLiftDriven = false;
        bool bWedgeTipAtPositiveX = true;

        bool IsCylinder() const { return Shape == EShape::Cylinder; }
        bool IsWedge() const { return Shape == EShape::Wedge; }
        float GetWedgeTopLocalZ(float LocalX) const
        {
            if (!IsWedge() || HalfExtent.X <= UE_KINDA_SMALL_NUMBER)
            {
                return HalfExtent.Z;
            }
            const float BackwardFraction = bWedgeTipAtPositiveX
                ? (HalfExtent.X - LocalX) / (2.0f * HalfExtent.X)
                : (LocalX + HalfExtent.X) / (2.0f * HalfExtent.X);
            const float Thickness = FMath::Lerp(
                WedgeTipThicknessCm,
                2.0f * HalfExtent.Z,
                FMath::Clamp(BackwardFraction, 0.0f, 1.0f));
            return -HalfExtent.Z + Thickness;
        }
        FVector GetCollisionHalfExtent() const
        {
            return IsCylinder()
                ? FVector(Radius, CylinderHalfLength, Radius)
                : HalfExtent;
        }
    };

    struct FForkliftRuntime
    {
        TWeakObjectPtr<USceneComponent> Root;
        TWeakObjectPtr<USceneComponent> LiftAssembly;
        TWeakObjectPtr<UPrimitiveComponent> LiftVisual;
        TWeakObjectPtr<USceneComponent> Pallet;
        TWeakObjectPtr<USceneComponent> Carton;
        FTransform InitialRoot;
        FVector InitialLiftRelativeLocation = FVector::ZeroVector;
        float InitialLiftVisualCenterZ = 0.0f;
        float GroundingOffsetCm = 0.0f;
        FTransform PalletRelative;
        FTransform CartonRelative;
        TWeakObjectPtr<USceneComponent> Wheels[4];
        TWeakObjectPtr<USceneComponent> WheelPivots[4];
        FVector WheelPivotInitialRelativeLocation[4] = {
            FVector::ZeroVector,
            FVector::ZeroVector,
            FVector::ZeroVector,
            FVector::ZeroVector};
        FQuat WheelPivotInitialRelative[4];
        TArray<TWeakObjectPtr<USceneComponent>> LiftDrivenComponents;
        TArray<FTransform> LiftDrivenRelativeTransforms;
        TArray<FFittedCollisionBox> CollisionBoxes;
        TArray<TWeakObjectPtr<UBoxComponent>> ChaosCollisionComponents;
        // Authoritative Chaos vehicle state. The imported USD subtree is
        // presentation-only and follows these bodies; it never drives them.
        TWeakObjectPtr<UBoxComponent> ChaosChassis;
        TWeakObjectPtr<UBoxComponent> ChaosCarriage;
        TWeakObjectPtr<UPhysicsConstraintComponent> ChaosLiftConstraint;
        FTransform InitialChaosChassis = FTransform::Identity;
        FTransform InitialChaosCarriage = FTransform::Identity;
        FVector ChaosChassisLocalCenter = FVector::ZeroVector;
        FVector ChaosCarriageLocalCenter = FVector::ZeroVector;
        float SuspensionCompressionCm[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float WheelNormalLoads[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int32 PalletDynamicBody = INDEX_NONE;
        int32 CartonDynamicBody = INDEX_NONE;
        float SpeedCmPerSecond = 0.0f;
        // Keyboard axes request full lock immediately; this is the resolved
        // steering-rack position used by vehicle motion and wheel visuals.
        float SteeringInput = 0.0f;
        // Actual kinematic surface motion from the last fixed step. This is
        // intentionally derived from the resolved root pose rather than the
        // requested wheel speed: a blocked forklift must not drag cargo while
        // its chassis is stationary.
        FVector SurfaceLinearVelocityCmPerSecond = FVector::ZeroVector;
        float SurfaceYawVelocityDegreesPerSecond = 0.0f;
        // Hydraulic command target and measured Chaos-carriage travel are
        // separate so HUD/debug geometry follows the physical fork assembly
        // even while the drive is catching up under load.
        float LiftCm = 0.0f;
        float ActualLiftCm = 0.0f;
        float LiftDeltaCm = 0.0f;
        bool bLiftUpperStopLatched = false;
        float WheelAngleDegrees = 0.0f;
        float RideHeightOffsetCm = 0.0f;
        float MaximumRideHeightCm = 0.0f;
        float WheelSupportOffsetsCm[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float BodyPitchDegrees = 0.0f;
        float BodyRollDegrees = 0.0f;
        float PitchVelocityDegrees = 0.0f;
        float RollVelocityDegrees = 0.0f;
        float MaximumAbsoluteTipDegrees = 0.0f;
        float ChassisMassKg = 3600.0f;
        float CombinedMassKg = 3600.0f;
        float PreviousSpeedCmPerSecond = 0.0f;
        FVector BaseCenterOfMassLocalCm = FVector(-18.0f, 0.0f, 58.0f);
        FVector CombinedCenterOfMassLocalCm = FVector(-18.0f, 0.0f, 58.0f);
        FVector ChassisInertiaTensorKgCm2 = FVector(10000000.0f);
        bool bWheelClimbActive = false;
        bool bForkReactionActive = false;
        float LastBlockedDriveSign = 0.0f;
        FString LastBlockedObstacle;
    };

    struct FWorkerRuntime
    {
        TWeakObjectPtr<USceneComponent> Root;
        TWeakObjectPtr<UCapsuleComponent> ChaosBody;
        TWeakObjectPtr<USceneComponent> VisualPivot;
        TWeakObjectPtr<USkeletalMeshComponent> SkeletalMesh;
        TArray<FName> FootBones;
        float SoleBelowLowestFootBoneCm = 0.0f;
        bool bLoggedFootGroundingRuntime = false;
        FTransform InitialTransform;
        FTransform InitialChaosTransform = FTransform::Identity;
        FTransform InitialVisualRelativeTransform = FTransform::Identity;
        TArray<FVector> Waypoints;
        TArray<float> DwellSeconds;
        FQuat HeadingOffset = FQuat::Identity;
        int32 DestinationIndex = 1;
        int32 RouteDirection = 1;
        int32 RouteReversalCount = 0;
        int32 AvoidanceTurnSign = 1;
        float DwellRemaining = 0.0f;
        float BlockedSeconds = 0.0f;
        float InitialHeadingYawDegrees = 0.0f;
        float TargetHeadingYawDegrees = 0.0f;
        float PendingHeadingYawDegrees = 0.0f;
        float PendingHeadingSeconds = 0.0f;
        float VisualHeadingYawDegrees = 0.0f;
        float MaximumVisualTurnRateDegreesPerSecond = 0.0f;
        float RouteReversalCooldownSeconds = 0.0f;
        float WorkerYieldRemainingSeconds = 0.0f;
        int32 SeparationEvents = 0;
        int32 RightOfWayYieldCount = 0;
    };

    struct FCollisionObstacle
    {
        FString Name;
        FVector Center = FVector::ZeroVector;
        FVector HalfExtent = FVector::ZeroVector;
        FQuat Rotation = FQuat::Identity;
    };

    struct FPropPhysicsProfile
    {
        FName Surface = TEXT("generic");
        FVector CenterOfMassLocalOffset = FVector::ZeroVector;
        FVector InertiaTensorKgCm2 = FVector(1000.0f);
        float MassKg = 1.0f;
        float StaticFriction = 0.60f;
        float DynamicFriction = 0.45f;
        float Restitution = 0.10f;
        float AirLinearDamping = 0.08f;
        float AirAngularDamping = 0.12f;
        float GroundAngularDamping = 4.0f;
        float ImpactResponseScale = 1.0f;
        float AngularResponseScale = 1.0f;
        float SleepLinearSpeedCm = 1.0f;
        float SleepAngularSpeedDegrees = 1.0f;
    };

    struct FDynamicBoxRuntime
    {
        FString Name;
        TWeakObjectPtr<USceneComponent> Root;
        FTransform InitialTransform;
        FTransform ForkliftRelativeTransform;
        FVector HalfExtent = FVector::ZeroVector;
        FVector LinearVelocity = FVector::ZeroVector;
        FVector AngularVelocityDegrees = FVector::ZeroVector;
        // The deterministic profile supplies material response without a
        // second, competing Chaos body on the authored visual hierarchy.
        FPropPhysicsProfile Physics;
        float ForkSupportCooldownSeconds = 0.0f;
        float ForkUnsupportedSeconds = 0.0f;
        int32 SupportedForklift = INDEX_NONE;
        int32 InitialSupportedForklift = INDEX_NONE;
        int32 SupportBodyIndex = INDEX_NONE;
        int32 InitialSupportBodyIndex = INDEX_NONE;
        float LastSupportCoverage = 1.0f;
        bool bShelfParcel = false;
        bool bPallet = false;
        bool bEvkProp = false;
        TArray<TWeakObjectPtr<UBoxComponent>> CompoundCollisionComponents;
        bool bAwake = false;
        bool bLoggedContact = false;
        bool bLoggedLedgeRelease = false;
    };

    void LoadRuntimeConfig();
    void BindNativeComponents();
    AActor* FindTaggedActor(const FName Tag) const;
    USceneComponent* FindTaggedComponent(const FName Tag) const;
    static void MakeMovable(USceneComponent* Component);
    bool WouldForkliftsOverlap(int32 MovingIndex, const FVector& CandidateLocation, const FRotator& CandidateRotation) const;
    bool WouldForkliftCollide(
        int32 MovingIndex,
        const FVector& CandidateLocation,
        const FRotator& CandidateRotation,
        FString& OutObstacle) const;
    bool CanWorkerOccupy(int32 MovingIndex, const FVector& CandidateLocation) const;
    float GetWorkerStaticClearanceCm(const FVector& CandidateLocation) const;
    int32 BuildCollisionGuard();
    bool ValidateCollisionGuard();
    void LogStageBindingFailure(const FString& Details);
    void UpdateCollisionStatus(int32 ForkliftIndex, const FString& Obstacle);
    void DrawCollisionDebug() const;
    bool ApplyWestShelfPlacement();
    void ConfigureForkliftMastMaterials();
    void ApplyForkliftPaintVariant();
    bool ConfigureInferenceScene();

    void FixedSimulationStep(float StepSeconds);
    void TickResolutionDataset(float DeltaSeconds);
    void ApplyResolutionDatasetVariant();
    void SimulateForklifts(float StepSeconds);
    void SimulateDynamicBoxes(float StepSeconds);
    void SimulateConveyor(float StepSeconds);
    void ConfigureAuthoritativeChaosPhysics();
    void ConfigureChaosPhysics();
    void SimulateChaosForklifts(float DeltaSeconds);
    void SimulateChaosConveyor(float StepSeconds);
    void SimulateChaosForkContacts(float StepSeconds);
    void MaintainChaosGravity();
    void SyncChaosTelemetry();
    void SimulateWorkers(float StepSeconds);
    void UpdateCargo(FForkliftRuntime& Forklift);
    void UpdateSafetySignal();
    void UpdateStackLights();
    void TickStackLightRig(float DeltaSeconds);
    void ConfigureStackLightRig();
    void ConfigureEvkProp();
    void TickEvkLeds(float DeltaSeconds);
    FPropPhysicsProfile BuildPropPhysicsProfile(
        const FString& Kind,
        const FVector& HalfExtent,
        int32 Variant) const;

    void SetupInferenceCapture();
    void CancelActiveInference();
    void TickInferenceCapture(float DeltaSeconds);
    void QueueCapture();
    void ReadbackAndCompressCapture();
    void OnFrameEncoded(
        FString EncodedPng,
        TArray<FColor> Pixels,
        int32 FrameWidth,
        int32 FrameHeight,
        uint32 CaptureGeneration);
    UTexture2D* CreateInferencePreviewTexture(
        const TArray<FColor>& Pixels,
        int32 FrameWidth,
        int32 FrameHeight);
    void SubmitInference();
    void DispatchPreparedInference(
        TArray<FString> EncodedMediaItems,
        TArray<FString> MediaMimeTypes,
        FString Prompt,
        FString TransportName,
        TArray<int32> SelectedIndices,
        double ActualWindowSeconds,
        int32 MediaWidth,
        int32 MediaHeight,
        int64 MediaBytes,
        uint32 RequestGeneration);
    void UploadEvkMediaAndDispatch(
        FString EncodedMedia,
        FString MediaMimeType,
        FString Prompt,
        FString TransportName,
        TArray<int32> SelectedIndices,
        double ActualWindowSeconds,
        int32 MediaWidth,
        int32 MediaHeight,
        int64 MediaBytes,
        uint32 RequestGeneration);
    void ProbeBackend();
    void HandleModelResponse(const FString& Body, bool bSucceeded, int32 ResponseCode);
    FString CurrentServerUrl() const;
    FString CurrentModelName() const;

    UPROPERTY()
    TObjectPtr<ASceneCapture2D> CaptureActor;

    UPROPERTY()
    TObjectPtr<UTextureRenderTarget2D> CaptureTarget;

    TSharedPtr<FRHIGPUTextureReadback> CaptureReadback;
    TSharedPtr<IHttpRequest, ESPMode::ThreadSafe> InferenceRequest;
    TSharedPtr<IHttpRequest, ESPMode::ThreadSafe> ProbeRequest;
    uint32 InferenceCaptureGeneration = 1;
    uint32 PendingCaptureGeneration = 0;
    int32 CaptureRenderWidth = 512;
    int32 CaptureRenderHeight = 288;
    int32 HostCaptureWidth = 512;
    int32 HostCaptureHeight = 288;
    int32 PendingCaptureOutputWidth = 384;
    int32 PendingCaptureOutputHeight = 216;

    FForkliftRuntime Forklifts[2];
    TArray<TWeakObjectPtr<USceneComponent>> Parcels;
    TArray<FTransform> ParcelInitialTransforms;
    TArray<float> ParcelPhases;
    TArray<FQuat> ParcelHeadingOffsets;
    TArray<float> ParcelDistances;
    TArray<float> ParcelVerticalPositions;
    TArray<float> ParcelVerticalVelocities;
    TArray<float> ParcelHalfHeights;
    TArray<FVector> ParcelHalfExtents;
    TArray<FPropPhysicsProfile> ParcelPhysicsProfiles;
    TArray<float> ParcelPitchDegrees;
    TArray<float> ParcelRollDegrees;
    TArray<float> ParcelPitchVelocities;
    TArray<float> ParcelRollVelocities;
    TArray<float> ParcelYawVelocities;
    TArray<bool> ParcelGrounded;
    TArray<FVector> ParcelLinearVelocities;
    TArray<int32> ParcelSupportedForklifts;
    TArray<bool> ParcelForkContactsLogged;
    TArray<TWeakObjectPtr<UPrimitiveComponent>> StackLensComponents[3];
    TArray<TWeakObjectPtr<UMaterialInstanceDynamic>> StackLensMaterials[3];
    TArray<TWeakObjectPtr<UMaterialBillboardComponent>> StackBillboardComponents[3];
    TArray<TWeakObjectPtr<UMaterialInstanceDynamic>> StackBillboardMaterials[3];
    TArray<TWeakObjectPtr<ULocalFogVolumeComponent>> StackHaloFogComponents[3];
    TArray<TWeakObjectPtr<USpotLightComponent>> StackWallSpotLights[3];
    TArray<TWeakObjectPtr<USpotLightComponent>> StackRoomSpotLights[3];
    TArray<TWeakObjectPtr<UPointLightComponent>> StackEmitterLights[3];
    TArray<TWeakObjectPtr<URectLightComponent>> StackWashLights;
    TArray<float> StackWashMultipliers;

    UPROPERTY()
    TArray<TObjectPtr<ARectLight>> RuntimeStackWashActors;

    UPROPERTY()
    TObjectPtr<AExponentialHeightFog> RuntimeSignalFogActor;

    UPROPERTY()
    TArray<TObjectPtr<ALocalFogVolume>> RuntimeStackHaloActors;

    UPROPERTY()
    TArray<TObjectPtr<ASpotLight>> RuntimeStackWallSpotActors;

    UPROPERTY()
    TArray<TObjectPtr<ASpotLight>> RuntimeStackRoomSpotActors;

    UPROPERTY()
    TArray<TObjectPtr<UMaterialBillboardComponent>> RuntimeStackBillboards;

    UPROPERTY()
    TArray<TObjectPtr<UStaticMeshComponent>> RuntimeEvkLedMeshes;

    UPROPERTY()
    TArray<TObjectPtr<UPointLightComponent>> RuntimeEvkLedLights;

    UPROPERTY()
    TArray<TObjectPtr<UMaterialInstanceDynamic>> RuntimeEvkLedMaterials;

    UPROPERTY()
    TArray<TObjectPtr<UMaterialInterface>> RuntimeForkliftPaintMaterials;

    UPROPERTY()
    TArray<TObjectPtr<UStaticMeshComponent>> RuntimeSafetyBorderComponents;

    UPROPERTY()
    TArray<TObjectPtr<UMaterialInstanceDynamic>> RuntimeInferenceSceneMaterials;

    UPROPERTY()
    TObjectPtr<ARectLight> RuntimeInferenceZoneFillActor;

    UPROPERTY()
    TArray<TObjectPtr<ALocalFogVolume>> RuntimeEvkLedFogActors;

    TArray<TWeakObjectPtr<ULocalFogVolumeComponent>> EvkLedFogComponents;
    TArray<float> EvkLedTimers;
    TArray<bool> EvkLedStates;
    FRandomStream EvkLedRandom;
    TWeakObjectPtr<USceneComponent> EvkVisualRoot;
    int32 EvkDynamicBody = INDEX_NONE;
    TWeakObjectPtr<USceneComponent> PackingTableTrayVisualRoot;
    int32 PackingTableTrayDynamicBody = INDEX_NONE;
    FBox PackingTableDesktopBounds = FBox(ForceInit);
    FBox PackingTableTrayBounds = FBox(ForceInit);
    // Final visible room footprint plus the supported apron leading to the
    // two distant invisible perimeter walls.
    FBox WarehouseFloorSupportBounds = FBox(ForceInit);
    // Authoritative world-space footprint and surface height of the visible
    // red safety mat. Ground truth and wheel support both use this same box so
    // their boundary cannot drift away from the rendered zone.
    FBox InferenceRedZoneBounds = FBox(ForceInit);

    TArray<TWeakObjectPtr<UPrimitiveComponent>> AuthoredCollisionComponents;
    TArray<FCollisionObstacle> CollisionObstacles;
    TArray<FDynamicBoxRuntime> DynamicBoxes;
    UPROPERTY(Transient)
    TArray<TObjectPtr<UBoxComponent>> RuntimeChaosBodies;
    UPROPERTY(Transient)
    TArray<TObjectPtr<UBoxComponent>> RuntimeChaosStaticColliders;
    UPROPERTY(Transient)
    TArray<TObjectPtr<UBoxComponent>> RuntimeChaosForkliftColliders;
    UPROPERTY(Transient)
    TArray<TObjectPtr<UPhysicsConstraintComponent>> RuntimeChaosConstraints;
    UPROPERTY(Transient)
    TArray<TObjectPtr<UBoxComponent>> RuntimeChaosConveyorBodies;
    UPROPERTY(Transient)
    TArray<TObjectPtr<UCapsuleComponent>> RuntimeChaosWorkerBodies;
    UPROPERTY(Transient)
    TArray<TObjectPtr<UPhysicalMaterial>> RuntimeChaosMaterials;
    UPROPERTY(Transient)
    TArray<TObjectPtr<AActor>> RuntimeChaosActors;
    TWeakObjectPtr<USceneComponent> DetectorCamera;
    FWorkerRuntime Workers[2];

    FString StageStatus = TEXT("Binding optimized native warehouse assets");
    FString StageError;
    FString CollisionStatus = TEXT("initializing");
    FString LastCollisionObstacle;
    FString LastLoggedStageFailure;
    FString LastLoggedInferenceState;
    FString GroundTruthSignal = TEXT("G");
    // Live deterministic motion truth is intentionally debounced so tiny
    // Chaos suspension/contact jitter cannot flash the HUD between G and A.
    // Inference requests use their exact captured transform window below.
    bool bGroundTruthMovingLatched = false;
    float GroundTruthMovingEvidenceSeconds = 0.0f;
    float GroundTruthStationaryEvidenceSeconds = 0.0f;
    FString RawModelSignal = TEXT("-");
    FString ModelSignal = TEXT("-");
    FString BackendStatus = TEXT("not checked");
    FString ActiveBackend = TEXT("host");
    FString ActiveModel;
    FString HostServerUrl = TEXT("http://127.0.0.1:18080");
    FString HostModel = TEXT("Cosmos-Reason2-2B-BF16.gguf");
    FString EvkServerUrl = TEXT("http://127.0.0.1:18181");
    FString EvkMediaBridgeUrl;
    FString EvkModel = TEXT("local/cosmos-reason2-2b");

    TArray<FString> EncodedFrames;
    // The host submits two original lossless PNGs about two seconds apart.
    // The EVK path selects the same two chronological observations as host
    // inference across the two-second buffer, then creates its pixel-lossless
    // RGB H.264 clip for the native-video endpoint.
    TArray<int32> EncodedFrameWidths;
    TArray<int32> EncodedFrameHeights;
    // A transform observation is captured at the same instant as each RGB
    // frame. It is used only as the Omniverse-style consistency tracker; it is
    // never included in the Reason2 request.
    TArray<FTransform> EncodedForkliftFrameTransforms;
    UPROPERTY(Transient)
    TArray<TObjectPtr<UTexture2D>> EncodedFrameTextures;
    TArray<FString> EncodedFrameTimes;
    TArray<double> EncodedFrameCaptureSeconds;

    TArray<FString> SubmittedEncodedFrames;

    UPROPERTY(Transient)
    TArray<TObjectPtr<UTexture2D>> SubmittedFrameTextures;
    TArray<FString> SubmittedFrameTimes;
    FString SubmittedGroundTruthSignal = TEXT("G");
    bool bSubmittedForkliftMotion = false;
    bool bSubmittedRedOverlap = false;
    float CommandThrottle = 0.0f;
    float CommandSteer = 0.0f;
    float CommandLift = 0.0f;
    float FixedAccumulator = 0.0f;
    float ComponentBindAccumulator = 0.25f;
    float CaptureAccumulator = 0.0f;
    float CaptureIntervalSeconds = 0.25f;
    float HostTemporalBaselineSeconds = 2.0f;
    float EvkTemporalBaselineSeconds = 2.0f;
    float BackendProbeAccumulator = 0.0f;
    // Runtime validation control. Zero stops the physical conveyor motor and
    // its initial/reset velocities without hiding any belt or parcel geometry.
    float ConveyorSpeedScale = 1.0f;
    float ConveyorDistanceCm = 0.0f;
    float ConveyorSurfaceZCm = 116.0f;
    float StackSignalWeights[3] = {0.0f, 0.0f, 0.0f};
    float StackTargetWeights[3] = {0.0f, 0.0f, 0.0f};
    double RequestStartSeconds = 0.0;
    double LastInferenceMilliseconds = 0.0;
    double LastStageFailureLogSeconds = -60.0;
    double LastBackendProbeLogSeconds = -60.0;
    double LastInferenceSummaryLogSeconds = -60.0;
    int32 ActiveForklift = 0;
    int32 CollisionBlockCount = 0;
    int32 InferenceSuccessCount = 0;
    int32 ParcelImpactCount = 0;
    int32 ParcelForkContactCount = 0;
    int32 CaptureFrameCount = 0;
    int32 EvkTemporalFrameCount = 2;
    bool bCommandBrake = false;
    bool bDrawCollisionDebug = false;
    bool bStageReady = false;
    bool bInferenceEnabled = true;
    bool bBackendHealthy = false;
    bool bCaptureReadbackPending = false;
    bool bEncodingFrame = false;
    bool bInferenceBusy = false;
    bool bIsShuttingDown = false;
    bool bLoggedFirstCapture = false;
    bool bLoggedFirstReadback = false;
    bool bLoggedFirstEncodedFrame = false;
    bool bLoggedFirstInferenceSubmission = false;
    bool bLoggedFirstInferencePreview = false;
    bool bLoggedFirstCaptureLuminance = false;
    bool bLoggedCaptureUnavailable = false;
    bool bLoggedChaosConveyorDrive = false;
    bool bSaveInferenceFrames = false;
    FString FfmpegExecutableOverride;
    bool bHasLoggedBackendProbe = false;
    bool bLastLoggedBackendHttpReady = false;
    bool bLastLoggedBackendModelReady = false;
    bool bRuntimeMotionTest = false;
    bool bPhysicsContactTest = false;
    bool bPhysicsContactTestInitialized = false;
    bool bShelfForkTest = false;
    bool bShelfForkTestInitialized = false;
    bool bEvkPhysicsTest = false;
    bool bEvkPhysicsTestInitialized = false;
    bool bForkliftClimbTest = false;
    bool bForkliftClimbTestInitialized = false;
    bool bForkEdgeBalanceTest = false;
    bool bForkEdgeBalanceTestInitialized = false;
    bool bForkWedgeTest = false;
    bool bForkWedgeTestInitialized = false;
    bool bResetLiftTest = false;
    bool bResetLiftTestTriggered = false;
    bool bHudScreenshotTest = false;
    bool bWorkerSoakTest = false;
    bool bWorkerSoakTestReported = false;
    bool bRuntimeMotionTestCaptured = false;
    bool bPresentation4K = false;
    bool bPresentation4KCaptured = false;
    bool bResolutionDataset = false;
    bool bResolutionDatasetInitialized = false;
    bool bResolutionDatasetWhiteFloor = false;
    bool bResolutionDatasetHideWorkers = false;
    bool bResolutionDatasetHideParcels = false;
    bool bResolutionDatasetAutoExit = false;
    bool bInteractiveValidationUnloaded = false;
    bool bInteractiveValidationUnloadedApplied = false;
    bool bInteractiveDriveValidation = false;
    int32 InteractiveDriveValidationPhase = INDEX_NONE;
    float InteractiveDriveValidationPhaseSeconds = 0.0f;
    int32 InteractiveDriveValidationLastTelemetrySecond = INDEX_NONE;
    FString ResolutionDatasetVariant = TEXT("baseline");
    // Optional sensor-only scene ablation. "minimal" renders only the active
    // forklift, room shell/floors, red safety mat/border, and conveyor. Extra
    // comma-separated groups can be restored without changing game physics.
    FString InferenceAblationVariant = TEXT("full");
    // Sensor optics/composition experiment. "authored-wide" uses the original
    // Omniverse DetectorEndline lens/off-axis projection and moves that fixed
    // camera backward for coverage without changing player view or physics.
    FString InferenceCameraVariant = TEXT("authored-wide");
    FString ForkliftPaintVariant = TEXT("isaac-yellow");
    int32 ResolutionDatasetFramesPerSignal = 10;
    bool bStackLightRigInitialized = false;
    bool bLoggedFirstParcelContact = false;
    bool bWestShelfPlacementApplied = false;
    bool bForkliftMastMaterialsConfigured = false;
    bool bInferenceSceneConfigured = false;
    bool bChaosPhysicsActive = false;
    float ChaosConveyorStartupGraceSeconds = 0.0f;
    float ChaosConveyorMotorRampSeconds = 0.0f;
    bool bChaosConveyorBodiesActivated = false;
    float RuntimeMotionTestElapsed = 0.0f;
    float PhysicsContactInitialParcelZ = 0.0f;
    float PhysicsContactMaximumParcelLiftCm = 0.0f;
    float ShelfForkTestInitialBodyZ = 0.0f;
    FVector ShelfForkTestInitialBodyLocation = FVector::ZeroVector;
    int32 ShelfForkTestBodyIndex = INDEX_NONE;
    float EvkPhysicsTestInitialZ = 0.0f;
    FVector EvkPhysicsTestInitialLocation = FVector::ZeroVector;
    int32 ForkliftClimbTestPalletBody = INDEX_NONE;
    int32 ForkEdgeBalanceTestPalletBody = INDEX_NONE;
    float ForkEdgeBalanceTestInitialZ = 0.0f;
    int32 ForkWedgeTestBodyIndex = INDEX_NONE;
    float ForkWedgeTestInitialZ = 0.0f;
    FVector ForkWedgeTestInitialLocation = FVector::ZeroVector;
    float ForkWedgeTestMaximumZ = 0.0f;
    float ResetLiftTestLiftBeforeResetCm = 0.0f;
    float ResetLiftTestMaximumLiftAfterResetCm = 0.0f;
    float WorkerSoakTestElapsed = 0.0f;
    float WorkerMinimumStaticClearanceCm[2] = {
        TNumericLimits<float>::Max(), TNumericLimits<float>::Max()};
    float WorkerMinimumPairClearanceCm = TNumericLimits<float>::Max();
    float WorkerConflictElapsedSeconds = 0.0f;
    int32 WorkerRightOfWayIndex = INDEX_NONE;
    int32 WorkerNextRightOfWayIndex = 0;
    bool bWorkerConflictHeadOn = false;
    float Presentation4KElapsed = 0.0f;
    float ResolutionDatasetElapsed = 0.0f;
    int32 ResolutionDatasetPhase = INDEX_NONE;
    int32 ResolutionDatasetLastTelemetrySecond = INDEX_NONE;
    int32 ResolutionDatasetLastForceTelemetrySecond = INDEX_NONE;
    int32 ResolutionDatasetFrameCounts[3] = {0, 0, 0};
    FTransform ResolutionDatasetInitialForkliftTransform = FTransform::Identity;
    FTransform ResolutionDatasetInitialChaosChassis = FTransform::Identity;
    FTransform ResolutionDatasetInitialChaosCarriage = FTransform::Identity;
    FString LastLoggedStackSignal;
};
