#pragma once

#include "Components/PrimitiveComponent.h"
#include "PhysicsEngine/BodyInstance.h"
#include "PhysicsProxy/SingleParticlePhysicsProxy.h"

// Only use the solver view inside AsyncPhysicsTickActor. UE 5.8 executes that
// callback with the game thread frozen; ordinary Tick keeps the component API.
// In particular, component transforms are presentation state, not substep state.
namespace QaiPhysics
{
class FBody
{
public:
    FBody(UPrimitiveComponent* InComponent, bool bInSolverStep)
        : Component(InComponent), bSolverStep(bInSolverStep)
    {
        if (bSolverStep && Component)
        {
            if (FBodyInstance* Instance = Component->GetBodyInstance())
            {
                auto Handle = Instance->GetBodyInstanceAsyncPhysicsTickHandle();
                if (Handle.IsValid())
                {
                    SolverBody = Handle.operator->();
                }
            }
        }
    }

    bool HasSolverBody() const { return SolverBody != nullptr; }
    FTransform GetComponentTransform() const
    {
        return SolverBody
            ? FTransform(SolverBody->R(), SolverBody->X(), Component->GetComponentScale())
            : Component->GetComponentTransform();
    }
    FVector GetComponentLocation() const { return GetComponentTransform().GetLocation(); }
    FQuat GetComponentQuat() const { return GetComponentTransform().GetRotation(); }
    FRotator GetComponentRotation() const { return GetComponentQuat().Rotator(); }
    FVector GetPhysicsLinearVelocity() const
    {
        return SolverBody ? FVector(SolverBody->V()) : Component->GetPhysicsLinearVelocity();
    }
    FVector GetPhysicsAngularVelocityInRadians() const
    {
        return SolverBody ? FVector(SolverBody->W()) : Component->GetPhysicsAngularVelocityInRadians();
    }
    FVector GetPhysicsAngularVelocityInDegrees() const
    {
        return GetPhysicsAngularVelocityInRadians() * (180.0 / UE_PI);
    }
    FVector CenterOfMassWorld() const
    {
        return SolverBody
            ? FVector(SolverBody->X() + SolverBody->R().RotateVector(SolverBody->CenterOfMass()))
            : Component->GetCenterOfMass();
    }
    FVector GetPhysicsLinearVelocityAtPoint(const FVector& Point) const
    {
        return SolverBody
            ? GetPhysicsLinearVelocity()
                + FVector::CrossProduct(GetPhysicsAngularVelocityInRadians(), Point - CenterOfMassWorld())
            : Component->GetPhysicsLinearVelocityAtPoint(Point);
    }
    void AddForce(const FVector& Force, FName BoneName = NAME_None, bool bAccelChange = false) const
    {
        if (!bSolverStep) { Component->AddForce(Force, BoneName, bAccelChange); }
        else if (SolverBody) { SolverBody->AddForce(Force); }
    }
    void AddForceAtLocation(const FVector& Force, const FVector& Point) const
    {
        if (!bSolverStep) { Component->AddForceAtLocation(Force, Point); }
        else if (SolverBody)
        {
            SolverBody->AddForce(Force);
            SolverBody->AddTorque(FVector::CrossProduct(Point - CenterOfMassWorld(), Force));
        }
    }
    void AddTorqueInRadians(const FVector& Torque, FName BoneName = NAME_None, bool bAccelChange = false) const
    {
        if (!bSolverStep) { Component->AddTorqueInRadians(Torque, BoneName, bAccelChange); }
        else if (SolverBody) { SolverBody->AddTorque(Torque); }
    }
    void SetPhysicsLinearVelocity(const FVector& Velocity) const
    {
        if (!bSolverStep) { Component->SetPhysicsLinearVelocity(Velocity); }
        else if (SolverBody) { SolverBody->SetV(Velocity); }
    }
    void SetWorldLocation(const FVector& Location, bool bSweep, FHitResult* Hit, ETeleportType Teleport) const
    {
        if (!bSolverStep) { Component->SetWorldLocation(Location, bSweep, Hit, Teleport); }
        else if (SolverBody) { SolverBody->SetX(Location); }
    }
    void WakeRigidBody() const
    {
        if (!bSolverStep) { Component->WakeRigidBody(); }
        else if (SolverBody && SolverBody->ObjectState() == Chaos::EObjectStateType::Sleeping)
        {
            SolverBody->SetObjectState(Chaos::EObjectStateType::Dynamic);
        }
    }
    void SetLinearDamping(float Value) const
    {
        if (!bSolverStep) { Component->SetLinearDamping(Value); }
        else if (SolverBody) { SolverBody->SetLinearEtherDrag(Value); }
    }
    void SetAngularDamping(float Value) const
    {
        if (!bSolverStep) { Component->SetAngularDamping(Value); }
        else if (SolverBody) { SolverBody->SetAngularEtherDrag(Value); }
    }

private:
    UPrimitiveComponent* Component = nullptr;
    Chaos::FRigidBodyHandle_Internal* SolverBody = nullptr;
    bool bSolverStep = false;
};
}
