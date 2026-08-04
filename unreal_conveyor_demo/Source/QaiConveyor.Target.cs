using UnrealBuildTool;

public class QaiConveyorTarget : TargetRules
{
    public QaiConveyorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        DefaultBuildSettings = BuildSettingsVersion.V7;
        IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
        ExtraModuleNames.Add("QaiConveyor");
        StaticAllocator = StaticAllocatorType.Ansi;
        bOverrideBuildEnvironment = true;
    }
}
