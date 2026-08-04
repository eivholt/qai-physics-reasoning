using UnrealBuildTool;

public class QaiConveyorEditorTarget : TargetRules
{
    public QaiConveyorEditorTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Editor;
        DefaultBuildSettings = BuildSettingsVersion.V7;
        IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
        ExtraModuleNames.Add("QaiConveyor");
        StaticAllocator = StaticAllocatorType.Ansi;
        bOverrideBuildEnvironment = true;
    }
}
