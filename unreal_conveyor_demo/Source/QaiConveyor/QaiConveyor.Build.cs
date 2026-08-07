using UnrealBuildTool;

public class QaiConveyor : ModuleRules
{
    public QaiConveyor(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "HTTP",
            "Json",
            "JsonUtilities",
            "ImageWrapper",
            "RenderCore",
            "RHI",
            "Slate"
        });

        // Use Unreal's already-created Microsoft GameInput interface for a
        // deterministic Windows gamepad poll. This bypasses controller-to-
        // local-player routing differences between editor and packaged builds.
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            PrivateDependencyModuleNames.Add("GameInputBase");
        }
    }
}
