"""Remove generated IQ9 CAD intermediates after the optimized proxy bake."""

import unreal


for directory in ("/Game/IQ9EVK/CADSource", "/Game/IQ9EVK/CADInspect"):
    removed = unreal.EditorAssetLibrary.delete_directory(directory)
    unreal.log(f"IQ9EVK_CAD_CLEANUP directory={directory} removed={removed}")

for asset in (
    "/Game/IQ9EVK/Runtime/SM_SM_IQ9_EVK_Proxy",
    "/Game/IQ9EVK/Runtime/MI_SM_IQ9_EVK_Proxy",
    "/Game/IQ9EVK/Runtime/T_SM_IQ9_EVK_Proxy_BaseColor",
    "/Game/IQ9EVK/Runtime/T_SM_IQ9_EVK_Proxy_Normal",
    "/Game/IQ9EVK/Runtime/T_SM_IQ9_EVK_Proxy_Roughness",
):
    if unreal.EditorAssetLibrary.does_asset_exist(asset):
        removed = unreal.EditorAssetLibrary.delete_asset(asset)
        unreal.log(f"IQ9EVK_LEGACY_PROXY_CLEANUP asset={asset} removed={removed}")
