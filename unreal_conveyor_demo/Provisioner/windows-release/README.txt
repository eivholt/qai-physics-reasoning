Reason2 Conveyor Safety — speed-v1 GenieX production release 20260819
====================================================================

Requirements
------------
- Windows 11 x64.
- Current NVIDIA driver for the full RTX presentation renderer.
- About 9 GB free space for the release plus installed host model/runtime.
- For EVK inference, the computer must be on the same LAN as 192.168.1.158.

Install and launch
------------------
1. Extract every file to a short path such as C:\Reason2-Conveyor. Do not run
   Setup from inside the ZIP; keep the full game executable path below the
   Windows 260-character CreateProcess limit.
2. Double-click Reason2-Conveyor-Setup.exe.
3. Windows may show an unsigned-app warning for this private build. Confirm only
   if the SHA256SUMS.txt hash matches the value supplied with the release.

Setup verifies every model/runtime artifact, starts the promoted Q8 host model,
installs the reboot-persistent GenieX full-W8 QAIRT EVK service, writes
runtime.json, and launches the packaged Unreal client. The first run can take
several minutes while artifacts are hashed and resident graphs are loaded.

Useful commands (PowerShell)
----------------------------
Repair/verify without launching:
  .\Reason2-Conveyor-Setup.exe ensure --app-dir .\Game --payload-root .\Payload --evk-host 192.168.1.158 --require-evk --no-launch

Launch after installation:
  .\Reason2-Conveyor-Setup.exe launch --app-dir .\Game --payload-root .\Payload

Create a redacted support bundle:
  .\Reason2-Conveyor-Setup.exe support-bundle --app-dir .\Game --payload-root .\Payload --reason manual

Runtime renderer controls
-------------------------
- F6 toggles Lumen GI and Lumen reflections. When off, the client uses no
  dynamic GI and falls back to screen-space reflections.
- F7 toggles global runtime ray tracing. With Lumen on, F7 switches between
  hardware and software Lumen. Unsupported GPUs show this control as fixed.
- Both controls are session-only and default to ON on the RTX 5090 profile.
- The lossless inference sensor follows the same Lumen state as the main view.

Inference sensor
----------------
- The fixed 448x256 lossless sensor is optically cropped to the monitored
  central straight lane and its immediately adjacent floor. The separate
  return conveyor is outside the model input; the player view is unchanged.

Accepted model contract
-----------------------
- Host: Cosmos-Reason2-2B-Parcel-Speed-v1, speed-v1 Q8 GGUF + F16 projector,
  versioned port 18084, 512-token context, 112 visual tokens, and flash attention.
- EVK: CL512 full-W8 QAIRT 2.45 graph under GenieX, one resident NPU worker,
  one generated class token, and direct lossless 448x256 PNG on port 18183.
- Unreal: D3D12, ray tracing and hardware Lumen default to enabled on capable RTX GPUs.
