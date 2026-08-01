# Codex control PoC for Isaac Sim 6.0.1

This integration connects Codex to a running local Isaac Sim process without
installing another Python package:

```text
Codex -> local stdio MCP server -> TCP 127.0.0.1:8226 -> Isaac Sim
```

The MCP server uses only the Python standard library. Isaac Sim executes the
small, typed USD and timeline operations through its built-in
`isaacsim.code_editor.python_server` extension. It does not expose arbitrary
Python execution as an MCP tool.

## 1. Finish the Isaac Sim installation

After extraction, run the NVIDIA post-install step once:

```powershell
& "C:\NVIDIA\isaac-sim-standalone-6.0.1\post_install.bat"
```

Run the compatibility checker before debugging the MCP path:

```powershell
& "C:\NVIDIA\isaac-sim-standalone-6.0.1\isaac-sim.compatibility_check.bat"
```

## 2. Start Isaac Sim with the Python server

From this repository:

```powershell
.\integrations\isaac_sim_mcp\launch_isaac_sim_poc.ps1
```

The launcher defaults to a 60 FPS application cap. Override it only when
needed, for example `-FrameRateLimit 30` for a lower-power inspection session.
The cap applies to Kit's application/render loops and does not change the USD
timeline timestep, physics frequency, or the five-FPS tutorial GIF encoding.

Equivalent direct command:

```powershell
& "C:\NVIDIA\isaac-sim-standalone-6.0.1\isaac-sim.bat" `
  --enable isaacsim.code_editor.python_server `
  --/app/runLoops/main/rateLimitEnabled=true `
  --/app/runLoops/main/rateLimitFrequency=60 `
  --/app/runLoops/main/rateLimitUseBusyLoop=false `
  --/app/runLoops/main/rateLimitUsePrecisionSleep=true
```

The first GUI launch may spend several minutes warming shaders. Wait until the
GUI is responsive. Keep the server bound to its default `127.0.0.1:8226`;
exposing it on `0.0.0.0` would allow remote arbitrary Python execution.

## 3. Check the native bridge

In a second PowerShell terminal:

```powershell
Test-NetConnection 127.0.0.1 -Port 8226

python .\integrations\isaac_sim_mcp\server.py --check-isaac
```

Expected application-level response:

```json
{
  "status": "ok",
  "output": "",
  "result": 2
}
```

## 4. Load the MCP tools in Codex

The project `.codex/config.toml` registers the local stdio server. Because MCP
configuration is read when a Codex task starts, restart Codex or open a new task
in this trusted repository after adding or changing the configuration.

In the new task, ask:

```text
Ping Isaac Sim, create the PoC scene, list /World/CodexPoC,
and move the blue marker to x=1, y=1, z=0.5.
```

The available tools are:

- `isaac_ping`
- `isaac_get_status`
- `isaac_set_frame_rate_limit`
- `isaac_enable_conveyor_safety`
- `isaac_create_conveyor_safety_scene`
- `isaac_capture_conveyor_safety_camera`
- `isaac_set_conveyor_safety_light`
- `isaac_get_conveyor_safety_state`
- `isaac_set_conveyor_safety_forklift_pose`
- `isaac_set_conveyor_safety_camera`
- `isaac_set_markings_enabled`
- `isaac_set_presentation_overlays_enabled`
- `isaac_save_stage_checkpoint`
- `isaac_load_stage_checkpoint`
- `isaac_create_lightweight_live_warehouse`
- `isaac_create_live_aisle_navigation`
- `isaac_set_live_aisle_congestion`
- `isaac_tick_live_aisle`
- `isaac_set_live_aisle_request_status`
- `isaac_apply_live_aisle_advisory`
- `isaac_capture_live_aisle_roof_frame`
- `isaac_create_poc_scene`
- `isaac_set_marker_pose`
- `isaac_set_timeline`
- `isaac_list_prims`
- `isaac_capture_viewport`
- `isaac_get_edge_supervisor_assets`
- `isaac_create_edge_supervisor_scene`
- `isaac_create_realistic_edge_supervisor_scene`
- `isaac_inspect_realistic_warehouse`
- `isaac_set_edge_supervisor_progress`
- `isaac_capture_edge_supervisor_cameras`
- `isaac_capture_edge_supervisor_sequence`
- `isaac_capture_realistic_edge_supervisor_cameras`
- `isaac_capture_realistic_edge_supervisor_sequence`

All PoC scene edits stay under `/World/CodexPoC`.
Viewport captures are restricted to `artifacts/isaac_sim_poc`.

## 5. Run the verified live supervisor

The publishable tutorial uses a lightweight warehouse assembled from official
NVIDIA rack, Nova Carter, forklift, pallet, and carton assets. It keeps the
roof cutaway readable and avoids the memory cost of the complete warehouse
stage.

The live bridge is
`scripts/run_live_isaac_evk_supervisor.py`. Its reliable `tracked` profile:

1. captures clean roof-camera windows;
2. reads trusted actor tracks and route occupancy from Isaac;
3. asks Cosmos-Reason2-2B for one feasible command;
4. validates the exact response;
5. applies it through a deterministic waypoint actuator;
6. leaves an independent local safety hold authoritative.

The corrected EVK-verified command is `REROUTE NORTH_BYPASS`. The south
bypass is blocked by pallets and is removed from the model grammar. The
forklift is rotated 90 degrees clockwise so its visual heading matches its
southbound motion.

Start the saved scene checkpoint against the persistent EVK service:

```powershell
python .\scripts\run_live_isaac_evk_supervisor.py `
  --inference-backend evk `
  --evk-target ubuntu@192.168.1.158 `
  --supervisor-profile tracked `
  --motion-controller kinematic_waypoint `
  --robot-speed 0.18 `
  --sensor-camera roof `
  --presentation-camera roof `
  --presentation-route-visualizations `
  --evk-video-width 384 `
  --frames-per-window 8 `
  --clear-baseline-responses 1 `
  --forklift-release-x -0.75 `
  --safety-hold-distance 2.2 `
  --capture-fps 2 `
  --model-fps 4 `
  --max-runtime-seconds 180 `
  --max-responses 18 `
  --presentation-fps 6 `
  --make-gif
```

For faster scene iteration, use `--inference-backend host` with the local BF16
service described in the full tutorial.

To retain interactive shelf, pallet, box, and lighting edits, add:

```powershell
--preserve-current-stage
```

The presentation recording contains white, cyan, and orange route lines. A
separate clean capture is sent to the model, so presentation geometry never
leaks into the inference window.

Rebuild the verified annotated media with:

```powershell
.\scripts\build_verified_isaac_tutorial_media.ps1
```

See the complete
[edge AI warehouse supervisor tutorial](../../docs/isaac_sim_edge_supervisor_tutorial.md)
for the captured run, exact timings, prompt contract, EVK command, acceptance
checks, media, and limitations. Earlier scripted and experimental recordings
remain under `artifacts/` and older `docs/media/isaac_sim_*` directories, but
they are no longer primary tutorial evidence.

For the user-controllable three-forklift conveyor and model-driven stack-light
demo, see the
[live Cosmos Reason2 conveyor tutorial](../../docs/isaac_sim_conveyor_reason2_tutorial.md).
