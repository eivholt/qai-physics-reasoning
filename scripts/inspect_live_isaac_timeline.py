from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.isaac_sim_mcp.server import IsaacTcpClient


SOURCE = r"""
import json
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom

timeline = omni.timeline.get_timeline_interface()
stage = omni.usd.get_context().get_stage()
parcel = stage.GetPrimAtPath(
    "/World/CodexPoC/ConveyorSafety/Conveyor/Parcels/Parcel1"
)


def parcel_position():
    value = UsdGeom.Xformable(parcel).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).ExtractTranslation()
    return [round(float(item), 6) for item in value]


states = [{
    "frame": -1,
    "playing": timeline.is_playing(),
    "time": float(timeline.get_current_time()),
    "parcel": parcel_position(),
}]
timeline.play()
states.append({
    "frame": 0,
    "playing": timeline.is_playing(),
    "time": float(timeline.get_current_time()),
    "parcel": parcel_position(),
})
for frame in range(1, 31):
    await omni.kit.app.get_app().next_update_async()
    if frame <= 10 or frame % 5 == 0:
        states.append({
            "frame": frame,
            "playing": timeline.is_playing(),
            "time": round(float(timeline.get_current_time()), 6),
            "parcel": parcel_position(),
        })
timeline.stop()
print(json.dumps({"states": states}))
""".strip()


def main() -> None:
    print(
        json.dumps(
            IsaacTcpClient(timeout_seconds=120.0).execute(SOURCE),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
