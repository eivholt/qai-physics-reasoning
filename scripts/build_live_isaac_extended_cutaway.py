from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.isaac_sim_mcp.conveyor_safety import (
    create_conveyor_safety_scene_source,
    enable_conveyor_safety_extension_source,
)
from integrations.isaac_sim_mcp.server import (
    IsaacTcpClient,
    _save_stage_checkpoint_source,
)


CHECKPOINT_NAME = "reason2_conveyor_safety_stable_cargo_r38.usda"


def main() -> None:
    client = IsaacTcpClient(timeout_seconds=900.0)
    created = client.execute(
        create_conveyor_safety_scene_source(
            {"new_stage": True, "start_playing": False}
        )
    )
    saved = client.execute(
        _save_stage_checkpoint_source({"filename": CHECKPOINT_NAME})
    )
    enabled = client.execute(enable_conveyor_safety_extension_source())
    print(
        json.dumps(
            {
                "created": created,
                "saved": saved,
                "extension": enabled,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
