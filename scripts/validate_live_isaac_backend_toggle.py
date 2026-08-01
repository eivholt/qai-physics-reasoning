from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from integrations.isaac_sim_mcp.server import IsaacTcpClient


SOURCE = r"""
import gc
import json
import sys

import omni.kit.app
import omni.timeline

module = sys.modules.get("qai.conveyor_safety.extension")
extension_class = getattr(module, "ConveyorSafetyExtension", None)
extension = next(
    (
        item
        for item in gc.get_objects()
        if extension_class is not None and isinstance(item, extension_class)
    ),
    None,
)
if extension is None:
    raise RuntimeError("Conveyor safety extension is not active")

timeline = omni.timeline.get_timeline_interface()
was_playing = timeline.is_playing()
original_auto_inference = extension._auto_inference_enabled
# Exercise the Play-mode switch without starting an external model process.
extension._auto_inference_enabled = False
if not was_playing:
    timeline.play()
    await omni.kit.app.get_app().next_update_async()

before = {
    "backend": extension._inference_backend,
    "server_url": extension._inference_server_url,
    "model": extension._inference_model,
    "button": extension._backend_button.text,
    "timeline_playing": timeline.is_playing(),
}
extension._toggle_inference_backend()
switched = {
    "backend": extension._inference_backend,
    "server_url": extension._inference_server_url,
    "model": extension._inference_model,
    "button": extension._backend_button.text,
    "timeline_playing": timeline.is_playing(),
}
extension._toggle_inference_backend()
restored = {
    "backend": extension._inference_backend,
    "server_url": extension._inference_server_url,
    "model": extension._inference_model,
    "button": extension._backend_button.text,
    "timeline_playing": timeline.is_playing(),
}
extension._auto_inference_enabled = original_auto_inference
if not was_playing:
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()

print(json.dumps({
    "before": before,
    "switched": switched,
    "restored": restored,
    "restored_exactly": before == restored,
    "timeline_restored": timeline.is_playing() == was_playing,
}))
"""


def main() -> None:
    response = IsaacTcpClient(timeout_seconds=30.0).execute(SOURCE)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
