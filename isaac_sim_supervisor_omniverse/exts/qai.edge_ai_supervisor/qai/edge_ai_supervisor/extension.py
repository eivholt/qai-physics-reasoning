"""Dockable live observability panel for the EVK warehouse supervisor."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import carb
import omni.ext
import omni.kit.app
import omni.ui as ui
from omni.kit.menu.utils import (
    MenuItemDescription,
    add_menu_items,
    remove_menu_items,
)


WINDOW_TITLE = "Edge AI Warehouse Supervisor"
POLL_INTERVAL_SECONDS = 0.2
STALE_AFTER_SECONDS = 5.0

_HEADER_STYLE = {
    "font_size": 18,
    "color": 0xFFF4F4F4,
}
_SECTION_STYLE = {
    "font_size": 14,
    "color": 0xFF63D7FF,
}
_MONO_STYLE = {
    "font_size": 14,
    "color": 0xFFE6E6E6,
}


def _repository_root() -> Path:
    # .../repo/isaac_sim_supervisor_omniverse/exts/ext/qai/package/file.py
    return Path(__file__).resolve().parents[5]


def _default_status_path() -> Path:
    return (
        _repository_root()
        / "artifacts"
        / "isaac_sim_live_aisle"
        / "live_status.json"
    )


def _format_position(value: Any) -> str:
    if not isinstance(value, list) or len(value) < 2:
        return "—"
    try:
        return f"x={float(value[0]):.2f}, y={float(value[1]):.2f}"
    except (TypeError, ValueError):
        return "—"


def _format_model_items(value: Any) -> str:
    """Render normalized model fields without exposing Python/JSON internals."""
    if not isinstance(value, list) or not value:
        return "none"

    items: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = str(item.get("name") or "unknown")
            offender = item.get("offender")
            items.append(
                f"{name} ({offender})" if offender else name
            )
        else:
            items.append(str(item))
    return ", ".join(items)


def _format_passage_state(value: Any) -> str:
    """Render one mandatory model passage field compactly."""
    if not isinstance(value, dict):
        return "not reported"
    status = str(value.get("status") or "UNKNOWN")
    offender = str(value.get("offender") or "UNKNOWN")
    return status if offender == "NONE" else f"{status} · {offender}"


class EdgeAiSupervisorExtension(omni.ext.IExt):
    """Isaac extension that renders the external controller's atomic status."""

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        self._status_path = _default_status_path()
        self._status: dict[str, Any] = {}
        self._last_status_mtime_ns = -1
        self._last_poll = 0.0
        self._labels: dict[str, ui.Label] = {}
        self._images: list[ui.Image] = []

        self._window = ui.Window(
            WINDOW_TITLE,
            width=570,
            height=820,
            visible=True,
        )
        self._build_ui()

        self._menu_items = [
            MenuItemDescription(
                name=WINDOW_TITLE,
                onclick_fn=self._show_window,
            )
        ]
        add_menu_items(self._menu_items, "Window")
        self._update_subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_update,
                name="qai.edge_ai_supervisor.status_poll",
            )
        )
        carb.log_info(
            f"{WINDOW_TITLE} reading live status from {self._status_path}"
        )

    def on_shutdown(self) -> None:
        if getattr(self, "_menu_items", None):
            remove_menu_items(self._menu_items, "Window")
        self._update_subscription = None
        self._window = None
        self._labels = {}
        self._images = []

    def _show_window(self) -> None:
        if self._window is not None:
            self._window.visible = True
            self._window.focus()

    def _section(self, title: str) -> None:
        ui.Label(title, height=18, style=_SECTION_STYLE)

    def _build_ui(self) -> None:
        with self._window.frame:
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=(
                    ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF
                ),
                vertical_scrollbar_policy=(
                    ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED
                ),
            ):
                with ui.VStack(spacing=2, style={"margin": 5}):
                    ui.Label(WINDOW_TITLE, height=23, style=_HEADER_STYLE)
                    self._labels["connection"] = ui.Label(
                        "Waiting for live controller status…",
                        height=17,
                        style={"font_size": 12, "color": 0xFF66C2FF},
                    )

                    self._section("EVK inference")
                    self._labels["evk_summary"] = ui.Label(
                        "Waiting for controller…",
                        height=72,
                        word_wrap=True,
                        alignment=ui.Alignment.LEFT_TOP,
                        style=_MONO_STYLE,
                    )

                    self._section(
                        "Last supervisor frames · routes excluded"
                    )
                    with ui.HStack(height=76, spacing=5):
                        for _ in range(3):
                            image = ui.Image(
                                "",
                                width=ui.Fraction(1),
                                height=74,
                                fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                                style={
                                    "background_color": 0xFF151515,
                                    "border_color": 0xFF444444,
                                    "border_width": 1,
                                },
                            )
                            self._images.append(image)

                    self._section("Decision and actuation")
                    self._labels["decision_summary"] = ui.Label(
                        "No decision yet.",
                        height=118,
                        word_wrap=True,
                        alignment=ui.Alignment.LEFT_TOP,
                        style=_MONO_STYLE,
                    )

                    self._section("Recent responses")
                    self._labels["history"] = ui.Label(
                        "No responses yet.",
                        height=96,
                        word_wrap=True,
                        alignment=ui.Alignment.LEFT_TOP,
                        style=_MONO_STYLE,
                    )

                    with ui.CollapsableFrame(
                        "Exact prompt",
                        collapsed=True,
                        height=0,
                        style={"font_size": 13},
                    ):
                        with ui.ScrollingFrame(
                            height=105,
                            horizontal_scrollbar_policy=(
                                ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF
                            ),
                            vertical_scrollbar_policy=(
                                ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED
                            ),
                            style={"background_color": 0xFF181818},
                        ):
                            self._labels["prompt"] = ui.Label(
                                "Waiting for controller…",
                                word_wrap=True,
                                style=_MONO_STYLE,
                            )

                    with ui.CollapsableFrame(
                        "Allow-listed outputs",
                        collapsed=True,
                        height=0,
                        style={"font_size": 13},
                    ):
                        self._labels["allowed_commands"] = ui.Label(
                            "—",
                            height=64,
                            word_wrap=True,
                            alignment=ui.Alignment.LEFT_TOP,
                            style=_MONO_STYLE,
                        )

                    ui.Spacer(height=1)
                    self._labels["status_path"] = ui.Label(
                        str(self._status_path),
                        height=17,
                        word_wrap=True,
                        style={"font_size": 11, "color": 0xFF777777},
                    )

    def _on_update(self, _event: Any) -> None:
        now = time.monotonic()
        if now - self._last_poll < POLL_INTERVAL_SECONDS:
            self._update_inference_timer()
            return
        self._last_poll = now
        self._read_status()
        self._update_inference_timer()

    def _read_status(self) -> None:
        try:
            stat = self._status_path.stat()
        except FileNotFoundError:
            self._labels["connection"].text = (
                "Waiting for live_status.json from the controller…"
            )
            return
        if stat.st_mtime_ns == self._last_status_mtime_ns:
            self._update_freshness()
            return
        try:
            payload = json.loads(self._status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._labels["connection"].text = f"Status read failed: {exc}"
            return
        if not isinstance(payload, dict):
            return
        self._last_status_mtime_ns = stat.st_mtime_ns
        self._status = payload
        self._render_status()

    def _update_freshness(self) -> None:
        updated = self._status.get("updated_epoch")
        if not isinstance(updated, (int, float)):
            return
        age = max(0.0, time.time() - float(updated))
        if age > STALE_AFTER_SECONDS and self._status.get("state") != "complete":
            self._labels["connection"].text = (
                f"STALE · no controller update for {age:.1f} s"
            )

    def _update_inference_timer(self) -> None:
        started = self._status.get("inference_started_epoch")
        if isinstance(started, (int, float)):
            elapsed = max(0.0, time.time() - float(started))
            self._render_evk_summary(running_elapsed=elapsed)

    def _render_evk_summary(
        self,
        running_elapsed: float | None = None,
    ) -> None:
        status = self._status
        model_latency = status.get("last_model_seconds")
        host_latency = status.get("last_host_total_seconds")
        if running_elapsed is not None:
            elapsed_text = f"{running_elapsed:.1f}s running"
        elif isinstance(host_latency, (int, float)):
            elapsed_text = f"{float(host_latency):.3f}s"
        else:
            elapsed_text = "idle"
        model_text = (
            f"{float(model_latency):.3f}s"
            if isinstance(model_latency, (int, float))
            else "—"
        )
        round_trip_text = (
            f"{float(host_latency):.3f}s"
            if isinstance(host_latency, (int, float))
            else "—"
        )
        self._labels["evk_summary"].text = (
            f"STATE {str(status.get('state') or 'unknown').upper()}  ·  "
            f"PHASE {status.get('phase') or '—'}  ·  "
            f"REQ {status.get('request_id') or '—'}\n"
            f"MODEL {status.get('model') or '—'}\n"
            f"EVK {status.get('evk_target') or '—'}\n"
            f"TIME {elapsed_text}  ·  MODEL {model_text}  ·  "
            f"ROUND TRIP {round_trip_text}"
        )

    def _render_decision_summary(self) -> None:
        status = self._status
        model_responses = status.get("last_model_responses")
        normalized = (
            model_responses
            if isinstance(model_responses, dict)
            else {}
        )
        actors = _format_model_items(normalized.get("traffic_actors"))
        passage_states = normalized.get("passage_states")
        passages = (
            passage_states
            if isinstance(passage_states, dict)
            else {}
        )
        left_passage = _format_passage_state(
            passages.get("left_passage")
        )
        right_passage = _format_passage_state(
            passages.get("right_passage")
        )
        gateway_action = status.get("gateway_action")
        gateway_route = status.get("gateway_route")
        gateway_source = status.get("gateway_source")
        gateway = (
            f"{gateway_action} {gateway_route} · {gateway_source}"
            if gateway_action
            else "—"
        )
        separation = status.get("separation_m")
        separation_text = (
            f"{float(separation):.2f}m"
            if isinstance(separation, (int, float))
            else "—"
        )
        self._labels["decision_summary"].text = (
            f"MODEL ACTORS              {actors}\n"
            f"LEFT PASSAGE              {left_passage}\n"
            f"RIGHT PASSAGE             {right_passage}\n"
            f"ADAPTER {status.get('last_raw_command') or '—'}\n"
            f"ACCEPTED {gateway}\n"
            f"ROBOT {status.get('active_action') or '—'} "
            f"{status.get('active_route') or '—'}  ·  GAP {separation_text}\n"
            f"SIM TELEMETRY robot "
            f"{_format_position(status.get('robot_position'))}  ·  "
            f"forklift {_format_position(status.get('forklift_position'))}"
        )

    def _render_status(self) -> None:
        status = self._status
        updated = status.get("updated_epoch")
        age = (
            max(0.0, time.time() - float(updated))
            if isinstance(updated, (int, float))
            else 0.0
        )
        self._labels["connection"].text = (
            f"LIVE · run {status.get('run_id', '—')} · {age:.1f} s ago"
        )
        self._render_evk_summary()
        self._render_decision_summary()

        frames = status.get("window_frames")
        frame_paths = frames if isinstance(frames, list) else []
        selected: list[str] = []
        if frame_paths:
            indices = (0, len(frame_paths) // 2, len(frame_paths) - 1)
            selected = [str(frame_paths[index]) for index in indices]
        for index, image in enumerate(self._images):
            image.source_url = selected[index] if index < len(selected) else ""

        history = status.get("response_history")
        rows = []
        if isinstance(history, list):
            for response in history[-6:]:
                if not isinstance(response, dict):
                    continue
                latency = response.get("host_total_seconds")
                suffix = (
                    f" · {float(latency):.2f}s"
                    if isinstance(latency, (int, float))
                    else ""
                )
                response_fields = response.get("model_responses")
                normalized = (
                    response_fields
                    if isinstance(response_fields, dict)
                    else {}
                )
                actors = _format_model_items(
                    normalized.get("traffic_actors")
                )
                passage_states = normalized.get("passage_states")
                passages = (
                    passage_states
                    if isinstance(passage_states, dict)
                    else {}
                )
                left_passage = _format_passage_state(
                    passages.get("left_passage")
                )
                right_passage = _format_passage_state(
                    passages.get("right_passage")
                )
                rows.append(
                    f"{response.get('request_id', '—')} "
                    f"[{response.get('phase', '—')}] "
                    f"actors={actors} · left={left_passage} · "
                    f"right={right_passage} → "
                    f"{response.get('raw_command', '—')}{suffix}"
                )
        self._labels["history"].text = "\n".join(rows) or "No responses yet."
        self._labels["prompt"].text = str(
            status.get("prompt") or "Waiting for controller…"
        )
        commands = status.get("allowed_commands")
        self._labels["allowed_commands"].text = (
            "\n".join(str(item) for item in commands)
            if isinstance(commands, list)
            else "—"
        )
