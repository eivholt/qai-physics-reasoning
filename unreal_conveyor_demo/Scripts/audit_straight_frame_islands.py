"""Report connected geometry islands in the blue frame slots of straight conveyors."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import unreal


MESH_PATHS = (
    "/Game/ConveyorRuntime/Scene/warehouse_conveyor_baked/World/CodexPoC/"
    "ConveyorSafety/Conveyor/Modules/SM_StraightWest.SM_StraightWest",
    "/Game/ConveyorRuntime/Scene/warehouse_conveyor_baked/World/CodexPoC/"
    "ConveyorSafety/Conveyor/Modules/ExtensionWest/SM_ConveyorBelt_A08_02."
    "SM_ConveyorBelt_A08_02",
)


def id_value(identifier: object) -> int:
    return int(identifier.get_editor_property("id_value"))  # type: ignore[attr-defined]


def audit_mesh(path: str) -> dict[str, object]:
    mesh = unreal.EditorAssetLibrary.load_asset(path)
    if not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError(f"Missing static mesh: {path}")
    description = mesh.get_static_mesh_description(0)
    if description is None:
        raise RuntimeError(f"No LOD0 mesh description: {path}")

    # Blue painted frame is material/polygon group one on A05 and zero on A08.
    blue_group = 1 if mesh.get_name() == "SM_StraightWest" else 0
    polygons = list(description.get_polygon_group_polygons(unreal.PolygonGroupID(blue_group)))
    parent: dict[int, int] = {}
    rank: dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        rank.setdefault(value, 0)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    polygon_vertices: list[tuple[int, list[int]]] = []
    for polygon in polygons:
        vertices = [id_value(vertex) for vertex in description.get_polygon_vertices(polygon)]
        if not vertices:
            continue
        first = vertices[0]
        find(first)
        for vertex in vertices[1:]:
            union(first, vertex)
        polygon_vertices.append((id_value(polygon), vertices))

    islands: dict[int, dict[str, object]] = {}
    for polygon_id, vertices in polygon_vertices:
        root = find(vertices[0])
        island = islands.setdefault(
            root,
            {
                "polygons": 0,
                "vertices": set(),
                "bounds_min": [float("inf")] * 3,
                "bounds_max": [float("-inf")] * 3,
                "polygon_ids": [],
            },
        )
        island["polygons"] = int(island["polygons"]) + 1
        island["polygon_ids"].append(polygon_id)  # type: ignore[union-attr]
        island["vertices"].update(vertices)  # type: ignore[union-attr]

    for island in islands.values():
        for vertex_id in island["vertices"]:  # type: ignore[union-attr]
            position = description.get_vertex_position(unreal.VertexID(vertex_id))
            for axis, value in enumerate((position.x, position.y, position.z)):
                island["bounds_min"][axis] = min(island["bounds_min"][axis], value)  # type: ignore[index]
                island["bounds_max"][axis] = max(island["bounds_max"][axis], value)  # type: ignore[index]
        island["vertices"] = len(island["vertices"])  # type: ignore[arg-type]

    ordered = sorted(
        islands.values(),
        key=lambda island: (
            -float(island["bounds_max"][2]),  # type: ignore[index]
            -int(island["polygons"]),
        ),
    )
    return {
        "mesh": path,
        "blue_group": blue_group,
        "polygon_count": len(polygons),
        "island_count": len(ordered),
        "islands": ordered,
    }


def main() -> None:
    report = {"meshes": [audit_mesh(path) for path in MESH_PATHS]}
    output = Path(unreal.Paths.project_saved_dir()) / "ImportAudit" / "straight_frame_islands.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    unreal.log(f"QAI_STRAIGHT_FRAME_ISLANDS={output}")


if __name__ == "__main__":
    main()
