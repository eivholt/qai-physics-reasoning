"""Compile and export Cosmos-Reason2-2B for a Qualcomm target."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import qai_hub as hub
from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.utils.args import export_parser
from qai_hub_models.utils.checkpoint import CheckpointType
from qai_hub_models.utils.export.dispatch import resolve_export_model
from qai_hub_models.utils.export.result import ComponentGroup

from . import DEFAULT_PRECISION, MODEL_ID, Model
from .model import (
    DEFAULT_EXPORT_CONTEXT_LENGTHS,
    FULL_EXPORT_CONTEXT_LENGTHS,
    Cosmos_Reason2_2B_PartBase,
)

SUPPORTED_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.w4a16: [
        TargetRuntime.GENIEX_QAIRT,
        TargetRuntime.GENIE,
    ],
    Precision.w4: [
        TargetRuntime.GENIEX_QAIRT,
        TargetRuntime.GENIE,
    ],
}

DEFAULT_EXPORT_DEVICE = "Dragonwing IQ-9075 EVK"
EXPORT_COMPONENTS = (
    "vision_encoder",
    "part1_of_4",
    "part2_of_4",
    "part3_of_4",
    "part4_of_4",
)
export_model = resolve_export_model(MODEL_ID)


@contextmanager
def _use_local_model_metadata() -> Iterator[Path]:
    """Resolve this out-of-tree model's YAMLs from the workspace package.

    QAIHM 0.58 copies ``QAIHM_MODELS_ROOT`` into three modules and builds its
    valid-model ID list at import time. Editable overlay packages extend the
    Python module path, but they do not change that filesystem metadata.
    Patch the three export-time readers and add this one ID only for the
    duration of this CLI export, then restore them. This avoids modifying
    site-packages and keeps built-in QAIHM behavior unchanged after return.
    """
    from qai_hub_models.configs import code_gen_yaml, info_yaml
    from qai_hub_models.utils.export import context as export_context

    local_models_root = Path(__file__).resolve().parents[1]
    local_model_dir = local_models_root / MODEL_ID
    missing = [
        str(local_model_dir / filename)
        for filename in ("info.yaml", "code-gen.yaml")
        if not (local_model_dir / filename).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Local QAIHM metadata is incomplete. Missing: "
            + ", ".join(missing)
        )

    modules = (code_gen_yaml, info_yaml, export_context)
    previous_roots = [module.QAIHM_MODELS_ROOT for module in modules]
    previous_model_ids = info_yaml.MODEL_IDS
    try:
        for module in modules:
            module.QAIHM_MODELS_ROOT = local_models_root
        if MODEL_ID not in previous_model_ids:
            info_yaml.MODEL_IDS = [*previous_model_ids, MODEL_ID]
        yield local_models_root
    finally:
        info_yaml.MODEL_IDS = previous_model_ids
        for module, previous_root in zip(
            modules, previous_roots, strict=True
        ):
            module.QAIHM_MODELS_ROOT = previous_root


def _ordered_compiled_models_by_component(
    compiled_models: Mapping[tuple[str, str], Any],
    model: Any,
) -> dict[str, list[Any]]:
    """Restore each component's declared graph order before QNN linking.

    QAI Hub Models 0.58 groups compiled models by iterating a result mapping.
    Completion timing can change that mapping's order. Genie selects prompt
    and token graphs by index, so every linked decoder partition must instead
    follow ``get_component_graph_names()`` exactly.
    """
    present_components = {
        component_name for component_name, _ in compiled_models
    }
    ordered: dict[str, list[Any]] = {}
    for component_name in model.component_names:
        if component_name not in present_components:
            continue
        expected_graphs = model.get_component_graph_names(component_name)
        actual_graphs = {
            graph_name
            for name, graph_name in compiled_models
            if name == component_name
        }
        if actual_graphs != set(expected_graphs):
            missing = sorted(set(expected_graphs) - actual_graphs)
            unexpected = sorted(actual_graphs - set(expected_graphs))
            raise ValueError(
                f"Cannot deterministically link {component_name}: "
                f"missing graphs {missing}; unexpected graphs {unexpected}"
            )
        ordered[component_name] = [
            compiled_models[(component_name, graph_name)]
            for graph_name in expected_graphs
        ]
    return ordered


def _run_ordered_multi_graph_collection_link(
    compiled_models: Mapping[tuple[str, str], hub.Model],
    device: hub.Device,
    model_name: str,
    model: Any,
    target_runtime: TargetRuntime,
    extra_options: str = "",
) -> ComponentGroup[hub.client.LinkJob]:
    """Submit link jobs in contractual order.

    The Hub service may still canonicalize the supplied model IDs. The EVK
    preflight therefore remains the authority on order inside each resulting
    context binary.
    """
    if not target_runtime.is_aot_compiled:
        raise ValueError(
            "Ordered multi-graph linking requires an AOT target runtime."
        )
    grouped = _ordered_compiled_models_by_component(compiled_models, model)
    link_jobs: ComponentGroup[hub.client.LinkJob] = ComponentGroup()
    for component_name, ordered_models in grouped.items():
        print(f"Linking {component_name} to context binary")
        link_jobs[component_name] = hub.submit_link_job(
            ordered_models,
            device=device,
            name=f"{model_name}_{component_name}",
            options=model.get_component_hub_link_options(
                component_name,
                target_runtime,
                extra_options,
            ),
        )
    return link_jobs


def _install_ordered_linker() -> None:
    """Patch QAIHM 0.58 to submit its link jobs in declared graph order."""
    from qai_hub_models.utils.export import multi_graph_collection_pipeline

    multi_graph_collection_pipeline.run_multi_graph_collection_link = (
        _run_ordered_multi_graph_collection_link
    )


def build_parser(cli_mode: bool = False) -> argparse.ArgumentParser:
    parser = export_parser(
        model_cls=Model,
        export_fn=export_model,
        supported_precision_runtimes=SUPPORTED_PRECISION_RUNTIMES,
        default_export_device=DEFAULT_EXPORT_DEVICE,
        omit_precision=True,
        cli_mode=cli_mode,
    )
    parser.add_argument(
        "--full-context-matrix",
        action="store_true",
        help=(
            "Compile contexts 512, 1024, 2048, and 4096. By default only "
            "the 512-token smoke matrix is compiled."
        ),
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=EXPORT_COMPONENTS,
        default=None,
        help=(
            "Export only these components. Use "
            "'--components vision_encoder --skip-downloading' to compile and "
            "link a replacement vision context without resubmitting text."
        ),
    )
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    with _use_local_model_metadata():
        if args is None:
            args = build_parser().parse_args()
        warnings.filterwarnings("ignore")
        checkpoint = getattr(args, "checkpoint", None)
        if checkpoint is not None:
            args.precision = CheckpointType.from_checkpoint(
                checkpoint
            ).precision(
                DEFAULT_PRECISION, checkpoint=checkpoint
            )
        else:
            args.precision = DEFAULT_PRECISION

        export_args = vars(args).copy()
        full_context_matrix = bool(
            export_args.pop("full_context_matrix", False)
        )
        Cosmos_Reason2_2B_PartBase.export_context_lengths = list(
            FULL_EXPORT_CONTEXT_LENGTHS
            if full_context_matrix
            else DEFAULT_EXPORT_CONTEXT_LENGTHS
        )
        _install_ordered_linker()
        export_model(MODEL_ID, **export_args)


if __name__ == "__main__":
    main()
