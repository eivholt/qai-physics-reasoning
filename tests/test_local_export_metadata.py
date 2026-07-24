from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import qai_hub_models.models as qaihm_models

# The adapter is an editable overlay on the installed QAIHM package.
WORKSPACE_MODELS = Path(__file__).resolve().parents[1] / "qai_hub_models" / "models"
if str(WORKSPACE_MODELS) not in qaihm_models.__path__:
    qaihm_models.__path__.insert(0, str(WORKSPACE_MODELS))

from qai_hub_models import Precision, TargetRuntime  # noqa: E402
from qai_hub_models.configs import code_gen_yaml, info_yaml  # noqa: E402
from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen  # noqa: E402
from qai_hub_models.configs.info_yaml import QAIHMModelInfo  # noqa: E402
from qai_hub_models.models.cosmos_reason2_2b import MODEL_ID  # noqa: E402
from qai_hub_models.models.cosmos_reason2_2b.export import (  # noqa: E402
    EXPORT_COMPONENTS,
    SUPPORTED_PRECISION_RUNTIMES,
    _resolve_export_vision_profile,
    _use_local_model_metadata,
    build_parser,
)
from qai_hub_models.models.cosmos_reason2_2b.model import (  # noqa: E402
    SUPPORTED_PRECISIONS,
)
from qai_hub_models.utils.export import context as export_context  # noqa: E402


class LocalExportMetadataTests(unittest.TestCase):
    def test_local_metadata_is_resolvable_and_roots_are_restored(self) -> None:
        modules = (code_gen_yaml, info_yaml, export_context)
        original_roots = [module.QAIHM_MODELS_ROOT for module in modules]
        original_model_ids = info_yaml.MODEL_IDS

        with _use_local_model_metadata() as local_root:
            self.assertEqual(local_root, WORKSPACE_MODELS)
            for module in modules:
                self.assertEqual(module.QAIHM_MODELS_ROOT, WORKSPACE_MODELS)

            info = QAIHMModelInfo.from_model(MODEL_ID)
            code_gen = QAIHMModelCodeGen.from_model(MODEL_ID)
            self.assertEqual(info.id, MODEL_ID)
            self.assertEqual(
                export_context.resolve_model_dir(MODEL_ID),
                WORKSPACE_MODELS / MODEL_ID,
            )
            self.assertIn(Precision.w4, code_gen.supported_precisions)
            self.assertIn(MODEL_ID, info_yaml.MODEL_IDS)

        for module, original_root in zip(
            modules, original_roots, strict=True
        ):
            self.assertEqual(module.QAIHM_MODELS_ROOT, original_root)
        self.assertIs(info_yaml.MODEL_IDS, original_model_ids)

    def test_w4_is_advertised_for_both_genie_runtimes(self) -> None:
        self.assertIn(Precision.w4, SUPPORTED_PRECISIONS)
        self.assertEqual(
            SUPPORTED_PRECISION_RUNTIMES[Precision.w4],
            [TargetRuntime.GENIEX_QAIRT, TargetRuntime.GENIE],
        )

    def test_export_parser_can_select_only_the_vision_component(self) -> None:
        with _use_local_model_metadata():
            args = build_parser().parse_args(
                [
                    "--components",
                    "vision_encoder",
                    "--skip-downloading",
                    "--image-size",
                    "224",
                    "384",
                ]
            )

        self.assertEqual(args.components, ["vision_encoder"])
        self.assertTrue(args.skip_downloading)
        self.assertEqual(args.image_size, [224, 384])
        self.assertEqual(
            EXPORT_COMPONENTS,
            (
                "vision_encoder",
                "part1_of_4",
                "part2_of_4",
                "part3_of_4",
                "part4_of_4",
            ),
        )

    def test_omitted_image_size_uses_checkpoint_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "args.json").write_text(
                json.dumps({"image_size": [224, 384]}),
                encoding="utf-8",
            )

            profile = _resolve_export_vision_profile(checkpoint, None)

        self.assertEqual(
            (profile.image_height, profile.image_width),
            (224, 384),
        )

    def test_explicit_image_size_must_match_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "args.json").write_text(
                json.dumps({"image_size": [224, 384]}),
                encoding="utf-8",
            )

            matching = _resolve_export_vision_profile(
                checkpoint, [224, 384]
            )
            with self.assertRaisesRegex(
                ValueError,
                "conflicts with checkpoint args.json image_size 224 384",
            ):
                _resolve_export_vision_profile(checkpoint, [512, 512])

        self.assertEqual(
            (matching.image_height, matching.image_width),
            (224, 384),
        )

    def test_legacy_and_default_checkpoints_fall_back_to_512(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            legacy_checkpoint = Path(temporary)
            (legacy_checkpoint / "args.json").write_text(
                json.dumps({"precision": "w4a16"}),
                encoding="utf-8",
            )

            legacy = _resolve_export_vision_profile(
                legacy_checkpoint, None
            )
            default = _resolve_export_vision_profile("DEFAULT", None)
            explicit = _resolve_export_vision_profile(
                legacy_checkpoint, [512, 512]
            )

        for profile in (legacy, default, explicit):
            self.assertEqual(
                (profile.image_height, profile.image_width),
                (512, 512),
            )

    def test_parser_distinguishes_omitted_image_size(self) -> None:
        with _use_local_model_metadata():
            parser = build_parser()
            args = parser.parse_args(["--skip-downloading"])
            help_text = parser.format_help()

        self.assertIsNone(args.image_size)
        self.assertIn(
            "checkpoint's args.json, or 512 512 for legacy/default",
            help_text,
        )


if __name__ == "__main__":
    unittest.main()
