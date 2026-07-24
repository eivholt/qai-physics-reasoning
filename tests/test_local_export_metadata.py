from __future__ import annotations

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
                ]
            )

        self.assertEqual(args.components, ["vision_encoder"])
        self.assertTrue(args.skip_downloading)
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


if __name__ == "__main__":
    unittest.main()
