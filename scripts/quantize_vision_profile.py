#!/usr/bin/env python3
"""Run Cosmos quantization with class-level vision dimensions configured.

The shared QAIHM quantizer already accepts ``--image-size HEIGHT WIDTH``.
This wrapper applies the same dimensions to the concrete Cosmos vision class
before delegating, preventing a later vision-only export from silently falling
back to the model's 512-square defaults.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from qai_hub_models.models.cosmos_reason2_2b import quantize
from qai_hub_models.models.cosmos_reason2_2b.vision_profile import (
    configure_cosmos_vision_profile,
)


def configure_from_argv(argv: Sequence[str]) -> tuple[int, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=(512, 512),
    )
    parsed, _ = parser.parse_known_args(list(argv))
    height, width = parsed.image_size
    configure_cosmos_vision_profile(height, width)
    return height, width


def main() -> None:
    configure_from_argv(sys.argv[1:])
    quantize.main()


if __name__ == "__main__":
    main()
