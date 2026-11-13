# Copyright (c) 2026 BAAI. All rights reserved.
"""Extend sglang's SUPPORTED_DEVICES with ``ptpu``."""

import logging

logger = logging.getLogger(__name__)


def patch_supported_devices():
    from sglang.srt.configs import device_config as dc

    if "ptpu" not in dc.SUPPORTED_DEVICES:
        dc.SUPPORTED_DEVICES = [*dc.SUPPORTED_DEVICES, "ptpu"]
        logger.info("patched SUPPORTED_DEVICES += [ptpu]")
