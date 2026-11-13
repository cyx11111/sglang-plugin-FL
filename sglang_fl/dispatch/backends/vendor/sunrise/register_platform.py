# Copyright (c) 2026 BAAI. All rights reserved.
"""Register Sunrise / PTPU AttentionBackend with SGLang.

``PlatformFL.init_backend()`` imports
``sglang_fl.dispatch.backends.vendor.sunrise.register_platform`` when the
FlagGems device vendor is ``sunrise``. The decorator below inserts the
creator into SGLang's ``ATTENTION_BACKENDS`` map so

    --attention-backend sunrise

(or the platform default from ``_ATTN_BACKEND_MAP``) resolves correctly.
"""

from __future__ import annotations

import logging

from sglang.srt.layers.attention.attention_registry import register_attention_backend

logger = logging.getLogger(__name__)


@register_attention_backend("sunrise")
def _create_sunrise_attn_backend(runner):
    from sglang_fl.dispatch.backends.vendor.sunrise.impl.attention.attention_backend import (
        SunriseAttnBackend,
    )

    logger.info("Creating SunriseAttnBackend (PTPU decode/prefix_caching)")
    return SunriseAttnBackend(runner)
