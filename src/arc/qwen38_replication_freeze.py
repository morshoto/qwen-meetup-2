"""Identity checks for the previously completed exploratory outcome."""
from __future__ import annotations

import hashlib
import json

from src.arc.qwen38_rsi_comparison import digest

ADAPTER_SHA256 = 'b363fd2bd15114577d96141b2e03500e7d3d241a4b4a64b6176f25dbb46ac9f6'
REPORT_SHA256 = '023ac3d4afa8982d43a24e2b328f4471e127a7c2f9a24109d582d9e89d193b70'


def verify_frozen_outcome(adapter_bytes: bytes, report_bytes: bytes) -> dict:
    """Check the immutable adapter and signed-content report, without changing either."""
    try:
        report = json.loads(report_bytes)
        valid = (isinstance(report, dict) and
                 report.get('sha256') == REPORT_SHA256 and
                 digest({k: v for k, v in report.items() if k != 'sha256'}) == REPORT_SHA256 and
                 report.get('status') == 'complete' and report.get('schedule_status') == 'complete' and
                 report.get('adapter_sha256') == ADAPTER_SHA256 and
                 hashlib.sha256(adapter_bytes).hexdigest() == ADAPTER_SHA256)
    except (ValueError, TypeError, UnicodeDecodeError):
        valid = False
    if not valid:
        raise ValueError('frozen outcome identity mismatch')
    return {'status': 'frozen_outcome_verified', 'adapter_sha256': ADAPTER_SHA256,
            'report_sha256': REPORT_SHA256}
