"""
Storage backends for evaluation results.

Supports:
- BucketStorage: S3/GCS bucket storage with incremental sync
"""

from .bucket import BucketStorage, PairResultData

__all__ = ["BucketStorage", "PairResultData"]
