# SPDX-License-Identifier: GPL-3.0-only
"""The shipped workflow applies the full TaoMate LoRA to an unmerged FL2VA cache."""
C3_MODEL_SHA256 = "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"


def require_c3_cache(data):
    # Premerged caches intentionally retain the original base model identity;
    # identity/size alone cannot prevent applying the same FC1 LoRA twice.
    if "premerged_lora" in data or data.get("mode") not in (None, "fc1_only"):
        raise RuntimeError("C3 requires an unmerged FC1 cache. A premerged TaoMate cache would apply the LoRA twice.")
    if data.get("model", {}).get("sha256") != C3_MODEL_SHA256:
        raise RuntimeError("C3 cache must originate from the exact unmerged FL2VA model in models.json.")
