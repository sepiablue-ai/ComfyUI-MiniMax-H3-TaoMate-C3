# SPDX-License-Identifier: GPL-3.0-only
"""Copy a legacy C3 API workflow to the portable loader/cache. Never edit the input."""
import argparse
import json
from pathlib import Path


def migrate(graph):
    result = json.loads(json.dumps(graph))
    if "nodes" in result:
        raise ValueError("Use the API-format C3 workflow, or open the supplied TaoMate_C3.json UI workflow.")
    classes = {node["class_type"] for node in result.values()}
    if not {"H3C2HeldFFN8", "H3C3HeldOut", "H3SparseAttention"}.issubset(classes):
        raise ValueError("This is not the adopted C3 graph. This utility does not convert C2 or Streaming VSA experiments to C3.")
    if any("VSA" in name for name in classes):
        raise ValueError("C3 uses H3SparseAttention, not a Streaming VSA patch.")
    for node in result.values():
        if node["class_type"] in {"H3V2PreconvertedLoader", "H3TaoMateW4A4Loader"}:
            node["class_type"] = "H3TaoMateW4A4Loader"
            node["inputs"]["cache_directory"] = "h3_preconverted/taomate_c3_fc1"
        if node["class_type"] == "LoadImage" and node["inputs"]["image"] == "baseline016_case01_frame000.png":
            node["inputs"]["image"] = "taomate_c3/first_frame.png"
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    result = migrate(json.loads(args.input.read_text(encoding="utf-8-sig")))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Wrote {args.output}; prompt, seed, sampler and numerical settings preserved.")


if __name__ == "__main__":
    main()
