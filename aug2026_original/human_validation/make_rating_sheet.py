# -*- coding: utf-8 -*-
"""
Build the blind rating sheet for the 29-judgment human validation.

Reads items_29.json, shuffles the items with a FIXED seed so the order is
reproducible but uncorrelated with the census table order, and writes
rating_sheet_blank.csv with an empty your_bin column. The reference bins are never
read here, so the sheet cannot leak them. ASCII-only.
"""
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20260610


def main():
    items = json.load(open(HERE / "items_29.json", encoding="utf-8"))["items"]
    assert len(items) == 29, "expected 29 items, got %d" % len(items)
    rng = random.Random(SEED)
    order = list(range(len(items)))
    rng.shuffle(order)
    out = HERE / "rating_sheet_blank.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["position", "item_id", "benchmark", "axis",
                    "construction_description", "your_bin_0_to_4"])
        for pos, idx in enumerate(order, 1):
            it = items[idx]
            w.writerow([pos, it["item_id"], it["benchmark"], it["axis"],
                        it["description"], ""])
    print("wrote", out, "(29 rows, seed %d)" % SEED)


if __name__ == "__main__":
    main()
