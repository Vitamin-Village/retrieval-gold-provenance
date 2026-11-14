# Pinned GraphRAG-Bench question files (not copied)

`code/_grb_provenance_trace.py` reads three question files of GitHub GraphRAG-Bench/GraphRAG-Benchmark from this folder. They are taken at commits that predate the repository's LICENSE file (MIT, added in commit 2328b37 on 2025-07-12), so they are listed here instead of copied. To restore them, save each file under the name in the first column, e.g.

    curl -L -o medical_questions_e18981f.json https://raw.githubusercontent.com/GraphRAG-Bench/GraphRAG-Benchmark/e18981f45d8fdaab6baabf1d5b71a864ebbb3983/Datasets/Questions/medical_questions.json

and check its SHA-256.

| file name here | commit | path in the repository | bytes | SHA-256 |
|---|---|---|---:|---|
| `medical_questions_e18981f.json` | e18981f45d8fdaab6baabf1d5b71a864ebbb3983 | `Datasets/Questions/medical_questions.json` | 1354276 | 17478681f1177fc503a7fca2fb9f8d321034a140d4c716df775b31df3d7eecd1 |
| `novel_questions_0b8f46a.json` | 0b8f46a0a646d9b60b3649a60a235b610fabf836 | `Datasets/Questions/novel_questions.json` | 1514240 | 4726018a9070e531725e3f848027f3d0ee08666984d4b022b9d5d930acf0b113 |
| `novel_questions_e18981f.json` | e18981f45d8fdaab6baabf1d5b71a864ebbb3983 | `Datasets/Questions/novel_questions.json` | 1047059 | 7911eb9a08315205a9258231563543f6d40aea546ce2c66553b4148178c92fa3 |
