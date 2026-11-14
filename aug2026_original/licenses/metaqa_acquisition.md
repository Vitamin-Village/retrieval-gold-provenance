# MetaQA Raw Data Acquisition Record

Acquired: 2026-06-13, via curl over HTTPS. All sources free/public.

## Sources

### 1. KB + vanilla QA (1/2/3-hop): HuggingFace mirror `camazlucas/MetaQA`

- Base URL: https://huggingface.co/datasets/camazlucas/MetaQA/resolve/main
- Files pulled: `kb/kb.txt`, `{1,2,3}-hop/vanilla/qa_{train,dev,test}.txt`
- Local names: `kb.txt`, `qa_{train,dev,test}_{1,2,3}hop.txt` (renamed to encode hop count; file contents unmodified)
- Provenance: mirror of the official MetaQA release (yuyuz/MetaQA, AAAI 2018 "Variational Reasoning for Question Answering with Knowledge Graph", arXiv:1709.04071). Official distribution is a Google Drive folder
  (https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE?resourcekey=0-kdv6ho5KcpEXdI2aUdLn_g) which is an old-style Drive ID; `gdown --folder` cannot traverse it (resourcekey not propagated), so the HF mirror was used.
- Integrity: every line count matches the official README table exactly (see Counts below), KB stats match the paper (~43k entities / ~135k triples / 9 relations).
- License: MetaQA is released under the Creative Commons Public License (per official README at https://github.com/yuyuz/MetaQA).

### 2. Text corpus wiki.txt: WikiMovies `movieqa.tar.gz` via Wayback Machine

- URL: http://web.archive.org/web/20220615183146/http://www.thespermwhale.com/jaseweston/babi/movieqa.tar.gz
  (original link http://www.thespermwhale.com/jaseweston/babi/movieqa.tar.gz now returns 404; https://dl.fbaipublicfiles.com/babi/movieqa.tar.gz returns 403)
- Archive kept as `movieqa.tar.gz` (57,070,041 bytes, tar timestamp 2016-06-15 — original Facebook bAbI/WikiMovies release).
- Extracted: `movieqa/knowledge_source/wiki.txt` -> `wiki.txt`; `movieqa/LICENSE.txt` -> `WIKIMOVIES_LICENSE.txt`; `movieqa/README.txt` -> `WIKIMOVIES_README.txt`. The rest of the tarball (full/wiki_ie/synthetic KBs, WikiMovies QA splits) remains inside the archive.
- License: Creative Commons (see `WIKIMOVIES_LICENSE.txt`, "CC License / THE MOVIEQA DATASET").

## Failed sources (tried first, recorded per protocol)

| Source | Result |
|---|---|
| `gdown --folder` on official MetaQA Google Drive | folder listing OK, subfolder retrieval fails (old-style ID + resourcekey unsupported) |
| http(s)://(www.)thespermwhale.com/jaseweston/babi/movieqa.tar.gz | 404 |
| https://dl.fbaipublicfiles.com/babi/movieqa.tar.gz | 403 |
| HF `rmanluo/RoG-metaqa` | does not exist (only RoG-webqsp / RoG-cwq) |
| HF `facebook/wiki_movies` | loader script only, no data (points to the dead thespermwhale URL) |

## Counts (verified locally)

| File | Lines | Expected (official README) | Match |
|---|---:|---:|---|
| kb.txt | 134,741 triples / 43,234 entities / 9 relations | ~135k triples, ~43k entities | OK |
| qa_train_1hop.txt | 96,106 | 96,106 | OK |
| qa_dev_1hop.txt | 9,992 | 9,992 | OK |
| qa_test_1hop.txt | 9,947 | 9,947 | OK |
| qa_train_2hop.txt | 118,980 | 118,980 | OK |
| qa_dev_2hop.txt | 14,872 | 14,872 | OK |
| qa_test_2hop.txt | 14,872 | 14,872 | OK |
| qa_train_3hop.txt | 114,196 | 114,196 | OK |
| qa_dev_3hop.txt | 14,274 | 14,274 | OK |
| qa_test_3hop.txt | 14,274 | 14,274 | OK |
| wiki.txt | 18,127 documents (115,982 lines) | ~18k documents | OK |

Format checks: all QA files are `question<TAB>answer1|answer2|...` with topic entity in [brackets]; 0 malformed lines in test splits. kb.txt is `subject|relation|object`. wiki.txt is numbered-line documents separated by blank lines, line 1 = movie title.

KB relations: directed_by, has_genre, has_imdb_rating, has_imdb_votes, has_tags, in_language, release_year, starred_actors, written_by.

## MD5

- kb.txt: 9f864bcfd71082b112c9370e4a1506a1
- wiki.txt: b5737d21b1f9dc8b70cc99a15417c67d
- qa_test_2hop.txt: 98f36350eb90c74cb534505d12d3aa41
- movieqa.tar.gz: cbe4c0cfd128e9f0d1b3039c22aef79c

## Citation

- MetaQA: Zhang, Dai, Kozareva, Smola, Song. Variational Reasoning for Question Answering with Knowledge Graph. AAAI 2018.
- WikiMovies: Miller et al. Key-Value Memory Networks for Directly Reading Documents. EMNLP 2016.
