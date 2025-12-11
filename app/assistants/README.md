---
language:
- de
tags:
- books
- rag
- text-retrieval
license: MIT
pretty_name: ragkeep-weimarer-klassik-books-de
configs:
- default
---

# ragkeep-weimarer-klassik-books-de

Released subset of Weimarer Klassik books curated in `ragkeep` and prepared by `ragprep`.

## Contents (HF subset)
- Released Markdown: `books/**/results/_released.md`
- HTML rendering: `books/**/results/html/<bookname>.html`
- TOC JSON: `books/**/results/toc.json`
- Provenance & corrections: `book-manifest.yaml`, `errata.txt`

`<bookname>` = canonical folder basename `Author#Title#Index`.

## Loading
```python
from datasets import load_dataset

md = load_dataset("michaelschmidt/ragkeep-weimarer-klassik-books-de",
                  data_files={"train": "books/**/results/_released.md"}, split="train")
idx = load_dataset("michaelschmidt/ragkeep-weimarer-klassik-books-de",
                  data_files={"index": "books/index.json"}, split="index")
```

## License
MIT

## Notes
- Full working tree (inputs, intermediates) lives in `ragkeep`; only the HF subset is mirrored here.
