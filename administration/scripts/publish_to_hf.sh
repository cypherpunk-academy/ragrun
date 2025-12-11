#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="$ROOT_DIR/ragkeep-deutsche-klassik-books-de"

if [ ! -d "$DATASET_DIR/.git" ]; then
  echo "Dataset repo not found at '$DATASET_DIR'. Please clone:"
  echo "  git clone https://huggingface.co/datasets/Lafisrap/ragkeep-deutsche-klassik-books-de \"$DATASET_DIR\""
  exit 1
fi

# Ensure LFS is enabled for HTML
( cd "$DATASET_DIR" && git lfs install && git lfs track "books/**/html/**" >/dev/null 2>&1 || true )
( cd "$DATASET_DIR" && git lfs track "books/**/results/html/**" >/dev/null 2>&1 || true )

# Clean destination books directory
rm -rf "$DATASET_DIR/books"
mkdir -p "$DATASET_DIR/books"

# Copy only the specific files mentioned
for book_dir in "$ROOT_DIR/books"/*; do
  if [ -d "$book_dir" ]; then
    book_name=$(basename "$book_dir")
    dest_book_dir="$DATASET_DIR/books/$book_name"
    mkdir -p "$dest_book_dir"
    
    # Copy release markdown from book root: any file that ends with _released.md (max 1)
    released_src=""
    for f in "$book_dir"/*_released.md; do
      if [ -f "$f" ]; then
        released_src="$f"
        break
      fi
    done
    if [ -n "$released_src" ]; then
      cp -f "$released_src" "$dest_book_dir/"
    fi
    # Copy html/ only from book root (do not copy from results)
    if [ -d "$book_dir/html" ]; then
      cp -rf "$book_dir/html" "$dest_book_dir/"
    fi
    [ -f "$book_dir/book.json" ] && cp -f "$book_dir/book.json" "$dest_book_dir/"
    [ -f "$book_dir/book-manifest.yaml" ] && cp -f "$book_dir/book-manifest.yaml" "$dest_book_dir/"
    [ -f "$book_dir/errata.txt" ] && cp -f "$book_dir/errata.txt" "$dest_book_dir/"
  fi
done

# Copy root-level files
cp -f "$ROOT_DIR/dataset_info.json" "$DATASET_DIR/"
cp -f "$ROOT_DIR/README.md" "$DATASET_DIR/"
cp -f "$ROOT_DIR/LICENSE" "$DATASET_DIR/"

echo "Prepared HF dataset repository."
