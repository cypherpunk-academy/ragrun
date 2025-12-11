"""Add missing book-id fields to book manifests.

The script scans all ``book-manifest.yaml`` files under ``books/``. When a
manifest lacks a ``book-id`` entry, a new UUIDv4 is inserted. The new field is
placed directly after ``book-type`` when present (otherwise after
``book-index`` or at the top).
"""

from pathlib import Path
from uuid import uuid4


def ensure_book_id(manifest_path: Path) -> str | None:
    """Add a book-id to the manifest if missing; return the new id or None."""
    lines = manifest_path.read_text(encoding="utf-8").splitlines()

    if any(line.strip().startswith("book-id:") for line in lines):
        return None

    new_id = str(uuid4())

    insert_at = None
    for idx, line in enumerate(lines):
        if line.startswith("book-type:"):
            insert_at = idx + 1
            break
        if line.startswith("book-index:") and insert_at is None:
            insert_at = idx + 1

    if insert_at is None:
        insert_at = 0

    lines.insert(insert_at, f"book-id: {new_id}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return new_id


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    books_dir = repo_root / "books"

    added = []
    for manifest in sorted(books_dir.rglob("book-manifest.yaml")):
        new_id = ensure_book_id(manifest)
        if new_id:
            added.append((manifest.relative_to(repo_root), new_id))

    if not added:
        print("All manifests already contain book-id.")
        return

    print(f"Added book-id to {len(added)} manifest(s):")
    for rel_path, new_id in added:
        print(f"- {rel_path}: {new_id}")


if __name__ == "__main__":
    main()
