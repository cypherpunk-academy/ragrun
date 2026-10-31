"""Site object diff: insert, update, unchanged, dest-only."""
from app.reader.site_sync import (
    diff_site_objects,
    keys_to_delete,
    keys_to_upload,
    normalize_etag,
)


def test_normalize_etag_strips_quotes_and_rejects_multipart():
    assert normalize_etag('"abc"') == "abc"
    assert normalize_etag("ABC") == "abc"
    assert normalize_etag('"abc-2"') is None
    assert normalize_etag("") is None


def test_diff_site_objects_classifies_keys():
    diff = diff_site_objects(
        {"a.html": "same", "b.html": "new-body", "c.html": "fresh"},
        {"a.html": '"same"', "b.html": '"old"', "gone.html": '"zzz"'},
    )
    assert diff["insert"] == ["c.html"]
    assert diff["update"] == ["b.html"]
    assert diff["unchanged"] == ["a.html"]
    assert diff["dest_only"] == ["gone.html"]


def test_replace_uploads_everything_and_deletes_extras():
    diff = diff_site_objects(
        {"a.html": "same", "b.html": "fresh"},
        {"a.html": '"same"', "old.html": '"zzz"'},
    )
    assert keys_to_upload("replace", diff) == ["b.html", "a.html"]
    assert keys_to_delete("replace", diff, prune=False) == ["old.html"]


def test_sync_uploads_changes_and_keeps_dest_only_unless_prune():
    diff = diff_site_objects(
        {"a.html": "same", "b.html": "fresh"},
        {"a.html": '"same"', "old.html": '"zzz"'},
    )
    assert keys_to_upload("sync", diff) == ["b.html"]
    assert keys_to_delete("sync", diff, prune=False) == []
    assert keys_to_delete("sync", diff, prune=True) == ["old.html"]
