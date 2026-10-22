from app.reader.html import annotate_text_html, render_chapter_html
from app.reader.urls import chapter_object_key, chapter_public_url, passage_link


def test_annotate_italics_and_quotes():
    text = "abc def ghi"
    html = annotate_text_html(
        text,
        {
            "italics": [{"start": 0, "end": 3}],
            "foreign_quotes": [{"start": 4, "end": 7}],
        },
    )
    assert "<i>abc</i>" in html
    assert "<q>«def»</q>" in html
    assert "ghi" in html


def test_annotate_page_ref_uses_passage_path(monkeypatch):
    from app.reader import urls as urls_mod

    monkeypatch.setattr(urls_mod.settings, "mcp_base_url", "https://api.ragxxx.com")
    html = annotate_text_html(
        "siehe S. 12 hier",
        {
            "page_refs": [
                {
                    "start": 6,
                    "end": 11,
                    "target_paragraph_id": "para-2",
                }
            ]
        },
    )
    assert 'href="https://api.ragxxx.com/passage/para-2"' in html
    assert "S. 12" in html


def test_annotate_escapes_plain_text():
    html = annotate_text_html("<script>", None)
    assert "&lt;script&gt;" in html
    assert "<script>" not in html


def test_render_chapter_marks_target():
    page = render_chapter_html(
        source_title="Die Philosophie der Freiheit",
        source_author="Rudolf Steiner",
        segment_title="Einleitung",
        paragraphs=[
            {
                "id": "p1",
                "paragraph_number": 1,
                "text_raw": "Erster Absatz.",
                "annotations": None,
            },
            {
                "id": "p2",
                "paragraph_number": 2,
                "text_raw": "Zweiter Absatz.",
                "annotations": {"italics": [{"start": 0, "end": 7}]},
            },
        ],
        target_paragraph_id="p2",
        inline_assets=True,
        app_passage_id="p2",
    )
    assert 'id="p2"' in page
    assert "is-target" in page
    assert "<i>Zweiter</i>" in page
    assert "ragapp://passage/p2" in page
    assert "Cormorant Garamond" in page


def test_chapter_urls(monkeypatch):
    from app.reader import urls as urls_mod

    monkeypatch.setattr(urls_mod.settings, "corpus_public_base_url", "https://corpus.ragxxx.com")
    monkeypatch.setattr(urls_mod.settings, "r2_prefix", "")
    monkeypatch.setattr(urls_mod.settings, "mcp_base_url", "https://api.ragxxx.com")
    assert chapter_object_key("sid", "einleitung") == "books/sid/einleitung.html"
    assert chapter_public_url("sid", "einleitung", "p1") == (
        "https://corpus.ragxxx.com/books/sid/einleitung.html?p=p1#p1"
    )
    assert passage_link("p1") == "https://api.ragxxx.com/passage/p1"
