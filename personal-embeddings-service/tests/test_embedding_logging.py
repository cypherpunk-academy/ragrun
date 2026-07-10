from app.api.endpoints.embeddings import _input_size_kb, _texts_from_request


def test_input_size_kb_empty():
    assert _input_size_kb([]) == 0.0


def test_input_size_kb_utf8():
    # "ä" is 2 bytes in UTF-8; total 5 bytes
    assert _input_size_kb(["ab", "ä"]) == 5 / 1024.0


def test_texts_from_request_single():
    assert _texts_from_request("hello") == ["hello"]


def test_texts_from_request_list():
    assert _texts_from_request(["a", "b"]) == ["a", "b"]
