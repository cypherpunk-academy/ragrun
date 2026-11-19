import tempfile
import unittest
from pathlib import Path

from app.inputs.video_url import read_download_title
from app.source import detect_source


class DetectSourceTest(unittest.TestCase):
    def test_youtube(self) -> None:
        self.assertEqual(
            detect_source("https://www.youtube.com/watch?v=abc"), "youtube"
        )
        self.assertEqual(detect_source("https://youtu.be/abc"), "youtube")

    def test_tiktok(self) -> None:
        self.assertEqual(
            detect_source("https://www.tiktok.com/@name/video/1"), "tiktok"
        )
        self.assertEqual(detect_source("https://vm.tiktok.com/ZGdXX5Aok/"), "tiktok")

    def test_facebook(self) -> None:
        samples = [
            "https://www.facebook.com/watch/?v=123",
            "https://m.facebook.com/reel/123",
            "https://fb.watch/abcDEF/",
            "https://www.fb.com/share/v/abc",
        ]
        for url in samples:
            self.assertEqual(detect_source(url), "facebook", url)

    def test_other(self) -> None:
        self.assertEqual(detect_source("https://vimeo.com/1"), "other")
        self.assertEqual(detect_source("https://notfacebook.com/watch"), "other")


class ReadDownloadTitleTest(unittest.TestCase):
    def test_reads_info_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.info.json"
            path.write_text('{"title": "  Ein Titel  "}', encoding="utf-8")
            self.assertEqual(read_download_title(Path(tmp)), "Ein Titel")

    def test_missing_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_download_title(Path(tmp)), "")
