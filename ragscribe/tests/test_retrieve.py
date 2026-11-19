import json
import tempfile
import unittest
from pathlib import Path

from app.retrieve import (
    filename_for,
    known_job_ids,
    parse_name_json,
    render_markdown,
    select_new,
    slug_part,
    unique_path,
)


JOB = "6813620c-4d31-43f0-8f93-69cff01cfb2d"


class RetrieveTest(unittest.TestCase):
    def test_slug_keeps_hyphen_and_folds_umlauts(self) -> None:
        self.assertEqual(slug_part("Abdul El-Sayed", "Unbekannt"), "Abdul_El-Sayed")
        self.assertEqual(
            slug_part("Palästinensische Selbstbestimmung", "Transkript"),
            "Palaestinensische_Selbstbestimmung",
        )
        self.assertEqual(filename_for("Abdul El-Sayed", "Medicare for All"), "Abdul_El-Sayed__Medicare_for_All.md")

    def test_parse_name_json_strips_fence(self) -> None:
        raw = '```json\n{"author": "Abdul El-Sayed", "topic": "Medicare for All"}\n```'
        self.assertEqual(parse_name_json(raw), ("Abdul El-Sayed", "Medicare for All"))

    def test_select_new_skips_known_and_unfinished(self) -> None:
        jobs = [
            {"id": JOB, "status": "ok", "text": "fertig", "created_at": "b"},
            {"id": "962d2592-afd3-427e-b3e4-c68de8f9eecf", "status": "ok", "text": "  ", "created_at": "c"},
            {"id": "9a34a8f3-a8a1-4bdd-aa30-cb40e6b2d5c2", "status": "queued", "text": "x", "created_at": "a"},
            {"id": "d2bf16f9-4ccc-4c4d-9354-fba20b6e04b5", "status": "ok", "text": "neu", "created_at": "d"},
        ]
        fresh = select_new(jobs, {JOB})
        self.assertEqual([j["id"] for j in fresh], ["d2bf16f9-4ccc-4c4d-9354-fba20b6e04b5"])

    def test_known_job_ids_and_unique_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inbox = Path(tmp)
            note = inbox / "schon.md"
            note.write_text(f"- Job: {JOB}\n", encoding="utf-8")
            self.assertEqual(known_job_ids(inbox), {JOB})
            taken = unique_path(inbox, "schon.md", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
            self.assertEqual(taken.name, "schon__aaaaaaaa.md")

    def test_render_keeps_transcript(self) -> None:
        body = render_markdown(
            {"id": JOB, "url": "https://vm.tiktok.com/x/", "text": "Rohtext.", "created_at": "t"},
            "Ada",
            "Thema",
        )
        self.assertIn(f"- Job: {JOB}", body)
        self.assertTrue(body.endswith("Rohtext.\n"))


if __name__ == "__main__":
    unittest.main()
