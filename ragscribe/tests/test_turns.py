import unittest

from app.turns import segments_to_turn_cued_text


class TurnTextTest(unittest.TestCase):
    def test_two_speakers(self) -> None:
        text = segments_to_turn_cued_text(
            [
                {"text": "Hello there.", "speaker": "SPEAKER_00"},
                {"text": "Still me.", "speaker": "SPEAKER_00"},
                {"text": "Other voice.", "speaker": "SPEAKER_01"},
                {"text": "  Continues.  ", "speaker": "SPEAKER_01"},
            ]
        )
        self.assertEqual(
            text,
            ">> Hello there. Still me.\n\n>> Other voice. Continues.\n",
        )

    def test_no_speaker(self) -> None:
        self.assertEqual(
            segments_to_turn_cued_text([{"text": "Intro only."}]),
            "Intro only.\n",
        )

    def test_one_speaker_skips_cues(self) -> None:
        text = segments_to_turn_cued_text(
            [
                {"text": "   ", "speaker": "SPEAKER_00"},
                {"text": "Real.", "speaker": "SPEAKER_00"},
            ]
        )
        self.assertEqual(text, "Real.\n")

    def test_one_speaker_merges(self) -> None:
        text = segments_to_turn_cued_text(
            [
                {"text": "Hello there.", "speaker": "SPEAKER_00"},
                {"text": "Still me.", "speaker": "SPEAKER_00"},
            ]
        )
        self.assertEqual(text, "Hello there. Still me.\n")


if __name__ == "__main__":
    unittest.main()
