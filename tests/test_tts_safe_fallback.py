import os
import unittest
from unittest import mock

from src.core import tts


class TestTTSSafeFallback(unittest.TestCase):
    def test_no_sounddevice_fallback_when_external_players_missing(self):
        original = os.environ.get("ALLOW_SOUNDDEVICE_TTS")
        os.environ["ALLOW_SOUNDDEVICE_TTS"] = "0"
        try:
            with mock.patch.object(tts.shutil, "which", return_value=None):
                called = {"sounddevice": False}

                def fake_sounddevice(path):
                    called["sounddevice"] = True

                with mock.patch.object(tts, "_reda_cu_sounddevice", side_effect=fake_sounddevice):
                    tts._reda_fisier_wav("/tmp/does-not-matter.wav")

                self.assertFalse(called["sounddevice"])
        finally:
            if original is None:
                os.environ.pop("ALLOW_SOUNDDEVICE_TTS", None)
            else:
                os.environ["ALLOW_SOUNDDEVICE_TTS"] = original


if __name__ == "__main__":
    unittest.main()
