import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception"))
from speaker import (  # noqa: E402
    ENROLL_SPEECH_S,
    SAMPLE_RATE,
    RequestConfirmation,
    SpeakerRecognizer,
    VoiceEnrollment,
    WakeHistory,
    pcm_to_float,
    trim_silence,
)

BINS = 64


def tone(frequency, seconds, amplitude=0.3):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def pcm(samples):
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


class FakeRecognizer(SpeakerRecognizer):
    """Empreinte factice : le spectre grossier du signal. Deux « voix » = deux fréquences."""

    def __init__(self):
        super().__init__(Path("."))
        self.threshold, self.margin = 0.45, 0.08

    def ready(self):
        return True

    def embed(self, samples):
        if len(samples) < SAMPLE_RATE * 0.5:
            return None
        spectrum = np.abs(np.fft.rfft(samples, n=4096))[:2048]
        vector = spectrum.reshape(BINS, -1).mean(axis=1).astype(np.float32)
        return vector / np.linalg.norm(vector)


class SpeakingVad:
    """Silero remplacé : tout bloc dont le son n’est pas nul est de la parole."""

    def process(self, block):
        return bool(np.abs(pcm_to_float(block)).max() > 0.01)


class SpeakerTests(unittest.TestCase):
    def test_trim_keeps_the_spoken_passage_in_one_piece(self):
        silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        word = tone(300, 0.6)
        # Un creux au milieu du mot ne le coupe pas en deux.
        word[3000:3600] = 0
        trimmed = trim_silence(np.concatenate([silence, word, silence]))
        self.assertGreaterEqual(len(trimmed), len(word))
        self.assertLess(len(trimmed), len(word) + int(0.25 * SAMPLE_RATE))
        self.assertEqual(len(trim_silence(np.zeros(SAMPLE_RATE, dtype=np.float32))), 0)

    def test_wake_history_keeps_two_seconds_and_empties_on_take(self):
        history = WakeHistory()
        for _ in range(40):
            history.add(3, pcm(tone(300, 0.08)))
        self.assertEqual(len(history.frames[3]), 25)
        self.assertGreater(len(history.take(3)), 0)
        self.assertEqual(len(history.take(3)), 0)
        history.drop(3)
        self.assertNotIn(3, history.frames)

    def test_identify_accepts_the_enrolled_voice_and_refuses_another(self):
        recognizer = FakeRecognizer()
        self.assertIsNone(recognizer.identify(tone(300, 2), "wake"))  # aucun profil
        recognizer.set_profiles([("luc", [recognizer.embed(tone(300, 3)).tolist()])])
        known = recognizer.identify(tone(300, 2), "request")
        self.assertEqual(known["profileId"], "luc")
        stranger = recognizer.identify(tone(2500, 2), "request")
        self.assertIsNone(stranger["profileId"])
        self.assertEqual(stranger["closestProfileId"], "luc")

    def test_the_wake_word_alone_has_a_lower_threshold(self):
        recognizer = FakeRecognizer()
        recognizer.set_profiles([("luc", [recognizer.embed(tone(300, 3)).tolist()])])
        # Score d’environ 1 : refusé sur une phrase avec un seuil de 1,05, accepté sur le mot seul
        # (seuil 0,95). Le mot d’appel dure moins d’une seconde et ressemble moins.
        recognizer.threshold = 1.05
        self.assertIsNone(recognizer.identify(tone(300, 2), "request")["profileId"])
        self.assertEqual(recognizer.identify(tone(300, 2), "wake")["profileId"], "luc")

    def test_two_close_profiles_need_a_clear_gap(self):
        recognizer = FakeRecognizer()
        recognizer.set_profiles([
            ("a", [recognizer.embed(tone(300, 3)).tolist()]),
            ("b", [recognizer.embed(tone(300, 3)).tolist()]),
        ])
        result = recognizer.identify(tone(300, 2), "request")
        self.assertIsNone(result["profileId"])
        self.assertGreater(result["score"], 0.9)

    def test_configure_ignores_out_of_range_values(self):
        recognizer = FakeRecognizer()
        recognizer.configure(1.5, -1)
        self.assertEqual((recognizer.threshold, recognizer.margin), (0.45, 0.08))
        recognizer.configure(0.6, 0.1)
        self.assertEqual((recognizer.threshold, recognizer.margin), (0.6, 0.1))

    def test_request_confirmation_waits_for_three_seconds_of_speech(self):
        confirmation = RequestConfirmation("i1")
        block = pcm(tone(300, 512 / SAMPLE_RATE))
        self.assertIsNone(confirmation.add(block, False))
        results = [confirmation.add(block, True) for _ in range(94)]
        self.assertTrue(all(result is None for result in results[:-1]))
        self.assertEqual(len(results[-1]), 94 * 512)
        self.assertIsNone(confirmation.add(block, True))  # une seule fois

    def feed(self, enrollment, samples, slot=0):
        data = pcm(samples)
        for offset in range(0, len(data) - 2559, 2560):
            enrollment.add(slot, data[offset:offset + 2560])

    def test_enrollment_waits_for_speech_and_three_wake_words(self):
        recognizer = FakeRecognizer()
        enrollment = VoiceEnrollment({"id": "e1"}, SpeakingVad)
        self.feed(enrollment, tone(300, ENROLL_SPEECH_S + 2))
        self.assertFalse(enrollment.complete())  # pas encore trois « Cornelius »
        for _ in range(3):
            enrollment.add_wake(tone(300, 0.7))
        enrollment.add_wake(tone(300, 0.2))  # trop court : ignoré
        self.assertTrue(enrollment.complete())
        progress = enrollment.progress()
        self.assertEqual((progress["wakes"], progress["requiredWakes"]), (3, 3))
        templates, reason = enrollment.templates(recognizer)
        self.assertIsNone(reason)
        self.assertEqual(len(templates), 12 + 3)
        self.assertTrue(all(len(vector) == BINS for vector in templates))

    def test_enrollment_keeps_the_stream_that_heard_the_most_speech(self):
        enrollment = VoiceEnrollment({"id": "e2"}, SpeakingVad)
        self.feed(enrollment, tone(300, 5), slot=1)
        self.feed(enrollment, tone(300, 9), slot=2)
        self.assertAlmostEqual(enrollment.speech_seconds(), 9, delta=0.2)

    def test_enrollment_refuses_too_little_speech(self):
        enrollment = VoiceEnrollment({"id": "e3"}, SpeakingVad)
        self.assertEqual(enrollment.templates(FakeRecognizer()), (None, "silence"))
        self.feed(enrollment, tone(300, 4))
        self.assertEqual(enrollment.templates(FakeRecognizer()), (None, "short"))


if __name__ == "__main__":
    unittest.main()
