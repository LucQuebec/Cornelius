import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception"))
from enrollment import Enrollment, FaceSample


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.enrollment = Enrollment({"id": "test"})
        self.feature = np.ones(128, dtype=np.float32) / np.sqrt(128)
        self.now = 0

    def capture(self, pose, feature=None, issue=None, enrollment=None, pitch=0.):
        self.now += .3
        target = enrollment or self.enrollment
        sample = FaceSample(self.feature if feature is None else feature, pose, issue, pitch=pitch)
        return target.add([sample], self.now)

    def test_requires_five_distinct_poses_and_preserves_each_view(self):
        for _ in range(6):
            self.assertIsNone(self.capture(0))
        self.assertEqual(self.enrollment.progress()["step"], 1)
        for _ in range(8):
            self.capture(0)  # Face-on frames cannot complete a side.
        self.assertEqual(len(self.enrollment.templates), 2)
        for _ in range(6):
            self.capture(.25)
        for _ in range(8):
            self.capture(.25)  # Same side cannot stand in for the opposite side.
        self.assertEqual(len(self.enrollment.templates), 4)
        for _ in range(6):
            self.assertIsNone(self.capture(-.25))
        self.assertEqual(self.enrollment.message, "Levez légèrement le menton.")
        for _ in range(4):
            self.capture(0, pitch=1.)  # Trop près de la position de face.
        self.assertEqual(len(self.enrollment.templates), 6)
        for _ in range(10):
            self.capture(0, pitch=12.)
        self.assertEqual(self.enrollment.message, "Baissez légèrement le menton.")
        for _ in range(8):
            self.capture(0, pitch=12.)  # Le même sens ne vaut pas pour l’autre.
        self.assertEqual(len(self.enrollment.templates), 8)
        result = None
        for _ in range(12):
            result = result or self.capture(0, pitch=-12.)
        self.assertEqual(len(result), 10)
        for vector in result:
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=6)

    def test_up_then_down_always_in_that_order(self):
        for pose in (0, .25, -.25):
            for _ in range(6):
                self.capture(pose)
        for _ in range(8):
            self.capture(0, pitch=-12.)  # Baissée en premier : ne compte pas pour le haut.
        self.assertEqual(self.enrollment.stage, 3)
        self.assertEqual(self.enrollment.message, "Levez légèrement le menton.")

    def test_trembling_measure_never_flips_the_instruction(self):
        for pose in (0, .25, -.25):
            for _ in range(6):
                self.capture(pose)
        messages = []
        for pitch in (8., 3., 9., -2., 7., 2., 10., 4., 6., 9., 3., 8.):  # Tête levée, mesure qui tremble.
            if self.enrollment.stage != 3:
                break
            self.capture(0, pitch=pitch)
            if self.enrollment.stage == 3:
                messages.append(self.enrollment.message)
        # Jamais « baissez » pendant l’étape du haut, et l’étape se termine malgré le tremblement.
        self.assertFalse([message for message in messages if "Baissez" in message])
        self.assertEqual(self.enrollment.stage, 4)

    def test_turned_heads_do_not_count_as_up_or_down(self):
        for pose in (0, .25, -.25):
            for _ in range(6):
                self.capture(pose)
        for _ in range(6):
            self.capture(.3, pitch=12.)
        self.assertEqual(self.enrollment.stage, 3)

    def test_a_brief_wobble_does_not_restart_the_step(self):
        for pose in (0, .25, -.25):
            for _ in range(6):
                self.capture(pose)
        for _ in range(3):
            self.capture(0, pitch=12.)
        kept = len(self.enrollment.samples)
        self.assertGreater(kept, 0)
        self.capture(0, pitch=1.)  # Un repère qui hésite une image…
        self.capture(0, pitch=-8.)  # … ou deux : la médiane l’absorbe, rien n’est perdu.
        self.assertGreaterEqual(len(self.enrollment.samples), kept)
        for _ in range(8):
            self.capture(0, pitch=1.)  # Revenu de face trop longtemps : on recommence.
        self.assertEqual(len(self.enrollment.samples), 0)

    def test_tells_what_to_correct(self):
        for pose in (0, .25, -.25):
            for _ in range(6):
                self.capture(pose)
        for _ in range(5):
            self.capture(0, pitch=40.)
        self.assertEqual(self.enrollment.message, "Un peu moins : le mouvement doit rester léger.")
        for _ in range(5):
            self.capture(0, pitch=2.)  # Le bon geste, trop petit.
        self.assertEqual(self.enrollment.message, "Levez le menton un peu plus.")
        self.capture(0, pitch=float("nan"))  # Angle inconnu : la consigne, sans compter la vue.
        self.assertEqual(len(self.enrollment.samples), 0)
        self.assertEqual(self.enrollment.message, "Levez légèrement le menton.")
        self.capture(.3, pitch=-12.)
        self.assertEqual(self.enrollment.message, "Gardez la tête droite, sans la tourner.")

    def test_tilted_views_are_compared_more_leniently(self):
        for pose in (0, .25, -.25):
            for _ in range(6):
                self.capture(pose)
        # Menton levé, le visage ressemble moins (0,3) : refusé de face, accepté pour cette étape.
        other = np.eye(128, dtype=np.float32)[0] - self.feature * self.feature[0]
        other /= np.linalg.norm(other)
        tilted = self.feature * .3 + other * np.sqrt(1 - .09)
        self.capture(0, feature=tilted, pitch=12.)
        self.assertEqual(len(self.enrollment.samples), 1)

    def test_samples_are_spaced_in_time(self):
        self.enrollment.add([FaceSample(self.feature, 0)], 0)
        self.enrollment.add([FaceSample(self.feature, 0)], .1)
        self.assertEqual(len(self.enrollment.samples), 1)
        self.enrollment.add([FaceSample(self.feature, 0)], .3)
        self.assertEqual(len(self.enrollment.samples), 2)

    def test_rejects_blur_multiple_faces_and_another_person(self):
        self.capture(0)
        self.capture(0, issue="Image floue")
        self.assertEqual(len(self.enrollment.samples), 0)
        self.capture(0)
        self.enrollment.add([], self.now)
        self.assertEqual(len(self.enrollment.samples), 0)
        for _ in range(6):
            self.capture(0)
        self.capture(.25, feature=-self.feature)
        self.assertEqual(len(self.enrollment.samples), 0)
        self.assertEqual(len(self.enrollment.templates), 2)
        self.assertIn("différent", self.enrollment.message)

    def test_completing_a_face_rejects_someone_else(self):
        other = np.zeros(128, dtype=np.float32); other[0] = 1
        completing = Enrollment({"id": "add", "mode": "add"}, existing=[other.tolist()])
        self.capture(0, enrollment=completing)
        self.assertEqual(len(completing.samples), 0)
        self.assertIn("ne correspond pas", completing.message)
        same = Enrollment({"id": "add", "mode": "add"}, existing=[self.feature.tolist()])
        self.capture(0, enrollment=same)
        self.assertEqual(len(same.samples), 1)


if __name__ == "__main__":
    unittest.main()
