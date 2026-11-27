"""Guided local enrollment: five distinct poses, quality and identity checks."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FaceSample:
    feature: np.ndarray
    pose: float
    issue: str | None = None
    # Inclinaison verticale de la tête en degrés (positive : menton levé ; NaN si inconnue).
    pitch: float = 0.0
    # Informations de reconnaissance, utilisées par le test en direct de l’administration.
    usable: bool = True
    identity: str | None = None
    scores: list = field(default_factory=list)
    required: float | None = None


FRONT = "Regardez la caméra de face."
SIDE = "Tournez légèrement la tête vers un côté."
OTHER_SIDE = "Tournez légèrement la tête vers l’autre côté."
UP = "Levez légèrement le menton."
DOWN = "Baissez légèrement le menton."
MORE_UP = "Levez le menton un peu plus."
MORE_DOWN = "Baissez le menton un peu plus."
STRAIGHT = "Gardez la tête droite, sans la tourner."
TOO_FAR = "Un peu moins : le mouvement doit rester léger."
STAGES = 5
# Écart d’inclinaison verticale, en degrés, par rapport à la position de face. Mesuré avec le
# modèle d’angle de la tête sur un miroir à caméra plongeante (2026-11-27) : de face -10,5° à
# moins d’un degré près ; menton levé naturellement +12°, baissé -13°.
PITCH_MIN = 5.0
PITCH_MAX = 35.0
# Ressemblance exigée avec la vue déjà prise la plus proche. Menton levé ou baissé, le visage
# ressemble moins aux autres vues : ces deux étapes sont plus tolérantes.
ANCHOR_MIN = 0.35
TILT_ANCHOR_MIN = 0.25
# Une mesure qui hésite (repères instables tête levée) ne fait pas recommencer l’étape aussitôt.
TILT_GRACE = 1.2
# L’angle tremble de quelques degrés d’une image à l’autre : médiane des dernières mesures.
PITCH_SMOOTHING = 5
# Geste déjà reconnu dans l’étape : il reste valable jusqu’à la moitié du minimum.
PITCH_KEEP = 3.0


class Enrollment:
    # Six images par position, espacées dans le temps, résumées en deux vues : on garde de
    # la variété (expression, micro-mouvements) sans conserver d’image.
    required = 6
    per_template = 3
    interval = 0.25
    # Les repères du visage hésitent parfois une image, tête baissée surtout : un écart plus
    # court ne fait pas recommencer l’étape.
    grace = 0.6

    def __init__(self, request, existing=None):
        self.request = request
        # Mode « compléter » : les nouvelles vues doivent ressembler au visage déjà mémorisé.
        self.existing = None if existing is None or len(existing) == 0 else np.asarray(existing)
        self.templates = []
        self.samples = []
        self.pitches = []
        self.recent = []
        self.baseline = None
        self.stage = 0
        self.direction = None
        self.vertical = None
        self.last_sample_at = -1
        self.missed_since = None
        self.message = FRONT

    def prompt(self):
        if self.stage == 3:
            return UP
        if self.stage == 4:
            return DOWN
        return (FRONT, SIDE, OTHER_SIDE)[min(self.stage, 2)]

    def progress(self):
        return {"step": min(self.stage, STAGES - 1), "collected": len(self.samples),
                "total": self.required, "message": self.message}

    def mismatch(self, sample):
        """None si la vue convient à l’étape, sinon la consigne à afficher."""
        stage = self.stage
        if stage == 0:
            return None if abs(sample.pose) <= 0.13 else FRONT
        if stage in (1, 2):
            direction = 1 if sample.pose > 0 else -1
            if stage == 2 and direction == self.direction:
                return self.prompt()
            if abs(sample.pose) > 0.42:
                return TOO_FAR
            return None if abs(sample.pose) >= 0.18 else self.prompt()
        if abs(sample.pose) > 0.2:
            return STRAIGHT
        if not np.isfinite(sample.pitch) or self.baseline is None or not np.isfinite(self.baseline):
            return self.prompt()
        # Toujours dans le même ordre : le haut, puis le bas. Les consignes ne changent pas de
        # place, et un tremblement de la mesure ne fait pas passer de l’une à l’autre.
        self.recent = (self.recent + [sample.pitch])[-PITCH_SMOOTHING:]
        offset = float(np.median(self.recent)) - self.baseline
        wanted = 1 if stage == 3 else -1
        if abs(offset) > PITCH_MAX:
            return TOO_FAR
        reached = offset * wanted
        if reached >= PITCH_MIN or (self.samples and reached >= PITCH_KEEP):
            return None
        if reached >= PITCH_MIN / 3:
            # Le bon geste, mais trop petit : on le dit, au lieu de répéter la consigne.
            return MORE_UP if wanted > 0 else MORE_DOWN
        return self.prompt()

    def add(self, samples, now):
        if len(samples) != 1:
            self.samples.clear()
            self.message = "Placez une seule personne devant la caméra."
            return None
        sample = samples[0]
        if sample.issue:
            self.samples.clear()
            self.message = sample.issue
            return None
        if self.existing is not None and float(np.max(self.existing @ sample.feature)) < 0.35:
            self.samples.clear()
            self.message = "Ce visage ne correspond pas à celui déjà mémorisé pour ce profil."
            return None
        # Comparée à la vue déjà prise la plus proche : tête baissée, le visage ressemble moins
        # à la vue de face qu’aux autres.
        anchors = self.templates or self.samples[:1]
        minimum = TILT_ANCHOR_MIN if self.stage >= 3 else ANCHOR_MIN
        if anchors and float(np.max(np.asarray(anchors) @ sample.feature)) < minimum:
            self.samples.clear()
            self.message = "Visage différent ou angle trop fort : revenez de face."
            return None
        problem = self.mismatch(sample)
        if problem:
            self.message = problem
            if self.missed_since is None:
                self.missed_since = now
            if now - self.missed_since >= (TILT_GRACE if self.stage >= 3 else self.grace):
                self.samples.clear()
                if self.stage == 0:
                    self.pitches.clear()
            return None
        self.missed_since = None
        self.message = self.prompt()
        if self.stage == 1:
            direction = 1 if sample.pose > 0 else -1
            if self.direction is not None and direction != self.direction:
                self.samples.clear()
            self.direction = direction
        if now - self.last_sample_at < self.interval:
            return None
        self.last_sample_at = now
        self.samples.append(sample.feature)
        if self.stage == 0 and np.isfinite(sample.pitch):
            self.pitches.append(sample.pitch)
        if len(self.samples) < self.required:
            return None
        for start in range(0, self.required, self.per_template):
            vector = np.mean(self.samples[start:start + self.per_template], axis=0)
            vector /= max(float(np.linalg.norm(vector)), 1e-9)
            self.templates.append(vector)
        self.samples.clear()
        self.recent = []
        if self.stage == 0:
            # Repère personnel pour le haut et le bas : l’inclinaison mesurée de face.
            self.baseline = float(np.median(self.pitches)) if self.pitches else float("nan")
        self.stage += 1
        if self.stage == STAGES:
            return [vector.tolist() for vector in self.templates]
        self.message = self.prompt()
        return None
