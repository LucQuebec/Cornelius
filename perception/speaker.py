"""Reconnaissance du locuteur : empreinte vocale CAM++ (3D-Speaker,
Apache-2.0) calculée par sherpa-onnx, entièrement en local, sans cloud.

- Enregistrement : la personne lit un court texte devant le micro du miroir ; la parole (Silero)
  est découpée en fenêtres de 3 s, chacune donne une empreinte ; les fenêtres trop éloignées du
  centre sont écartées.
- Identification : au mot d'appel (les 2 dernières secondes du flux de veille), puis confirmation
  sur les premières secondes de la demande. Score = moyenne des trois meilleures ressemblances
  (cosinus) avec les empreintes d'un profil ; il faut dépasser le seuil ET distancer le second.

La voix sert à personnaliser (mémoire, sources), jamais à autoriser une action sensible : une
voix s'imite et s'enregistre. Ni le son ni les empreintes ne sont journalisés.
"""
import os
from collections import deque
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
WAKE_FRAME_SAMPLES = 1280          # 80 ms par trame de veille
WAKE_HISTORY_FRAMES = 25           # 2 s de son gardées par flux de veille
WINDOW_S = 3.0                     # fenêtre d'enregistrement
STEP_S = 1.5
ENROLL_SPEECH_S = 20.0             # parole nécessaire pour enregistrer une voix
MIN_TEMPLATES = 4
MAX_TEMPLATES = 12
# « Cornelius » lus dans le texte : empreintes courtes, comparables au mot d'appel (2026-11-27).
MAX_WAKE_TEMPLATES = 4
WAKE_TARGET = 3                    # attendus avant de finir, sauf lecture très longue
ENROLL_MAX_SPEECH_S = 35.0
CONFIRM_SPEECH_S = 3.0             # parole de la demande utilisée pour confirmer
DEFAULT_THRESHOLD = 0.45
DEFAULT_MARGIN = 0.08
# Le mot d'appel seul dure moins d'une seconde : son score est plus bas que sur une phrase.
# Banc du 2026-11-27 (extraits de 0,6 à 1 s) : voix justes 0,48 à 0,75, inconnues 0,18 à 0,24.
# Une voix inconnue a atteint 0,335 sur 0,8 s : seuil du mot seul à 0,35 (0,45 − 0,10).
WAKE_THRESHOLD_OFFSET = 0.10
MODEL_NAME = "campplus.onnx"


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else None


def pcm_to_float(pcm):
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768


def trim_silence(samples, frame=320):
    """Garde le passage parlé d'un seul tenant (de la première à la dernière trame de 20 ms
    au-dessus de 1/10 de la plus forte, avec 100 ms de marge) : le mot d'appel entier, sans le
    silence autour. Rien si le tout est trop faible. (Le 2026-11-27, ne garder que les trames
    fortes réduisait « Cornelius » à 0,5 s et le score sous le seuil.)"""
    count = len(samples) // frame
    if count == 0:
        return samples[:0]
    blocks = samples[:count * frame].reshape(count, frame)
    energy = np.sqrt((blocks ** 2).mean(axis=1))
    peak = float(energy.max())
    if peak < 0.005:
        return samples[:0]
    loud = np.flatnonzero(energy >= peak * 0.1)
    first, last = max(0, loud[0] - 5), min(count, loud[-1] + 6)
    return blocks[first:last].reshape(-1)


class SpeakerRecognizer:
    def __init__(self, models):
        self.path = Path(models) / MODEL_NAME
        self.extractor = None
        self.detail = None
        self.profiles = []            # [(profile_id, matrice des empreintes normalisées)]
        self.threshold = float(os.getenv("CORNELIUS_VOICE_THRESHOLD", DEFAULT_THRESHOLD))
        self.margin = float(os.getenv("CORNELIUS_VOICE_MARGIN", DEFAULT_MARGIN))

    def ready(self):
        if self.extractor is not None:
            return True
        if self.detail is not None:
            return False
        try:
            import sherpa_onnx
            if not self.path.exists():
                raise FileNotFoundError(MODEL_NAME)
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.path), num_threads=2, provider="cpu")
            self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            return True
        except Exception as error:
            self.detail = "Reconnaissance de la voix indisponible (" + type(error).__name__ + ")."
            return False

    def status(self):
        return {"type": "voice_status", "ready": self.ready(), "detail": self.detail,
                "profiles": len(self.profiles)}

    def set_profiles(self, profiles):
        loaded = []
        for profile_id, templates in profiles:
            rows = [v for v in (_normalize(t) for t in templates) if v is not None]
            if rows:
                loaded.append((profile_id, np.stack(rows)))
        self.profiles = loaded

    def configure(self, threshold=None, margin=None):
        if isinstance(threshold, (int, float)) and 0.2 <= threshold <= 0.9:
            self.threshold = float(threshold)
        if isinstance(margin, (int, float)) and 0 <= margin <= 0.5:
            self.margin = float(margin)

    def embed(self, samples):
        """Empreinte normalisée d'un extrait (float32, 16 kHz), ou None s'il est trop court."""
        if not self.ready() or len(samples) < SAMPLE_RATE * 0.5:
            return None
        stream = self.extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(samples, dtype=np.float32))
        stream.input_finished()
        return _normalize(self.extractor.compute(stream))

    def identify(self, samples, source):
        """Rend {profileId|None, score, second, source, seconds} ou None si rien à comparer."""
        if not self.profiles:
            return None
        vector = self.embed(samples)
        if vector is None:
            return None
        scored = []
        for profile_id, templates in self.profiles:
            similarities = np.sort(templates @ vector)[::-1][:3]
            scored.append((float(similarities.mean()), profile_id))
        scored.sort(reverse=True)
        best, best_id = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        threshold = self.threshold - (WAKE_THRESHOLD_OFFSET if source == "wake" else 0)
        accepted = best >= threshold and best - second >= self.margin
        return {"profileId": best_id if accepted else None, "closestProfileId": best_id,
                "score": round(best, 3),
                "second": round(second, 3), "source": source,
                "seconds": round(len(samples) / SAMPLE_RATE, 2)}


class WakeHistory:
    """Les 2 dernières secondes de chaque flux de veille, pour identifier qui a dit « Cornelius »."""

    def __init__(self):
        self.frames = {}

    def add(self, slot, pcm):
        self.frames.setdefault(slot, deque(maxlen=WAKE_HISTORY_FRAMES)).append(pcm)

    def drop(self, slot):
        self.frames.pop(slot, None)

    def take(self, slot):
        frames = self.frames.get(slot)
        if not frames:
            return np.zeros(0, dtype=np.float32)
        samples = pcm_to_float(b"".join(frames))
        frames.clear()
        return trim_silence(samples)


class RequestConfirmation:
    """Les premières secondes de parole de la demande (trames VAD de 512 échantillons)."""

    def __init__(self, interaction_id):
        self.interaction = interaction_id
        self.chunks = []
        self.samples = 0
        self.done = False

    def add(self, pcm, speaking):
        if self.done or not speaking:
            return None
        chunk = pcm_to_float(pcm)
        self.chunks.append(chunk)
        self.samples += len(chunk)
        if self.samples < CONFIRM_SPEECH_S * SAMPLE_RATE:
            return None
        self.done = True
        return np.concatenate(self.chunks)


class VoiceEnrollment:
    """Lecture du texte : la parole des flux de veille, triée par Silero, flux par flux."""

    def __init__(self, request, vad_factory):
        self.request = request
        self.vad_factory = vad_factory
        self.streams = {}             # slot -> [vad, reste < 512 échantillons, morceaux de parole, total]
        self.wakes = []               # extraits du mot « Cornelius » entendus pendant la lecture
        self.last_reported = None

    def add_wake(self, samples):
        if len(samples) >= SAMPLE_RATE * 0.4:
            self.wakes.append(samples)

    def add(self, slot, pcm):
        entry = self.streams.get(slot)
        if entry is None:
            entry = self.streams[slot] = [self.vad_factory(), b"", [], 0]
        vad, rest, chunks, _ = entry
        data = rest + pcm
        usable = len(data) // 1024 * 1024
        for offset in range(0, usable, 1024):
            block = data[offset:offset + 1024]
            if vad.process(block):
                chunks.append(pcm_to_float(block))
                entry[3] += 512
        entry[1] = data[usable:]

    def best(self):
        if not self.streams:
            return None
        return max(self.streams.values(), key=lambda entry: entry[3])

    def speech_seconds(self):
        best = self.best()
        return 0.0 if best is None else best[3] / SAMPLE_RATE

    def progress(self):
        seconds = int(self.speech_seconds())
        if (seconds, len(self.wakes)) == self.last_reported:
            return None
        self.last_reported = (seconds, len(self.wakes))
        return {"type": "voice_enrollment_progress", "id": self.request["id"],
                "speechSeconds": seconds, "requiredSeconds": int(ENROLL_SPEECH_S),
                "wakes": len(self.wakes), "requiredWakes": WAKE_TARGET}

    def complete(self):
        speech = self.speech_seconds()
        return speech >= ENROLL_SPEECH_S and (len(self.wakes) >= WAKE_TARGET
                                              or speech >= ENROLL_MAX_SPEECH_S)

    def templates(self, recognizer):
        """Empreintes retenues (listes de floats), ou (None, raison)."""
        best = self.best()
        if best is None:
            return None, "silence"
        speech = np.concatenate(best[2]) if best[2] else np.zeros(0, dtype=np.float32)
        window, step = int(WINDOW_S * SAMPLE_RATE), int(STEP_S * SAMPLE_RATE)
        vectors = []
        for start in range(0, max(0, len(speech) - window) + 1, step):
            vector = recognizer.embed(speech[start:start + window])
            if vector is not None:
                vectors.append(vector)
        if len(vectors) < MIN_TEMPLATES:
            return None, "short"
        matrix = np.stack(vectors)
        centroid = _normalize(matrix.mean(axis=0))
        kept = matrix[matrix @ centroid >= 0.5]
        if len(kept) < MIN_TEMPLATES:
            return None, "mixed"
        # Les plus représentatives d'abord ; au plus MAX_TEMPLATES.
        order = np.argsort(-(kept @ centroid))[:MAX_TEMPLATES]
        rows = [kept[i] for i in order]
        # Puis les « Cornelius » de la même voix (un mot seul ressemble moins : seuil plus bas).
        for samples in self.wakes[:MAX_WAKE_TEMPLATES]:
            vector = recognizer.embed(trim_silence(samples))
            if vector is not None and float(vector @ centroid) >= 0.3:
                rows.append(vector)
        return [[round(float(x), 6) for x in row] for row in rows], None
