import type { SessionMode } from './types.js';

/**
 * Ouverture de la conversation, en deux questions indépendantes :
 * - ce qui l’ouvre : la présence, le regard ou le mot « Cornelius » ;
 * - qui peut l’ouvrir : tout le monde ou les seuls membres du foyer reconnus.
 *
 * Les quatre réglages historiques (étape 1 à 4, personnes reconnues, mot d’appel,
 * reconnaissance au mot d’appel) se recouvraient : l’étape 1 avec « personnes reconnues »
 * valait l’étape 2, et la voix existait deux fois (mot d’appel exigé dans une étape caméra,
 * ou étape 4). Chaque combinaison ci-dessous a maintenant une seule écriture.
 *
 * | Ce qui ouvre | Qui           | Étape | Reconnues | Mot d’appel | Caméra en veille |
 * |--------------|---------------|-------|-----------|-------------|------------------|
 * | présence     | tout le monde | 1     | non       | non         | allumée          |
 * | présence     | membres       | 2     | non       | non         | allumée          |
 * | regard       | tout le monde | 3     | non       | non         | allumée          |
 * | regard       | membres       | 3     | oui       | non         | allumée          |
 * | « Cornelius »| tout le monde | 4     | non       | oui         | éteinte          |
 * | « Cornelius »| membres       | 2     | non       | oui         | allumée          |
 *
 * « Cornelius » réservé aux membres garde la caméra allumée en veille : c’est le seul moyen de
 * savoir, au mot d’appel, qui parle avant d’ouvrir la conversation.
 *
 * Option « caméra seulement au mot d’appel » : avec « Cornelius », la caméra
 * dort en veille dans tous les cas (étape 4) ; la reconnaissance de la voix dit qui parle, et
 * « membres » devient l’étape 4 avec « personnes reconnues » (une voix reconnue est exigée).
 */
export const OPENING_TRIGGERS = ['presence', 'gaze', 'voice'] as const;
export type OpeningTrigger = (typeof OPENING_TRIGGERS)[number];
export const OPENING_AUDIENCES = ['everyone', 'members'] as const;
export type OpeningAudience = (typeof OPENING_AUDIENCES)[number];

export interface Opening {
  trigger: OpeningTrigger;
  audience: OpeningAudience;
  /** « Cornelius » pour tout le monde : reconnaître la personne pendant la conversation. */
  recognizeVoice: boolean;
  /** « Cornelius » : caméra éteinte en veille, allumée seulement pendant la conversation. */
  cameraOnWake?: boolean;
}

export interface OpeningSettings {
  mode: SessionMode;
  knownPeopleOnly: boolean;
  wakeWord: boolean;
  voiceRecognition: boolean;
  cameraOnWake?: boolean;
}

export function isOpening(value: unknown): value is Opening {
  if (typeof value !== 'object' || value === null) return false;
  const entry = value as Record<string, unknown>;
  return (
    Object.keys(entry).every((key) =>
      ['trigger', 'audience', 'recognizeVoice', 'cameraOnWake'].includes(key),
    ) &&
    (entry.cameraOnWake === undefined || typeof entry.cameraOnWake === 'boolean') &&
    (OPENING_TRIGGERS as readonly unknown[]).includes(entry.trigger) &&
    (OPENING_AUDIENCES as readonly unknown[]).includes(entry.audience) &&
    typeof entry.recognizeVoice === 'boolean'
  );
}

/** Lit les réglages enregistrés, y compris les combinaisons anciennes qui se recouvraient. */
export function openingFromSettings(settings: OpeningSettings): Opening {
  const { mode, knownPeopleOnly, wakeWord, voiceRecognition } = settings;
  const cameraOnWake = settings.cameraOnWake === true;
  // Absent quand désactivée : les ouvertures sans l’option gardent leur forme d’avant.
  const wakeCamera = cameraOnWake ? { cameraOnWake } : {};
  if (mode === 4)
    return {
      trigger: 'voice',
      audience: knownPeopleOnly && cameraOnWake ? 'members' : 'everyone',
      recognizeVoice: voiceRecognition,
      ...wakeCamera,
    };
  const members = mode === 2 || knownPeopleOnly;
  // Ancien « mot d’appel exigé » dans une étape caméra : la personne était déjà reconnue.
  if (wakeWord)
    return {
      trigger: 'voice',
      audience: members ? 'members' : 'everyone',
      recognizeVoice: members ? voiceRecognition : true,
      ...wakeCamera,
    };
  return {
    trigger: mode === 3 ? 'gaze' : 'presence',
    audience: members ? 'members' : 'everyone',
    recognizeVoice: voiceRecognition,
    ...wakeCamera,
  };
}

export function settingsForOpening(opening: Opening): OpeningSettings {
  const members = opening.audience === 'members';
  const voiceRecognition = opening.recognizeVoice;
  const cameraOnWake = opening.cameraOnWake === true;
  const wakeCamera = cameraOnWake ? { cameraOnWake } : {};
  if (opening.trigger === 'voice' && cameraOnWake)
    return { mode: 4, knownPeopleOnly: members, wakeWord: true, voiceRecognition, ...wakeCamera };
  if (opening.trigger === 'voice')
    return members
      ? { mode: 2, knownPeopleOnly: false, wakeWord: true, voiceRecognition, ...wakeCamera }
      : { mode: 4, knownPeopleOnly: false, wakeWord: true, voiceRecognition, ...wakeCamera };
  if (opening.trigger === 'gaze')
    return { mode: 3, knownPeopleOnly: members, wakeWord: false, voiceRecognition, ...wakeCamera };
  return {
    mode: members ? 2 : 1,
    knownPeopleOnly: false,
    wakeWord: false,
    voiceRecognition,
    ...wakeCamera,
  };
}

export function sameOpeningSettings(a: OpeningSettings, b: OpeningSettings): boolean {
  return (
    a.mode === b.mode &&
    a.knownPeopleOnly === b.knownPeopleOnly &&
    a.wakeWord === b.wakeWord &&
    a.voiceRecognition === b.voiceRecognition &&
    (a.cameraOnWake === true) === (b.cameraOnWake === true)
  );
}

/**
 * La caméra reste-t-elle allumée en veille ? Oui, sauf avec « Cornelius » pour tout le monde
 * sans reconnaissance : avec la reconnaissance, elle guette déjà la personne pour la reconnaître
 * dès qu’elle dit « Cornelius », sans attendre le démarrage de la caméra.
 */
export function cameraWatchesInStandby(opening: Opening): boolean {
  if (opening.trigger === 'voice' && opening.cameraOnWake) return false;
  return !voiceOnlyOpening(opening) || opening.recognizeVoice;
}

/**
 * « Cornelius » pour tout le monde, ou caméra seulement au mot d’appel (étape 4) : seule la voix
 * ouvre, et seul le silence ferme.
 */
export function voiceOnlyOpening(opening: Opening): boolean {
  return (
    opening.trigger === 'voice' &&
    (opening.audience === 'everyone' || opening.cameraOnWake === true)
  );
}

/**
 * Ouverture publiée par le serveur ; un serveur antérieur ne publiait que les anciens réglages,
 * relus ici. Le type d’entrée reste structurel pour ne pas dépendre de `MirrorState`.
 */
export function currentOpening(state: {
  opening?: Opening;
  session: { mode: SessionMode };
  knownPeopleOnly?: boolean;
  wakeWord: { enabled: boolean };
  voiceRecognition?: boolean;
  cameraOnWake?: boolean;
}): Opening {
  return (
    state.opening ??
    openingFromSettings({
      mode: state.session.mode,
      knownPeopleOnly: state.knownPeopleOnly === true,
      wakeWord: state.wakeWord.enabled,
      voiceRecognition: state.voiceRecognition === true,
      cameraOnWake: state.cameraOnWake === true,
    })
  );
}
