import { useEffect, useState } from 'react';
import { Check, CircleAlert, Mic, RotateCcw, Trash2 } from 'lucide-react';
import type { DevicesStatus, MirrorState } from '../../../shared/types';
import { api } from '../../api';
import type { RunAction } from './types';
import { formatWhen } from './format';
import { ConfirmAction } from './shared';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Alert, AlertDescription } from '../ui/alert';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/card';

/**
 * Texte lu à voix haute pour mémoriser une voix (environ 30 s). Des sons variés, et
 * « Cornelius » plusieurs fois : c’est sur ce mot que la voix est reconnue au quotidien.
 */
const READING = [
  'Bonjour Cornelius. Je m’appelle {name} et j’habite ici.',
  'Le matin, j’aime écouter la radio en buvant mon café, puis je regarde la météo avant de partir.',
  'Cornelius, quel temps fera-t-il demain ?',
  'Les enfants jouent dans le jardin pendant que le chat dort au soleil.',
  'Cornelius, rappelle-moi d’acheter du pain.',
  'Le soir, nous préparons le souper ensemble et parlons de notre journée.',
  'Cornelius, merci.',
];

/** Reconnaissance du locuteur : chaque personne lit le texte devant le miroir. */
export function VoiceEnrollment({
  state,
  status,
  run,
}: {
  state: MirrorState;
  status: DevicesStatus | null;
  run: RunAction;
}) {
  const voice = status?.perception.voice;
  const [threshold, setThreshold] = useState('');
  const [margin, setMargin] = useState('');
  const [forgetting, setForgetting] = useState<string | null>(null);
  useEffect(() => {
    if (voice) {
      setThreshold(String(voice.threshold));
      setMargin(String(voice.margin));
    }
  }, [voice?.threshold, voice?.margin]);
  if (!status?.perception.connected) return null;
  if (!voice) return null; // Serveur sans la reconnaissance de la voix.
  const name = (id: string | null) =>
    (id && state.profiles.find((profile) => profile.id === id)?.name) || '—';
  const enrolling = voice.enrollment;
  const result = !enrolling ? voice.result : null;
  const enrolled = new Map(voice.voices.map((entry) => [entry.profileId, entry]));

  return (
    <Card id="voice-enrollment" className="scroll-mt-4">
      <CardHeader>
        <CardTitle>Reconnaissance de la voix</CardTitle>
        <CardDescription>
          Chaque personne lit un court texte près du miroir. Cornelius la reconnaît ensuite quand
          elle dit « Cornelius », même sans caméra. Aucun son n’est conservé : seulement des
          empreintes numériques, sur cet ordinateur. La voix personnalise, elle ne sert jamais de
          mot de passe.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {!voice.ready && (
          <Alert role="status">
            <AlertDescription>
              <p>{voice.detail ?? 'Reconnaissance de la voix en cours de chargement.'}</p>
            </AlertDescription>
          </Alert>
        )}

        {enrolling && (
          <div className="space-y-3 rounded-lg border p-4" aria-live="polite">
            <p className="font-medium">
              {name(enrolling.profileId)}, lisez ce texte à voix haute, à votre rythme, près du
              miroir :
            </p>
            <blockquote className="space-y-1 border-l-4 pl-4 text-lg leading-relaxed">
              {READING.map((line) => (
                <p key={line}>{line.replace('{name}', name(enrolling.profileId))}</p>
              ))}
            </blockquote>
            <p className="text-sm text-muted-foreground">
              Parole entendue : {Math.min(enrolling.speechSeconds, enrolling.requiredSeconds)} s sur{' '}
              {enrolling.requiredSeconds} s
              {enrolling.requiredWakes
                ? ` · « Cornelius » entendus : ${Math.min(enrolling.wakes ?? 0, enrolling.requiredWakes)} sur ${enrolling.requiredWakes}`
                : ''}
              . Relisez le texte si la barre n’est pas pleine.
            </p>
            <progress
              className="h-2 w-full"
              max={enrolling.requiredSeconds}
              value={Math.min(enrolling.speechSeconds, enrolling.requiredSeconds)}
            />
            <Button
              size="sm"
              variant="outline"
              onClick={() => void run('Enregistrement annulé.', () => api.cancelVoiceEnrollment())}
            >
              Annuler
            </Button>
          </div>
        )}

        {result && state.profiles.some((profile) => profile.id === result.profileId) && (
          <Alert role="status">
            <AlertDescription>
              <p className="flex items-start gap-2">
                {result.ok ? (
                  <Check className="mt-0.5 size-4 shrink-0 text-primary" aria-hidden="true" />
                ) : (
                  <CircleAlert className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden="true" />
                )}
                {name(result.profileId)} : {result.message}
                {result.ok && ' Dites « Cornelius » pour vérifier qu’elle est bien reconnue.'}
              </p>
            </AlertDescription>
          </Alert>
        )}

        <ul className="divide-y rounded-lg border">
          {state.profiles.map((profile) => {
            const entry = enrolled.get(profile.id);
            return (
              <li key={profile.id} className="flex flex-wrap items-center gap-3 p-3">
                <span className="min-w-32 font-medium">{profile.name}</span>
                <span className="flex-1 text-sm text-muted-foreground">
                  {entry
                    ? `Voix mémorisée ${formatWhen(entry.enrolledAt)} (${entry.templates} empreintes)`
                    : 'Aucune voix mémorisée'}
                </span>
                {entry ? (
                  <div className="flex flex-wrap gap-2">
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={!voice.ready || !!enrolling}
                      aria-label={'Refaire la voix de ' + profile.name}
                      title="Remplace la voix mémorisée par une nouvelle lecture du texte."
                      onClick={() =>
                        void run('Lecture du texte lancée.', () => api.enrollVoice(profile.id))
                      }
                    >
                      <RotateCcw />
                      Refaire
                    </Button>
                    <Button
                      size="sm"
                      variant="danger-outline"
                      disabled={enrolling?.profileId === profile.id}
                      aria-label={'Oublier la voix de ' + profile.name}
                      onClick={() => setForgetting(profile.id)}
                    >
                      <Trash2 />
                      Oublier
                    </Button>
                  </div>
                ) : (
                  <Button
                    size="sm"
                    disabled={!voice.ready || !!enrolling}
                    aria-label={'Mémoriser la voix de ' + profile.name}
                    onClick={() =>
                      void run('Lecture du texte lancée.', () => api.enrollVoice(profile.id))
                    }
                  >
                    <Mic />
                    Mémoriser la voix
                  </Button>
                )}
                {forgetting === profile.id && (
                  <ConfirmAction
                    message={`Oublier la voix de ${profile.name} ? Cornelius ne la reconnaîtra plus avant une nouvelle lecture du texte.`}
                    confirmLabel="Oublier la voix"
                    onCancel={() => setForgetting(null)}
                    onConfirm={() => {
                      void run('Voix oubliée.', () => api.forgetVoice(profile.id)).then(() =>
                        setForgetting(null),
                      );
                    }}
                  />
                )}
              </li>
            );
          })}
        </ul>

        {voice.history.length > 0 && (
          <div className="space-y-2">
            <p className="text-sm font-medium">Dernières identifications</p>
            <table className="w-full text-sm">
              <thead className="text-left text-muted-foreground">
                <tr>
                  <th className="font-normal">Quand</th>
                  <th className="font-normal">Sur</th>
                  <th className="font-normal">Reconnu</th>
                  <th className="font-normal">Plus proche</th>
                  <th className="font-normal">Score</th>
                  <th className="font-normal">Second</th>
                </tr>
              </thead>
              <tbody>
                {voice.history.map((entry) => (
                  <tr key={entry.at + entry.source}>
                    <td>{new Date(entry.at).toLocaleTimeString('fr-CA')}</td>
                    <td>{entry.source === 'wake' ? '« Cornelius »' : 'demande'}</td>
                    <td>{entry.profileId ? name(entry.profileId) : 'personne'}</td>
                    <td>{name(entry.closestProfileId)}</td>
                    <td>{entry.score.toFixed(2)}</td>
                    <td>{entry.second.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <form
          className="flex flex-wrap items-end gap-3"
          onSubmit={(event) => {
            event.preventDefault();
            void run('Seuil de la voix enregistré.', () =>
              api.setVoiceThreshold(Number(threshold), Number(margin)),
            );
          }}
        >
          <div className="space-y-1">
            <Label htmlFor="voice-threshold">Seuil (0,20 à 0,90 ; − 0,10 sur « Cornelius »)</Label>
            <Input
              id="voice-threshold"
              className="w-28"
              inputMode="decimal"
              value={threshold}
              onChange={(event) => setThreshold(event.target.value.replace(',', '.'))}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="voice-margin">Écart avec le second</Label>
            <Input
              id="voice-margin"
              className="w-28"
              inputMode="decimal"
              value={margin}
              onChange={(event) => setMargin(event.target.value.replace(',', '.'))}
            />
          </div>
          <Button size="sm" variant="outline" type="submit">
            Enregistrer
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
