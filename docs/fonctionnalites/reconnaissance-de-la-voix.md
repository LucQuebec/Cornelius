# La reconnaissance de la voix

Cornelius reconnaît la personne qui dit « Cornelius », sans caméra. Le profil reconnu ouvre la conversation avec sa mémoire, ses sources et ses agents, comme un visage reconnu. Tout se passe sur cet ordinateur.

## Installer

La reconnaissance de la voix accompagne le [mot d’appel](mot-d-appel.md). `scripts/install-linux.sh` l’installe avec lui. Sous Windows, ou sur une installation existante :

```sh
.venv/bin/python -m pip install -r perception/requirements-voice.txt
.venv/bin/python perception/download_models.py --no-vision
```

Puis relancez Cornelius. Sans ce paquet ou sans le modèle, la carte de l’administration indique que la reconnaissance de la voix est indisponible ; le reste fonctionne comme avant.

## Mémoriser une voix

Administration → **Le foyer et les visages** → **Reconnaissance de la voix** → **Mémoriser la voix**. La personne lit le texte affiché, près du miroir. Il faut 20 s de parole et trois « Cornelius » entendus ; un compteur suit la lecture. Pendant la lecture, « Cornelius » n’ouvre pas de conversation.

Le son vient du micro du miroir, par le flux du mot d’appel : le miroir doit être ouvert, mot d’appel activé. C’est le même micro qui sert ensuite à reconnaître la voix.

## Fonctionnement

- **Moteur** : empreinte vocale CAM++ de 3D-Speaker (entraînée sur VoxCeleb, Apache 2.0, 29 Mo), calculée par [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) sur le processeur : 17 ms pour 1,5 s de son sur un i5-9400.
- **Mémorisation** : la parole (détectée par Silero) est découpée en fenêtres de 3 s, une empreinte par fenêtre ; celles qui s’éloignent du centre sont écartées. Les « Cornelius » lus dans le texte donnent jusqu’à 4 empreintes courtes, comparables au mot d’appel. Au plus 16 empreintes par profil, dans la table `profile_voices`.
- **Au mot d’appel** : la perception garde les 2 dernières secondes de chaque flux de veille. À la détection, le passage parlé est gardé d’un seul tenant, avec 100 ms de marge, puis comparé aux empreintes des profils : score = moyenne des trois meilleures ressemblances (cosinus). Il faut dépasser le seuil (0,45 ; 0,35 sur le mot seul, qui dure moins d’une seconde) et distancer le second profil (écart 0,08). Le profil est choisi avant l’ouverture.
- **Sur la demande** : si la conversation est encore anonyme, les 3 premières secondes de parole confirment. Dans les étapes 1 et 4, le profil rejoint la conversation sans la redémarrer.
- **Priorités** : un choix explicite et un visage reconnu devant le miroir l’emportent sur la voix.
- **Miroir** : le prénom reconnu s’affiche tout de suite, avant que la caméra ne s’allume. Quand elle voit une seule personne encore inconnue, c’est la personne reconnue à la voix : le miroir garde son prénom et la mémoire lui attribue la conversation.
- **Réglage** : la carte montre les 20 dernières identifications (scores seulement) et règle le seuil et l’écart.

## Caméra seulement au mot d’appel

**Caméra et reconnaissance** → **Caméra du miroir** → **Allumer la caméra seulement quand Cornelius entend son nom**. L’option est proposée quand « Cornelius » ouvre la conversation.

- En veille, la caméra est éteinte : le miroir libère la caméra, et son voyant s’éteint.
- Elle s’allume pendant chaque conversation, pour reconnaître le visage et pour la vision locale, puis s’éteint à la fin.
- Cornelius salue comme d’habitude : on dit « Cornelius », puis on attend son bonjour.
- Réservé aux membres du foyer : une voix reconnue ouvre tout de suite. Sinon (aucune voix mémorisée, voix trop basse, reconnaissance non installée), la caméra s’allume et un visage reconnu dans les 6 s ouvre encore la conversation, après le temps de démarrage de la caméra. Ni voix ni visage reconnu : rien ne s’ouvre, et l’administration affiche « Mot d’appel sans suite ».

Techniquement, l’option est l’étape 4 ; réservée aux membres, elle garde « personnes reconnues » (voir `shared/opening.ts`). Les pages **Caméra et reconnaissance** et **Le foyer et les visages** gardent la caméra allumée tant qu’elles sont ouvertes, le temps des réglages.

## Limites

- La voix **personnalise** ; elle ne prouve pas l’identité. Une voix s’imite et s’enregistre : aucune action sensible ne doit en dépendre.
- Satellites : pas encore. Un satellite garde le profil qui lui est rattaché.
- Une voix plus basse, la distance ou un rhume font baisser le score : refaire la mémorisation au besoin, avec un « Cornelius » dit comme au quotidien.

## Vie privée

Aucun son n’est conservé ni journalisé. Les empreintes restent sur cet ordinateur. Elles sont effacées avec le profil, ou par **Oublier la voix**, et retirées aussi des sauvegardes de la base.

## Mesures

Sur un i5-9400 sans GPU, avec des voix françaises enregistrées (trois mémorisées, deux inconnues) :

| Extrait | Voix mémorisées reconnues | Voix inconnues refusées | Score moyen (mémorisée / inconnue) |
| ------- | ------------------------- | ----------------------- | ---------------------------------- |
| 0,8 s   | 89 sur 96                 | 15 sur 15               | 0,55 à 0,70 / 0,22                 |
| 1,5 s   | 47 sur 48                 | 7 sur 7                 | 0,57 à 0,73 / 0,20                 |
| 3 s     | 25 sur 25                 | 3 sur 3                 | 0,74 à 0,88 / 0,26                 |

Sur un vrai miroir (micro intégré à la webcam), « Cornelius » dit à voix normale obtient de 0,41 à 0,49.
