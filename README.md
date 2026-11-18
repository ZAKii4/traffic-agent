# Traffic Agent — comptage de véhicules par vision

![CI](https://github.com/ZAKii4/traffic-agent/actions/workflows/ci.yml/badge.svg)

Plateforme d'analyse du trafic routier à partir de vidéos : détection (YOLO26), suivi (ByteTrack) et comptage des véhicules par classe et par sens de circulation.

## Démarrage

```bash
uv sync
uv run traffic-agent count --video vehicles.mp4 --camera demo-autoroute
```

Les lignes de comptage se configurent dans `configs/cameras.yaml`, en coordonnées relatives (0 à 1).

## Résultats

| Vidéo | Vérité terrain | Compté | Erreur |
| --- | --- | --- | --- |
| Démo autoroute (21 s, 4K) | 10 | 10 | 0 % |

## Qualité

`make lint` (ruff) et `make test` (pytest, 19 tests) tournent à chaque pull request.

## Auteurs

Zakaria Khamlich et Malak Zaoufy — EMSI, 5e année IA/Data, G1, groupe 9.
