"""Interface en ligne de commande : traffic-agent count --camera ... --video ..."""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from traffic_agent.config import load_config

app = typer.Typer(help="Agent de surveillance routière : comptage de véhicules.")


@app.callback()
def main() -> None:
    """Point d'entrée (sous-commandes ci-dessous)."""


@app.command()
def count(
    video: Annotated[Path, typer.Option(exists=True, help="Fichier vidéo à analyser")],
    camera: Annotated[str, typer.Option(help="Identifiant de caméra dans la configuration")],
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/cameras.yaml"),
    out: Annotated[Path, typer.Option(help="Un événement JSON par ligne")] = Path(
        "runs/events.jsonl"
    ),
    weights: Annotated[str, typer.Option(help="Poids du modèle de détection")] = "yolo26n.pt",
) -> None:
    """Compte les véhicules qui franchissent les lignes de la caméra choisie."""
    from traffic_agent.vision.detector import YoloDetector
    from traffic_agent.vision.pipeline import count_video

    camera_config = load_config(config).camera(camera)
    detector = YoloDetector(weights=weights)
    out.parent.mkdir(parents=True, exist_ok=True)
    totals: Counter[tuple[str, str, str]] = Counter()
    with out.open("w", encoding="utf-8") as f:
        for event in count_video(str(video), camera_config, detector):
            f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
            totals[(event.line_id, event.direction, event.class_name)] += 1
            typer.echo(
                f"{event.video_time_s:6.1f} s  {event.line_id:12s} "
                f"{event.direction:3s}  #{event.track_id:<4d} {event.class_name}"
            )
    typer.echo("\nLigne         Sens  Classe      Nombre")
    for (line_id, direction, class_name), n in sorted(totals.items()):
        typer.echo(f"{line_id:13s} {direction:5s} {class_name:11s} {n:6d}")
    typer.echo(f"\nTotal : {sum(totals.values())} franchissements -> {out}")
