"""Comptage de franchissements de ligne, indépendant de tout modèle de détection."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Line:
    """Segment de comptage en pixels. Le sens « in » va du côté négatif vers le côté positif."""

    x1: float
    y1: float
    x2: float
    y2: float

    def side(self, x: float, y: float) -> float:
        """Signe du produit vectoriel : de quel côté du segment se trouve le point."""
        return (self.x2 - self.x1) * (y - self.y1) - (self.y2 - self.y1) * (x - self.x1)

    def within(self, x: float, y: float) -> bool:
        """Vrai si la projection du point tombe sur le segment (et pas sur son prolongement)."""
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        t = ((x - self.x1) * dx + (y - self.y1) * dy) / (dx * dx + dy * dy)
        return 0.0 <= t <= 1.0


@dataclass(frozen=True)
class CrossingEvent:
    camera_id: str
    line_id: str
    track_id: int
    class_name: str
    direction: str  # "in" ou "out"
    video_time_s: float


@dataclass
class LineCounter:
    """Compte chaque véhicule au plus une fois, à partir de son point bas-centre."""

    camera_id: str
    line_id: str
    line: Line
    _last_side: dict[int, float] = field(default_factory=dict)
    _counted: set[int] = field(default_factory=set)

    def update(
        self, track_id: int, class_name: str, x: float, y: float, video_time_s: float
    ) -> CrossingEvent | None:
        side = self.line.side(x, y)
        previous = self._last_side.get(track_id)
        self._last_side[track_id] = side
        if previous is None or track_id in self._counted or not self.line.within(x, y):
            return None
        if previous * side >= 0:
            return None
        self._counted.add(track_id)
        direction = "in" if previous < 0 < side else "out"
        return CrossingEvent(
            self.camera_id, self.line_id, track_id, class_name, direction, video_time_s
        )
