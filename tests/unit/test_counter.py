import pytest

from traffic_agent.vision.counter import CrossingEvent, Line, LineCounter

LINE = Line(0, 100, 200, 100)  # segment horizontal à y = 100, de x = 0 à x = 200


def feed(
    counter: LineCounter, track_id: int, points: list[tuple[float, float]]
) -> list[CrossingEvent]:
    events = [counter.update(track_id, "car", x, y, float(t)) for t, (x, y) in enumerate(points)]
    return [e for e in events if e is not None]


def new_counter() -> LineCounter:
    return LineCounter("cam", "ligne", LINE)


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([(50, 80), (50, 120)], "in"),
        ([(50, 120), (50, 80)], "out"),
    ],
)
def test_direction(points: list[tuple[float, float]], expected: str) -> None:
    assert [e.direction for e in feed(new_counter(), 1, points)] == [expected]


def test_hesitation_sur_la_ligne_comptee_une_seule_fois() -> None:
    zigzag = [(50, 90), (50, 110), (50, 95), (50, 105)]
    assert len(feed(new_counter(), 1, zigzag)) == 1


def test_passage_hors_segment_ignore() -> None:
    assert feed(new_counter(), 1, [(300, 80), (300, 120)]) == []


def test_premiere_position_jamais_comptee() -> None:
    assert feed(new_counter(), 1, [(50, 120)]) == []


def test_deux_vehicules_comptes_separement() -> None:
    counter = new_counter()
    assert len(feed(counter, 1, [(50, 80), (50, 120)])) == 1
    assert len(feed(counter, 2, [(60, 80), (60, 120)])) == 1


def test_evenement_contient_le_contexte() -> None:
    event = feed(new_counter(), 7, [(50, 80), (50, 120)])[0]
    assert (event.camera_id, event.line_id, event.track_id, event.video_time_s) == (
        "cam",
        "ligne",
        7,
        1.0,
    )


def test_deplacement_sans_franchissement_non_compte() -> None:
    assert feed(new_counter(), 1, [(50, 60), (50, 70), (50, 90)]) == []
