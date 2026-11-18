"""Configuration unique de la démo : lignes (coordonnées relatives 0-1) et vérité terrain."""

VEHICLE_CLASSES = [1, 2, 3, 5, 7]
IMGSZ = 1280

LINES = {
    "vers_camera": (0.02, 0.75, 0.47, 0.75),
    "s_eloigne": (0.48, 0.45, 0.90, 0.45),
}

GROUND_TRUTH = {"vers_camera": 7, "s_eloigne": 3}
