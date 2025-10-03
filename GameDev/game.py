# hex_world.py
# Simple pointy-top hex map generator with biomes + spawn placement
# Run: python hex_world.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random

# ---------- Config ----------
HEX_SIZE = 15          # pixel radius of each hex
CHUNK_RADIUS = 7      # how many axial steps from (0,0) to generate
N_PLAYERS = 4          # number of starting positions to place
ELEV_MEAN = 0.55       # mean elevation for random terrain
ELEV_STD = 0.23        # std dev for random terrain
SEED = 42              # set to None for different map each run

# ---------- Hex helpers ----------
SQRT3 = np.sqrt(3.0)

class Hex:
    """Axial coordinates for pointy-top hexes."""
    __slots__ = ("q", "r")
    def __init__(self, q: int, r: int):
        self.q = q
        self.r = r
    def __hash__(self):
        return hash((self.q, self.r))
    def __eq__(self, other):
        return isinstance(other, Hex) and self.q == other.q and self.r == other.r
    def __repr__(self):
        return f"Hex(q={self.q}, r={self.r})"
    def neighbors(self):
        # Axial directions (pointy-top)
        dirs = [(+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1)]
        return [Hex(self.q + dq, self.r + dr) for dq, dr in dirs]

def hex_to_pixel(h: Hex, size: float):
    """Axial -> pixel for pointy-top orientation."""
    x = size * (SQRT3 * h.q + SQRT3/2.0 * h.r)
    y = size * (1.5 * h.r)
    return (x, y)

def hex_polygon_vertices(h: Hex, size: float):
    """Return a list of (x,y) vertices for the hex polygon."""
    cx, cy = hex_to_pixel(h, size)
    verts = []
    # Pointy-top: start angle at -30 deg then step by 60
    for i in range(6):
        angle = np.deg2rad(60 * i - 30)
        vx = cx + size * np.cos(angle)
        vy = cy + size * np.sin(angle)
        verts.append((vx, vy))
    return verts

# ---------- Biome coloring ----------
def biome_color(elev: float) -> str:
    """
    Very simple thresholds:
      <0.20 water
      <0.30 beach
      <0.60 plains
      <0.80 forest
      else mountain
    """
    if elev < 0.20: return "#3a7bd5"  # water
    if elev < 0.30: return "#ffe29f"  # beach
    if elev < 0.60: return "#7ec850"  # plains
    if elev < 0.80: return "#4a8f29"  # forest
    return "#888888"                  # mountain

# ---------- Terrain generation ----------
def gen_chunk(center_q: int, center_r: int, radius: int) -> dict:
    """
    Generate a square-like chunk in axial coords (not perfect hex radius mask).
    Returns dict[Hex] -> elevation in [0,1].
    """
    rng = np.random.default_rng(SEED) if SEED is not None else np.random.default_rng()
    elev = {}
    for dq in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            h = Hex(center_q + dq, center_r + dr)
            # Gaussian noise terrain, clipped to [0,1]
            val = float(np.clip(rng.normal(ELEV_MEAN, ELEV_STD), 0.0, 1.0))
            elev[h] = val
    return elev

# ---------- Start placement ----------
def axial_sqdist(a: Hex, b: Hex) -> int:
    """A fast 'spread' metric for greedy spacing; not true hex distance but works."""
    dq = a.q - b.q
    dr = a.r - b.r
    return dq * dq + dr * dr

def place_starts(hexes: list, elevs: dict, n_players: int = 4) -> list:
    """
    Greedy 'farthest next' placement:
      - consider only landish hexes (0.3..0.7 elevation)
      - first pick is arbitrary max candidate
      - each next pick maximizes min distance to existing picks
    """
    candidates = [h for h in hexes if 0.30 <= elevs[h] <= 0.70]
    if not candidates:
        return []
    starts = []
    # first: pick the highest-elevation candidate inside the band (nice land)
    first = max(candidates, key=lambda h: elevs[h])
    starts.append(first)
    while len(starts) < n_players and candidates:
        best = max(
            candidates,
            key=lambda h: min(axial_sqdist(h, s) for s in starts)
        )
        starts.append(best)
        candidates.remove(best)
    return starts

# ---------- Drawing ----------
def draw_hex(ax, h: Hex, size: float, color: str, edge: str = "black", lw: float = 0.25):
    verts = hex_polygon_vertices(h, size)
    poly = patches.Polygon(verts, closed=True, facecolor=color, edgecolor=edge, linewidth=lw)
    ax.add_patch(poly)

def draw_world(hex_size=HEX_SIZE, chunk_radius=CHUNK_RADIUS, n_players=N_PLAYERS):
    elev_map = gen_chunk(0, 0, chunk_radius)

    # Landscape plot 
    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    ax.set_aspect("equal")
    ax.axis("off")

    # draw terrain
    for h, elev in elev_map.items():
        draw_hex(ax, h, hex_size, biome_color(elev))

    # place and draw starts
    starts = place_starts(list(elev_map.keys()), elev_map, n_players)
    for s in starts:
        x, y = hex_to_pixel(s, hex_size)
        ax.plot(x, y, "o", markersize=10)  # default matplotlib color

    # framing
    # set limits with a small margin
    xs, ys = zip(*(hex_to_pixel(h, hex_size) for h in elev_map.keys()))
    margin = hex_size * 4
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    plt.tight_layout()
    plt.savefig("hex_world.png", dpi=300)   # save first
    plt.show()                                # then display
    plt.close(fig)



if __name__ == "__main__":
    draw_world()
