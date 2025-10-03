# Create a 2-page A4 rules sheet (Enemies + Buildings) in the parchment theme.
# Outputs: a 2-page PDF and two PNG pages.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from textwrap import fill
import matplotlib.patches as patches

# ---------- Theme helpers ----------
A4_INCH = (8.27, 11.69)

def parchment_ax(fig):
    ax = fig.add_axes([0,0,1,1])
    ax.set_axis_off()
    # parchment background (beige + noise)
    ax.add_patch(patches.Rectangle((0,0),1,1,transform=ax.transAxes, facecolor="#EBD9BA", zorder=0))
    # add subtle noise
    rng = np.random.default_rng(42)
    noise = rng.uniform(-0.03, 0.03, size=(200, 200))
    ax.imshow(noise, extent=[0,1,0,1], cmap="gray", alpha=0.15, zorder=1, interpolation="bilinear")
    # border
    border = patches.FancyBboxPatch((0.02,0.02),0.96,0.96, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    edgecolor="#48290F", facecolor="none", linewidth=6, zorder=2)
    inner  = patches.FancyBboxPatch((0.035,0.035),0.93,0.93, boxstyle="round,pad=0.01,rounding_size=0.02",
                                    edgecolor="#7A4B23", facecolor="none", linewidth=2, zorder=2)
    ax.add_patch(border); ax.add_patch(inner)
    return ax

def draw_title(ax, title, subtitle=None):
    ax.text(0.5, 0.955, title, ha="center", va="top",
            fontsize=24, fontweight="bold", color="#3A2A17")
    if subtitle:
        ax.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=11, color="#5A4631")

def make_table(ax, df, bbox, col_widths=None, fontsize=11):
    tbl = ax.table(cellText=df.values,
                   colLabels=df.columns.tolist(),
                   cellLoc="left", colLoc="center",
                   bbox=bbox)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    # header style
    for (r,c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2E2A24")
            cell.set_text_props(weight="bold", color="white")
            cell.set_height(cell.get_height()*1.12)
        else:
            cell.set_facecolor("#F6EAD0" if r%2==0 else "#FDF7E8")
    return tbl

def wrap_df(df, widths):
    out = df.copy()
    for col, width in widths.items():
        out[col] = out[col].apply(lambda x: fill(str(x), width=width))
    return out

# ---------- Data ----------
enemies = [
    # ["Forest Spider", "2", "1 (Melee)", "2",
    #  "Patrols nearest 3-tile radius from spawn. Attacks nearest Worker. If killed → drops 1 Biomass."],
    # ["Corvus Bird", "4", "2 (Area/Swoop)", "1",
    #  "Appears via Event. Swoop: choose a 3-tile line; units there roll 1d6 + Defense ≥ 5 or take 1 damage. Leaves after."],
    # ["Swamp Leech", "1", "1 (Melee)", "2",
    #  "Spawns on swamp/jungle. When it damages a unit, reduce Food supply by 1."],
    # ["Scavenger Rat", "3", "0 (Steals)", "1",
    #  "Steals 1 Food from any resource tile or open storage. If adjacent to Colony/Outpost, 50% chance to steal. Flees after."],
    # ["Fungal Spore", "0", "0", "1",
    #  "Infects its spawn tile on Event. Workers there must spend an action to remove or lose 1 Food at Upkeep. Cure: Medical Fungarium or spend 1 Biomass at Colony."],
    # ["Giant Beetle", "2", "2 (Melee; targets structures first)", "3",
    #  "Roams toward nearest Colony/Outpost within 3 tiles; attacks structures first (Outposts/roads treated as HP 2). Drops 1 Material on death."]
]
df_enemies = pd.DataFrame(enemies, columns=["Name", "Move", "Attack", "HP", "Spawn / Behavior"])

buildings = [
    # ["Storage Chamber", "Upgrade", "2 Materials", "—", "+3 Storage"],
    # ["Brood Chamber", "Upgrade", "3 Materials", "—", "+1 Population Cap (+1 Food upkeep)"],
    # ["Road Segment", "Build", "1 Material", "—", "Place road along chosen edge"],
    # ["Outpost", "Build", "3 Materials", "Continuous road required", "+2 Storage + extends influence 2 tiles"],
    # ["Foraging Techniques", "Upgrade (Tech)", "1 Material + 1 Bio", "—", "Workers gather +1 when gathering"],
    # ["Pheromone Network", "Upgrade (Tech)", "3 Bio", "—", "Auto-transfer +1 resource along roads during Logistics"],
    # ["Medical Fungarium", "Upgrade", "2 Bio", "—", "Instant cure action for Fungal Spore"]
]
df_buildings = pd.DataFrame(buildings, columns=["Building", "Type", "Resource Requirements", "Requirements", "Effect"])

# Wrap text to fit columns for readability
df_enemies_w = wrap_df(df_enemies, {
    "Name": 16, "Move": 4, "Attack": 14, "HP": 3, "Spawn / Behavior": 52
})
df_buildings_w = wrap_df(df_buildings, {
    "Building": 18, "Type": 14, "Resource Requirements": 22, "Requirements": 24, "Effect": 34
})

# ---------- Page 1: Enemies ----------
fig1 = plt.figure(figsize=A4_INCH)
ax1 = parchment_ax(fig1)
draw_title(ax1, "Enemies", "Stat key: Move (hexes/activation) • Attack (base damage & type) • HP (hits to defeat)")
make_table(ax1, df_enemies_w, bbox=[0.055, 0.08, 0.89, 0.83], fontsize=11)

# ---------- Page 2: Buildings ----------
fig2 = plt.figure(figsize=A4_INCH)
ax2 = parchment_ax(fig2)
draw_title(ax2, "Buildings & Tech", "Costs shown in Materials/Bio. “—” means no extra requirement.")
make_table(ax2, df_buildings_w, bbox=[0.055, 0.12, 0.89, 0.79], fontsize=12)

# ---------- Save outputs ----------
pdf_path = "./Insectpire_Enemies_and_Buildings_A4.pdf"
png1 = "./Insectpire_Enemies_A4.png"
png2 = "./Insectpire_Buildings_A4.png"
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig1, bbox_inches="tight")
    pdf.savefig(fig2, bbox_inches="tight")
fig1.savefig(png1, dpi=300, bbox_inches="tight")
fig2.savefig(png2, dpi=300, bbox_inches="tight")

(pdf_path, png1, png2)
