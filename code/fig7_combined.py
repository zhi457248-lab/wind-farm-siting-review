"""
fig7_combined.py -- Reproduce Table 3 and Figure 7 from the original SCADA data.

Reads the six raw wind-farm xlsx files and computes the summary statistics
reported in Table 3 (tab:sites) and the three panels of Figure 7:
  (a) mean hub-height wind speed vs. capacity factor;
  (b) per-site Pearson correlation between hub-height wind speed and active
      power output;
  (c) normalised power curves (mean active power / nominal capacity).

Key processing rules (these are the exact rules used to obtain the numbers
in the paper):
  * Hub-height wind speed is the column "Wind speed - at the height of wheel
    hub (m/s)".
  * Mean hub-height wind speed is averaged over records with ws > 0 only.
  * Capacity factor is mean active power over records with power > 0 only,
    divided by the nominal capacity.
  * Pearson r is computed on the subset of records where both ws > 0 and
    power > 0.
  * Timestamps written as "YYYY-MM-DD 24:00:00" (an Excel-style end-of-day
    convention appearing in Site 6) are converted to 00:00:00 of the next day.

Outputs:
  /Users/juicy/风电选址综述/fig7_combined.png
  /Users/juicy/风电选址综述/fig7_combined.pdf
  /Users/juicy/风电选址综述/tab_sites.csv
  /Users/juicy/风电选址综述/tab_scada_correlations.csv

Data source:
  /Users/juicy/vs code/wind_siting/data/raw/wind_farms/
  (the two file variants per site, with/without nominal-capacity suffix,
  are byte-for-byte identical; the script prefers the short-name variant).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DATA = Path("/Users/juicy/vs code/wind_siting/data/raw/wind_farms")
OUT_DIR = Path("/Users/juicy/风电选址综述")

CAPACITIES = {1: 99, 2: 200, 3: 99, 4: 66, 5: 36, 6: 96}
SITE_LABELS = {
    1: "S1 (99MW)",
    2: "S2 (200MW)",
    3: "S3 (99MW)",
    4: "S4 (66MW)",
    5: "S5 (36MW)",
    6: "S6 (96MW)",
}
COLORS = {
    1: "#1b9e77",
    2: "#1f78b4",
    3: "#d95f02",
    4: "#b15928",
    5: "#7570b3",
    6: "#66a61e",
}
MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v", 6: "+"}

RAW_COLUMNS = [
    "Time",
    "ws10",
    "wd10",
    "ws30",
    "wd30",
    "ws50",
    "wd50",
    "ws_hub",
    "wd_hub",
    "temp",
    "pres",
    "rh",
    "power",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_time(value):
    """Convert Excel timestamps; handle the 24:00 end-of-day convention."""
    if isinstance(value, str) and " 24:00:00" in value:
        base = value.replace(" 24:00:00", " 00:00:00")
        return pd.Timestamp(base) + pd.Timedelta(days=1)
    return pd.to_datetime(value, errors="coerce")


def load_site(site_id):
    """Load one SCADA xlsx file and standardise columns."""
    short_name = BASE_DATA / f"Wind farm site {site_id} .xlsx"
    long_name = BASE_DATA / (
        f"Wind farm site {site_id} "
        f"(Nominal capacity-{CAPACITIES[site_id]}MW).xlsx"
    )
    path = short_name if short_name.exists() else long_name
    if not path.exists():
        raise FileNotFoundError(f"Site {site_id} data not found: {path}")

    df = pd.read_excel(path, sheet_name=0, header=0, usecols=list(range(13)))
    df.columns = RAW_COLUMNS

    df["Time"] = df["Time"].map(parse_time)
    df["ws_hub"] = pd.to_numeric(df["ws_hub"], errors="coerce")
    df["power"] = pd.to_numeric(df["power"], errors="coerce")
    df["site_id"] = site_id

    # Keep all records, including duplicates, to match the published n = 70,176.
    return df


def compute_site_stats(df):
    """Return Table-3 statistics for a single site DataFrame."""
    cap = CAPACITIES[df["site_id"].iloc[0]]
    ws = df["ws_hub"]
    pw = df["power"]

    ws_pos = ws > 0
    pw_pos = pw > 0
    both = ws_pos & pw_pos

    return {
        "n_records": len(df),
        "n_valid_time": df["Time"].notna().sum(),
        "mean_hub_ws": ws[ws_pos].mean(),
        "capacity_factor": pw[pw_pos].mean() / cap * 100.0,
        "pearson_r": ws[both].corr(pw[both]),
        "capacity_MW": cap,
    }


def build_binned_curve(df, cap, ws_min=0.0, ws_max=20.0, bin_width=0.5):
    """Return binned (hub wind speed, normalised power) curve."""
    valid = df[(df["ws_hub"].notna()) & (df["power"].notna()) & (df["power"] >= 0)]
    bins = np.arange(ws_min, ws_max + bin_width, bin_width)
    valid = valid[(valid["ws_hub"] >= ws_min) & (valid["ws_hub"] <= ws_max)]

    valid["bin"] = pd.cut(valid["ws_hub"], bins, include_lowest=True)
    grouped = (
        valid.groupby("bin", observed=False)
        .agg(mean_power=("power", "mean"), count=("power", "size"))
        .reset_index()
    )
    grouped["bin_mid"] = grouped["bin"].map(lambda b: b.mid)
    grouped["norm_power"] = grouped["mean_power"] / cap * 100.0

    # Only plot bins with at least 5 observations to avoid noisy tails
    grouped = grouped[grouped["count"] >= 5]
    return grouped["bin_mid"].values, grouped["norm_power"].values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SCADA validation: reproducing Table 3 and Figure 7")
    print("=" * 70)

    # Load and summarise
    site_data = {i: load_site(i) for i in range(1, 7)}
    summary = {i: compute_site_stats(site_data[i]) for i in range(1, 7)}

    table = pd.DataFrame(
        {
            "Site": [f"Site {i}" for i in range(1, 7)],
            "Capacity (MW)": [summary[i]["capacity_MW"] for i in range(1, 7)],
            "Mean hub-height wind speed (m/s)": [
                summary[i]["mean_hub_ws"] for i in range(1, 7)
            ],
            "Capacity factor (%)": [
                summary[i]["capacity_factor"] for i in range(1, 7)
            ],
            "Pearson r": [summary[i]["pearson_r"] for i in range(1, 7)],
        }
    )
    table["Mean hub-height wind speed (m/s)"] = table[
        "Mean hub-height wind speed (m/s)"
    ].round(2)
    table["Capacity factor (%)"] = table["Capacity factor (%)"].round(1)
    table["Pearson r"] = table["Pearson r"].round(3)

    print("\nTable 3 (tab:sites)")
    print(table.to_string(index=False))
    print(
        f"\nMean Pearson r = {table['Pearson r'].mean():.4f}; "
        f"range = {table['Pearson r'].min():.3f} (Site {table['Pearson r'].idxmin()+1}) "
        f"to {table['Pearson r'].max():.3f} (Site {table['Pearson r'].idxmax()+1})"
    )

    # Save tables
    table.to_csv(OUT_DIR / "tab_sites.csv", index=False)
    corr_table = pd.DataFrame(
        {
            "Site": [f"Site {i}" for i in range(1, 7)],
            "Pearson r": [summary[i]["pearson_r"] for i in range(1, 7)],
        }
    )
    corr_table.to_csv(OUT_DIR / "tab_scada_correlations.csv", index=False)

    # ------------------------------------------------------------------
    # Figure 7
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=200)
    fig.patch.set_facecolor("white")

    # Panel (a): wind speed vs capacity factor
    ax = axes[0]
    x = np.array([summary[i]["mean_hub_ws"] for i in range(1, 7)])
    y = np.array([summary[i]["capacity_factor"] for i in range(1, 7)])
    for i in range(1, 7):
        ax.scatter(
            x[i - 1],
            y[i - 1],
            color=COLORS[i],
            marker=MARKERS[i],
            s=120,
            zorder=3,
            label=SITE_LABELS[i],
        )
        ax.annotate(
            f"S{i}",
            (x[i - 1], y[i - 1]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            color=COLORS[i],
            fontweight="bold",
        )
    # Linear trend
    z = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() - 0.3, x.max() + 0.3, 100)
    ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.4, lw=1.2, zorder=1)
    ax.set_xlabel("Mean hub wind speed (m/s)", fontsize=10)
    ax.set_ylabel("Capacity factor (%)", fontsize=10)
    ax.set_title("(a) Wind speed vs capacity factor", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(4.0, 8.5)
    ax.set_ylim(19, 46)

    # Panel (b): Pearson r bar chart
    ax = axes[1]
    labels = [SITE_LABELS[i] for i in range(1, 7)]
    values = [summary[i]["pearson_r"] for i in range(1, 7)]
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=[COLORS[i] for i in range(1, 7)],
                   edgecolor="black", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.55, 0.95)
    ax.set_xlabel("Pearson r", fontsize=10)
    ax.set_title("(b) Wind-power correlation", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    # Panel (c): normalised power curves
    ax = axes[2]
    for i in range(1, 7):
        xb, yb = build_binned_curve(site_data[i], CAPACITIES[i])
        ax.plot(
            xb,
            yb,
            color=COLORS[i],
            linewidth=2,
            label=SITE_LABELS[i],
            linestyle={1: "-", 2: "--", 3: ":", 4: "-.", 5: "--", 6: ":"}[i],
        )
    ax.set_xlabel("Hub wind speed (m/s)", fontsize=10)
    ax.set_ylabel("Normalised output (% of capacity)", fontsize=10)
    ax.set_title("(c) Normalised power curves", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 20)
    ax.set_ylim(-2, 72)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig7_combined.png", dpi=300, bbox_inches="tight",
               facecolor="white")
    fig.savefig(OUT_DIR / "fig7_combined.pdf", dpi=300, bbox_inches="tight",
               facecolor="white")
    print("\nSaved fig7_combined.png/pdf and tab_sites.csv/tab_scada_correlations.csv")


if __name__ == "__main__":
    main()
