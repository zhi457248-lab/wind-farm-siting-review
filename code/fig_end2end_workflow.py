"""
End-to-end workflow figure: PRISMA → Stage1 → Stage2 → WFSSF → Case validation → Site 6 counterfactual
One figure that tells the entire research story.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---- Style ----------------------------------------------------------------
PALETTE = {
    'screen': '#E8F1F8',  # PRISMA
    'cluster': '#FCE4D6',  # Stage 1
    'extract': '#FFF2CC',  # Stage 2
    'frame': '#E2EFDA',    # WFSSF
    'case': '#D9E1F2',     # Case study
    'cf': '#F8CBAD',       # Counterfactual
    'arrow': '#404040',
    'text': '#1F1F1F',
}
NATURE_LIKE = ['#E8F1F8', '#FCE4D6', '#FFF2CC', '#E2EFDA', '#D9E1F2', '#F8CBAD']

fig, ax = plt.subplots(figsize=(15, 6.5), dpi=200)
ax.set_xlim(0, 15)
ax.set_ylim(0, 7)
ax.axis('off')


def panel(x, y, w, h, color, title, body, fontsize_title=11, fontsize_body=8.5):
    """Rounded box with title + body text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.04,rounding_size=0.12',
        linewidth=1.3, edgecolor='#404040', facecolor=color, alpha=0.92
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.32, title,
            ha='center', va='top', fontsize=fontsize_title,
            fontweight='bold', color=PALETTE['text'])
    ax.text(x + w/2, y + h/2 - 0.15, body,
            ha='center', va='center', fontsize=fontsize_body,
            color=PALETTE['text'])


def arrow(x1, y1, x2, y2, label='', color='#404040'):
    """Connector arrow with optional label."""
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.25,head_length=0.4',
        linewidth=1.6, color=color, mutation_scale=12
    )
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, label,
                ha='center', va='bottom', fontsize=7.5,
                color=color, style='italic')


# ---- Six panels (left → right) -------------------------------------------
PANEL_W = 2.15
PANEL_H = 5.0
PANEL_Y = 1.0
GAP = 0.20

xs = [0.3 + i * (PANEL_W + GAP) for i in range(6)]

# Panel 1: PRISMA screening
panel(xs[0], PANEL_Y, PANEL_W, PANEL_H, PALETTE['screen'],
      '1. PRISMA\nScreening',
      '5,012 records\nidentified\n\n↓\n4,558 screened\n\n454 included\n(since 2001)\n\nWoS + Scopus\n+ CNKI (90)')

# Panel 2: Stage 1 (LLM clustering)
panel(xs[1], PANEL_Y, PANEL_W, PANEL_H, PALETTE['cluster'],
      '2. Stage 1\nLLM Clustering',
      '454 abstracts\n→ top-3 indicators\n\nClaude API\n(NLP-assisted)\n\n7 thematic\ncategories\n\nTop-15 indicators\nidentified')

# Panel 3: Stage 2 (full-text extraction)
panel(xs[2], PANEL_Y, PANEL_W, PANEL_H, PALETTE['extract'],
      '3. Stage 2\nFull-Text Extract',
      '77-study nested\nfull-text subset\n\n544 raw mentions\n↓\n60 controlled\nindicators\n\n314 mentions\n(7 categories)\nκ = 0.81')

# Panel 4: WFSSF framework
panel(xs[3], PANEL_Y, PANEL_W, PANEL_H, PALETTE['frame'],
      '4. WFSSF\nFramework',
      '4-tier sequential\nnon-compensatory\n\nT1 Resource\nT2 Accessibility\nT3 Impact\nT4 Risk gate\n\nOn/Off bifurcation\nat Tier 3')

# Panel 5: Case validation (6 sites)
panel(xs[4], PANEL_Y, PANEL_W, PANEL_H, PALETTE['case'],
      '5. Case\nValidation',
      '6 wind farms\n(36-200 MW)\n70,176 SCADA\nrecords/site\n\nPearson r = 0.785\n(p < 0.001)\n\nFramework\nconcordance:\nX / 6 sites')

# Panel 6: Site 6 counterfactual
panel(xs[5], PANEL_Y, PANEL_W, PANEL_H, PALETTE['cf'],
      '6. Counterfactual\n(Site 6)',
      'High wind speed\n→ GIS-MCDM\nselects Site 6\n\nWFSSF excludes\nat T2 (no infra)\n\nWind-speed\nparadox resolved')

# ---- Connector arrows -----------------------------------------------------
y_mid = PANEL_Y + PANEL_H/2
for i in range(5):
    arrow(xs[i] + PANEL_W, y_mid,
          xs[i+1], y_mid,
          color=PALETTE['arrow'])

# Title
ax.text(7.5, 6.65, 'End-to-End Research Workflow',
        ha='center', va='top', fontsize=15, fontweight='bold',
        color=PALETTE['text'])

# Subtitle / data flow at bottom
ax.text(7.5, 0.55,
        r'$\bf{Data\ flow:}$ 5,012 records $\rightarrow$ 454 abstracts $\rightarrow$ 77 full-text '
        r'$\rightarrow$ 60 indicators $\rightarrow$ 4-tier framework $\rightarrow$ 6-site validation',
        ha='center', va='center', fontsize=9, color='#505050')
ax.text(7.5, 0.20,
        r'$\bf{Output:}$ 60-term controlled vocabulary, 4-tier WFSSF, 6-site empirical concordance, '
        r'counterfactual failure-mode catalogue',
        ha='center', va='center', fontsize=8.5, color='#505050', style='italic')

plt.tight_layout()
plt.savefig('/Users/juicy/风电选址综述/fig_end2end_workflow.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/juicy/风电选址综述/fig_end2end_workflow.pdf',
            dpi=300, bbox_inches='tight', facecolor='white')
print('Saved fig_end2end_workflow.png/pdf')
