"""
End-to-end workflow figure: PRISMA -> Stage1 -> Stage2 -> WFSSF -> Case validation -> Site 6 counterfactual.
Two-row snake layout to avoid the monotonous single-row strip.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---- Professional, colour-blind-friendly palette -------------------------
PALETTE = {
    'screen':  '#E8F4FD',  # PRISMA
    'cluster': '#D6EAF8',  # Stage 1
    'extract': '#FCF3CF',  # Stage 2
    'frame':   '#D5F5E3',  # WFSSF
    'case':    '#E8DAEF',  # Case study
    'cf':      '#FADBD8',  # Counterfactual
    'arrow':   '#5D6D7E',
    'text':    '#1F1F1F',
}

fig, ax = plt.subplots(figsize=(13, 7.2), dpi=200)
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.5)
ax.axis('off')


def panel(x, y, w, h, color, title, body, fontsize_title=11, fontsize_body=8.5):
    """Rounded box with title + body text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.04,rounding_size=0.12',
        linewidth=1.4, edgecolor='#404040', facecolor=color, alpha=0.95
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.30, title,
            ha='center', va='top', fontsize=fontsize_title,
            fontweight='bold', color=PALETTE['text'])
    ax.text(x + w/2, y + h/2 - 0.18, body,
            ha='center', va='center', fontsize=fontsize_body,
            color=PALETTE['text'])


def arrow(x1, y1, x2, y2, color='#5D6D7E'):
    """Connector arrow."""
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.25,head_length=0.4',
        linewidth=1.7, color=color, mutation_scale=12
    )
    ax.add_patch(a)


# ---- 2 x 3 snake layout ---------------------------------------------------
W, H = 3.4, 2.55
GAP_X, GAP_Y = 0.55, 0.90

# Top row (left -> right): stages 1, 2, 3
x_top = [0.70, 0.70 + W + GAP_X, 0.70 + 2*(W + GAP_X)]
y_top = 4.35

# Bottom row (right -> left): stages 6, 5, 4
x_bot = [x_top[2], x_top[1], x_top[0]]
y_bot = 1.05

# Top row panels
panel(x_top[0], y_top, W, H, PALETTE['screen'],
      '1. PRISMA Screening',
      '5,012 records identified\n\u2193\n4,558 screened\n454 included (since 2001)\nWoS + Scopus + CNKI (90)')

panel(x_top[1], y_top, W, H, PALETTE['cluster'],
      '2. Stage 1 LLM Clustering',
      '454 abstracts\n\u2192 top-3 indicators\nClaude API (NLP-assisted)\n7 thematic categories\nTop-15 indicators identified')

panel(x_top[2], y_top, W, H, PALETTE['extract'],
      '3. Stage 2 Full-Text Extract',
      '77-study nested subset\n544 raw mentions\n\u2193\n60 controlled indicators\n314 mentions, \u03ba = 0.81')

# Bottom row panels (reverse order for snake)
panel(x_bot[0], y_bot, W, H, PALETTE['cf'],
      '6. Counterfactual (Site 6)',
      'High wind speed\n\u2192 GIS-MCDM selects Site 6\nWFSSF excludes at T2\n(no infrastructure)\nWind-speed paradox resolved')

panel(x_bot[1], y_bot, W, H, PALETTE['case'],
      '5. Case Validation',
      '6 wind farms (36-200 MW)\n70,176 SCADA records/site\nPearson r = 0.785\n(p < 0.001)\nFramework concordance: 6/6')

panel(x_bot[2], y_bot, W, H, PALETTE['frame'],
      '4. WFSSF Framework',
      '4-tier sequential\nnon-compensatory\nT1 Resource  /  T2 Accessibility\nT3 Impact  /  T4 Risk gate\nOn/Off bifurcation at Tier 3')

# ---- Arrows: snake --------------------------------------------------------
# Top row left -> right
yt = y_top + H/2
for i in range(2):
    arrow(x_top[i] + W, yt, x_top[i+1], yt)

# Down from Stage 2 to WFSSF (right side)
arrow(x_top[2] + W/2, y_top, x_bot[2] + W/2, y_bot + H)

# Bottom row right -> left
yb = y_bot + H/2
for i in range(2):
    arrow(x_bot[i], yb, x_bot[i+1] + W, yb)

# ---- Bottom data flow / output --------------------------------------------
ax.text(6.5, 0.30,
        r'$\bf{Data\ flow:}$ 5,012 records $\rightarrow$ 454 abstracts $\rightarrow$ 77 full-text '
        r'$\rightarrow$ 60 indicators $\rightarrow$ 4-tier framework $\rightarrow$ 6-site validation',
        ha='center', va='center', fontsize=9, color='#505050')
ax.text(6.5, 0.06,
        r'$\bf{Output:}$ 60-term controlled vocabulary, 4-tier WFSSF, 6-site empirical concordance, '
        r'counterfactual failure-mode catalogue',
        ha='center', va='center', fontsize=8.5, color='#505050', style='italic')

plt.tight_layout()
plt.savefig('/Users/juicy/风电选址综述/fig_end2end_workflow.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/juicy/风电选址综述/fig_end2end_workflow.pdf',
            dpi=300, bbox_inches='tight', facecolor='white')
print('Saved fig_end2end_workflow.png/pdf')
