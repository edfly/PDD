#!/usr/bin/env python3
"""Dependency-free vector PDF figure for the cross-platform energy projection.

matplotlib savefig is killed by the sandbox in this environment, so this
script emits a valid PDF by hand (no native libraries). Reads
results/energy_silicon_projection.json and writes figures/energy_silicon_projection.pdf.
Grouped bars (3 method groups x 4 substrate bars) on a log y-axis.

Revision (2026-10-10): white background, enlarged canvas and fonts for legibility
when scaled into a double-column journal page, value labels on every bar, and a
rotated y-axis title.
"""
import json, math

POST = "D:/2026/paper/PBB/post"
DATA = f"{POST}/results/energy_silicon_projection.json"
OUT = f"{POST}/figures/energy_silicon_projection.pdf"

data = json.load(open(DATA))
proj = data["projection"]

methods = [("Proto-GIN", "Proto-GIN (r=1.0)"),
            ("Spike-GNN (Euc.)", "Spike-GNN (Euc.)"),
            ("SpikeRPGNN", "SpikeRPGNN")]
keys = ["E_loihi_mean", "E_truenorth_mean", "E_analytic_mean", "E_dense_gpu_mean"]
sub_names = ["Loihi (23.6)", "TrueNorth (26)", "Analytic (0.9)", "Dense GPU (4.6)"]
colors = [(0.122, 0.467, 0.706), (1.0, 0.498, 0.055),
          (0.173, 0.627, 0.173), (0.498, 0.498, 0.498)]

W, H = 660.0, 400.0
x0, x1, y0, y1 = 86.0, 636.0, 70.0, 348.0
ymin, ymax = 1.0, 100.0


def ymap(v):
    return y0 + (math.log10(v) - math.log10(ymin)) / \
        (math.log10(ymax) - math.log10(ymin)) * (y1 - y0)


def fmt(v):
    if v >= 10:
        return "%.1f" % v
    return "%.2f" % v


def esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


C = []
def add(s):
    C.append(s)


# white background (fill entire canvas)
add("1 1 1 rg\n0 0 %.1f %.1f re f" % (W, H))
# thin border so the white panel is visible against the page
add("0.8 0.8 0.8 RG\n0.6 w\n0 0 %.1f %.1f re S" % (W, H))

# axes
add("0 0 0 RG\n1.2 w\n%.1f %.1f m %.1f %.1f l %.1f %.1f l S"
    % (x0, y0, x1, y0, x0, y1))

# y gridlines + tick labels
add("0.7 0.7 0.7 RG\n0.4 w")
for t in [1, 2, 5, 10, 20, 50, 100]:
    yy = ymap(t)
    add("%.1f %.1f m %.1f %.1f l S" % (x0, yy, x1, yy))
    add("0.15 0.15 0.15 rg\nBT /F1 14 Tf %.1f %.1f Td (%s) Tj ET"
        % (x0 - 34, yy - 4, esc(str(t))))

# rotated y-axis title ("Energy per graph (uJ)")
ytxt = "Energy per graph (uJ, log scale)"
add("0.15 0.15 0.15 rg\nBT /F2 15 Tf [0 1 -1 0 %.1f %.1f] Tm (%s) Tj ET"
    % (24, (y0 + y1) / 2.0, esc(ytxt)))

# bars
ng = len(methods)
gw = (x1 - x0) / ng
bw, gap = 21.0, 8.0
block = 4 * bw + 3 * gap
for g, (mk, label) in enumerate(methods):
    e = proj[mk]
    cx = x0 + (g + 0.5) * gw
    for i, k in enumerate(keys):
        v = max(e[k], ymin * 1.001)
        bx = cx - block / 2 + i * (bw + gap)
        bh = ymap(v) - y0
        r, gg, b = colors[i]
        add("%.3f %.3f %.3f rg\n%.1f %.1f %.1f %.1f re f"
            % (r, gg, b, bx, y0, bw, bh))
        add("0 0 0 RG\n0.4 w\n%.1f %.1f %.1f %.1f re S"
            % (bx, y0, bw, bh))
        # value label just above the bar
        add("0 0 0 rg\nBT /F1 13 Tf %.1f %.1f Td (%s) Tj ET"
            % (bx + bw / 2.0 - 14, ymap(v) + 5, esc(fmt(v))))

# x labels (method groups)
for g, (mk, label) in enumerate(methods):
    cx = x0 + (g + 0.5) * gw
    add("0 0 0 rg\nBT /F2 16 Tf %.1f %.1f Td (%s) Tj ET"
        % (cx - 62, y0 - 22, esc(label)))

# title
add("0 0 0 rg\nBT /F3 20 Tf %.1f %.1f Td (%s) Tj ET"
    % (x0, H - 26, esc("Cross-platform energy projection (1-shot, measured SOP)")))

# legend (top, horizontal)
lx, ly = x0 + 8, H - 50
for i, nm in enumerate(sub_names):
    r, gg, b = colors[i]
    bx = lx + i * 150
    add("%.3f %.3f %.3f rg\n%.1f %.1f 15 15 re f" % (r, gg, b, bx, ly))
    add("0 0 0 RG\n0.4 w\n%.1f %.1f 15 15 re S" % (bx, ly))
    add("0 0 0 rg\nBT /F1 14 Tf %.1f %.1f Td (%s) Tj ET"
        % (bx + 20, ly + 2, esc(nm)))

content = "\n".join(C).encode("latin-1")

# ---- assemble PDF ----
objects = []
objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
objects.append(
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.0f %.0f] "
    b"/Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R >> >> /Contents 7 0 R >>"
    % (W, H))
objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>")
objects.append(b"<< /Length %d >>\nstream\n" % len(content) + content +
               b"\nendstream")

pdf = b"%PDF-1.4\n"
offsets = [0]
for i, obj in enumerate(objects, start=1):
    offsets.append(len(pdf))
    pdf += b"%d 0 obj\n" % i + obj + b"\nendobj\n"

xref_pos = len(pdf)
n = len(objects) + 1
pdf += b"xref\n0 %d\n" % n
pdf += b"0000000000 65535 f \n"
for off in offsets[1:]:
    pdf += b"%010d 00000 n \n" % off
pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % n
pdf += b"startxref\n%d\n%%%%EOF\n" % xref_pos

open(OUT, "wb").write(pdf)
print("wrote", OUT, "(%d bytes)" % len(pdf))
