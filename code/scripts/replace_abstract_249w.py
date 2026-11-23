# -*- coding: utf-8 -*-
r"""Replace the abstract block in the main manuscript with the compressed
version from the final audit report (section 9), preserving every
gate-checked phrase (controllability, sigma=0, p=0.0013/0.0002, 2.2--2.3x,
all 15, "not accuracy gains", 34.8%, T=8, +-50%).

The replacement is marker-based (\begin{abstract} ... \end{abstract}) so no
long literal string has to be transcribed. A strict word count (LaTeX commands
stripped, math delimiters stripped but math CONTENT counted as words -- the
strictest reasonable interpretation of the journal's 250-word limit) must come
out at or under 250.
"""
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(ROOT, "manuscript",
                   "SpikeRPGNN.tex")

NEW_ABSTRACT = (
    "Graph neural networks incur dense floating-point costs scaling with depth "
    "and graph size; spiking networks realize efficiency gains only under "
    "controlled firing activity. Prior spiking GNNs set rates heuristically, "
    "accommodating rather than optimizing this trade-off. We propose "
    "\\textbf{SpikeRPGNN}, which casts it as an \\textbf{Energy--Accuracy Pareto "
    "optimization} problem with manifold-valued spiking neurons on a "
    "hyperbolic--Euclidean product manifold. A spike-rate regularizer "
    "$\\mathcal{L}_{sr}=\\lambda_{sr}(\\bar{r}-r^{*})^{2}$ scalarizes the "
    "bi-objective problem and steers the firing rate to target $r^{*}$; a "
    "geometric-consistency bound and a scalarization-theory instantiation "
    "identify when the controlled point is Pareto optimal on the threshold "
    "slice. The increment over a fixed-threshold spiking layer is calibration, "
    "not accuracy: at $\\lambda_{sr}{=}0$ the realized rate falls to $0.14$ "
    "(PROTEINS) and $0.09$ (ENZYMES), so budgeting for the commanded $0.4$ "
    "would misestimate energy by $2$--$4\\times$, whereas \\emph{all 15} "
    "regularized configurations land inside the target band within one standard "
    "deviation of unregularized accuracy. A dedicated $\\sigma=0$ control shows "
    "controllability does not depend exclusively on membrane noise: at "
    "$\\sigma=0$ the regularizer still moves the rate toward target on PROTEINS "
    "and ENZYMES (paired $t$-test $p=0.0013$ and $0.0002$), while on MUTAG "
    "noiseless dynamics quantize to period-2 plateaus; the noise instead makes "
    "that point reproducible, shrinking across-run rate spread by "
    "$2.2$--$2.3\\times$. We claim controllability, not accuracy gains or lower "
    "energy: the analytic energy at the commanded point is $34.8\\%$ below "
    "capacity-matched dense GNNs once the $T=8$ integration cost is charged, a "
    "sign not robust to $\\pm 50\\%$ errors in the per-operation constants."
)


def word_count(s):
    """Strip LaTeX commands, then count whitespace-separated tokens that carry
    alphanumerics. Math content is counted as words (strictest reading)."""
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)          # commands
    s = re.sub(r"[{}\\$&_^~]", " ", s)             # braces, math delimiters
    s = s.replace("--", " ")
    return len([t for t in s.split() if re.search(r"[0-9A-Za-z]", t)])


def main():
    with open(TEX, encoding="utf-8") as f:
        tex = f.read()
    m = re.search(r"\\begin\{abstract\}\n(.*?)\n\\end\{abstract\}", tex, re.S)
    if not m:
        print("ABORT: abstract block not found")
        return 2
    old = m.group(1)
    old_wc = word_count(old)
    new_wc = word_count(NEW_ABSTRACT)
    print(f"current abstract : {old_wc} words (strict counter)")
    print(f"replacement      : {new_wc} words (strict counter; journal limit 250)")
    if new_wc > 250:
        print("ABORT: replacement exceeds 250 words under the strict counter")
        return 2
    bak = TEX + f".bak_abstract_{time.strftime('%Y%m%d_%H%M%S')}"
    with open(bak, "w", encoding="utf-8") as f:
        f.write(tex)
    tex2 = tex[:m.start(1)] + NEW_ABSTRACT + tex[m.end(1):]
    tmp = TEX + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(tex2)
    os.replace(tmp, TEX)
    print(f"[ok] abstract replaced; backup: {bak}")
    # post-check: the gate-checked phrases must all survive
    must = ["controllability", "all 15", "not accuracy gains", "34.8",
            "T=8", "period-2", "0.0013", "0.0002", "2.2", "2.3",
            "SpikeRPGNN", "Pareto", "0.14", "0.09"]
    miss = [w for w in must if w not in NEW_ABSTRACT]
    print(f"[{'ok' if not miss else 'FAIL'}] gate phrases preserved "
          f"(missing: {miss})")
    return 0 if not miss else 2


if __name__ == "__main__":
    sys.exit(main())
