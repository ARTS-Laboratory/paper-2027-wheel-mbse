"""Where the artifacts live, resolved once so no module has to count `..` hops.

STDLIB ONLY, AND THAT IS A CONSTRAINT RATHER THAN A STYLE NOTE.  `wheel_fea` imports this,
and `tests/test_import_hygiene.py` runs `import wheel_fea` in an interpreter with no jax,
no pygad and no matplotlib — because the CadQuery env has none of them and the STEP
exporter has to import the same module the optimizer does.  Anything imported here is
imported there.

The layout this describes:

    <ROOT>/                     best_solution.json, stage2_elites.json  — the provenance
                                chain, read by both envs, so they stay at the top
      src/                      the modules.  `__file__` of anything here is one hop down
      studies/                  the study drivers AND their .json/.jpg output, together
      export/                   what the CadQuery env produces: .step, the manifest, the
                                poster figure
      tests/

A study driver's own `HERE` is `studies/`, which is where its `--out` should land, so those
call sites are deliberately left alone.  It is the INPUTS — the genome, the manifest — that
need an anchor that does not move with the caller, and that is what `ROOT` and `EXPORT` are
for.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
STUDIES = os.path.join(ROOT, "studies")
EXPORT = os.path.join(ROOT, "export")

# The two genome files every env reads.  Named because "best_solution.json" appears at 38
# call sites and a typo in one of them is a silently different wheel.
BEST_SOLUTION = os.path.join(ROOT, "best_solution.json")
STAGE2_ELITES = os.path.join(ROOT, "stage2_elites.json")
