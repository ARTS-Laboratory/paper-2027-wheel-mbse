"""One phase slot's solve, in its own interpreter.  Spawned by `wheel_pool.PhasePool`.

THE ENV BLOCK BELOW RUNS BEFORE THE FIRST IMPORT, AND THAT ORDER IS THE WHOLE POINT.
OpenBLAS, MKL and XLA read their thread settings once, at import, and never look again —
so an `os.environ` write after `import numpy` is a no-op that looks like it worked.  The
pool also passes these through `Popen(env=...)`, which is the path that actually runs in
production; the block here is what makes a worker launched by hand, from a bare
interpreter with no make and no pytest, behave the same as one the pool started.

`XLA_FLAGS` is the one that is load-bearing rather than merely tidy — see
`wheel_pool.PINNED_ENV` for the measurement.  Without it this process's adjoint GRADIENT
disagrees with the parent's in the last bits, while every forward value agrees exactly.

WHAT THIS PROCESS IS FOR.  It receives `(genes, cfg, phase, orientation, force, delta0)`,
builds its own mesh, runs one `service_qoi_value_and_grad`, and returns the leaves
`wheel_objective.t3_terms` reads — three floats and three 14-vectors.  It does NOT return
the mesh, the sparse problem or the displacement field: `WheelMesh._coord_fn` holds a
jitted function after the first `mesh_coords` call and cannot be pickled at all, and the
field is megabytes for a number the parent never looks at.  Anything the parent needs off
the field — the p-norm at a probe exponent — is computed here, by the same
`wheel_objective._probe_values` the serial path calls, so the two cannot drift.

STDOUT IS THE DATA CHANNEL.  Nothing in this process may print to it.  stderr is inherited
from the parent, so a traceback still reaches the terminal.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wheel_pool as WP                            # noqa: E402  — stdlib+numpy, no jax

for _name, _value in WP.PINNED_ENV.items():
    os.environ.setdefault(_name, _value)

import traceback                                                      # noqa: E402

import jax_config                    # noqa: F401,E402  — x64 before the first trace
import numpy as np                                                    # noqa: E402

import wheel_adjoint as WA                                            # noqa: E402
import wheel_objective as WO                                          # noqa: E402
import wheel_wheel as WW                                              # noqa: E402


def run_phase(task):
    """The reply for one phase slot, shaped like the adjoint dict the serial path reads.

    The shape is not a convenience.  `t3_terms`' loop body reads `o["axle_drop"]["value"]`,
    `o["pnorm_stress"]["grad"]`, `o["_meta"]["max_stress_mpa"]` and so on; returning those
    same nested keys means the loop body is LITERALLY THE SAME LINES on the pooled path as
    on the serial one, and the only difference between the two branches is where `o` came
    from.  That is what makes the bit-identity gate a statement about the transport rather
    than about two implementations that happen to agree today.
    """
    genes = np.asarray(task["genes"], dtype=float)
    cfg = task["cfg"]
    orientation = task["orientation"]
    mesh = WW.build_wheel(genes, cfg, phase_deg=float(task["phase"]),
                          orientation=orientation)

    qoi = ("pnorm_stress",
           lambda prob: WA._qoi_pnorm_stress(prob, p=task["stress_gauss_p"]))
    o = WA.service_qoi_value_and_grad(
        genes, cfg, (qoi,), force=task["force"], mesh=mesh, delta0=task["delta0"],
        **task["problem_kw"])

    meta = o["_meta"]
    probe = WO._probe_values(meta["prob"], meta["res"]["u"], task["probe_p"])
    return {
        "axle_drop": {"value": o["axle_drop"]["value"], "grad": o["axle_drop"]["grad"]},
        "pnorm_stress": {"value": o["pnorm_stress"]["value"],
                         "grad": o["pnorm_stress"]["grad"],
                         "coupling_grad": o["pnorm_stress"]["coupling_grad"]},
        "_meta": {"max_stress_mpa": meta["max_stress_mpa"],
                  "contact_force_n": meta["contact_force_n"]},
        "_probe": probe,
    }


def main(argv):
    slot = argv[1] if len(argv) > 1 else "?"
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        task = WP._recv(stdin)
        if task is None or task.get("kind") == "stop":
            return 0
        try:
            WP._send(stdout, {"ok": True, "result": run_phase(task)})
        except Exception as exc:                        # noqa: BLE001
            # EVERY exception is reported, not just the solver's.  A worker that dies on
            # an unexpected error would show up in the parent as a hang followed by a
            # respawn, which reads as flaky infrastructure; reported, it arrives at
            # `descend` as the `solve_reject` it already knows how to handle, with the
            # traceback on stderr for whoever has to fix it.
            traceback.print_exc()
            print(f"[phase worker {slot}] the task above failed", file=sys.stderr,
                  flush=True)
            WP._send(stdout, {"ok": False, "error": WP._error_frame(exc)})


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
