# `gui/` — an optional control surface

    make gui                 # the desktop app
    make gui-browser         # the same thing in a browser, on http://127.0.0.1:8731/

The mission/requirements layer, a live preview of the wheel, and detached runs with
progress bars. Two ways in, **one server**: the desktop shell starts `gui/server.py` on a
port the OS picks and points a window at it, and the browser route starts the same file on
a fixed port and opens a tab. Nothing in the panels knows which one it is under.

**It is optional and nothing else depends on it.** `gui/` imports *from* `src/`; nothing in
`src/`, `studies/` or `tests/` imports `gui/`. It adds nothing to `requirements-opt.txt` or
`requirements-cad.txt` — the server is stdlib only and the frontend is hand-written HTML,
CSS and JS with no build step. `pytest` collects nothing here — 957 collected either way.

Three tracked files are touched, and none of them does anything on its own: ignored lines
in `.gitignore` for `gui/runs/` and the shell's `node_modules/` and `dist/`, and three
targets in the Makefile that test for what they need first and print a message rather than
failing when the directory is gone. So `rm -rf gui/` is a supported state, not a broken
install — though those edits are still there to remove if you want the tree back exactly as
it was.

## The desktop shell

`gui/desktop/` is an Electron app: four small files, an icon, and no application code of
its own. It owns a window and a child process; every panel, number and decision still comes
from the Python server.

    cd gui/desktop && npm install        # once — Electron is ~150 MB and is gitignored
    make gui                             # or: cd gui/desktop && npm start
    make gui-dist                        # an installer for the platform you are ON

**Cross-building is the exception.** `make gui-dist` builds for the machine it runs on. A
Windows installer *from Linux* needs `wine` on the PATH; a macOS `.dmg` needs macOS.
Neither can be faked, and only the Linux AppImage has been built and run here.

`deb` is deliberately not in the Linux target list: a `.deb` **requires** a maintainer name
and email, and this repo carries no author metadata anywhere — no `LICENSE`, no `author` in
`pyproject.toml`. Add `"deb"` to `build.linux.target` and a `"maintainer"` beside it if you
want one; whose address goes in is not a decision a build file should make for you.

**On Linux, `npm start` may refuse to run** with *"The SUID sandbox helper binary was found,
but is not configured correctly."* That is Chromium's sandbox helper needing root ownership
in an unpackaged tree, and it is a property of the checkout rather than of this app:

    sudo chown root:root gui/desktop/node_modules/electron/dist/chrome-sandbox
    sudo chmod 4755      gui/desktop/node_modules/electron/dist/chrome-sandbox

Packaged builds do not have the problem. `--no-sandbox` also silences it and is not the fix.

### The shell and the pipeline can be on different machines

    ssh -N -L 8731:127.0.0.1:8731 you@the-box          # on the laptop, left running
    WHEEL_SERVER_URL=http://127.0.0.1:8731 npm start   # a native window, remote pipeline

`WHEEL_SERVER_URL` is **attach mode**: the shell skips finding a checkout and skips
starting anything, and just points its window at a server that is already running. This is
how you drive the pipeline from a machine that has no checkout — which is not a
convenience, it is the only way, because the venv interpreters are Linux ELF and an sshfs
mount of the repo would hand macOS a `python` it cannot exec.

**Point it at a loopback tunnel.** That is the server's rule rather than the shell's:
`server.py` binds `127.0.0.1` because the endpoint starts processes and has no
authentication, and its docstring says to tunnel rather than move the bind address. Nothing
in the shell re-binds anything.

In attach mode the three File-menu verbs that reach for a local path are greyed, and
nothing answering at the URL gets the failure page with the tunnel command on it rather
than Chromium's.

**Where the repo is** is the one question a packaged app has that `npm start` does not.
Unpacked, the answer is two directories up. Installed from a `.dmg` or an `.exe`, the app is
somewhere else entirely and the checkout it should drive is a thing only the user knows, so
it is asked for once (`File → Choose checkout…`) and remembered, and `WHEEL_REPO`
overrides. A window that cannot find a checkout, or finds one with no `.venv-opt`, says so
on its own page with the command that fixes it — it does not open blank.

**The page draws its own title bar.** The `header` strip is the drag region; macOS insets
the traffic lights over its left end and Windows and Linux overlay their controls on the
right. Every rule that arranges this is scoped to `html.desktop`, which is set only when the
preload bridge is present, so the browser route renders byte-identical CSS and none of it
applies.

## Jobs outlive the app

This is the design constraint everything else follows from. A run is never a child of the
server, and the server is a child of the window. Close the browser, close the window, stop
the server, `kill -9` it — the descent keeps descending. Each run gets its own directory
under `gui/runs/`:

    gui/runs/20260909-221959_mbsebase_jorn/
      job.json        the spec, the unit name, the state
      run.sh          exactly what ran   (run.cmd on Windows)
      pid             written by the wrapper, about itself
      rc              the exit code, written after the command returns
      stdout.log
      …               whatever the run was told to write here instead of the repo root

Start the server again and it reads those directories back; there is no state in the
process. `rc` existing is the definition of finished, which is why a job that was killed
reads `interrupted` rather than `failed`: nothing reported a status, and that is a third
thing worth knowing.

**Three detach mechanisms, one registry**, and nothing below the launch decision knows which
ran. `systemd-run --user` where there is a systemd user manager — the same incantation the
Makefile pastes into its long-run recipe comments, and **the only one of the three that also
gives a cgroup and a `MemoryMax`**, which is why it is preferred wherever it exists.
`start_new_session=True` on Linux without one and on macOS. `DETACHED_PROCESS |
CREATE_NEW_PROCESS_GROUP` on Windows.

Because only the first can cap memory, `plan()` **refuses** on total RAM — which every
platform reports exactly — and only **warns** on available RAM, which off Linux is an
approximation. The hard guard does not rest on the soft number.

User units die at logout (`Linger=no` on this box). `loginctl enable-linger` if you want a
descent to survive logging out; closing the app does not need it.

## What it will not do

- **It never writes `best_solution.json` and never commits.** Promotion is a multi-file
  atomic act with a checklist attached — `tests/test_promotion.py` prints it — and the
  useful thing a panel can do is surface the selection tier and that checklist at the
  moment someone is promoting, not perform it.
- **It defaults to not overwriting committed evidence.** A study artifact records what was
  measured at the commit that filed it. The MBSE drivers run with their `--out` pointed
  into the job's run directory unless you tick `refresh_artifact`, and anything that would
  rewrite a committed JSON says so before it starts.
- **It gives every job its own output paths.** `--out` and `--best-out` default to fixed
  names in the repo root and are written unconditionally, so two runs at the defaults
  clobber each other.
- **The window goes to one place and stays there.** This server starts processes, so links
  open in the OS browser and the page cannot navigate anywhere else or open a second window.

## Costs are shown with their basis

Before launching, the panel prints the estimated wall time and peak RSS against free
memory, and blocks what does not fit — a `medium` descent prices at ~84 GiB on a 61 GiB
box. Every constant behind those numbers is either measured and cited or extrapolated and
labelled; see the cost-model block at the top of `catalog.py`. Two of them were measured
here rather than inherited:

- a `smoke` descent exceeded **22.9 GiB** and had not reached step 0 after six minutes, so
  the memory model is affine on a large mesh-independent baseline rather than proportional
  to element count — a proportional model priced it at 7.2 GiB and the resulting cgroup cap
  put the run into swap;
- the first evaluation is a **~1180 s** one-off jit trace (step 0 took 1391 s against a
  ~210 s steady state in `stage3_refillet_shipped_r2.log`), which dominates any short run.

## The files

| file | what it is |
|---|---|
| `server.py` | stdlib `http.server`; holds no state, answers from disk. Localhost only — it starts processes and has no authentication. `--port 0` for an OS-chosen port |
| `catalog.py` | the runnable targets as data: argv, cost, progress kind, what they overwrite |
| `jobs.py` | detached launch, the registry, liveness, reattach — and the three-platform half of all four |
| `progress.py` | the three progress readers — trajectory JSON, GA stdout, phase matching |
| `model.py` | mission → requirements → compliance table, all via `wheel_requirements` |
| `preview.py` | genome → polygons, via the project's own geometry kernel |
| `static/` | the page. No framework, no build |
| `desktop/main.js` | starts the server, owns the window, finds the checkout — or attaches to a remote one |
| `desktop/menu.js` | the native menubar; its View items drive the page's own tabs over IPC |
| `desktop/preload.js` | the whole bridge: what platform, and a tab callback. Nothing else |
| `desktop/shell.html` | the first paint and the last resort — splash, and the failure page |
| `desktop/build/make_icon.py` | draws `icon.png` from the shipped genome, stdlib PNG. Re-run it after a promotion and the icon follows the part |

It runs under `.venv-opt/bin/python` because that env can import the pipeline modules; the
STEP exporter is launched as a subprocess in `.venv-cad`, exactly as `make export` does.
Resident set is ~200 MB, most of it jax, loaded lazily on the first MBSE request.

## Two things it reads that are worth knowing

The preview is drawn by `generate_bezier_centerline → thicken_3taper_curve → place_sector`
— the same calls the mesher and the STEP exporter build the real part from — so it is exact
rather than an approximation, and a full wheel rebuilds in about 0.6 ms. One sector goes
over the wire and the browser repeats it twelve times, which is how `place_sector` builds
the wheel anyway.

The MBSE panel calls `wheel_requirements` directly and never reads
`studies/study_mbse_*.json`. Those artifacts are anchored at genome `09e8188` and PLAN.md
§119 declined to refresh them on purpose; they are correct as a record and wrong as a
source. The points→weights map re-derives at every promotion because `c_smoothness` reads
the reference genome's own `smoothness` term, so it is recomputed on every derive and never
cached.
