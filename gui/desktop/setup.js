/* Building the two virtualenvs ON THE MACHINE THE APP IS RUNNING ON.
 *
 * THIS FILE EXISTS BECAUSE THE ALTERNATIVE WAS A REMOTE.  The shell used to answer a
 * checkout with no `.venv-opt` by printing "build it with: make env" and stopping, which
 * on a machine with no venv and no terminal open is not an instruction, it is a wall --
 * and the way round that wall was to point the window at some other computer that did
 * have one.  That made the app a viewer of a machine rather than the interface to one.
 * So the shell builds the envs itself: the app is the interface to the computer it is
 * running on, and a computer that is not set up yet is a computer it can set up.
 *
 * IT RUNS EXACTLY WHAT `make env` RUNS, and that is the whole design.  Makefile:247-255 is
 *
 *     python3 -m venv .venv-opt
 *     .venv-opt/bin/python -m pip install --upgrade pip
 *     .venv-opt/bin/python -m pip install -r requirements-opt.txt
 *
 * and the two recipes below are that, twice, with `bin/python` spelled `Scripts\python.exe`
 * on Windows.  Nothing here resolves a dependency, chooses a version or pins anything: the
 * requirements files are the source of truth and this only feeds them to pip.  Do not let
 * it grow an opinion about their contents.
 *
 * WHY IT DOESN'T JUST SHELL OUT TO `make`: Windows has no make, and macOS only has one if
 * the Xcode command line tools are installed.  Both of those are machines this is supposed
 * to work on.  The recipe is three lines; re-typing them is cheaper than requiring a build
 * tool to install a package manager.
 */
'use strict';

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const IS_WIN = process.platform === 'win32';
const IS_MAC = process.platform === 'darwin';

/* The two envs, in build order, and with the one distinction that decides what a failure
 * means.  `.venv-opt` is what `gui/server.py` itself runs under, so without it there is no
 * control surface at all.  `.venv-cad` is only ever reached as a subprocess by the STEP
 * exporter (`catalog.py`'s `export` target), so a machine without it is a machine that can
 * do everything but write a .step -- worth saying, not worth refusing to start over. */
const ENVS = [
  {
    dir: '.venv-opt',
    req: 'requirements-opt.txt',
    required: true,
    label: 'the optimiser / FEA environment',
    /* What "built" means, beyond a directory existing.  See `verify` below. */
    imports: ['numpy', 'scipy', 'jax'],
  },
  {
    dir: '.venv-cad',
    req: 'requirements-cad.txt',
    required: false,
    label: 'the STEP exporter environment',
    imports: ['cadquery'],
  },
];

/* Same rule as gui/catalog.py and main.js, which are the files this one has to agree
 * with. */
const venvPython = (root, dir) =>
  path.join(root, dir, ...(IS_WIN ? ['Scripts', 'python.exe'] : ['bin', 'python']));

/* ---------------------------------------------------------------------------
 * RUNNING A CHILD, ONE LINE AT A TIME
 * ------------------------------------------------------------------------ */
function run(cmd, args, cwd, onLine) {
  return new Promise((resolve, reject) => {
    onLine(`$ ${[cmd, ...args].join(' ')}`);
    let child;
    try {
      child = spawn(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    } catch (e) { return reject(e); }

    /* pip draws a progress bar with carriage returns when it thinks it has a terminal.
     * stdio is a pipe here so it does not, but splitting on \r as well costs nothing and
     * means a library that does it anyway cannot turn a 200 MB download into one line. */
    const watch = stream => {
      let buf = '';
      stream.setEncoding('utf8');
      stream.on('data', chunk => {
        buf += chunk;
        let cut;
        while ((cut = buf.search(/\r\n|[\r\n]/)) >= 0) {
          const eol = buf.slice(cut, cut + 2) === '\r\n' ? 2 : 1;
          onLine(buf.slice(0, cut));
          buf = buf.slice(cut + eol);
        }
      });
      stream.on('end', () => { if (buf) onLine(buf); });
    };
    watch(child.stdout); watch(child.stderr);

    child.on('error', e => reject(e));
    child.on('exit', code =>
      code === 0 ? resolve() : reject(new Error(`${path.basename(cmd)} exited ${code}`)));
  });
}

/* ---------------------------------------------------------------------------
 * FINDING AN INTERPRETER TO BUILD FROM
 * ------------------------------------------------------------------------ */

/* THE PATH IS NOT ENOUGH, AND macOS IS WHY.  An app opened from Finder inherits a PATH of
 * `/usr/bin:/bin:/usr/sbin:/sbin` and nothing else -- no Homebrew, no pyenv, no
 * python.org.  `/usr/bin/python3` is there, but it is Apple's own, which has trailed these
 * pins by several minor versions on every recent release (3.9.6), and requirements-opt.txt
 * is pinned at numpy 2.5 / jax 0.11.  So the usual install locations are probed by absolute
 * path as well.  Every one of these is a directory some installer really uses; none is a
 * guess about this particular machine, and a candidate that is not there fails to spawn and
 * is skipped. */
function candidates() {
  if (IS_WIN)
    /* `py` is the version launcher and the only reliable entry point on Windows: a bare
     * `python3` there is usually the Microsoft Store's stub, which is not an interpreter
     * and opens the Store when you run it. */
    return [['py', ['-3.12']], ['py', ['-3']], ['python', []], ['python3', []]];

  const onPath = [['python3.12', []], ['python3.13', []], ['python3.11', []],
                  ['python3', []], ['python', []]];
  if (!IS_MAC) return [...onPath, ['/usr/bin/python3.12', []], ['/usr/bin/python3', []]];

  const framework = v =>
    [`/Library/Frameworks/Python.framework/Versions/${v}/bin/python3`, []];
  return [
    ...['3.12', '3.13', '3.11'].flatMap(v => [
      [`/opt/homebrew/bin/python${v}`, []],       // Homebrew, Apple silicon
      [`/usr/local/bin/python${v}`, []],          // Homebrew, Intel
      framework(v),                               // python.org installer
    ]),
    ...onPath,
  ];
}

function probe([cmd, pre]) {
  return new Promise(resolve => {
    let child;
    const args = [...pre, '-c', 'import sys;print("%d.%d" % sys.version_info[:2])'];
    try {
      child = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'ignore'] });
    } catch { return resolve(null); }
    let out = '';
    child.stdout.setEncoding('utf8').on('data', c => { out += c; });
    child.on('error', () => resolve(null));
    child.on('exit', code => {
      const m = code === 0 && out.match(/^(\d+)\.(\d+)/);
      resolve(m ? { cmd, pre, major: +m[1], minor: +m[2], version: `${m[1]}.${m[2]}` }
                : null);
    });
  });
}

/* Both requirements files say "Python 3.12" in their header, so 3.12 wins outright when it
 * is on the machine.  Otherwise the newest 3.x found is used and the log says so rather
 * than refusing: a pin that happens to resolve on 3.13 should not be blocked by this file
 * guessing, and a pin that does not will fail in pip with pip's own message, which names
 * the package and is a better error than any I could invent here. */
async function hostPython(onLine) {
  const seen = new Set();
  const found = [];
  for (const c of candidates()) {
    const key = [c[0], ...c[1]].join(' ');
    if (seen.has(key)) continue;
    seen.add(key);
    const hit = await probe(c);
    if (hit && hit.major === 3 && hit.minor >= 10) found.push(hit);
  }
  if (!found.length) return null;

  found.sort((a, b) => (b.minor === 12) - (a.minor === 12) || b.minor - a.minor);
  const py = found[0];
  onLine(`# building from ${py.cmd}${py.pre.length ? ' ' + py.pre.join(' ') : ''}` +
         `  (Python ${py.version})`);
  if (py.minor !== 12)
    onLine('# note: requirements-opt.txt and requirements-cad.txt are written for ' +
           'Python 3.12. If a pin has no wheel for this version, pip will say which.');
  return py;
}

/* ---------------------------------------------------------------------------
 * THE ENVS
 * ------------------------------------------------------------------------ */

/* A DIRECTORY IS NOT AN ENVIRONMENT, AND THIS IS THE CHECK THAT KNOWS THE DIFFERENCE.
 * `python -m venv` succeeds in a second; the pip install after it takes minutes and can
 * fail, leaving a `bin/python` that exists and imports nothing.  Existence is therefore
 * not the test -- measured, a bare venv in this tree passes `fs.existsSync` and then lets
 * `gui/server.py` START, because that server is stdlib-only on purpose and imports the
 * pipeline lazily.  The window opens, and the first panel that wants `wheel_requirements`
 * is where you find out.  So "built" means the packages are actually installed.
 *
 * `find_spec` RATHER THAN A REAL IMPORT, so this is cheap enough to run on every launch
 * and not just after something has already gone wrong: measured on this box, 9 ms against
 * 281 ms for `import numpy, scipy, jax`.  It answers "did pip put these here", which is
 * precisely the question a half-finished install gets wrong; a package that is present but
 * broken is a different failure and shows up as a traceback, with a traceback's detail. */
function verify(root, env) {
  const py = venvPython(root, env.dir);
  if (!fs.existsSync(py)) return Promise.resolve(false);
  const probeSrc =
    'import importlib.util as u, sys;' +
    `sys.exit(0 if all(u.find_spec(m) for m in ${JSON.stringify(env.imports)}) else 1)`;
  return new Promise(resolve => {
    let child;
    try {
      child = spawn(py, ['-c', probeSrc], { stdio: 'ignore' });
    } catch { return resolve(false); }
    child.on('error', () => resolve(false));
    child.on('exit', code => resolve(code === 0));
  });
}

async function build(root, env, host, onLine) {
  onLine('');
  onLine(`### ${env.dir} -- ${env.label}`);
  await run(host.cmd, [...host.pre, '-m', 'venv', path.join(root, env.dir)], root, onLine);
  const py = venvPython(root, env.dir);
  await run(py, ['-m', 'pip', 'install', '--upgrade', 'pip'], root, onLine);
  await run(py, ['-m', 'pip', 'install', '-r', env.req], root, onLine);
  if (!await verify(root, env))
    throw new Error(`pip finished but ${env.imports.join(', ')} will not import`);
  onLine(`### ${env.dir} ready`);
}

/* Build whatever this machine is missing.  Resolves with the list of envs that are usable
 * afterwards; rejects only if the REQUIRED one could not be built, because that is the
 * only failure that leaves nothing to open a window onto. */
async function ensureEnvs(root, onLine) {
  const state = [];
  for (const env of ENVS) state.push([env, await verify(root, env)]);

  const missing = state.filter(([, ok]) => !ok).map(([env]) => env);
  if (!missing.length) {
    onLine('# both environments are already built.');
    return ENVS.map(e => e.dir);
  }

  onLine(`# this machine is missing ${missing.map(e => e.dir).join(' and ')}.`);
  onLine('# running what `make env` runs, here, now. This downloads a few hundred MB');
  onLine('# and takes a handful of minutes on a first run.');

  const host = await hostPython(onLine);
  if (!host)
    throw new Error(
      'No Python 3.10 or newer was found on this machine.\n\n' +
      'The pipeline is a Python program; the app can build its environments but it ' +
      'cannot install a Python to build them with. Install one from python.org (or ' +
      '`brew install python@3.12` on macOS), then try again.');

  const ready = [];
  for (const [env, ok] of state) {
    if (ok) { ready.push(env.dir); continue; }
    try {
      await build(root, env, host, onLine);
      ready.push(env.dir);
    } catch (e) {
      if (env.required) throw new Error(`${env.dir} could not be built.\n\n${e.message}`);
      onLine('');
      onLine(`### ${env.dir} FAILED: ${e.message}`);
      onLine('### Only the STEP exporter needs it. Everything else will work.');
    }
  }
  return ready;
}

module.exports = { ENVS, ensureEnvs, venvPython, verify };
