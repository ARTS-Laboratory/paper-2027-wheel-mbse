/* The desktop shell.
 *
 * IT OWNS A WINDOW AND A CHILD PROCESS, AND NOTHING ELSE.  Every panel, every number and
 * every decision still comes from `gui/server.py`; this file starts that server on a port
 * the OS picks, points a window at it, and gets out of the way.  Keeping it that thin is
 * what lets the same server still be opened in a browser -- `gui/wheelgui --open` is
 * unchanged and untouched by any of this.
 *
 * THE SERVER IS A CHILD, THE JOBS ARE NOT.  Quitting kills the server (below, in
 * `before-quit`) and that is safe precisely because `jobs.py` refuses to make a run a
 * child of anything: a descent launched from this window is a transient systemd unit or a
 * new session of its own, and closing the app does not touch it.  The next launch reads
 * it back off disk.  Do not "improve" this by keeping the app alive to babysit runs.
 *
 * IT DRIVES THE MACHINE IT IS RUNNING ON, AND ONLY THAT ONE.  There is no attach mode, no
 * remote URL and no tunnel: the window, the server, the venvs and the runs are all on this
 * computer, and the Mac build drives the Mac.  That is a stronger constraint than it
 * sounds, because it removes the escape hatch -- a machine with no `.venv-opt` cannot be
 * answered by pointing the window at a machine that has one, so the app has to be able to
 * BUILD one.  See `setup.js`, which runs what `make env` runs.
 *
 * WHERE THE REPO IS, is the one question a packaged app has that `npm start` does not.
 * Unpacked, the answer is two directories up.  Installed from a .dmg or an .exe, the app
 * is somewhere else entirely and the checkout it should drive is a thing only the user
 * knows, so it is asked for once and remembered.  See `resolveRepo`.
 */
'use strict';

const { app, BrowserWindow, dialog, shell, nativeTheme, ipcMain } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const buildMenu = require('./menu');
const { ENVS, ensureEnvs, verify } = require('./setup');

const IS_MAC = process.platform === 'darwin';
const IS_WIN = process.platform === 'win32';

/* The local page: the splash, the failure state and the setup log all live in this one
 * file, and it is the only thing this window ever loads that the server did not serve. */
const SHELL = path.join(__dirname, 'shell.html');
const SHELL_URL = pathToFileURL(SHELL).href;

/* Three colours from static/style.css, repeated here because the window is painted before
 * any stylesheet loads.
 *
 * GROUND is `--ground`, the page body, and it is the window's own background -- getting it
 * wrong is a flash of the wrong colour on every launch.  SURFACE is `--surface`, which is
 * what the HEADER is painted in, and it is the one the overlay must use: the strip the OS
 * draws its buttons on sits on top of that header, so an overlay tinted `--ground` shows
 * as a darker rectangle at the end of the title bar. */
const GROUND = { light: '#faf9f7', dark: '#16161a' };
const SURFACE = { light: '#ffffff', dark: '#1d1d22' };
const INK = { light: '#1a1a18', dark: '#e8e6e1' };
const theme = () => (nativeTheme.shouldUseDarkColors ? 'dark' : 'light');
const overlay = () => ({ color: SURFACE[theme()], symbolColor: INK[theme()], height: 42 });

let win = null;
let server = null;          // the ChildProcess running gui/server.py
let serverUrl = null;
let serverLog = [];         // the last lines it printed, kept for the failure page

/* ---------------------------------------------------------------------------
 * SETTINGS -- one small JSON file, no dependency
 * ------------------------------------------------------------------------ */
const settingsPath = () => path.join(app.getPath('userData'), 'settings.json');

function loadSettings() {
  try { return JSON.parse(fs.readFileSync(settingsPath(), 'utf8')); } catch { return {}; }
}

function saveSettings(patch) {
  const next = { ...loadSettings(), ...patch };
  try {
    fs.mkdirSync(path.dirname(settingsPath()), { recursive: true });
    fs.writeFileSync(settingsPath(), JSON.stringify(next, null, 2));
  } catch (e) { console.error('could not save settings:', e.message); }
  return next;
}

/* ---------------------------------------------------------------------------
 * THE REPO, AND THE INTERPRETER INSIDE IT
 * ------------------------------------------------------------------------ */
const isRepo = root =>
  !!root && fs.existsSync(path.join(root, 'gui', 'server.py'));

/* A venv is `bin/python` everywhere but Windows. Same rule as gui/catalog.py, which is
 * the file that has to agree with this one. */
const pythonIn = root => path.join(root, '.venv-opt',
  ...(IS_WIN ? ['Scripts', 'python.exe'] : ['bin', 'python']));

function resolveRepo() {
  const dev = path.resolve(__dirname, '..', '..');
  for (const candidate of [process.env.WHEEL_REPO, loadSettings().repo, dev])
    if (isRepo(candidate)) return candidate;
  return null;
}

async function askForRepo() {
  const r = await dialog.showOpenDialog(win, {
    title: 'Where is the wheel checkout?',
    message: 'Pick the directory holding gui/, src/ and the Makefile.',
    properties: ['openDirectory'],
  });
  if (r.canceled || !r.filePaths.length) return null;
  const root = r.filePaths[0];
  if (!isRepo(root)) {
    await dialog.showMessageBox(win, {
      type: 'error', message: 'That is not a wheel checkout.',
      detail: `No gui/server.py under ${root}.`,
    });
    return null;
  }
  saveSettings({ repo: root });
  return root;
}

/* ---------------------------------------------------------------------------
 * THE SERVER
 * ------------------------------------------------------------------------ */
function startServer(root) {
  return new Promise((resolve, reject) => {
    const py = pythonIn(root);
    if (!fs.existsSync(py))
      return reject(new Error(
        `No interpreter at ${py}.\n\n` +
        `The panels import wheel_requirements, wheel_geometry and wheel_fea, so they ` +
        `need the optimisation env.`));

    serverLog = [];
    /* `-u`: the ready line has to arrive before the pipe buffer fills, and server.py
     * flushes it by hand for the same reason. Belt and braces on purpose -- a shell that
     * hangs forever on a line that was written but not flushed is a bad first run. */
    server = spawn(py, ['-u', path.join('gui', 'server.py'), '--port', '0'],
                   { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });

    let settled = false;
    const done = (fn, arg) => { if (!settled) { settled = true; fn(arg); } };

    const watch = stream => {
      let buf = '';
      stream.setEncoding('utf8');
      stream.on('data', chunk => {
        buf += chunk;
        let nl;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
          serverLog.push(line);
          if (serverLog.length > 200) serverLog.shift();
          const hit = line.match(/^WHEELGUI_READY (\S+)/);
          if (hit) { serverUrl = hit[1]; done(resolve, hit[1]); }
        }
      });
    };
    watch(server.stdout); watch(server.stderr);

    server.on('error', e => done(reject, e));
    server.on('exit', code => {
      server = null;
      done(reject, new Error(
        `The server exited with code ${code} before it was ready.\n\n` +
        serverLog.join('\n')));
      /* An exit AFTER we were ready is a crash mid-session, and a window pointed at a
       * dead port shows a browser error page. Say what happened instead. */
      if (settled && win && !win.isDestroyed()) showShell({
        msg: `The server stopped (exit ${code}).\n\n` + serverLog.join('\n') });
    });

    /* jax is imported lazily, so the port answers long before the MBSE panel costs
     * anything. Sixty seconds is a cold filesystem, not a slow import. */
    setTimeout(() => done(reject, new Error(
      'The server did not report a port within 60 s.\n\n' + serverLog.join('\n'))), 60000);
  });
}

function stopServer() {
  if (!server) return;
  const p = server; server = null;
  try { p.kill(); } catch { /* already gone */ }
}

/* ---------------------------------------------------------------------------
 * THE WINDOW
 * ------------------------------------------------------------------------ */
function createWindow() {
  const { bounds } = loadSettings();
  win = new BrowserWindow({
    width: 1440, height: 940, minWidth: 880, minHeight: 620,
    ...(bounds || {}),
    show: false,
    backgroundColor: GROUND[theme()],
    /* Windows and Linux read the window's own icon for the taskbar; macOS takes it from
     * the bundle and ignores this. Unpackaged, it is the only icon there is. */
    ...(IS_MAC ? {} : { icon: path.join(__dirname, 'build', 'icon.png') }),
    /* The page draws its own title bar (see `header` in static/index.html), so on macOS
     * the traffic lights are inset over it and on Windows/Linux the system controls are
     * overlaid on it. `titleBarOverlay` colours have to be repainted when the OS theme
     * flips -- see the nativeTheme listener below. */
    titleBarStyle: IS_MAC ? 'hiddenInset' : 'hidden',
    ...(IS_MAC ? { trafficLightPosition: { x: 16, y: 14 } }
              : { titleBarOverlay: overlay() }),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true, nodeIntegration: false, sandbox: true,
      spellcheck: false,
    },
  });

  win.once('ready-to-show', () => win.show());
  win.on('close', () => {
    if (win && !win.isFullScreen() && !win.isMinimized())
      saveSettings({ bounds: win.getNormalBounds() });
  });
  win.on('closed', () => { win = null; });

  /* THE WINDOW GOES TO ONE PLACE AND STAYS THERE.  This server starts processes; a page
   * inside it that could navigate somewhere else, or open a second window onto anything
   * it liked, would be handing that to whatever it navigated to. Links go to the OS
   * browser, which is where a link to a datasheet belongs anyway. */
  const external = url => { shell.openExternal(url); return { action: 'deny' }; };
  win.webContents.setWindowOpenHandler(({ url }) => external(url));
  win.webContents.on('will-navigate', (e, url) => {
    if (serverUrl && url.startsWith(serverUrl)) return;
    e.preventDefault(); shell.openExternal(url);
  });

  return win;
}

/* Three states, one file, chosen by the fragment: no hash is the splash, and a hash is a
 * JSON payload `{ msg, setup }`.  `setup` is what puts the "Set up this machine" button on
 * the page, and it is set from a MEASUREMENT rather than from the shape of the error --
 * see `showStartFailure`. */
function showShell(payload) {
  if (!win || win.isDestroyed()) return Promise.resolve();
  return win.loadFile(SHELL, payload
    ? { hash: encodeURIComponent(JSON.stringify(payload)) }
    : {});
}

/* The server would not start.  Whether that is worth offering to set the machine up for is
 * not answerable from the message -- a missing interpreter and a half-installed one read
 * completely differently and want the same button -- so ask the env itself rather than
 * pattern-matching the text. */
async function showStartFailure(root, message) {
  const ok = await verify(root, ENVS[0]);
  return showShell({ msg: message, setup: !ok });
}

async function launchInto(root) {
  try {
    const url = await startServer(root);
    await win.loadURL(url);
    saveSettings({ repo: root });
  } catch (e) {
    await showStartFailure(root, e.message);
  }
}

async function boot() {
  /* GUARDED, because `boot` is also the retry path.  View → Reload calls this when there
   * is no server URL to reload, and an unguarded `createWindow()` there would open a
   * SECOND window onto the same failure and leak the first. */
  if (!win || win.isDestroyed()) createWindow();
  await showShell(null);                          // the splash, painted at once

  const root = resolveRepo();
  if (!root)
    return showShell({ msg:
      'No wheel checkout found.\n\nThis app is a control surface over a checkout of ' +
      'the wheel repository: it needs the directory holding gui/, src/ and the ' +
      'Makefile.\n\nUse File → Choose checkout… to point it at one.' });

  /* THE CHECK IS `verify`, NOT `fs.existsSync`, AND THAT IS A BUG FIX.  An interrupted
   * pip leaves a `.venv-opt/bin/python` that exists and has nothing in it, and existence
   * was passing that -- so `gui/server.py` started (it is stdlib-only by design and
   * imports the pipeline lazily), the window opened, and the missing packages surfaced
   * as a panel error minutes later instead of as the one thing that was actually wrong.
   * `verify` asks pip's own question with `find_spec` and costs 9 ms, so there is no
   * reason for the launch path to settle for a weaker answer. */
  const built = await verify(root, ENVS[0]);
  if (!built) {
    const py = pythonIn(root);
    return showShell({
      setup: true,
      msg: (fs.existsSync(py)
        ? `There is an interpreter at ${py}, but numpy, scipy and jax are not all ` +
          `installed in it -- which is what an interrupted pip install leaves behind.\n\n`
        : `The checkout at ${root} has no .venv-opt, so there is no interpreter here ` +
          `that can import wheel_requirements, wheel_geometry and wheel_fea.\n\n`) +
        `The app can finish this on this machine. It runs what \`make env\` runs: ` +
        `create the two virtualenvs, then pip-install requirements-opt.txt and ` +
        `requirements-cad.txt into them. Anything already downloaded is reused.`,
    });
  }

  await launchInto(root);
}

async function relaunchInto(root) {
  stopServer();
  serverUrl = null;
  await showShell(null);
  await launchInto(root);
}

/* ---------------------------------------------------------------------------
 * SETTING THIS MACHINE UP
 * ------------------------------------------------------------------------ */
let settingUp = false;

/* Reachable ONLY from shell.html, and the check is on the sender rather than on trust.
 *
 * This window's preload is shared with everything `gui/server.py` serves -- it has to be,
 * there is one webPreferences per window -- so without this guard a page delivered over
 * HTTP could invoke pip.  That page is our own and is already allowed to start 50-hour
 * descents through the server's API, so this is not the difference between safe and
 * unsafe; it is the difference between a bridge whose reach is stated and one whose reach
 * is argued.  `preload.js` says the page can ask what platform it is on and nothing else,
 * and this keeps that sentence true. */
async function handleSetup(event) {
  if (!event.senderFrame || !event.senderFrame.url.startsWith(SHELL_URL))
    throw new Error('setup is only available from the shell page');
  if (settingUp) throw new Error('a setup is already running');

  const root = resolveRepo();
  if (!root) return { ok: false, error: 'No checkout to set up.' };

  settingUp = true;
  const send = line => {
    if (win && !win.isDestroyed()) win.webContents.send('wheel:setup-line', line);
  };
  try {
    const ready = await ensureEnvs(root, send);
    send('');
    send('# done. Starting the control surface\u2026');
    /* Navigate AFTER the invoke resolves, not before: loading the server URL destroys the
     * page that is waiting on this promise, and a renderer that never sees its own result
     * cannot show the last line of the log it just printed. */
    setTimeout(() => relaunchInto(root), 500);
    return { ok: true, ready };
  } catch (e) {
    send('');
    send(`### setup failed: ${e.message}`);
    return { ok: false, error: e.message };
  } finally {
    settingUp = false;
  }
}

/* ---------------------------------------------------------------------------
 * LIFECYCLE
 * ------------------------------------------------------------------------ */
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (win) { if (win.isMinimized()) win.restore(); win.focus(); }
  });

  app.whenReady().then(() => {
    buildMenu({
      onTab: tab => win && win.webContents.send('menu:tab', tab),
      onReload: () => { if (serverUrl) win.loadURL(serverUrl); else boot(); },
      onChooseRepo: async () => {
        const root = await askForRepo();
        if (root) await relaunchInto(root);
      },
      onOpenRuns: () => {
        const root = resolveRepo();
        if (root) shell.openPath(path.join(root, 'gui', 'runs'));
      },
      onOpenRepo: () => { const r = resolveRepo(); if (r) shell.openPath(r); },
    });

    /* The overlay buttons are painted by the OS, not by CSS, so a theme flip has to be
     * pushed to them by hand or they stay light on a dark header. */
    nativeTheme.on('updated', () => {
      if (!win || win.isDestroyed()) return;
      win.setBackgroundColor(GROUND[theme()]);
      if (!IS_MAC) win.setTitleBarOverlay(overlay());
    });

    ipcMain.handle('wheel:info', () => ({
      platform: process.platform,
      version: app.getVersion(),
      repo: resolveRepo(),
      electron: process.versions.electron,
    }));
    ipcMain.handle('wheel:setup', handleSetup);

    boot();
    app.on('activate', () => { if (!BrowserWindow.getAllWindows().length) boot(); });
  });

  app.on('window-all-closed', () => { if (!IS_MAC) app.quit(); });
  app.on('before-quit', stopServer);
  app.on('will-quit', stopServer);
}
