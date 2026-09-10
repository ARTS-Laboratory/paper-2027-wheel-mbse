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
const buildMenu = require('./menu');

const IS_MAC = process.platform === 'darwin';
const IS_WIN = process.platform === 'win32';

/* ATTACH MODE: drive a server that is ALREADY RUNNING, somewhere else.
 *
 * The normal path spawns `.venv-opt/bin/python gui/server.py` next to a checkout, which is
 * exactly what a second machine does not have -- and cannot fake, because those venv
 * binaries are Linux ELF and an sshfs mount of the repo would hand macOS an interpreter it
 * cannot exec.  So the shell and the pipeline are allowed to sit on different machines:
 *
 *     ssh -N -L 8731:127.0.0.1:8731 you@the-box          # on the laptop
 *     WHEEL_SERVER_URL=http://127.0.0.1:8731 npm start
 *
 * THE URL SHOULD BE A LOOPBACK TUNNEL, and that is the server's rule rather than this
 * file's: `server.py` binds 127.0.0.1 because the endpoint starts processes and has no
 * authentication, and its docstring says to tunnel rather than move the bind address.
 * Nothing here re-binds anything, so pointing this at a non-loopback host only works if
 * someone has already overridden that -- which is their decision to have made, not a
 * default this reaches around. */
const ATTACH = (process.env.WHEEL_SERVER_URL || '').trim();

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
        `need the optimisation env. Build it with:\n\n    make env`));

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
      if (settled && win && !win.isDestroyed()) showFailure(
        `The server stopped (exit ${code}).\n\n` + serverLog.join('\n'));
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

function showFailure(message) {
  if (!win || win.isDestroyed()) return;
  win.loadFile(path.join(__dirname, 'shell.html'),
               { hash: encodeURIComponent(message) });
}

function attachOrigin() {
  /* `origin + '/'` and not the string as given: `will-navigate` compares with
   * `startsWith`, so a URL typed without a trailing slash would fail to match the very
   * page it just loaded and the shell would send its own index to the OS browser. */
  try {
    const u = new URL(ATTACH);
    return /^https?:$/.test(u.protocol) ? u.origin + '/' : null;
  } catch {
    return null;
  }
}

async function bootAttached() {
  const url = attachOrigin();
  if (!url)
    return showFailure(
      `WHEEL_SERVER_URL is not an http(s) URL:\n\n    ${ATTACH}\n\n` +
      `Expected something like http://127.0.0.1:8731 -- the local end of an ssh tunnel ` +
      `to the machine holding the checkout.`);
  serverUrl = url;
  try {
    await win.loadURL(url);
  } catch (e) {
    showFailure(
      `Nothing answered at ${url}\n\n${e.message}\n\n` +
      `Attach mode does not start a server; it expects one to be running already. ` +
      `Check that the tunnel is up and that the far end is serving:\n\n` +
      `    ssh -N -L 8731:127.0.0.1:8731 you@the-box\n` +
      `    make gui-browser        # on the box, if nothing is listening yet`);
  }
}

async function boot() {
  createWindow();
  await win.loadFile(path.join(__dirname, 'shell.html'));   // the splash, painted at once

  if (ATTACH) return bootAttached();

  let root = resolveRepo();
  if (!root) {
    showFailure('No wheel checkout found.\n\nThis app is a control surface over a ' +
                'checkout of the wheel repository: it needs the directory holding gui/, ' +
                'src/ and the Makefile.\n\nUse File → Choose checkout… to point ' +
                'it at one.');
    return;
  }
  try {
    const url = await startServer(root);
    await win.loadURL(url);
    saveSettings({ repo: root });
  } catch (e) {
    showFailure(e.message);
  }
}

async function relaunchInto(root) {
  stopServer();
  serverUrl = null;
  await win.loadFile(path.join(__dirname, 'shell.html'));
  try {
    const url = await startServer(root);
    await win.loadURL(url);
  } catch (e) { showFailure(e.message); }
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
      /* In attach mode the checkout is on the other end of a tunnel, so the three verbs
       * that reach for a local path are greyed rather than left to fail quietly on a
       * directory this machine does not have. */
      attached: !!ATTACH,
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
      repo: ATTACH ? null : resolveRepo(),   // null: the checkout is not on this machine
      attached: ATTACH || null,
      electron: process.versions.electron,
    }));

    boot();
    app.on('activate', () => { if (!BrowserWindow.getAllWindows().length) boot(); });
  });

  app.on('window-all-closed', () => { if (!IS_MAC) app.quit(); });
  app.on('before-quit', stopServer);
  app.on('will-quit', stopServer);
}
