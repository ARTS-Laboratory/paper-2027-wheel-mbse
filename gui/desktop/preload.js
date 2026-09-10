/* The only bridge between the page and the machine, and it is deliberately small.
 *
 * The page is served over HTTP by a process that starts descents. Everything it needs it
 * already gets from its own API; what it CANNOT get from there is which platform it is
 * being drawn on -- which decides where the window controls sit and therefore how much
 * room the header has to leave for them -- and when the menubar asks it to change tab.
 * So: what platform, and a tab callback.
 *
 * `setup` IS THE ONE VERB HERE THAT RUNS ANYTHING, AND IT IS NOT FOR THE SERVED PAGE.  It
 * belongs to `shell.html` -- the local page shown when there is no server yet, where the
 * whole question is whether this machine has the environments to start one. It is exposed
 * here only because a window has ONE preload and shell.html shares it; `main.js` refuses
 * the call unless the sender really is that page. It takes no arguments for the same
 * reason: there is no path, no package and no URL for a caller to choose.
 *
 * `window.wheel` being absent is the signal that this is the browser route rather than
 * the desktop one, and static/app.js reads it exactly that way.
 */
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('wheel', {
  desktop: true,
  platform: process.platform,
  info: () => ipcRenderer.invoke('wheel:info'),
  onTab: fn => ipcRenderer.on('menu:tab', (_e, tab) => fn(tab)),

  /* shell.html only -- see above. */
  setup: () => ipcRenderer.invoke('wheel:setup'),
  onSetupLine: fn => ipcRenderer.on('wheel:setup-line', (_e, line) => fn(line)),
});
