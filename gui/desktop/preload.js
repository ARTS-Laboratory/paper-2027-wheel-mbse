/* The only bridge between the page and the machine, and it is deliberately three things.
 *
 * The page is served over HTTP by a process that starts descents. Everything it needs it
 * already gets from its own API; what it CANNOT get from there is which platform it is
 * being drawn on -- which decides where the window controls sit and therefore how much
 * room the header has to leave for them -- and when the menubar asks it to change tab.
 * So: what platform, and a tab callback. Nothing that touches the filesystem, nothing
 * that runs anything.
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
});
