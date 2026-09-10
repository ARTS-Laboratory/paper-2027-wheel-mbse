/* The menubar.
 *
 * A menubar is not decoration on macOS -- an app without one gets Electron's default,
 * which advertises "Electron" as the product and offers to open the Electron docs. It is
 * also the only place the window-level verbs belong: the page itself is the server's, and
 * putting "Choose checkout" or "Open runs folder" in it would mean the browser route had
 * buttons that could not work.
 *
 * The tab items drive the SAME four tabs the page renders, over IPC. They do not
 * duplicate the tab strip's logic; they ask it to do its job.
 */
'use strict';

const { app, Menu } = require('electron');

const IS_MAC = process.platform === 'darwin';
const TABS = [
  ['Mission & Requirements', 'mission', '1'],
  ['Design & Preview', 'design', '2'],
  ['Run', 'run', '3'],
  ['Status', 'status', '4'],
];

module.exports = function buildMenu(on) {
  const template = [
    ...(IS_MAC ? [{ role: 'appMenu' }] : []),
    {
      label: 'File',
      submenu: [
        /* Greyed in attach mode: the checkout is on another machine, and a file dialog
         * that can only ever open the wrong directory is worse than a disabled item. */
        { label: 'Choose checkout…', click: on.onChooseRepo, enabled: !on.attached },
        { type: 'separator' },
        { label: 'Open runs folder', click: on.onOpenRuns, enabled: !on.attached },
        { label: 'Open checkout', click: on.onOpenRepo, enabled: !on.attached },
        { type: 'separator' },
        IS_MAC ? { role: 'close' } : { role: 'quit' },
      ],
    },
    { role: 'editMenu' },
    {
      label: 'View',
      submenu: [
        ...TABS.map(([label, tab, key]) => ({
          label, accelerator: `CmdOrCtrl+${key}`, click: () => on.onTab(tab),
        })),
        { type: 'separator' },
        /* NOT `role: 'reload'`. That reloads the current URL, and when the server failed
         * to start the current URL is the local failure page -- reloading it would
         * re-show the same message rather than retrying, which is the one thing anyone
         * pressing reload on that page wants. */
        { label: 'Reload', accelerator: 'CmdOrCtrl+R', click: on.onReload },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    { role: 'windowMenu' },
    {
      role: 'help',
      submenu: [
        { label: 'About Wheel', click: () => app.showAboutPanel() },
      ],
    },
  ];
  app.setAboutPanelOptions({
    applicationName: 'Wheel',
    applicationVersion: app.getVersion(),
    copyright: 'A control surface over the compliant-wheel pipeline.\n' +
               'Runs launched here are detached: closing this app does not stop them.' +
               (on.attached ? '\n\nAttached to a server on another machine.' : ''),
  });
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
};
