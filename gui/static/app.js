'use strict';
/* The client. Plain ES modules-free JS on purpose: no build step, nothing to install,
 * and the whole gui/ directory stays deletable. */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
}

async function api(path, body, method) {
  const opt = { method: method || (body ? 'POST' : 'GET'),
                headers: { 'Content-Type': 'application/json' } };
  if (body) opt.body = JSON.stringify(body);
  const r = await fetch(path, opt);
  const text = await r.text();
  let data; try { data = JSON.parse(text); } catch { data = { error: text }; }
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

/* Significant-figure formatting. A requirement set holds 66.7233 N beside 2300 MPa
 * beside 0.35; a fixed decimal count makes one of them unreadable. */
function num(v, sig = 5) {
  if (v === null || v === undefined || v === '') return '—';
  if (typeof v !== 'number') return String(v);
  if (!isFinite(v)) return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(2);
  const d = Math.max(0, sig - 1 - Math.floor(Math.log10(a)));
  return v.toFixed(Math.min(d, 6)).replace(/\.?0+$/, '') || '0';
}

function dur(s) {
  if (s === null || s === undefined || !isFinite(s)) return '—';
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${(s / 60).toFixed(1)} min`;
  if (s < 172800) return `${(s / 3600).toFixed(1)} h`;
  return `${(s / 86400).toFixed(1)} d`;
}

function verdict(v) {
  return el('span', { class: 'verdict v-' + String(v).replace(/\s+/g, '-') }, v);
}

function note(kind, ...kids) { return el('div', { class: 'note ' + kind }, ...kids); }

function modal(title, bodyNodes, actions) {
  const root = $('#modal-root');
  const close = () => (root.innerHTML = '');
  const box = el('div', { class: 'modal' },
    el('h3', {}, title), ...bodyNodes,
    el('div', { class: 'actions' },
      ...actions.map(a => el('button',
        { class: 'act ' + (a.kind || 'ghost'),
          onclick: () => { if (a.run) a.run(); if (a.closes !== false) close(); } },
        a.label))));
  root.replaceChildren(el('div', { class: 'scrim', onclick: e => { if (e.target === e.currentTarget) close(); } }, box));
  return close;
}

/* ===================================================================== */
/* TABS                                                                  */
/* ===================================================================== */
let TAB = 'mission';
function selectTab(name) {
  const b = $(`#tabs button[data-tab="${name}"]`);
  if (!b) return;
  TAB = name;
  $$('#tabs button').forEach(x => x.setAttribute('aria-selected', x === b));
  $$('.panel').forEach(p => (p.hidden = p.id !== 'p-' + TAB));
  if (TAB === 'status') renderStatus();
}
$$('#tabs button').forEach(b => b.onclick = () => selectTab(b.dataset.tab));

/* The desktop shell's View menu drives THESE tabs rather than keeping its own idea of
 * which one is showing -- a menubar and a tab strip that each tracked the selection
 * separately would disagree the first time anyone used the other one. `window.wheel` is
 * absent in a browser and this is simply skipped. */
if (window.wheel) window.wheel.onTab(selectTab);

/* ===================================================================== */
/* MISSION & REQUIREMENTS                                                */
/* ===================================================================== */
const M = { form: null, mission: {}, points: null, axes: [], calibrated: null };

async function initMission() {
  const { form, calibrated } = await api('/api/mbse/form');
  M.form = form; M.axes = form.axes; M.calibrated = calibrated;
  for (const f of form.fields) M.mission[f.name] = f.default;

  $('#mission-form').replaceChildren(...form.fields.map(f => {
    const input = f.kind === 'choice'
      ? el('select', { onchange: e => { M.mission[f.name] = e.target.value; derive(); } },
          ...f.choices.map(c => el('option', { value: c, selected: c === f.default }, c)))
      : el('input', { type: 'number', value: f.default,
                      step: f.kind === 'int' ? '1' : 'any',
                      oninput: e => { M.mission[f.name] = e.target.value; derive(); } });
    return el('label', { class: 'field' },
      el('span', { class: 'name' }, f.name, f.unit ? el('span', { class: 'unit' }, ' ' + f.unit) : ''),
      input, el('span', { class: 'help' }, f.help));
  }));

  buildPoints();
  $('#pts-calibrated').onclick = () => { M.points = { ...calibrated.points }; buildPoints(); derive(); };
  $('#pts-off').onclick = () => { M.points = null; buildPoints(); derive(); };
  $('#btn-verify').onclick = () => runCompliance();
  $('#btn-save-req').onclick = saveRequirements;
  $('#btn-descend').onclick = descendUnder;
  await derive();
  await runCompliance();
}

function buildPoints() {
  const on = M.points !== null;
  const pts = M.points || M.calibrated.points;
  $('#points').replaceChildren(...M.axes.map(a => {
    const slider = el('input', { type: 'range', min: 0, max: 100, step: 0.5,
      value: pts[a], disabled: !on,
      oninput: e => rebalance(a, parseFloat(e.target.value)) });
    return [el('span', { class: 'pn' }, a),
            slider,
            el('span', { class: 'pv', id: 'pv-' + a }, num(pts[a], 4))];
  }).flat());
  $('#pts-total').textContent = num(M.axes.reduce((s, a) => s + pts[a], 0), 5);
  $('#pts-state').textContent = on
    ? `Anchored at ${M.calibrated.reference} — its smoothness term is the one scale the weight table cannot supply, so this map re-derives at every promotion.`
    : 'Weights untouched: DEFAULT_WEIGHTS exactly, not the calibrated allocation re-applied. Those two agree to a couple of ulp, and a couple of ulp is not a thing a default should introduce.';
}

/* Zero-sum by construction. The remaining budget is spread over the OTHER axes in
 * proportion to what they already hold, so moving one slider does not silently reorder
 * the rest; when they are all at zero there is no proportion to keep and it goes even. */
function rebalance(axis, value) {
  const pts = { ...(M.points || M.calibrated.points) };
  value = Math.max(0, Math.min(100, value));
  const others = M.axes.filter(a => a !== axis);
  const rest = 100 - value;
  const sum = others.reduce((s, a) => s + pts[a], 0);
  pts[axis] = value;
  for (const a of others) pts[a] = sum > 1e-9 ? rest * (pts[a] / sum) : rest / others.length;
  M.points = pts;
  for (const a of M.axes) {
    const pv = $('#pv-' + a); if (pv) pv.textContent = num(pts[a], 4);
    const sl = $$('#points input')[M.axes.indexOf(a)]; if (sl) sl.value = pts[a];
  }
  $('#pts-total').textContent = num(M.axes.reduce((s, a) => s + pts[a], 0), 5);
  derive();
}

let deriveTimer = null, lastDerived = null;
function derive() {
  clearTimeout(deriveTimer);
  return new Promise(res => {
    deriveTimer = setTimeout(async () => {
      try {
        const d = await api('/api/mbse/derive', { mission: M.mission, points: M.points });
        lastDerived = d; renderDerived(d); res(d);
      } catch (e) { renderDeriveError(e.message); res(null); }
    }, 120);
  });
}

function renderDeriveError(msg) {
  $('#derivations').replaceChildren(el('tbody', {}, el('tr', {}, el('td', {},
    note('bad', el('b', {}, 'refused. '), msg)))));
}

function renderDerived(d) {
  $('#derivations').replaceChildren(
    el('thead', {}, el('tr', {}, el('th', {}, 'quantity'), el('th', { class: 'n' }, 'value'),
      el('th', {}, ''), el('th', {}, 'how'))),
    el('tbody', {}, ...d.derivations.map(r => el('tr', {},
      el('td', {}, el('span', { class: 'num' }, r.name)),
      el('td', { class: 'n' }, num(r.value, 6)),
      el('td', { class: 'faint tiny' }, r.unit),
      el('td', { class: 'muted' }, r.why)))));

  $('#req-hash').textContent = 'req_hash ' + d.req_hash;
  const units = { force_n: 'N', target_deflection_mm: 'mm', allowable_stress_mpa: 'MPa',
                  min_wall_mm: 'mm', e_mpa: 'MPa', nu: '' };
  $('#requirements').replaceChildren(el('tbody', {},
    ...Object.entries(d.requirements).map(([k, v]) => el('tr', {},
      el('td', {}, el('span', { class: 'num' }, k)),
      el('td', { class: 'n' }, num(v, 7)),
      el('td', { class: 'faint tiny' }, units[k])))));

  const t = d.thermal;
  $('#thermal').replaceChildren(
    (t.e_retention < 1 || t.sigma_retention < 1)
      ? note('warn', el('b', {}, 'thermal knockdown active. '),
          `modulus x ${num(t.e_retention, 4)}, allowable x ${num(t.sigma_retention, 4)}. `,
          el('span', { class: 'muted tiny' }, 'Quasi-static knockdown only — creep and rate effects are not modelled.'))
      : el('span'));

  $('#weights').replaceChildren(
    el('thead', {}, el('tr', {}, el('th', {}, 'term'), el('th', { class: 'n' }, 'weight'),
      el('th', { class: 'n' }, 'default'))),
    el('tbody', {}, ...d.weights.map(w => el('tr', {},
      el('td', {}, el('span', { class: 'num' }, w.term),
        w.moved ? el('span', { class: 'verdict v-MISSED', style: 'margin-left:6px' }, 'moved') : ''),
      el('td', { class: 'n' }, num(w.value, 6)),
      el('td', { class: 'n faint' }, num(w.default, 6))))));
}

async function runCompliance() {
  const tbl = $('#compliance');
  try {
    const c = await api('/api/mbse/compliance',
      { mission: M.mission, points: M.points });
    renderCompliance(c);
  } catch (e) {
    $('#compliance-note').replaceChildren(note('bad', e.message));
    tbl.replaceChildren();
  }
}

function renderCompliance(c) {
  const notes = [];
  if (c.error) notes.push(note('bad', c.error));
  if (c.strict_refused)
    notes.push(note('warn', el('b', {}, 'cross-comparison. '),
      'This record was scored under a different requirement set, so strict verification ' +
      'refused. The table below is read with strict=False: a utilisation is a fraction of ' +
      'an allowable and an axle drop is measured against a target, so these numbers look ' +
      'alike and mean different things. Re-score to compare properly.'));
  if (c.provenance && c.provenance.startsWith('unstated'))
    notes.push(note('', el('b', {}, 'provenance unstated. '),
      `${c.record} predates the requirements block, so nothing records what it was scored ` +
      `under. Its numbers are read as-is.`));
  if (c.compliant === true) notes.push(note('ok', el('b', {}, 'COMPLIANT. '), 'Every shall passes.'));
  if (c.compliant === false) notes.push(note('bad', el('b', {}, 'NOT COMPLIANT. '),
    'At least one shall is breached — see the FAIL rows.'));
  $('#compliance-note').replaceChildren(...notes);

  const row = r => el('tr', { class: r.kind },
    el('td', {}, el('span', { class: 'num' }, r.id),
      el('div', { class: 'muted tiny' }, r.statement)),
    el('td', {}, el('span', { class: 'kind' }, r.kind)),
    el('td', { class: 'n' }, num(r.value, 5)),
    el('td', { class: 'n faint' }, r.limit === null || r.limit === undefined ? '—' : num(r.limit, 5)),
    el('td', { class: 'n' }, r.margin === null || r.margin === undefined ? '—' : num(r.margin, 4)),
    el('td', {}, verdict(r.verdict)),
    el('td', { class: 'muted tiny' }, r.method,
      r.evidence ? el('div', { class: 'num tiny faint' },
        `${r.evidence.quantity} = ${num(r.evidence.value, 5)} ${r.evidence.sense} ${num(r.evidence.limit, 5)}`) : ''));

  $('#compliance').replaceChildren(
    el('thead', {}, el('tr', {},
      el('th', {}, 'requirement'), el('th', {}, ''), el('th', { class: 'n' }, 'value'),
      el('th', { class: 'n' }, 'limit'), el('th', { class: 'n' }, 'margin'),
      el('th', {}, 'verdict'), el('th', {}, 'method'))),
    el('tbody', {}, ...(c.rows || []).map(row)));
}

async function saveRequirements() {
  try {
    const r = await api('/api/mbse/save', { mission: M.mission, points: M.points });
    modal('Requirements written', [
      el('p', {}, 'A Stage-3 descent can be pointed at this file.'),
      el('pre', {}, r.path), el('p', { class: 'muted' }, 'req_hash ' + r.req_hash)],
      [{ label: 'Close' }]);
  } catch (e) { modal('Refused', [note('bad', e.message)], [{ label: 'Close' }]); }
}

async function descendUnder() {
  const r = await api('/api/mbse/save', { mission: M.mission, points: M.points });
  openLaunch('stage3', { requirements: r.path, config: 'coarse', steps: 60 });
}

/* ===================================================================== */
/* DESIGN & PREVIEW                                                      */
/* ===================================================================== */
const D = { genes: {}, bounds: [], consts: null };

async function initDesign() {
  const g = await api('/api/genome');
  D.genes = g.genes; D.bounds = g.bounds; D.consts = g.constants;
  buildGeneSliders();
  $('#btn-reload-genome').onclick = async () => {
    const g2 = await api('/api/genome');
    D.genes = g2.genes; buildGeneSliders(); refreshPreview();
  };
  refreshPreview();
}

function buildGeneSliders() {
  $('#genes').replaceChildren(...D.bounds.map(b => {
    const v = D.genes[b.name];
    const row = el('div', { class: 'gene', id: 'gene-' + b.name },
      el('span', { class: 'g-name' }, b.name),
      el('input', { type: 'range', min: b.low, max: b.high,
        step: (b.high - b.low) / 1000, value: v,
        oninput: e => {
          D.genes[b.name] = parseFloat(e.target.value);
          $('#gene-' + b.name + ' .g-val').textContent = num(D.genes[b.name], 5);
          refreshPreview();
        } }),
      el('span', { class: 'g-val' }, num(v, 5)));
    return row;
  }));
}

let previewTimer = null;
function refreshPreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    try {
      const o = await api('/api/preview', { genes: D.genes });
      drawWheel(o); drawTaper(o);
      $('#preview-timing').textContent =
        `${o.sector.length} pts/sector · rebuilt in ${o.build_ms} ms`;
      const sat = Object.entries(o.saturated);
      $('#sat-note').replaceChildren(sat.length
        ? note('warn', el('b', {}, 'at a bound: '),
            sat.map(([k, w]) => `${k} (${w})`).join(', '), '. ',
            el('span', { class: 'muted' },
              'A saturated gene is a boundary optimum. A pinned t* means the printable ' +
              'wall floor is the active constraint, which is a real answer; a pinned cy* ' +
              'looks like the box is too tight and measurably is not.'))
        : el('span'));
      for (const b of D.bounds)
        $('#gene-' + b.name).classList.toggle('sat', b.name in o.saturated);
    } catch (e) {
      $('#preview-timing').textContent = e.message;
    }
  }, 30);
}

const pathOf = pts => pts.map((p, i) => (i ? 'L' : 'M') + p[0] + ' ' + p[1]).join('') + 'Z';
const openPathOf = pts => pts.map((p, i) => (i ? 'L' : 'M') + p[0] + ' ' + p[1]).join('');
const svg = (tag, attrs = {}) => {
  const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

function drawWheel(o) {
  const c = o.constants, root = $('#wheel');
  root.replaceChildren();
  const R = c.rim_outer_radius_mm * 1.08;
  root.setAttribute('viewBox', `${-R} ${-R} ${2 * R} ${2 * R}`);

  for (const [r, cls] of [[c.hub_radius_mm, 'w-ring'], [c.rim_radius_mm, 'w-ring'],
                          [c.rim_outer_radius_mm, 'w-ground']])
    root.append(svg('circle', { cx: 0, cy: 0, r, class: cls }));

  /* Twelve rotations of sector 0 — which is how place_sector builds the wheel and how
   * the mesher assembles it, so this is the wheel and not a picture of one. */
  const step = 360 / c.n_spokes;
  const spoke = pathOf(o.sector), centre = openPathOf(o.centerline);
  for (let i = 0; i < c.n_spokes; i++) {
    const g = svg('g', { transform: `rotate(${i * step})` });
    g.append(svg('path', { d: spoke, class: 'w-spoke' }));
    g.append(svg('path', { d: centre, class: 'w-centre' }));
    for (const f of o.fillets)
      g.append(svg('path', { d: openPathOf(f.points), class: 'w-fillet' }));
    root.append(g);
  }
  /* The control polygon on sector 0 only: twelve copies of it would be noise. */
  root.append(svg('path', { d: openPathOf(o.handles), class: 'w-hull' }));
  o.handles.forEach((p, i) => {
    root.append(svg('circle', { cx: p[0], cy: p[1], r: 0.7, class: 'w-cp' }));
    const t = svg('text', { x: p[0] + 1.2, y: p[1] - 1.0, class: 'w-label' });
    t.textContent = 'P' + i; root.append(t);
  });
}

function drawTaper(o) {
  const { s, t } = o.taper, root = $('#taper');
  const max = Math.max(...t) * 1.15 || 1;
  const d = t.map((v, i) => (i ? 'L' : 'M') + (s[i] * 100) + ' ' + (30 - 30 * v / max)).join('');
  root.replaceChildren(
    svg('path', { d, style: 'fill:none;stroke:var(--accent);stroke-width:1.2;vector-effect:non-scaling-stroke' }));
  for (const x of [33.333, 66.667])
    root.append(svg('line', { x1: x, y1: 0, x2: x, y2: 30,
      style: 'stroke:var(--rule-strong);stroke-width:.5;stroke-dasharray:2 2;vector-effect:non-scaling-stroke' }));
  $('#taper-range').textContent =
    `t0 ${num(t[0], 4)} → ${num(Math.min(...t), 4)} min → t3 ${num(t[t.length - 1], 4)} mm`;
}

/* ===================================================================== */
/* RUN                                                                   */
/* ===================================================================== */
let TARGETS = [], SELECTED = null;

async function initRun() {
  TARGETS = await api('/api/targets');
  const groups = {};
  for (const t of TARGETS) (groups[t.group] ||= []).push(t);
  const names = { pipeline: 'Pipeline', mbse: 'MBSE', gates: 'Gates' };
  $('#target-list').replaceChildren(...Object.entries(groups).map(([g, ts]) =>
    el('div', { style: 'margin-bottom:14px' },
      el('h3', { class: 'muted tiny', style: 'text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px' },
        names[g] || g),
      ...ts.map(t => el('div', { class: 'job', style: 'cursor:pointer',
          onclick: () => openLaunch(t.key) },
        el('span', { class: 'jl' }, t.label),
        t.heavy ? verdict('heavy') : el('span'),
        el('span', { class: 'meta' }, t.blurb))))));
  $('#btn-any').onclick = () => {
    const v = $('#any-target').value.trim();
    if (v) openLaunch('make:' + v);
  };
}

async function openLaunch(key, preset) {
  const t = TARGETS.find(x => x.key === key)
    || { key, label: key.replace('make:', 'make '), params: [], heavy: true };
  const params = { ...(preset || {}) };
  for (const p of t.params) if (!(p.name in params)) params[p.name] = p.default;

  const inputs = t.params.map(p => {
    const inp = p.kind === 'choice'
      ? el('select', { onchange: e => { params[p.name] = e.target.value; recost(); } },
          ...p.choices.map(c => el('option', { value: c, selected: c === params[p.name] }, c)))
      : el('input', { value: params[p.name] === '' ? '' : params[p.name],
          type: p.kind === 'str' ? 'text' : 'number', step: p.kind === 'int' ? '1' : 'any',
          style: p.kind === 'str' ? 'width:100%;text-align:left' : '',
          oninput: e => { params[p.name] = e.target.value; recost(); } });
    return el('label', { class: 'field' },
      el('span', { class: 'name' }, p.name), inp, el('span', { class: 'help' }, p.help));
  });

  const costBox = el('div');
  const close = modal(t.label,
    [el('p', { class: 'muted' }, t.blurb || ''), ...inputs, costBox],
    [{ label: 'Cancel' },
     { label: 'Launch', kind: 'act', closes: true,
       run: async () => {
         try { await api('/api/jobs', { key, params }); await pollJobs(); }
         catch (e) {
           modal('Refused', [note('bad', e.message),
             el('p', { class: 'muted small' },
               'Launch anyway only if you know this box can take it.')],
             [{ label: 'Cancel' },
              { label: 'Launch anyway', kind: 'danger',
                run: () => api('/api/jobs', { key, params, force: true }).then(pollJobs) }]);
         }
       } }]);

  async function recost() {
    try {
      const p = await api('/api/plan', { key, params });
      const rows = [];
      if (p.seconds !== undefined) {
        rows.push(el('dt', {}, 'estimated'), el('dd', {}, dur(p.seconds)));
        rows.push(el('dt', {}, 'peak RSS'), el('dd', {}, num(p.peak_gib, 4) + ' GiB'));
        rows.push(el('dt', {}, 'free now'), el('dd', {}, num(p.available_gib, 4) + ' GiB'));
        if (p.memory_max_gib)
          rows.push(el('dt', {}, 'MemoryMax'), el('dd', {}, num(p.memory_max_gib, 4) + ' GiB'));
      }
      costBox.replaceChildren(
        el('pre', {}, p.command),
        rows.length ? el('dl', { class: 'kv' }, ...rows) : el('span'),
        p.basis ? el('p', { class: 'faint tiny' }, 'basis: ' + p.basis) : el('span'),
        ...p.warnings.map(w => note('warn', w)),
        ...p.blockers.map(b => note('bad', el('b', {}, 'blocked: '), b)));
    } catch (e) { costBox.replaceChildren(note('bad', e.message)); }
  }
  recost();
}

function jobRow(j) {
  const p = j.progress || {};
  const frac = p.fraction;
  const cls = j.state === 'finished' ? 'done'
    : (j.state === 'failed' || j.state === 'interrupted') ? 'bad' : '';
  const bar = el('div', { class: 'bar ' + cls + (frac === null || frac === undefined ? ' indet' : '') },
    el('i', { style: `width:${((frac ?? 0) * 100).toFixed(1)}%` }));
  const meta = [];
  if (p.total) meta.push(`${p.done}/${p.total}`);
  if (p.eta_seconds) meta.push('eta ' + dur(p.eta_seconds));
  for (const [k, v] of Object.entries(p.detail || {}))
    if (v !== null && v !== undefined) meta.push(`${k} ${typeof v === 'number' ? num(v, 5) : v}`);
  meta.push(j.started_at.replace('T', ' '));
  return el('div', { class: 'job' + (SELECTED === j.id ? ' sel' : ''),
      onclick: () => { SELECTED = j.id; pollJobs(); } },
    el('span', { class: 'jl' }, j.label, j.heavy ? ' ' : '', j.heavy ? verdict('heavy') : ''),
    verdict(j.state),
    el('span', { class: 'meta' }, ...meta.map(m => el('span', {}, m))),
    (j.state === 'running' || j.state === 'queued' || frac !== null) ? bar : el('span'));
}

async function pollJobs() {
  let list;
  try { list = await api('/api/jobs'); } catch { return; }
  $('#hd-jobs').textContent = list.filter(j => j.state === 'running' || j.state === 'queued').length
    + ' / ' + list.length;
  $('#joblist').replaceChildren(...(list.length
    ? list.map(jobRow)
    : [el('p', { class: 'muted' }, 'No jobs yet. Launch one on the left; it will keep running if you close this page.')]));
  if (SELECTED) await renderJobDetail(SELECTED);
}

async function renderJobDetail(id) {
  let j;
  try { j = await api('/api/jobs/' + id); } catch { return; }
  const p = j.progress || {};
  const kv = [];
  const push = (k, v) => { kv.push(el('dt', {}, k), el('dd', {}, v)); };
  push('state', j.state);
  push('started', j.started_at.replace('T', ' '));
  if (j.ended_at) push('ended', j.ended_at.replace('T', ' '));
  if (j.exit_code !== null && j.exit_code !== undefined) push('exit', String(j.exit_code));
  if (p.total) push('progress', `${p.done} / ${p.total}`);
  if (p.eta_seconds) push('eta', dur(p.eta_seconds));
  if (p.elapsed_seconds) push('elapsed', dur(p.elapsed_seconds));
  if (j.unit) push('unit', j.unit);
  for (const [k, v] of Object.entries(p.detail || {}))
    if (v !== null && v !== undefined) push(k, typeof v === 'number' ? num(v, 6) : v);
  if (p.tier !== null && p.tier !== undefined)
    push('selection tier', p.tier === 0 ? '0 — shippable' : `${p.tier} — not a promotion candidate`);

  const phases = p.phases
    ? el('div', { class: 'legend' }, ...p.phases.map(ph =>
        el('span', { class: ph.done ? '' : 'faint' }, (ph.done ? '● ' : '○ ') + ph.name)))
    : el('span');

  const series = (p.series || []).filter(s => typeof s.loss === 'number');
  const spark = series.length > 2 ? lossSpark(series) : el('span');

  $('#job-detail').replaceChildren(el('div', { class: 'card' },
    el('h2', {}, j.label),
    el('p', { class: 'hint num tiny' }, (j.argv || []).join(' ')),
    el('dl', { class: 'kv' }, ...kv),
    phases, spark,
    (p.events || []).length
      ? note('warn', el('b', {}, 'events: '), p.events.map(e => `${e.kind}@${e.step}`).join(', '))
      : el('span'),
    el('div', { class: 'actions' },
      (j.state === 'running' || j.state === 'queued')
        ? el('button', { class: 'act danger',
            onclick: () => api('/api/jobs/' + id + '/cancel', {}).then(pollJobs) }, 'Cancel')
        : el('button', { class: 'act ghost',
            onclick: () => { api('/api/jobs/' + id + '/forget', {}).then(() => { SELECTED = null; pollJobs(); }); } }, 'Forget'),
      el('span', { class: 'faint tiny num' }, j.rundir)),
    el('h2', { style: 'margin-top:14px' }, 'Log'),
    el('pre', { class: 'log' }, j.log || '(nothing yet)')));
}

function lossSpark(series) {
  const ys = series.map(s => s.loss), n = ys.length;
  const lo = Math.min(...ys), hi = Math.max(...ys), span = (hi - lo) || 1;
  const d = ys.map((y, i) => (i ? 'L' : 'M') + (i / (n - 1) * 100) + ' ' + (28 - 26 * (y - lo) / span)).join('');
  const s = svg('svg', { class: 'spark', viewBox: '0 0 100 30', preserveAspectRatio: 'none' });
  s.append(svg('path', { d }));
  return el('div', {}, el('div', { class: 'muted tiny', style: 'margin-top:10px' },
    `loss ${num(hi, 6)} → ${num(ys[n - 1], 6)}`), s);
}

/* ===================================================================== */
/* STATUS                                                                */
/* ===================================================================== */
async function renderStatus() {
  const s = await api('/api/status');
  const g = s.genome || {};
  const left = [];
  left.push(el('div', { class: 'card' }, el('h2', {}, 'Shipped genome'),
    g.error ? note('bad', g.error) : el('dl', { class: 'kv' },
      el('dt', {}, 'file'), el('dd', {}, g.path),
      el('dt', {}, 'hash'), el('dd', {}, g.hash),
      el('dt', {}, 'loss'), el('dd', {}, num(g.loss, 7)),
      ...Object.entries(g.metrics || {}).slice(0, 10).flatMap(([k, v]) =>
        [el('dt', {}, k), el('dd', {}, num(v, 6))])),
    g.has_requirements_block === false
      ? note('', el('b', {}, 'no requirements block. '),
          'This record predates the requirements layer, so nothing on it records what it ' +
          'was scored under. verify() reports its provenance as unstated rather than ' +
          'implying a match.')
      : el('span')));

  const right = [];
  const e = s.export;
  right.push(el('div', { class: 'card' }, el('h2', {}, 'CAD export'),
    !e ? note('warn', 'No manifest in export/ — nothing has been exported yet.')
      : el('div', {},
        e.stale
          ? note('bad', el('b', {}, 'stale. '),
              `The manifest was built from genome ${e.genome_hash}, and the shipped genome ` +
              `is now ${g.hash}. Re-run the export before trusting the solid.`)
          : note('ok', el('b', {}, 'fresh. '), 'The manifest matches the shipped genome.'),
        el('dl', { class: 'kv' },
          el('dt', {}, 'exported'), el('dd', {}, e.exported_at || '—'),
          el('dt', {}, 'took'), el('dd', {}, dur(e.export_seconds)),
          el('dt', {}, 'mass'), el('dd', {}, num(e.mass_g, 5) + ' g'),
          el('dt', {}, 'valid'), el('dd', {}, String(e.valid))),
        (e.fillets || []).length ? el('div', {},
          el('h3', { class: 'muted tiny', style: 'margin:12px 0 4px;text-transform:uppercase' },
            'fillets — asked for vs built'),
          el('table', {}, el('thead', {}, el('tr', {},
              el('th', { class: 'n' }, 'requested'), el('th', { class: 'n' }, 'built'),
              el('th', { class: 'n' }, 'Kt error %'))),
            el('tbody', {}, ...e.fillets.map(f => el('tr', {},
              el('td', { class: 'n' }, num(f.requested, 4)),
              el('td', { class: 'n' }, num(f.built, 4)),
              el('td', { class: 'n' }, num(f.kt_error_pct, 3)))))),
          el('p', { class: 'faint tiny' },
            'OCC does not always deliver the radius it was asked for, and the manifest is ' +
            'the only place that difference is recorded.')) : el('span'))));

  $('#status-left').replaceChildren(...left);
  $('#status-right').replaceChildren(...right);
}

/* ===================================================================== */
async function boot() {
  try {
    const m = await api('/api/machine');
    $('#hd-mem').textContent = `${m.available_gib} / ${m.total_gib} GiB`;
  } catch {}
  try {
    const s = await api('/api/status');
    $('#hd-genome').textContent = (s.genome && s.genome.hash) || '—';
  } catch {}
  await initRun();
  await pollJobs();
  initDesign().catch(e => console.error(e));
  initMission().catch(e => console.error(e));
  setInterval(pollJobs, 2500);
  setInterval(async () => {
    try { const m = await api('/api/machine');
      $('#hd-mem').textContent = `${m.available_gib} / ${m.total_gib} GiB`; } catch {}
  }, 10000);
}
boot();
