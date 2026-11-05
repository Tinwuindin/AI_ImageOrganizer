// Navegación, estado global de las tareas y montaje de cada vista.

import { api } from './api.js';
import { formatNumber, h, icon, plural, progressBar, replace } from './dom.js';
import { jobIsActive, onJobStarted } from './jobs.js';
import { mountAnalysis } from './views/analysis.js';
import { mountCategories } from './views/categories.js';
import { mountDuplicates } from './views/duplicates.js';
import { mountLibrary } from './views/library.js';
import { mountOrganizer } from './views/organizer.js';

const ROUTES = {
  biblioteca: { label: 'Biblioteca', icon: 'library', mount: mountLibrary },
  analisis: { label: 'Análisis', icon: 'analysis', mount: mountAnalysis },
  organizar: { label: 'Organizar', icon: 'organize', mount: mountOrganizer },
  duplicados: { label: 'Duplicados', icon: 'duplicates', mount: mountDuplicates },
  categorias: { label: 'Categorías', icon: 'categories', mount: mountCategories },
};

const JOB_NAMES = { analysis: 'Analizando', organize: 'Moviendo archivos', undo: 'Deshaciendo' };
const POLL_ACTIVE_MS = 800;
const POLL_IDLE_MS = 5000;

const viewRoot = document.getElementById('view');
const nav = document.getElementById('nav');
const pill = document.getElementById('job-pill');
const statsLine = document.getElementById('sidebar-stats');

let currentView = null;
let lastJob = null;

function renderNav(active) {
  replace(nav, Object.entries(ROUTES).map(([key, route]) => h('a', {
    href: `#/${key}`,
    'aria-current': key === active ? 'page' : null,
  }, icon(route.icon), h('span', null, route.label))));
}

function navigate() {
  const key = window.location.hash.replace('#/', '').split('?')[0];
  const route = ROUTES[key] ?? ROUTES.biblioteca;
  const activeKey = ROUTES[key] ? key : 'biblioteca';
  currentView?.unmount?.();
  renderNav(activeKey);
  viewRoot.textContent = '';
  viewRoot.scrollTop = 0;
  currentView = route.mount(viewRoot);
  if (lastJob) currentView?.onJob?.(lastJob);
}

function renderPill(job) {
  if (!jobIsActive(job)) {
    pill.hidden = true;
    return;
  }
  pill.hidden = false;
  pill.href = job.kind === 'analysis' ? '#/analisis' : '#/organizar';
  replace(pill,
    h('strong', null, job.state === 'cancelling' ? 'Cancelando…' : JOB_NAMES[job.kind] ?? 'Trabajando'),
    job.total ? `${formatNumber(job.done)} de ${formatNumber(job.total)}` : 'Preparando…',
    progressBar(job.done, job.total),
  );
}

async function refreshStats() {
  try {
    const stats = await api('/api/images/stats');
    statsLine.textContent = stats.library ? plural(stats.library, 'imagen', 'imágenes') : '';
  } catch {
    statsLine.textContent = '';
  }
}

async function pollJob() {
  let delay = POLL_IDLE_MS;
  try {
    const job = await api('/api/pipeline/status');
    const finished = jobIsActive(lastJob) && !jobIsActive(job);
    lastJob = job;
    renderPill(job);
    currentView?.onJob?.(job);
    if (finished) refreshStats();
    if (jobIsActive(job)) delay = POLL_ACTIVE_MS;
  } catch {
    // Se reintenta en el siguiente ciclo.
  }
  setTimeout(pollJob, delay);
}

onJobStarted((job) => {
  lastJob = job;
  renderPill(job);
  currentView?.onJob?.(job);
});

window.addEventListener('hashchange', navigate);
navigate();
refreshStats();
pollJob();
