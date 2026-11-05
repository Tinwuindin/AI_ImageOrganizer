// Análisis: elección de carpetas y avance de cada etapa.

import { api, pickFolder } from '../api.js';
import {
  formatDuration, formatNumber, h, icon, plural, progressBar, replace, showError,
  storedValue, storeValue, toast,
} from '../dom.js';
import { jobIsActive, jobStarted } from '../jobs.js';

const STORAGE_KEY = 'organizador.carpetas';

const STAGES = [
  { key: 'scan', title: 'Escaneo', hint: 'Archivos, hash SHA-256, fecha EXIF y miniaturas.' },
  { key: 'duplicates', title: 'Duplicados exactos', hint: 'Copias idénticas, aunque tengan otro nombre.' },
  { key: 'embeddings', title: 'Vectores visuales', hint: 'EmbeddingGemma 2 en la GPU.' },
  { key: 'clustering', title: 'Agrupación', hint: 'Reúne las imágenes parecidas.' },
  { key: 'classification', title: 'Clasificación', hint: 'Gemma 4 analiza representantes de cada grupo y los casos ambiguos.' },
];

export function mountAnalysis(root) {
  let roots = storedValue(STORAGE_KEY, []);
  let job = null;

  const folderList = h('div', { class: 'folder-list' });
  const pathInput = h('input', {
    class: 'input grow',
    placeholder: 'Ruta de una carpeta, por ejemplo D:\\Fotos',
    'aria-label': 'Ruta de una carpeta',
  });
  const startButton = h('button', { class: 'button primary', onClick: start }, 'Analizar');
  const cancelButton = h('button', { class: 'button danger', onClick: cancel, hidden: true }, 'Cancelar');
  const requirements = h('div');
  const stages = h('div', { class: 'stages' });
  const stagesNote = h('p', { class: 'muted small' });
  const results = h('div', { class: 'card', hidden: true });
  const problems = h('div', { class: 'card', hidden: true });

  root.append(h('div', { class: 'page' },
    h('header', { class: 'page-header' },
      h('h1', null, 'Análisis'),
      h('p', null, 'Todo ocurre en este equipo. Puedes cancelar cuando quieras: al volver a analizar, se retoma donde quedó.'),
    ),
    requirements,
    h('section', { class: 'card' },
      h('h2', null, 'Carpetas'),
      folderList,
      h('form', {
        class: 'row',
        onSubmit: (event) => {
          event.preventDefault();
          addFolder(pathInput.value.trim());
          pathInput.value = '';
        },
      },
        pathInput,
        h('button', { class: 'button', type: 'submit' }, 'Agregar'),
        h('button', { class: 'button', type: 'button', onClick: browse }, icon('folder'), 'Elegir carpeta…'),
      ),
      h('div', { class: 'row' }, startButton, cancelButton),
    ),
    h('section', { class: 'card' }, h('h2', null, 'Progreso'), stagesNote, stages),
    results,
    problems,
  ));

  function addFolder(path) {
    if (!path || roots.includes(path)) return;
    roots = [...roots, path];
    storeValue(STORAGE_KEY, roots);
    renderFolders();
  }

  async function browse() {
    try {
      const path = await pickFolder();
      if (path) addFolder(path);
    } catch (error) {
      showError(error);
    }
  }

  function renderFolders() {
    replace(folderList, roots.length
      ? roots.map((path) => h('div', { class: 'folder-item' },
        h('span', { class: 'path grow' }, path),
        h('button', {
          class: 'button ghost',
          'aria-label': `Quitar ${path}`,
          onClick: () => {
            roots = roots.filter((item) => item !== path);
            storeValue(STORAGE_KEY, roots);
            renderFolders();
          },
        }, '✕'),
      ))
      : h('p', { class: 'muted small' }, 'Aún no has elegido ninguna carpeta.'));
    updateButtons();
  }

  function updateButtons() {
    const active = jobIsActive(job);
    startButton.disabled = active || roots.length === 0;
    startButton.textContent = active && job.kind === 'analysis' ? 'Analizando…' : 'Analizar';
    cancelButton.hidden = !active || job.kind !== 'analysis';
    cancelButton.disabled = job?.state === 'cancelling';
  }

  async function start() {
    try {
      jobStarted(await api('/api/pipeline/start', { method: 'POST', body: { roots } }));
    } catch (error) {
      showError(error);
    }
  }

  async function cancel() {
    try {
      await api('/api/pipeline/cancel', { method: 'POST' });
      toast('Cancelando: se guarda lo avanzado.');
    } catch (error) {
      showError(error);
    }
  }

  function stageState(index, current) {
    if (!job || job.kind !== 'analysis' || job.state === 'idle') return 'pending';
    if (job.state === 'completed') return 'done';
    if (index < current) return 'done';
    if (index > current) return 'pending';
    return jobIsActive(job) ? 'active' : 'stopped';
  }

  function stageNumbers() {
    const parts = [];
    if (job.total) parts.push(`${formatNumber(job.done)} de ${formatNumber(job.total)}`);
    if (job.rate_per_second) {
      parts.push(job.rate_per_second >= 1
        ? `${job.rate_per_second.toFixed(1)} img/s`
        : `${(1 / job.rate_per_second).toFixed(1)} s/img`);
    }
    if (job.eta_seconds) parts.push(`quedan ${formatDuration(job.eta_seconds)}`);
    return parts.join(' · ');
  }

  function renderStages() {
    const current = STAGES.findIndex((stage) => stage.key === job?.stage);
    replace(stages, STAGES.map((stage, index) => {
      const status = stageState(index, current);
      return h('div', { class: `stage ${status}` },
        h('div', { class: 'dot' }, status === 'done' ? '✓' : status === 'stopped' ? '!' : index + 1),
        h('div', null,
          h('div', { class: 'stage-title' },
            h('span', null, stage.title),
            status === 'active' ? h('span', { class: 'numbers' }, stageNumbers()) : null,
          ),
          h('div', { class: 'hint' }, stage.hint),
          status === 'active' ? progressBar(job.done, job.total) : null,
        ),
      );
    }));
    stagesNote.textContent = '';
    if (job?.kind === 'analysis') {
      if (job.state === 'cancelling') stagesNote.textContent = 'Cancelando: termina el lote en curso.';
      if (job.state === 'cancelled') stagesNote.textContent = 'Análisis cancelado. Lo avanzado quedó guardado.';
      if (job.state === 'failed') stagesNote.textContent = `El análisis se detuvo: ${job.error}`;
      if (job.state === 'completed') stagesNote.textContent = `Terminado en ${formatDuration(job.elapsed_seconds)}.`;
    }
  }

  function renderResults() {
    const summary = job?.summary;
    const finished = job?.kind === 'analysis' && !jobIsActive(job) && job.state !== 'idle';
    results.hidden = !finished || !summary?.scan;
    if (results.hidden) return;
    const classification = summary.classification ?? {};
    const stat = (value, label) => h('div', { class: 'stat' }, h('b', null, formatNumber(value)), h('span', null, label));
    replace(results,
      h('h2', null, 'Resultado'),
      h('div', { class: 'stats' },
        stat(summary.scan.discovered, 'imágenes encontradas'),
        stat(summary.scan.added, 'nuevas'),
        stat(summary.duplicate_groups ?? 0, 'grupos de duplicados'),
        stat(summary.clusters ?? 0, 'grupos visuales'),
        stat(classification.analyzed ?? 0, 'analizadas por Gemma'),
        stat(classification.inherited ?? 0, 'heredadas de su grupo'),
      ),
      job.state === 'completed'
        ? h('div', { class: 'row' }, h('a', { class: 'button primary', href: '#/biblioteca' }, 'Ver biblioteca'), h('a', { class: 'button', href: '#/organizar' }, 'Organizar archivos'))
        : null,
    );
  }

  async function renderProblems() {
    try {
      const { items, total } = await api('/api/images', { params: { status: 'error', page_size: 50 } });
      problems.hidden = total === 0;
      replace(problems,
        h('h2', null, plural(total, 'imagen con problemas', 'imágenes con problemas')),
        h('p', { class: 'muted small' }, 'No se pudieron leer. Se reintentan en cada análisis y nunca se mueven.'),
        h('table', { class: 'table' },
          h('thead', null, h('tr', null, h('th', null, 'Archivo'), h('th', null, 'Motivo'))),
          h('tbody', null, items.map((item) => h('tr', null,
            h('td', { class: 'path' }, item.path),
            h('td', null, item.error),
          ))),
        ),
      );
    } catch (error) {
      showError(error);
    }
  }

  async function checkRequirements() {
    try {
      const { ready, missing } = await api('/api/system/info');
      replace(requirements, ready ? null : h('div', { class: 'notice' },
        `Faltan archivos: ${missing.join(', ')}. Descárgalos una vez con `,
        h('code', null, 'python scripts/download_models.py'),
        '.',
      ));
    } catch (error) {
      showError(error);
    }
  }

  renderFolders();
  renderStages();
  checkRequirements();
  renderProblems();

  let wasActive = false;
  return {
    onJob(update) {
      job = update;
      renderStages();
      renderResults();
      updateButtons();
      const active = jobIsActive(job);
      if (wasActive && !active) renderProblems();
      wasActive = active;
    },
  };
}
