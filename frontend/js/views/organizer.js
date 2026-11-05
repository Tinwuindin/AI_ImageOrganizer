// Organizar: propuesta, revisión y ejecución, con historial para deshacer.

import { api, pickFolder, thumbnailById } from '../api.js';
import {
  confirmDialog, formatDate, formatNumber, h, icon, plural, progressBar, replace, showError,
  storedValue, storeValue, toast,
} from '../dom.js';
import { jobIsActive, jobStarted } from '../jobs.js';

const STORAGE_KEY = 'organizador.destino';

const TEMPLATES = [
  ['{category}/{subcategory}', 'Categoría / Subcategoría'],
  ['{category}', 'Categoría'],
  ['{category}/{year}', 'Categoría / Año'],
  ['{year}/{category}', 'Año / Categoría'],
  ['{year}/{month}/{category}', 'Año / Mes / Categoría'],
];

const PLAN_STATES = {
  executed: 'Ejecutado',
  undone: 'Deshecho',
  running: 'En curso',
};

const REASON_LABELS = {
  low_confidence: 'Baja confianza',
  duplicate: 'Duplicado',
};

export function mountOrganizer(root) {
  let plan = null;
  let job = null;

  const destination = h('input', {
    class: 'input grow',
    value: storedValue(STORAGE_KEY, ''),
    placeholder: 'Carpeta de destino, por ejemplo D:\\Fotos ordenadas',
    'aria-label': 'Carpeta de destino',
  });
  const template = h('select', { class: 'select', 'aria-label': 'Estructura de carpetas' },
    TEMPLATES.map(([value, label]) => h('option', { value }, label)));
  const review = h('div');
  const progress = h('div');
  const history = h('section', { class: 'card', hidden: true });

  root.append(h('div', { class: 'page' },
    h('header', { class: 'page-header' },
      h('h1', null, 'Organizar'),
      h('p', null, 'Primero revisas una propuesta; nada se mueve hasta que lo confirmes. Nunca se sobrescribe ni se borra ningún archivo, y puedes deshacer cada ejecución.'),
    ),
    h('section', { class: 'card' },
      h('h2', null, 'Nueva propuesta'),
      h('div', { class: 'row' },
        destination,
        h('button', {
          class: 'button',
          type: 'button',
          onClick: async () => {
            try {
              const path = await pickFolder();
              if (path) destination.value = path;
            } catch (error) {
              showError(error);
            }
          },
        }, icon('folder'), 'Elegir…'),
      ),
      h('div', { class: 'row' },
        h('label', { class: 'field' }, 'Estructura de carpetas', template),
      ),
      h('button', { class: 'button primary', onClick: generate }, 'Generar propuesta'),
    ),
    progress,
    review,
    history,
  ));

  async function generate() {
    const path = destination.value.trim();
    if (!path) {
      toast('Indica una carpeta de destino.', 'error');
      return;
    }
    storeValue(STORAGE_KEY, path);
    try {
      plan = await api('/api/organizer/plans', {
        method: 'POST',
        body: { destination: path, template: template.value },
      });
      if (plan.total === 0) toast('No hay nada que mover: todo está ya en su lugar.');
      renderPlan();
    } catch (error) {
      showError(error);
    }
  }

  function renderPlan() {
    if (!plan || plan.status !== 'draft') {
      replace(review);
      return;
    }
    const groups = new Map();
    for (const folder of plan.folders) {
      const name = folder.folder.split('/')[0];
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(folder);
    }
    const lowConfidence = plan.folders.reduce((sum, folder) => sum + (folder.low_confidence || 0), 0);

    replace(review,
      h('section', { class: 'card' },
        h('div', { class: 'row spread no-wrap' },
          h('div', { class: 'shrink' },
            h('h2', null, 'Revisa la propuesta'),
            h('p', { class: 'muted small' }, 'Destino: ', h('span', { class: 'path' }, plan.destination)),
          ),
          h('button', { class: 'button ghost', onClick: discard }, 'Descartar'),
        ),
        lowConfidence
          ? h('div', { class: 'notice' }, `${plural(lowConfidence, 'imagen', 'imágenes')} con baja confianza ${lowConfidence === 1 ? 'quedó desmarcada' : 'quedaron desmarcadas'}. Revísalas antes de incluirlas.`)
          : null,
        [...groups.entries()].map(([name, folders]) => h('div', { class: 'folder-group' },
          h('h3', null, name, h('span', { class: 'muted small' },
            plural(folders.reduce((sum, item) => sum + item.total, 0), 'imagen', 'imágenes'))),
          folders.map(folderRow),
        )),
      ),
      h('div', { class: 'sticky-action' },
        h('span', null, `Se moverán ${plural(plan.included, 'archivo')} a ${plural(plan.folders.filter((item) => item.included > 0).length, 'carpeta')}.`),
        h('button', {
          class: 'button primary',
          disabled: plan.included === 0 || jobIsActive(job),
          onClick: execute,
        }, 'Mover archivos'),
      ),
    );
  }

  function folderRow(folder) {
    const items = h('div', { class: 'plan-items', hidden: true });
    const allIncluded = folder.included === folder.total;
    const checkbox = h('input', {
      type: 'checkbox',
      checked: folder.included > 0,
      'aria-label': `Incluir ${folder.folder}`,
      onChange: () => updateFolder(folder.folder, { included: checkbox.checked }),
    });
    checkbox.indeterminate = folder.included > 0 && !allIncluded;
    const subfolder = folder.folder.split('/').slice(1).join('/') || folder.folder;

    return h('div', { class: `plan-folder ${folder.included === 0 ? 'plan-excluded' : ''}` },
      h('div', { class: 'plan-folder-head' },
        h('label', { class: 'check' }, checkbox),
        h('div', { class: 'plan-folder-name' },
          subfolder, ' ',
          h('span', { class: 'muted small' }, folder.included === folder.total
            ? formatNumber(folder.total)
            : `${formatNumber(folder.included)} de ${formatNumber(folder.total)}`),
          folder.low_confidence ? h('span', null, ' ', h('span', { class: 'badge warning' }, `${folder.low_confidence} por revisar`)) : null,
        ),
        h('div', { class: 'row' },
          h('div', { class: 'plan-samples' }, folder.sample_ids.slice(0, 4).map((id) => h('img', {
            src: thumbnailById(id), alt: '', loading: 'lazy',
          }))),
          h('button', {
            class: 'button ghost small',
            onClick: () => renameFolder(folder.folder),
          }, 'Renombrar'),
          h('button', {
            class: 'button ghost small',
            'aria-expanded': 'false',
            onClick: (event) => toggleItems(event.currentTarget, items, folder),
          }, 'Ver'),
        ),
      ),
      items,
    );
  }

  async function toggleItems(button, container, folder) {
    const open = container.hidden;
    container.hidden = !open;
    button.setAttribute('aria-expanded', String(open));
    button.textContent = open ? 'Ocultar' : 'Ver';
    if (!open || container.childElementCount) return;
    try {
      const { operations } = await api(`/api/organizer/plans/${plan.id}/operations`, {
        params: { folder: folder.folder, page_size: 200 },
      });
      replace(container,
        h('div', { class: 'grid' }, operations.map(operationTile)),
        folder.total > operations.length
          ? h('p', { class: 'muted small' }, `Se muestran las primeras ${operations.length}.`)
          : null,
      );
    } catch (error) {
      showError(error);
    }
  }

  function operationTile(operation) {
    let included = operation.included;
    const name = operation.source_path.split(/[\\/]/).pop();
    const tick = h('span', { class: 'tick' }, included ? '✓' : '');
    const tile = h('button', {
      class: `tile ${included ? '' : 'excluded'}`,
      title: operation.source_path,
      'aria-label': name,
      'aria-pressed': String(included),
      onClick: async () => {
        try {
          included = !included;
          await api(`/api/organizer/operations/${operation.id}`, { method: 'PATCH', body: { included } });
          tile.classList.toggle('excluded', !included);
          tile.setAttribute('aria-pressed', String(included));
          tick.textContent = included ? '✓' : '';
          plan = await api(`/api/organizer/plans/${plan.id}`);
          refreshSummary();
        } catch (error) {
          included = !included;
          showError(error);
        }
      },
    },
      h('img', { src: thumbnailById(operation.image_id), alt: '', loading: 'lazy' }),
      REASON_LABELS[operation.reason] ? h('span', { class: 'flag' }, REASON_LABELS[operation.reason]) : null,
      tick,
      h('span', { class: 'caption' }, name),
    );
    return tile;
  }

  // Actualiza el total sin cerrar las carpetas que la persona tiene abiertas.
  function refreshSummary() {
    const action = review.querySelector('.sticky-action');
    if (!action) return;
    action.firstChild.textContent = `Se moverán ${plural(plan.included, 'archivo')} a ${plural(plan.folders.filter((item) => item.included > 0).length, 'carpeta')}.`;
    action.querySelector('button').disabled = plan.included === 0 || jobIsActive(job);
  }

  async function updateFolder(folder, changes) {
    try {
      plan = await api(`/api/organizer/plans/${plan.id}/folders`, {
        method: 'PATCH',
        body: { folder, ...changes },
      });
      renderPlan();
    } catch (error) {
      showError(error);
      renderPlan();
    }
  }

  function renameFolder(folder) {
    const input = h('input', { class: 'input', value: folder, 'aria-label': 'Nombre de la carpeta' });
    const dialog = h('dialog', { class: 'confirm' },
      h('form', {
        method: 'dialog',
        onSubmit: () => {
          const value = input.value.trim();
          if (value && value !== folder) updateFolder(folder, { new_folder: value });
        },
      },
        h('h2', null, 'Renombrar carpeta'),
        h('p', { class: 'muted small' }, 'Usa / para crear subcarpetas.'),
        input,
        h('div', { class: 'row' },
          h('button', { class: 'button', type: 'button', onClick: () => dialog.close() }, 'Cancelar'),
          h('button', { class: 'button primary', type: 'submit' }, 'Guardar'),
        ),
      ),
    );
    dialog.addEventListener('close', () => dialog.remove());
    document.body.append(dialog);
    dialog.showModal();
    input.select();
  }

  async function discard() {
    try {
      await api(`/api/organizer/plans/${plan.id}`, { method: 'DELETE' });
      plan = null;
      renderPlan();
    } catch (error) {
      showError(error);
    }
  }

  async function execute() {
    const accepted = await confirmDialog({
      title: `¿Mover ${plural(plan.included, 'archivo')}?`,
      body: `Se moverán a ${plan.destination}. Podrás deshacerlo desde el historial.`,
      accept: 'Mover archivos',
    });
    if (!accepted) return;
    try {
      jobStarted(await api(`/api/organizer/plans/${plan.id}/execute`, { method: 'POST' }));
      plan = null;
      renderPlan();
    } catch (error) {
      showError(error);
    }
  }

  async function undo(item) {
    const accepted = await confirmDialog({
      title: '¿Deshacer esta organización?',
      body: `${plural(item.done, 'archivo')} ${item.done === 1 ? 'volverá' : 'volverán'} a su carpeta original.`,
      accept: 'Deshacer',
      danger: true,
    });
    if (!accepted) return;
    try {
      jobStarted(await api(`/api/organizer/plans/${item.id}/undo`, { method: 'POST' }));
    } catch (error) {
      showError(error);
    }
  }

  async function loadHistory() {
    try {
      const { plans } = await api('/api/organizer/plans');
      const draft = plans.find((item) => item.status === 'draft');
      if (draft && !plan) {
        plan = await api(`/api/organizer/plans/${draft.id}`);
        renderPlan();
      }
      const finished = plans.filter((item) => item.status !== 'draft');
      history.hidden = finished.length === 0;
      replace(history,
        h('h2', null, 'Historial'),
        h('table', { class: 'table' },
          h('thead', null, h('tr', null,
            h('th', null, 'Fecha'), h('th', null, 'Destino'),
            h('th', null, 'Movidos'), h('th', null, 'Estado'), h('th', null, ''),
          )),
          h('tbody', null, finished.map((item) => h('tr', null,
            h('td', null, formatDate(item.executed_at ?? item.created_at)),
            h('td', { class: 'path' }, item.destination),
            h('td', null, formatNumber(item.done ?? 0), item.failed ? h('span', null, ' ', h('span', { class: 'badge danger' }, `${item.failed} con error`)) : null),
            h('td', null, PLAN_STATES[item.status] ?? item.status),
            h('td', null, item.status === 'executed' && item.done > 0
              ? h('button', { class: 'button', disabled: jobIsActive(job), onClick: () => undo(item) }, 'Deshacer')
              : null),
          ))),
        ),
      );
    } catch (error) {
      showError(error);
    }
  }

  function renderProgress() {
    if (!jobIsActive(job) || job.kind === 'analysis') {
      replace(progress);
      return;
    }
    replace(progress, h('section', { class: 'card' },
      h('h2', null, job.kind === 'undo' ? 'Deshaciendo…' : 'Moviendo archivos…'),
      h('p', { class: 'muted small' }, `${formatNumber(job.done)} de ${formatNumber(job.total)}`),
      progressBar(job.done, job.total),
    ));
  }

  function announce() {
    const summary = job.summary ?? {};
    if (job.kind === 'organize' && summary.total !== undefined) {
      toast(summary.failed
        ? `Se movieron ${formatNumber(summary.moved)} archivos; ${formatNumber(summary.failed)} no se pudieron mover. Revisa el historial.`
        : `Se ${summary.moved === 1 ? 'movió' : 'movieron'} ${plural(summary.moved, 'archivo')}.`, summary.failed ? 'error' : '');
    }
    if (job.kind === 'undo' && summary.total !== undefined) {
      toast(summary.failed
        ? `${formatNumber(summary.failed)} archivos no se pudieron restaurar: su lugar original está ocupado.`
        : `Se ${summary.restored === 1 ? 'restauró' : 'restauraron'} ${plural(summary.restored, 'archivo')}.`, summary.failed ? 'error' : '');
    }
    if (job.state === 'failed') toast(`La tarea falló: ${job.error}`, 'error');
  }

  loadHistory();

  let wasActive = false;
  return {
    onJob(update) {
      job = update;
      renderProgress();
      const active = jobIsActive(job);
      if (wasActive && !active) {
        announce();
        loadHistory();
      }
      wasActive = active;
    },
  };
}
