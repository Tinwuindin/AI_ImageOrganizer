// Utilidades de interfaz. Todo el contenido se inserta con nodos de texto: los
// nombres de archivo y el texto leído de las imágenes nunca se interpretan como HTML.

const ICONS = {
  library: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  analysis: '<path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1"/>',
  organize: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M12 10.5v5M9.5 13l2.5 2.5 2.5-2.5"/>',
  duplicates: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
  categories: '<path d="M4 6h16M4 12h10M4 18h6"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
};

export function icon(name) {
  const template = document.createElement('template');
  // Solo constantes de este archivo: nunca datos.
  template.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
  return template.content.firstChild;
}

export function h(tag, props, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') element.className = value;
    else if (key.startsWith('on') && typeof value === 'function') {
      element.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key in element && key !== 'list') element[key] = value;
    else element.setAttribute(key, value === true ? '' : value);
  }
  append(element, children);
  return element;
}

function append(parent, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    parent.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function replace(element, ...children) {
  element.textContent = '';
  append(element, children);
}

const numbers = new Intl.NumberFormat('es-MX');
export const formatNumber = (value) => numbers.format(value ?? 0);

export function plural(count, singular, pluralForm = `${singular}s`) {
  return `${formatNumber(count)} ${count === 1 ? singular : pluralForm}`;
}

export function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return '';
  if (seconds < 60) return `${Math.max(Math.round(seconds), 1)} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  return `${hours} h ${minutes % 60} min`;
}

export function formatDate(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat('es-MX', { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

export function toast(message, kind = '') {
  const element = h('div', { class: `toast ${kind}` }, message);
  document.getElementById('toasts').append(element);
  setTimeout(() => element.remove(), kind === 'error' ? 7000 : 3500);
}

export function showError(error) {
  toast(error?.message ?? 'Ocurrió un error inesperado.', 'error');
}

// Ventana de confirmación. Resuelve true si la persona acepta.
export function confirmDialog({ title, body, accept, danger = false }) {
  return new Promise((resolve) => {
    const dialog = h('dialog', { class: 'confirm' },
      h('h2', null, title),
      h('p', { class: 'muted' }, body),
      h('div', { class: 'row' },
        h('button', { class: 'button', onClick: () => dialog.close('cancel') }, 'Cancelar'),
        h('button', {
          class: `button ${danger ? 'danger' : 'primary'}`,
          onClick: () => dialog.close('accept'),
        }, accept),
      ),
    );
    dialog.addEventListener('close', () => {
      dialog.remove();
      resolve(dialog.returnValue === 'accept');
    });
    document.body.append(dialog);
    dialog.showModal();
  });
}

export function progressBar(done, total) {
  const percent = total > 0 ? Math.min(100, (done / total) * 100) : null;
  const fill = h('span');
  if (percent !== null) fill.style.width = `${percent}%`;
  return h('div', {
    class: `bar ${percent === null ? 'indeterminate' : ''}`,
    role: 'progressbar',
    'aria-valuenow': percent === null ? null : Math.round(percent),
    'aria-valuemin': 0,
    'aria-valuemax': 100,
  }, fill);
}

export function emptyState(title, body, action) {
  return h('div', { class: 'empty' }, h('h2', null, title), h('p', null, body), action);
}

export function storedValue(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value === null ? fallback : JSON.parse(value);
  } catch {
    return fallback;
  }
}

export function storeValue(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // El almacenamiento local es solo una comodidad.
  }
}
