// Biblioteca: navegación por categorías, búsqueda en lenguaje natural y detalle.

import { api, previewUrl, thumbnailUrl } from '../api.js';
import {
  emptyState, formatBytes, formatDate, formatNumber, h, icon, plural, replace, showError, toast,
} from '../dom.js';

const PAGE_SIZE = 60;

const SOURCE_LABELS = {
  direct: 'Analizada por Gemma',
  cluster: 'Heredada de su grupo',
  manual: 'Asignada a mano',
};

export function mountLibrary(root) {
  const state = {
    category: null,
    subcategory: null,
    sort: 'date_desc',
    lowConfidence: false,
    query: '',
    page: 1,
    items: [],
    total: 0,
    exhausted: false,
    loading: false,
    token: 0,
    categories: [],
    minConfidence: 0,
  };

  const filters = h('aside', { class: 'filters', 'aria-label': 'Categorías' });
  const summary = h('span', { class: 'muted small grow' });
  const grid = h('div', { class: 'grid' });
  const sentinel = h('div', { class: 'sentinel' });
  const emptyHolder = h('div');
  const searchInput = h('input', {
    class: 'input',
    type: 'search',
    placeholder: 'Describe lo que buscas: «mi perro en la playa», «recibos de gasolina»…',
    'aria-label': 'Buscar en lenguaje natural',
  });
  const sortSelect = h('select', {
    class: 'select',
    'aria-label': 'Ordenar',
    onChange: () => { state.sort = sortSelect.value; reload(); },
  },
    h('option', { value: 'date_desc' }, 'Más recientes'),
    h('option', { value: 'date_asc' }, 'Más antiguas'),
    h('option', { value: 'name' }, 'Por nombre'),
    h('option', { value: 'size' }, 'Más pesadas'),
    h('option', { value: 'confidence' }, 'Menor confianza'),
  );
  const lowConfidence = h('input', {
    type: 'checkbox',
    onChange: () => { state.lowConfidence = lowConfidence.checked; reload(); },
  });

  root.append(h('section', { class: 'library' },
    filters,
    h('div', { class: 'library-main' },
      h('form', {
        class: 'searchbar',
        role: 'search',
        onSubmit: (event) => {
          event.preventDefault();
          state.query = searchInput.value.trim();
          reload();
        },
      },
        h('div', { class: 'search-field' }, icon('search'), searchInput),
        h('button', { class: 'button primary', type: 'submit' }, 'Buscar'),
      ),
      h('div', { class: 'row' },
        summary,
        h('label', { class: 'check small' }, lowConfidence, 'Por revisar'),
        sortSelect,
      ),
      emptyHolder,
      grid,
      sentinel,
    ),
  ));

  searchInput.addEventListener('search', () => {
    if (!searchInput.value && state.query) {
      state.query = '';
      reload();
    }
  });

  function renderFilters() {
    const total = state.categories.reduce((sum, node) => sum + node.total, 0);
    const button = (label, count, pressed, onClick, extra = '') => h('button', {
      class: `filter ${extra}`, 'aria-pressed': String(pressed), onClick,
    }, h('span', { class: 'label', title: label }, label), h('span', { class: 'count' }, formatNumber(count)));

    const select = (category, subcategory = null) => () => {
      state.category = category;
      state.subcategory = subcategory;
      renderFilters();
      reload();
    };

    replace(filters,
      h('h2', null, 'Categorías'),
      button('Todas', total, state.category === null, select(null)),
      state.categories.filter((node) => node.total > 0).map((node) => [
        button(node.name, node.total,
          state.category === node.name && state.subcategory === null, select(node.name)),
        state.category === node.name
          ? node.subcategories.slice(0, 40).map((item) => button(
            item.name, item.total, state.subcategory === item.name,
            select(node.name, item.name), 'sub',
          ))
          : null,
      ]),
    );
  }

  function tile(item, index) {
    const review = item.confidence !== null && item.confidence < state.minConfidence;
    return h('button', {
      class: 'tile',
      title: item.filename,
      'aria-label': item.description || item.filename,
      onClick: () => openDetail(state.items, index, {
        categories: state.categories,
        onChange: refreshCategories,
      }),
    },
      h('img', { src: thumbnailUrl(item), alt: '', loading: 'lazy', decoding: 'async' }),
      review ? h('span', { class: 'flag' }, 'Revisar') : null,
      h('span', { class: 'caption' }, item.subcategory || item.category || item.filename),
    );
  }

  function showItems(items) {
    const offset = state.items.length;
    state.items.push(...items);
    grid.append(...items.map((item, index) => tile(item, offset + index)));
  }

  async function loadPage() {
    if (state.loading || state.exhausted) return;
    state.loading = true;
    const token = state.token;
    sentinel.textContent = 'Cargando…';
    try {
      if (state.query) {
        const result = await api('/api/search', {
          params: { q: state.query, category: state.category, limit: 120 },
        });
        if (token !== state.token) return;
        state.total = result.results.length;
        state.exhausted = true;
        showItems(result.results);
        if (!result.semantic && result.results.length) {
          toast('La búsqueda por significado no está disponible: se buscó solo por texto.');
        }
      } else {
        const result = await api('/api/images', {
          params: {
            category: state.category,
            subcategory: state.subcategory,
            sort: state.sort,
            low_confidence: state.lowConfidence ? 'true' : null,
            page: state.page,
            page_size: PAGE_SIZE,
          },
        });
        if (token !== state.token) return;
        state.total = result.total;
        state.page += 1;
        state.exhausted = state.items.length + result.items.length >= result.total;
        showItems(result.items);
      }
      renderSummary();
    } catch (error) {
      showError(error);
      state.exhausted = true;
    } finally {
      if (token === state.token) {
        state.loading = false;
        sentinel.textContent = '';
      }
    }
  }

  function renderSummary() {
    summary.textContent = state.query
      ? `${plural(state.total, 'resultado')} más ${state.total === 1 ? 'parecido' : 'parecidos'} a «${state.query}»`
      : plural(state.total, 'imagen', 'imágenes');
    if (state.total > 0) {
      replace(emptyHolder);
    } else if (state.query) {
      replace(emptyHolder, emptyState('Sin resultados', 'Prueba con otras palabras o quita el filtro de categoría.'));
    } else if (state.category || state.lowConfidence) {
      replace(emptyHolder, emptyState('Nada por aquí', 'No hay imágenes con este filtro.'));
    } else {
      replace(emptyHolder, emptyState(
        'Tu biblioteca está vacía',
        'Elige una carpeta con tus fotos y el asistente las analizará en este equipo. Nada sale de tu computadora.',
        h('a', { class: 'button primary', href: '#/analisis' }, 'Analizar una carpeta'),
      ));
    }
  }

  function reload() {
    // Los resultados de una búsqueda ya vienen ordenados por parecido.
    sortSelect.disabled = Boolean(state.query);
    lowConfidence.disabled = Boolean(state.query);
    state.token += 1;
    state.page = 1;
    state.items = [];
    state.total = 0;
    state.exhausted = false;
    state.loading = false;
    grid.textContent = '';
    loadPage();
  }

  async function refreshCategories() {
    try {
      const { categories } = await api('/api/categories');
      state.categories = categories;
      renderFilters();
    } catch (error) {
      showError(error);
    }
  }

  const observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) loadPage();
  }, { root, rootMargin: '600px' });
  observer.observe(sentinel);

  refreshCategories();
  api('/api/system/info')
    .then((info) => { state.minConfidence = info.min_confidence; })
    .catch(showError)
    .finally(reload);

  let wasActive = false;
  return {
    unmount: () => observer.disconnect(),
    onJob(job) {
      const active = job.state === 'running';
      if (wasActive && !active) {
        refreshCategories();
        reload();
      }
      wasActive = active;
    },
  };
}

function metadataRows(item) {
  const rows = [
    ['Fecha', formatDate(item.date_taken)],
    ['Tamaño', item.width ? `${item.width} × ${item.height} · ${formatBytes(item.size)}` : formatBytes(item.size)],
    ['Origen', SOURCE_LABELS[item.source]],
    ['Confianza', item.confidence !== null && item.confidence !== undefined
      ? `${Math.round(item.confidence * 100)} %` : null],
    ['Grupo', item.cluster_id !== null && item.cluster_id !== undefined ? `#${item.cluster_id}` : null],
    ['Ruta', item.path],
  ];
  return rows.filter(([, value]) => value).flatMap(([label, value]) => [
    h('dt', null, label),
    h('dd', { class: label === 'Ruta' ? 'path' : null }, value),
  ]);
}

// Detalle de una imagen, con corrección de categoría e imágenes parecidas.
export function openDetail(items, startIndex, { categories, onChange }) {
  let index = startIndex;

  const image = h('img', { alt: '' });
  const previous = h('button', {
    class: 'detail-nav prev', 'aria-label': 'Anterior', onClick: () => move(-1),
  }, '‹');
  const next = h('button', {
    class: 'detail-nav next', 'aria-label': 'Siguiente', onClick: () => move(1),
  }, '›');
  const info = h('div', { class: 'detail-info' });
  const dialog = h('dialog', { class: 'detail' },
    h('div', { class: 'detail-layout' },
      h('div', { class: 'detail-media' }, image, previous, next),
      info,
    ),
  );

  function move(step) {
    const target = index + step;
    if (target >= 0 && target < items.length) {
      index = target;
      render();
    }
  }

  function render() {
    const item = items[index];
    image.src = previewUrl(item.id);
    image.alt = item.description || item.filename;
    previous.disabled = index === 0;
    next.disabled = index === items.length - 1;

    const categorySelect = h('select', { class: 'select', 'aria-label': 'Categoría' },
      categories.map((category) => h('option', {
        value: category.name, selected: category.name === item.category,
      }, category.name)),
    );
    const subcategoryInput = h('input', {
      class: 'input', value: item.subcategory ?? '', 'aria-label': 'Subcategoría', maxLength: 80,
    });
    const similar = h('div', { class: 'mini-grid' });

    replace(info,
      h('div', { class: 'close-row' },
        h('h2', null, item.filename),
        h('button', { class: 'button ghost', 'aria-label': 'Cerrar', onClick: () => dialog.close() }, '✕'),
      ),
      item.description ? h('p', null, item.description) : null,
      item.source === 'cluster'
        ? h('p', { class: 'muted small' }, 'Clasificación heredada de una imagen muy parecida de su grupo; esta imagen no pasó por Gemma.')
        : null,
      item.tags?.length ? h('div', { class: 'tags' }, item.tags.map((tag) => h('span', { class: 'badge' }, tag))) : null,
      item.ocr_text ? h('div', null, h('h3', null, 'Texto en la imagen'), h('p', { class: 'small' }, item.ocr_text)) : null,
      h('form', {
        class: 'row',
        onSubmit: async (event) => {
          event.preventDefault();
          try {
            const updated = await api(`/api/images/${item.id}/classification`, {
              method: 'PATCH',
              body: { category: categorySelect.value, subcategory: subcategoryInput.value },
            });
            items[index] = { ...item, ...updated };
            render();
            toast('Clasificación actualizada.');
            onChange?.();
          } catch (error) {
            showError(error);
          }
        },
      },
        h('label', { class: 'field grow' }, 'Categoría', categorySelect),
        h('label', { class: 'field grow' }, 'Subcategoría', subcategoryInput),
        h('button', { class: 'button', type: 'submit' }, 'Guardar'),
      ),
      h('dl', null, metadataRows(item)),
      h('div', null, h('h3', null, 'Imágenes parecidas'), similar),
    );

    api(`/api/images/${item.id}/similar`, { params: { limit: 8 } }).then(({ results }) => {
      replace(similar, results.map((match) => h('button', {
        class: 'tile',
        title: match.filename,
        'aria-label': match.filename,
        onClick: () => {
          items.splice(index + 1, 0, match);
          move(1);
        },
      }, h('img', { src: thumbnailUrl(match), alt: '', loading: 'lazy' }))));
    }).catch(() => replace(similar));
  }

  dialog.addEventListener('keydown', (event) => {
    if (event.target.matches('input, select, textarea')) return;
    if (event.key === 'ArrowLeft') move(-1);
    if (event.key === 'ArrowRight') move(1);
  });
  dialog.addEventListener('close', () => dialog.remove());
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close();
  });

  document.body.append(dialog);
  render();
  dialog.showModal();
}
