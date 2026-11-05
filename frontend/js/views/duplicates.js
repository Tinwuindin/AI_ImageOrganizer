// Duplicados exactos: qué copia se conserva en cada grupo.

import { api, thumbnailById } from '../api.js';
import { emptyState, formatBytes, formatNumber, h, plural, replace, showError, toast } from '../dom.js';

const PAGE_SIZE = 30;

export function mountDuplicates(root) {
  const summary = h('p');
  const list = h('div', { class: 'stack' });
  const more = h('div');
  let page = 1;
  let loaded = 0;

  root.append(h('div', { class: 'page' },
    h('header', { class: 'page-header' },
      h('h1', null, 'Duplicados'),
      h('p', null, 'Archivos idénticos byte por byte. Al organizar, las copias sobrantes van a una carpeta _Duplicados: nada se elimina, tú decides después.'),
      summary,
    ),
    list,
    more,
  ));

  async function keep(group, member) {
    try {
      await api(`/api/images/duplicates/${group.id}/keep`, {
        method: 'POST',
        body: { image_id: member.id },
      });
      group.canonical_image_id = member.id;
      toast('Se conservará esta copia.');
    } catch (error) {
      showError(error);
    }
  }

  function groupCard(group) {
    return h('section', { class: 'card dup-group' },
      h('img', { src: thumbnailById(group.canonical_image_id), alt: '', loading: 'lazy' }),
      h('div', null,
        h('h3', null, `${group.copies} copias · ${formatBytes(group.size)} cada una`),
        h('p', { class: 'muted small' }, 'Elige la copia que se queda en su categoría:'),
        h('fieldset', { style: 'border:0;margin:6px 0 0;padding:0' },
          h('legend', { hidden: true }, 'Copia que se conserva'),
          group.members.map((member) => h('label', { class: 'dup-member' },
            h('input', {
              type: 'radio',
              name: `grupo-${group.id}`,
              checked: member.id === group.canonical_image_id,
              onChange: () => keep(group, member),
            }),
            h('span', { class: 'path' }, member.path),
          )),
        ),
      ),
    );
  }

  async function load() {
    try {
      const data = await api('/api/images/duplicates', { params: { page, page_size: PAGE_SIZE } });
      if (page === 1) {
        summary.textContent = data.total
          ? `${plural(data.total, 'grupo')} · ${formatBytes(data.recoverable_bytes)} en copias sobrantes`
          : '';
        replace(list, data.total ? null : emptyState('Sin duplicados', 'No se encontraron archivos idénticos en lo analizado.'));
      }
      list.append(...data.groups.map(groupCard));
      loaded += data.groups.length;
      page += 1;
      replace(more, loaded < data.total
        ? h('button', { class: 'button', onClick: load }, `Ver más (quedan ${formatNumber(data.total - loaded)})`)
        : null);
    } catch (error) {
      showError(error);
    }
  }

  load();
  return {};
}
