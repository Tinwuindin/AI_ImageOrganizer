// Categorías que el asistente puede usar al clasificar.

import { api } from '../api.js';
import { confirmDialog, formatNumber, h, plural, replace, showError, toast } from '../dom.js';

const FALLBACK = 'Otros';

export function mountCategories(root) {
  const list = h('section', { class: 'card' });
  const name = h('input', { class: 'input', placeholder: 'Nombre', 'aria-label': 'Nombre', maxLength: 60, required: true });
  const description = h('input', {
    class: 'input grow',
    placeholder: 'Qué incluye: el modelo se guía por esta descripción',
    'aria-label': 'Descripción',
    maxLength: 300,
  });

  root.append(h('div', { class: 'page' },
    h('header', { class: 'page-header' },
      h('h1', null, 'Categorías'),
      h('p', null, 'Gemma elige siempre una de esta lista y tiende a preferir las primeras, así que van de lo específico a lo genérico. Una descripción concreta mejora el resultado. Los cambios aplican a las imágenes que aún no se han clasificado.'),
    ),
    h('form', {
      class: 'card',
      onSubmit: async (event) => {
        event.preventDefault();
        try {
          await api('/api/categories', {
            method: 'POST',
            body: { name: name.value, description: description.value },
          });
          name.value = '';
          description.value = '';
          load();
        } catch (error) {
          showError(error);
        }
      },
    },
      h('h2', null, 'Nueva categoría'),
      h('div', { class: 'category-row' }, name, description, h('button', { class: 'button primary', type: 'submit' }, 'Agregar')),
    ),
    list,
  ));

  function row(category, categories) {
    const isFallback = category.name === FALLBACK;
    const nameInput = h('input', {
      class: 'input', value: category.name, 'aria-label': 'Nombre', disabled: isFallback, maxLength: 60,
    });
    const descriptionInput = h('input', {
      class: 'input', value: category.description, 'aria-label': `Descripción de ${category.name}`, maxLength: 300,
    });

    async function save() {
      const changed = nameInput.value.trim() !== category.name
        || descriptionInput.value.trim() !== category.description;
      if (!changed || !nameInput.value.trim()) return;
      try {
        await api(`/api/categories/${category.id}`, {
          method: 'PATCH',
          body: { name: nameInput.value, description: descriptionInput.value },
        });
        toast('Categoría actualizada.');
        load();
      } catch (error) {
        showError(error);
        load();
      }
    }

    async function remove() {
      const accepted = await confirmDialog({
        title: `¿Eliminar «${category.name}»?`,
        body: category.total
          ? `${plural(category.total, 'imagen pasará', 'imágenes pasarán')} a «${FALLBACK}».`
          : 'No tiene imágenes asignadas.',
        accept: 'Eliminar',
        danger: true,
      });
      if (!accepted) return;
      try {
        await api(`/api/categories/${category.id}`, { method: 'DELETE' });
        load();
      } catch (error) {
        showError(error);
      }
    }

    nameInput.addEventListener('change', save);
    descriptionInput.addEventListener('change', save);

    return h('div', { class: 'category-row' },
      nameInput,
      descriptionInput,
      h('div', { class: 'row' },
        h('span', { class: 'badge', title: 'Imágenes en esta categoría' }, formatNumber(category.total)),
        h('button', {
          class: 'button ghost danger',
          disabled: isFallback,
          title: isFallback ? 'Esta categoría es necesaria' : null,
          'aria-label': `Eliminar ${category.name}`,
          onClick: remove,
        }, 'Eliminar'),
      ),
    );
  }

  async function load() {
    try {
      const { categories } = await api('/api/categories');
      replace(list, h('h2', null, plural(categories.length, 'categoría')), categories.map((item) => row(item, categories)));
    } catch (error) {
      showError(error);
    }
  }

  load();
  return {};
}
