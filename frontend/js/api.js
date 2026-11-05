// Cliente de la API local.

export async function api(path, { method = 'GET', body, params } = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== null && value !== undefined && value !== '') url.searchParams.set(key, value);
  }
  const options = { method };
  if (body !== undefined) {
    options.headers = { 'Content-Type': 'application/json' };
    options.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error('No se pudo conectar con el organizador. ¿Sigue en ejecución?');
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(typeof detail === 'string' ? detail : `Error ${response.status}`);
  }
  return payload;
}

export function thumbnailUrl(image) {
  const version = image.sha256 ? `?v=${image.sha256.slice(0, 12)}` : '';
  return `/api/images/${image.id}/thumbnail${version}`;
}

export const thumbnailById = (id) => `/api/images/${id}/thumbnail`;
export const previewUrl = (id) => `/api/images/${id}/preview`;

export async function pickFolder() {
  const { path } = await api('/api/system/pick-folder', { method: 'POST' });
  return path;
}
