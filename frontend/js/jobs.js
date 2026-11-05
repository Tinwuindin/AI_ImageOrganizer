// Estado compartido de las tareas en segundo plano.

const ACTIVE_STATES = new Set(['running', 'cancelling']);

let listener = () => {};

export const jobIsActive = (job) => ACTIVE_STATES.has(job?.state);

export function onJobStarted(callback) {
  listener = callback;
}

// Una vista lo llama al lanzar una tarea para reflejarla sin esperar al sondeo.
export function jobStarted(job) {
  listener(job);
}
