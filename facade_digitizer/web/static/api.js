// Запросы к серверу. Токен страницы — в каждом изменяющем запросе (защита от чужих
// сайтов в том же браузере); действия рабочего стола идут строго по одному и несут
// ревизию состояния, по которому их решил оператор.

const TOKEN = document.querySelector('meta[name="facade-token"]').content;

export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

export async function request(method, url, body) {
  const options = { method, headers: { 'X-Facade-Token': TOKEN } };
  if (body instanceof FormData) {
    options.body = body;
  } else if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new ApiError('Сервер не отвечает. Проверьте, что программа запущена.', 0, null);
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    throw new ApiError(data?.error || `Ошибка сервера (${response.status})`, response.status, data);
  }
  return data;
}

export const get = (url) => request('GET', url);
export const post = (url, body = {}) => request('POST', url, body);

export function upload(url, file) {
  const form = new FormData();
  form.append('file', file, file.name);
  return request('POST', url, form);
}

// Очередь действий: следующее уходит, когда пришёл ответ на предыдущее, и берёт
// ревизию из самого свежего состояния. Два быстрых клика не обгоняют друг друга.
export function createActor({ getState, applyState, onError }) {
  let chain = Promise.resolve();
  return function act(name, args = {}) {
    chain = chain.then(async () => {
      const state = getState();
      if (!state || !state.desk) return;
      try {
        const result = await post('/api/action', { desk: state.desk, rev: state.rev, name, args });
        applyState(result.state);
      } catch (error) {
        if (error.data?.state) applyState(error.data.state);
        onError(error, name);
      }
    });
    return chain;
  };
}
