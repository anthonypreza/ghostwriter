const $ = id => document.getElementById(id);
let runs = [], selected = null, samples = [], currentSample = null, busy = false;
let previousTokenizer = null;
let availableDevices = [];

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map(x => `${x.loc.at(-1)}: ${x.msg}`).join('; ') : data.detail;
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return data;
}
function status(message, error = false) {
  $('request-status').textContent = message;
  $('request-status').className = `request-status${error ? ' error' : busy ? ' busy' : ''}`;
}
function options(select, items, label = x => x, value = x => x) {
  select.replaceChildren(...items.map(item => new Option(label(item), value(item))));
}
function selectedTokenizer() { return selected?.tokenizers.find(t => t.name === $('tokenizer').value); }
function applyTokenizer() {
  const tokenizer = selectedTokenizer();
  if (!tokenizer) return;
  // A model change should never erase a prompt the user has written.
  if (!previousTokenizer || $('prompt').value === previousTokenizer.prompt) $('prompt').value = tokenizer.prompt;
  if (!previousTokenizer || Number($('max-tokens').value) === previousTokenizer.max_new_tokens) $('max-tokens').value = tokenizer.max_new_tokens;
  previousTokenizer = tokenizer;
  const context = selected.seq_len;
  $('run-facts').textContent = `${selected.model.toUpperCase()} · ${tokenizer.token_level.toUpperCase()} · ${context ? `${context}-token context` : 'Context from checkpoint'}`;
  $('active-model').textContent = `${selected.model.toUpperCase()} · ${tokenizer.token_level.toUpperCase()}`;
  $('choose-model').setAttribute('aria-label', `Choose model. Current: ${selected.model.toUpperCase()}, ${tokenizer.token_level.toUpperCase()}.`);
  updateSummary();
}
function chooseRun(run) {
  selected = run;
  options($('checkpoint'), run.checkpoints);
  options($('tokenizer'), run.tokenizers, t => `${t.name} · ${t.token_level}`, t => t.name);
  if (run.tokenizers.some(t => t.name === 'tokenizer.json')) $('tokenizer').value = 'tokenizer.json';
  const context = Number.isInteger(run.seq_len) && run.seq_len > 0 ? run.seq_len : null;
  $('context').value = context ?? '';
  $('context').readOnly = context !== null;
  $('context-help').textContent = context ? 'Matches the model’s training window automatically.' : 'Leave empty to use the checkpoint. For legacy weights, enter the original training window.';
  $('model-settings').textContent = JSON.stringify({run: run.id, model: run.model_settings, training: run.training_settings, evaluation: run.evaluation, best_epoch: run.best_epoch}, null, 2);
  $('generation-fields').disabled = busy;
  applyTokenizer();
  renderRuns();
  if (!busy) status('');
}
function renderRuns() {
  const query = $('search').value.toLowerCase();
  const filtered = runs.filter(run => [run.id, run.model, ...run.tokenizers.map(t => t.token_level)].join(' ').toLowerCase().includes(query));
  $('run-count').textContent = `${runs.length} saved ${runs.length === 1 ? 'run' : 'runs'}`;
  $('library-status').hidden = filtered.length > 0;
  $('library-status').textContent = runs.length ? 'No runs match your search.' : 'No saved models yet. Train with the pipeline, then refresh this list.';
  $('runs').replaceChildren(...filtered.map(run => {
    const button = document.createElement('button');
    button.className = 'run-card';
    button.setAttribute('aria-pressed', String(selected?.id === run.id));
    button.disabled = busy;
    const heading = document.createElement('div'); heading.className = 'card-heading';
    const name = document.createElement('span'); name.className = 'model-name'; name.textContent = run.model.toUpperCase();
    const badge = document.createElement('span'); badge.className = 'token-badge'; badge.textContent = run.tokenizers[0].token_level;
    heading.append(name, badge);
    const id = document.createElement('span'); id.className = 'run-id'; id.textContent = run.id; button.title = run.id;
    const metric = document.createElement('span'); metric.className = 'card-metric';
    const ppl = run.evaluation?.val?.perplexity;
    const facts = [run.seq_len ? `${run.seq_len}-token context` : null, run.model_settings?.embedding_size ? `${run.model_settings.embedding_size} embedding` : null, Number.isFinite(ppl) ? `Val PPL ${ppl.toFixed(2)}` : null];
    metric.textContent = facts.filter(Boolean).join(' · ');
    button.append(heading, id, metric);
    button.addEventListener('click', () => { chooseRun(run); $('model-picker').close(); $('prompt').focus(); });
    return button;
  }));
}
async function refresh() {
  $('refresh').disabled = true;
  try {
    const data = await api('/api/runs');
    runs = data.runs;
    const previousDevice = availableDevices.length ? $('device').value : null;
    availableDevices = data.devices;
    options($('device'), data.devices, device => device === 'cuda' ? 'GPU (CUDA / ROCm)' : 'CPU');
    $('device').value = data.devices.includes(previousDevice) ? previousDevice : (data.devices.includes('cuda') ? 'cuda' : 'cpu');
    const run = runs.find(r => r.id === selected?.id) || runs[0];
    if (run) chooseRun(run);
    else { selected = null; $('generation-fields').disabled = true; renderRuns(); }
  } catch (error) { status(`Could not load runs: ${error.message}`, true); }
  finally { $('refresh').disabled = false; }
}
function showSample(sample) {
  currentSample = sample;
  const settings = sample.settings;
  $('empty').hidden = true; $('result').hidden = false;
  $('sample-number').textContent = `Sample ${sample.number}`;
  $('result-info').textContent = `${sample.modelName || settings.token_level.toUpperCase()} · ${settings.token_level.toUpperCase()} · ${sample.elapsed_seconds.toFixed(2)}s`;
  $('lyrics').replaceChildren();
  if (!sample.text) $('lyrics').textContent = '(No visible text generated.)';
  else if (settings.plain || settings.color === 'never') $('lyrics').textContent = sample.text;
  else {
    for (const part of sample.text.split(/(<[A-Z_]+>)/)) {
      if (/^<[A-Z_]+>$/.test(part)) { const span = document.createElement('span'); span.className = 'section-marker'; span.textContent = part; $('lyrics').append(span); }
      else $('lyrics').append(document.createTextNode(part));
    }
  }
  $('result-settings').textContent = `Run ${settings.run} · Temperature ${settings.temperature} · Top-k ${settings.top_k} · Limit ${settings.max_new_tokens} · Context ${settings.context_length} · ${settings.device.toUpperCase()} · Seed ${settings.seed}`;
  $('command').textContent = sample.command;
  $('history').hidden = samples.length < 2;
  $('history').replaceChildren(...samples.map(item => {
    const button = document.createElement('button'); button.textContent = `Sample ${item.number}`;
    button.setAttribute('aria-pressed', String(item === sample));
    button.addEventListener('click', () => showSample(item)); return button;
  }));
}
$('generation-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (busy || !selected) return;
  const payload = {
    run: selected.id, checkpoint: $('checkpoint').value, tokenizer: $('tokenizer').value,
    prompt: $('prompt').value, temperature: Number($('temperature').value), top_k: Number($('top-k').value),
    max_new_tokens: Number($('max-tokens').value), seed: $('seed').value.trim() || null,
    context_length: !$('context').readOnly && $('context').value ? Number($('context').value) : null,
    device: $('device').value, color: $('color').value, plain: $('plain').checked,
  };
  busy = true; $('generation-fields').disabled = true; $('refresh').disabled = true; $('choose-model').disabled = true; $('generate').textContent = 'Generating…'; renderRuns();
  const started = performance.now();
  const update = () => status(`Generating… ${((performance.now() - started) / 1000).toFixed(1)}s`);
  update(); const timer = setInterval(update, 200);
  let message, failed = false;
  try {
    const sample = await api('/api/generate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    sample.modelName = selected.model.toUpperCase();
    sample.number = (samples[0]?.number || 0) + 1;
    samples.unshift(sample); samples = samples.slice(0, 10);
    showSample(sample); message = `Sample ${sample.number} ready in ${sample.elapsed_seconds.toFixed(2)}s.`;
  } catch (error) { message = error.message; failed = true; }
  finally {
    clearInterval(timer); busy = false; $('generation-fields').disabled = !selected; $('refresh').disabled = false;
    $('choose-model').disabled = false; $('generate').textContent = samples.length ? 'Generate again' : 'Generate';
    renderRuns(); status(message, failed);
  }
});
async function copy(text, button) {
  const original = button.textContent;
  try { await navigator.clipboard.writeText(text); button.textContent = 'Copied'; }
  catch { status('Clipboard unavailable. Select and copy the visible text instead.', true); }
  setTimeout(() => { button.textContent = original; }, 1500);
}
$('copy-text').addEventListener('click', event => copy(currentSample.text, event.currentTarget));
$('copy-command').addEventListener('click', event => copy(currentSample.command, event.currentTarget));
$('download').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([currentSample.text], {type: 'text/plain;charset=utf-8'}));
  const link = document.createElement('a'); link.href = url; link.download = `ghostwriter-${currentSample.settings.run}-${currentSample.number}.txt`;
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
function updateSummary() {
  $('sampling-summary').textContent = `Temperature ${$('temperature').value} · ${$('max-tokens').value} tokens`;
}
function setOptions(open) {
  $('options-panel').hidden = !open;
  $('toggle-options').setAttribute('aria-expanded', String(open));
}
$('toggle-options').addEventListener('click', () => setOptions($('options-panel').hidden));
$('reset-options').addEventListener('click', () => {
  $('temperature').value = '1.0'; $('temperature-range').value = '1'; $('top-k').value = '40';
  $('max-tokens').value = selectedTokenizer()?.max_new_tokens ?? 200;
  $('seed').value = ''; $('device').value = availableDevices.includes('cuda') ? 'cuda' : 'cpu'; $('color').value = 'auto'; $('plain').checked = false;
  updateSummary();
});
$('generation-form').addEventListener('invalid', event => {
  setOptions(true);
  const details = event.target.closest('details');
  if (details) details.open = true;
}, true);
$('temperature-range').addEventListener('input', () => { $('temperature').value = $('temperature-range').value; updateSummary(); });
$('temperature').addEventListener('input', () => { $('temperature-range').value = $('temperature').value; updateSummary(); });
$('max-tokens').addEventListener('input', updateSummary);
$('choose-model').addEventListener('click', () => {
  $('search').value = ''; renderRuns(); $('model-picker').showModal(); $('search').focus();
});
$('close-picker').addEventListener('click', () => $('model-picker').close());
$('model-picker').addEventListener('keydown', event => {
  if (event.key === 'Escape') { event.preventDefault(); $('model-picker').close(); }
});
$('model-picker').addEventListener('click', event => { if (event.target === $('model-picker')) {
  const bounds = $('model-picker').getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) $('model-picker').close();
}});
document.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter' && !busy && selected && !$('model-picker').open) {
    event.preventDefault(); $('generation-form').requestSubmit();
  }
});
$('prompt-hint').textContent = /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘ Enter to generate' : 'Ctrl Enter to generate';
$('search').addEventListener('input', renderRuns);
$('tokenizer').addEventListener('change', applyTokenizer);
$('refresh').addEventListener('click', refresh);
refresh();
