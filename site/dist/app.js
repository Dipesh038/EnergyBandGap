const form = document.querySelector('#predict-form');
const formulaInput = document.querySelector('#formula');
const errorBox = document.querySelector('#formula-error');
const button = document.querySelector('#predict-button');
const status = document.querySelector('#model-status');
let modelPromise;
let metadataPromise;
let observationsPromise;
let predictionVersion = 0;

function normalizeFormula(formula, allowedElements) {
  const subscriptDigits = '₀₁₂₃₄₅₆₇₈₉';
  const compact = formula.trim().replace(/\s+/g, '').replace(/[₀-₉]/g, digit => String(subscriptDigits.indexOf(digit)));
  // Preserve explicit symbols such as CO (C + O). Repair uppercase runs only
  // when reading every letter separately would require an invalid element.
  return compact.replace(/[A-Z]{2,}/g, run => {
    if ([...run].every(letter => allowedElements.has(letter))) return run;
    const solutions = new Map();
    function segment(position) {
      if (position === run.length) return [];
      if (solutions.has(position)) return solutions.get(position);
      let best = null;
      for (const length of [1, 2]) {
        if (position + length > run.length) continue;
        const symbol = run[position] + (length === 2 ? run[position + 1].toLowerCase() : '');
        if (!allowedElements.has(symbol)) continue;
        const rest = segment(position + length);
        if (rest && (!best || rest.length + 1 < best.length)) best = [symbol, ...rest];
      }
      solutions.set(position, best);
      return best;
    }
    return segment(0)?.join('') || run;
  });
}

function readNumber(text, position) {
  const match = /^(?:\d+(?:\.\d*)?|\.\d+)/.exec(text.slice(position));
  if (!match) return [1, position];
  const value = Number(match[0]);
  if (!Number.isFinite(value) || value <= 0) throw new Error('Amounts must be positive numbers.');
  return [value, position + match[0].length];
}

function parseFormula(formula, allowedElements) {
  const text = normalizeFormula(formula, allowedElements);
  if (!text) throw new Error('Enter a chemical formula first.');
  if (text.length > 120) throw new Error('Please enter a shorter formula.');
  const groups = [{ opener: '', counts: new Map() }];
  let pos = 0;
  while (pos < text.length) {
    const char = text[pos];
    if (char === '(' || char === '[') {
      groups.push({ opener: char, counts: new Map() });
      pos++;
    } else if (char === ')' || char === ']') {
      if (groups.length === 1 || (char === ')' ? '(' : '[') !== groups[groups.length - 1].opener) throw new Error('The parentheses in this formula do not match.');
      pos++;
      const [factor, next] = readNumber(text, pos);
      pos = next;
      const group = groups.pop().counts;
      if (!group.size) throw new Error('An empty group cannot be predicted.');
      const parent = groups[groups.length - 1].counts;
      for (const [element, count] of group) parent.set(element, (parent.get(element) || 0) + count * factor);
    } else if (/[A-Z]/.test(char)) {
      const match = /^[A-Z][a-z]?/.exec(text.slice(pos));
      const element = match[0];
      if (!allowedElements.has(element)) throw new Error(`${element} is not an element in this model.`);
      pos += element.length;
      const [count, next] = readNumber(text, pos);
      pos = next;
      const group = groups[groups.length - 1].counts;
      group.set(element, (group.get(element) || 0) + count);
    } else if (/\s/.test(char)) {
      pos++;
    } else {
      throw new Error(`I could not read “${char}”. Enter a chemical formula (for example, HCl), rather than a compound name.`);
    }
  }
  if (groups.length !== 1) throw new Error('The parentheses in this formula do not match.');
  const counts = groups[0].counts;
  if (!counts.size) throw new Error('Enter at least one element.');
  const total = [...counts.values()].reduce((a, b) => a + b, 0);
  const fractions = new Map([...counts].map(([element, count]) => [element, count / total]));
  return { fractions, counts, formula: text, corrected: text !== formula.trim() };
}

function loadMetadata() {
  if (!metadataPromise) metadataPromise = fetch('model-meta.json').then(checkedJson).catch(error => { metadataPromise = undefined; throw error; });
  return metadataPromise;
}

async function loadModel() {
  if (!modelPromise) {
    modelPromise = Promise.all([
      loadMetadata(),
      fetch('forest.bin').then(async response => {
        if (!response.ok) throw new Error('Could not load the trained model.');
        return response.arrayBuffer();
      })
    ]).then(([meta, buffer]) => {
      if (meta.format !== 'rf-preorder-v1' || buffer.byteLength !== meta.node_count * meta.node_bytes) throw new Error('The model files are incomplete.');
      return { meta, view: new DataView(buffer), indices: new Map(meta.elements.map((name, index) => [name, index])) };
    }).catch(error => { modelPromise = undefined; throw error; });
  }
  return modelPromise;
}

function checkedJson(response) {
  if (!response.ok) throw new Error('Could not load the project data.');
  return response.json();
}

function predict(model, fractions) {
  // scikit-learn casts prediction inputs to float32 before tree traversal.
  const features = new Float32Array(model.meta.elements.length);
  for (const [element, fraction] of fractions) features[model.indices.get(element)] = fraction;
  const view = model.view;
  let sum = 0;
  for (let tree = 0; tree < model.meta.tree_count; tree++) {
    const offset = model.meta.tree_offsets[tree];
    let node = 0;
    while (true) {
      const byte = (offset + node) * 12;
      const feature = view.getUint16(byte, true);
      const right = view.getUint16(byte + 2, true);
      const value = view.getFloat64(byte + 4, true);
      if (feature === 65535) { sum += value; break; }
      node = features[feature] <= value ? node + 1 : right;
    }
  }
  return sum / model.meta.tree_count;
}

function showComposition(fractions, formula) {
  const panel = document.querySelector('#composition-panel');
  document.querySelector('#composition-title').textContent = `${formula} composition`;
  const bars = document.querySelector('#composition-bars');
  bars.replaceChildren();
  [...fractions].sort((a, b) => b[1] - a[1]).forEach(([element, fraction]) => {
    const row = document.createElement('div');
    row.className = 'composition-row';
    const name = document.createElement('span'); name.textContent = element;
    const track = document.createElement('div'); track.className = 'track';
    const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${fraction * 100}%`;
    const percent = document.createElement('span'); percent.textContent = `${(fraction * 100).toFixed(1)}%`;
    track.append(fill); row.append(name, track, percent); bars.append(row);
  });
  panel.hidden = false;
}

async function showObservation(formula, gap, version) {
  const box = document.querySelector('#observed-box');
  box.hidden = true;
  try {
    if (!observationsPromise) observationsPromise = fetch('observations.json').then(checkedJson);
    const observations = await observationsPromise;
    if (version !== predictionVersion) return;
    const found = observations[formula];
    box.replaceChildren();
    box.classList.toggle('no-reference', !found);
    if (!found) {
      const message = document.createElement('div');
      message.textContent = 'No exact experimental CSV entry for this formula. This is a model estimate without a direct reference value here.';
      box.append(message);
      box.hidden = false;
      return;
    }
    const heading = document.createElement('div');
    heading.className = 'comparison-title';
    heading.textContent = 'MODEL VS EXPERIMENTAL CSV REFERENCE';
    const grid = document.createElement('div');
    grid.className = 'comparison-grid';
    const difference = Math.abs(gap - found.mean);
    for (const [label, value, highlighted] of [
      ['Predicted', `${gap.toFixed(3)} eV`, false],
      ['CSV reference', `${found.mean.toFixed(3)} eV`, false],
      ['Absolute difference', difference > 0 && difference < 0.0005 ? '<0.001 eV' : `${difference.toFixed(3)} eV`, true]
    ]) {
      const metric = document.createElement('div');
      metric.className = `comparison-metric${highlighted ? ' is-difference' : ''}`;
      const caption = document.createElement('span');
      caption.textContent = label;
      const number = document.createElement('strong');
      number.textContent = value;
      metric.append(caption, number);
      grid.append(metric);
    }
    const note = document.createElement('div');
    note.className = 'comparison-note';
    note.textContent = `Absolute difference = |predicted − CSV reference|. ${found.count > 1 ? `The reference is the mean of ${found.count} entries. ` : ''}This is a reference comparison, not an independent accuracy test.`;
    box.append(heading, grid, note);
    box.hidden = false;
  } catch { /* Prediction remains available if optional reference data fails. */ }
}

function clearResult() {
  predictionVersion++;
  errorBox.hidden = true;
  button.disabled = false;
  button.firstElementChild.textContent = 'Predict band gap';
  document.querySelector('#formula-note').hidden = true;
  document.querySelector('#result-empty').hidden = false;
  document.querySelector('#result-content').hidden = true;
  document.querySelector('#result-panel').classList.remove('has-result');
  document.querySelector('#observed-box').hidden = true;
  document.querySelector('#composition-panel').hidden = true;
  status.textContent = 'READY TO PREDICT';
  return predictionVersion;
}

async function runPrediction(event) {
  event.preventDefault();
  const version = clearResult();
  const typedFormula = formulaInput.value.trim();
  button.disabled = true;
  button.firstElementChild.textContent = 'Checking formula…';
  status.textContent = 'CHECKING FORMULA';
  try {
    const meta = await loadMetadata();
    if (version !== predictionVersion) return;
    const parsed = parseFormula(typedFormula, new Set(meta.elements));
    const { fractions, formula, corrected } = parsed;
    button.firstElementChild.textContent = 'Loading model…';
    status.textContent = 'LOADING MODEL';
    const model = await loadModel();
    if (version !== predictionVersion) return;
    const gap = predict(model, fractions);
    document.querySelector('#gap-value').textContent = gap.toFixed(2);
    document.querySelector('#result-formula').textContent = `For ${formula}`;
    if (corrected) {
      const note = document.querySelector('#formula-note');
      note.textContent = `Interpreted “${typedFormula}” as “${formula}”.`;
      note.hidden = false;
    }
    document.querySelector('#band-gap-label').textContent = `${gap.toFixed(2)} eV gap`;
    document.querySelector('#band-gap-space').style.height = `${Math.min(140, Math.max(28, 28 + gap * 18))}px`;
    document.querySelector('#result-empty').hidden = true;
    document.querySelector('#result-content').hidden = false;
    document.querySelector('#result-panel').classList.add('has-result');
    status.textContent = 'PREDICTION COMPLETE';
    showComposition(fractions, formula);
    showObservation(formula, gap, version);
  } catch (error) {
    if (version === predictionVersion) {
      errorBox.textContent = error.message || 'Prediction failed. Please try again.';
      errorBox.hidden = false;
      status.textContent = 'READY TO PREDICT';
    }
  } finally {
    if (version === predictionVersion) {
      button.disabled = false;
      button.firstElementChild.textContent = 'Predict band gap';
    }
  }
}

form.addEventListener('submit', runPrediction);
formulaInput.addEventListener('input', clearResult);
document.querySelectorAll('[data-formula]').forEach(example => example.addEventListener('click', () => {
  formulaInput.value = example.dataset.formula;
  form.requestSubmit();
}));

if (typeof module !== 'undefined') module.exports = { parseFormula, predict, loadModel };
