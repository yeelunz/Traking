const STANDALONE_GROUP_KEY = '__standalone__';
const SHOW_LOCALIZATION_DEBUG = false;

const state = {
  experiments: [],
  filtered: [],
  groups: [],
  selectedGroup: null,
  searchTerm: '',
  current: null,
  currentView: { mode: 'single', expId: null, label: '' },
  noteAutosaveTimer: null,
  currentClassificationPredictions: [],
  assetExpId: null,
  detectionChart: null,
  segmentationChart: null,
  classificationChart: null,
  groupDetectionChart: null,
  groupSegmentationChart: null,
  groupClassificationChart: null,
  losoDetectionChart: null,
  losoSegmentationChart: null,
  losoClassificationChart: null,
  detectionCategory: 'detection_visualizations',
  segmentationCategory: 'segmentation_overlays',
  groupChartSelections: {},
  losoChartSelections: {}
};

const DETECTION_SUMMARY_FIELDS = [
  { label: 'IoU', key: 'iou_mean', std: 'iou_std', percent: true },
  { label: 'Center Error (pix)', key: 'ce_mean', std: 'ce_std' },
  { label: 'SR@0.5', key: 'success_rate_50', percent: true },
  { label: 'SR@0.75', key: 'success_rate_75', percent: true },
  { label: 'AUROC', key: 'success_auc' },
  { label: 'ROI Fallback', key: 'roi_fallback_rate', percent: true },
  { label: 'FPS', key: 'fps' },
  { label: 'Drift Rate', key: 'drift_rate' }
];

const SEGMENTATION_SUMMARY_FIELDS = [
  { label: 'Dice', key: 'dice_mean', std: 'dice_std', percent: true },
  { label: 'IoU', key: 'iou_mean', std: 'iou_std', percent: true },
  { label: 'Centroid (pix)', key: 'centroid_mean', std: 'centroid_std' },
  { label: 'FPS', key: 'fps_mean', std: 'fps_std' }
];

const CLASSIFICATION_SUMMARY_FIELDS = [
  { label: 'Accuracy', key: 'accuracy', percent: true },
  { label: 'Balanced Acc', key: 'balanced_accuracy', percent: true },
  { label: 'Precision', key: 'precision_positive', percent: true },
  { label: 'Recall', key: 'recall_positive', percent: true },
  { label: 'Specificity', key: 'specificity', percent: true, allowNull: true, chartExclude: true },
  { label: 'F1', key: 'f1_positive', percent: true },
  { label: 'ROC AUC', key: 'roc_auc', percent: true, allowNull: true },
  { label: 'Brier Score', key: 'brier_score', decimals: 4 },
  { label: 'Reliability', key: 'reliability', decimals: 4 },
  { label: 'Resolution', key: 'resolution', decimals: 4 },
  { label: 'Uncertainty', key: 'uncertainty', decimals: 4 },
  { label: 'Threshold', key: 'threshold_used', decimals: 4, chartExclude: true },
  { label: "Youden's J", key: 'youden_j', decimals: 4, chartExclude: true },
];

const TRAJECTORY_FILTER_FIELDS = [
  { label: 'Jitter CX', key: 'jitter_cx', decimals: 2 },
  { label: 'Jitter CY', key: 'jitter_cy', decimals: 2 },
  { label: 'Jitter W', key: 'jitter_w', decimals: 2 },
  { label: 'Jitter H', key: 'jitter_h', decimals: 2 },
  { label: 'Smoothness CX', key: 'smoothness_cx', decimals: 2 },
  { label: 'Smoothness CY', key: 'smoothness_cy', decimals: 2 },
  { label: 'Area Stability', key: 'area_stability', percent: true },
  { label: 'Path Length', key: 'path_length', decimals: 1 },
];

const SUMMARY_COLORS = ['#6aa5ff', '#f8c146', '#66dfc5', '#ff8ba7', '#bd93f9', '#94f7c5'];

function withCacheBust(url, cacheBust) {
  if (!cacheBust) return url;
  const joiner = url.includes('?') ? '&' : '?';
  return `${url}${joiner}_=${Date.now()}`;
}

async function fetchJSON(url, options = {}) {
  const { cacheBust, ...fetchOptions } = options || {};
  const res = await fetch(withCacheBust(url, cacheBust), {
    ...fetchOptions,
    ...(cacheBust ? { cache: 'no-store' } : {}),
  });
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json();
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function ensureScheduleNotePanel() {
  let panel = document.getElementById('schedule-note-panel');
  if (panel) return panel;
  const main = document.querySelector('main');
  if (!main) return null;
  panel = document.createElement('section');
  panel.id = 'schedule-note-panel';
  panel.className = 'hidden';
  panel.innerHTML = `
    <div class="section-header">
      <div>
        <h2 id="schedule-note-title">排程 Note</h2>
        <p id="schedule-note-path"></p>
      </div>
      <div class="note-actions">
        <span id="schedule-note-status" class="note-status"></span>
      </div>
    </div>
    <textarea id="schedule-note-input" rows="5" placeholder="在這裡記錄這個排程是做什麼的…"></textarea>
  `;
  main.insertBefore(panel, main.firstChild);
  return panel;
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined) return '—';
  return Number(value).toFixed(digits);
}

function formatWithStd(value, std, digits = 3) {
  if (value === null || value === undefined) return '—';
  const base = formatNumber(value, digits);
  if (std === null || std === undefined) return base;
  return `${base} ± ${formatNumber(std, digits)}`;
}

function formatPercent(value, digits = 2) {
  if (value === null || value === undefined) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

function formatPercentWithStd(value, std, digits = 2) {
  if (value === null || value === undefined) return '—';
  const base = formatPercent(value, digits);
  if (std === null || std === undefined) return base;
  return `${base} ± ${(std * 100).toFixed(digits)}%`;
}

function formatPercentValue(value, digits = 2) {
  if (value === null || value === undefined) return '—';
  return `${Number(value).toFixed(digits)}%`;
}

function hasChartJs() {
  return typeof Chart !== 'undefined';
}

function clamp01(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return null;
  return Math.max(0, Math.min(1, num));
}

function formatDatasetValueForExport(value, dataset) {
  if (value === null || value === undefined || Number.isNaN(value)) return '';
  const numeric = Number(value);
  if (dataset.percent) {
    return `${numeric.toFixed(2)}%`;
  }
  return formatNumber(numeric, 3);
}

function rowsToCSV(rows) {
  return rows
    .map((row) =>
      row
        .map((cell) => {
          if (cell === null || cell === undefined) return '';
          const text = String(cell);
          if (/[",\n]/.test(text)) {
            return `"${text.replace(/"/g, '""')}"`;
          }
          return text;
        })
        .join(',')
    )
    .join('\n');
}

async function copyTextToClipboard(text) {
  if (!text) return;
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  try {
    document.execCommand('copy');
  } finally {
    document.body.removeChild(textarea);
  }
}

async function copyChartData(chartKey, format = 'csv') {
  if (!chartKey) {
    alert('找不到圖表');
    return;
  }
  const chart = state[chartKey];
  if (!chart) {
    alert('圖表尚未載入，請稍後再試');
    return;
  }
  const labels = chart.data?.labels || [];
  if (!labels.length) {
    alert('目前沒有可匯出的資料');
    return;
  }
  const datasetEntries = (chart.data.datasets || [])
    .map((dataset, index) => {
      const visible = typeof chart.isDatasetVisible === 'function' ? chart.isDatasetVisible(index) !== false : dataset.hidden !== true;
      return { dataset, index, visible };
    })
    .filter((entry) => entry.visible);
  if (!datasetEntries.length) {
    alert('所有序列都被隱藏，無法匯出。');
    return;
  }
  const header = ['Label', ...datasetEntries.map((entry) => entry.dataset.label || `Series ${entry.index + 1}`)];
  const rows = [header];
  labels.forEach((label, rowIndex) => {
    const row = [label ?? `Row ${rowIndex + 1}`];
    datasetEntries.forEach(({ dataset }) => {
      const source = dataset.data;
      const value = source && typeof source === 'object' ? source[rowIndex] : undefined;
      row.push(formatDatasetValueForExport(value, dataset));
    });
    rows.push(row);
  });
  let payload = '';
  if (format === 'json') {
    const jsonRows = rows.slice(1).map((row) => {
      const obj = { label: row[0] };
      row.slice(1).forEach((value, idx) => {
        obj[header[idx + 1]] = value;
      });
      return obj;
    });
    payload = JSON.stringify(jsonRows, null, 2);
  } else {
    payload = rowsToCSV(rows);
  }
  try {
    await copyTextToClipboard(payload);
    alert('圖表資料已複製到剪貼簿');
  } catch (err) {
    console.error(err);
    alert('複製失敗：' + err.message);
  }
}

function sanitizeFilename(name) {
  const fallback = 'chart';
  if (!name) return fallback;
  const cleaned = name.replace(/[^a-z0-9_\-]+/gi, '_').replace(/^_+|_+$/g, '');
  return cleaned || fallback;
}

function downloadChartImage(chartKey, canvasId, filenameBase) {
  const chartInstance = chartKey ? state[chartKey] : null;
  const canvas = chartInstance?.canvas || document.getElementById(canvasId);
  if (!canvas || !canvas.width || !canvas.height) {
    alert('圖表尚未載入，無法匯出。');
    return;
  }
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width = canvas.width;
  exportCanvas.height = canvas.height;
  const ctx = exportCanvas.getContext('2d');
  if (!ctx) {
    alert('瀏覽器不支援匯出畫布。');
    return;
  }
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
  ctx.drawImage(canvas, 0, 0);
  const link = document.createElement('a');
  const safeName = sanitizeFilename(filenameBase);
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  link.download = `${safeName}_${timestamp}.png`;
  link.href = exportCanvas.toDataURL('image/png', 1.0);
  link.click();
}

function setupDownloadButtons() {
  const buttons = document.querySelectorAll('[data-chart-key][data-canvas-id]');
  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const chartKey = btn.dataset.chartKey;
      const canvasId = btn.dataset.canvasId;
      const filename = btn.dataset.filename || chartKey;
      downloadChartImage(chartKey, canvasId, filename);
    });
  });
}

function setupDataExportButtons() {
  const buttons = document.querySelectorAll('[data-export-chart]');
  buttons.forEach((btn) => {
    btn.addEventListener('click', async () => {
      const chartKey = btn.dataset.exportChart;
      const format = btn.dataset.exportFormat || 'csv';
      const originalText = btn.textContent;
      btn.disabled = true;
      try {
        await copyChartData(chartKey, format);
        btn.textContent = '已複製';
        setTimeout(() => {
          btn.textContent = originalText;
        }, 1500);
      } finally {
        btn.disabled = false;
      }
    });
  });
}

function renderExperiments() {
  const list = document.getElementById('experiment-list');
  list.innerHTML = '';
  if (!state.filtered.length) {
    const empty = document.createElement('li');
    empty.textContent = '沒有可顯示的實驗';
    empty.classList.add('empty');
    list.appendChild(empty);
  }
  state.filtered.forEach((exp) => {
    const li = document.createElement('li');
    const foldSuffix = exp.is_loso && exp.fold_count ? ` (LOSO × ${exp.fold_count})` : '';
    li.innerHTML = `<strong>${escapeHtml(exp.name)}${escapeHtml(foldSuffix)}</strong><br /><small>${escapeHtml(exp.relative_path)}</small>`;
    li.dataset.id = exp.id;
    if (state.current && state.current.id === exp.id) {
      li.classList.add('active');
    }
    li.addEventListener('click', () => selectExperiment(exp.id));
    list.appendChild(li);
  });
  const experimentCount = document.getElementById('experiment-count');
  if (experimentCount) {
    experimentCount.textContent = `${state.filtered.length} 筆`;
  }
}

function getSelectedGroup() {
  return state.groups.find((g) => g.key === state.selectedGroup) || null;
}

function getScheduleNoteContext() {
  const group = getSelectedGroup();
  if (!group) return null;
  if (group.key === STANDALONE_GROUP_KEY) {
    if (group.experiments.length !== 1) {
      return null;
    }
    const exp = group.experiments[0];
    return {
      group,
      groupPath: '',
      expId: exp.id,
      label: exp.name || exp.relative_path || exp.id,
      pathLabel: exp.relative_path || exp.id,
      note: exp.schedule_note || '',
    };
  }
  return {
    group,
    groupPath: group.key,
    expId: group.experiments[0]?.id || null,
    label: group.label,
    pathLabel: group.key,
    note: group.note || '',
  };
}

function renderScheduleNotePanel() {
  ensureScheduleNotePanel();
  const panel = document.getElementById('schedule-note-panel');
  const title = document.getElementById('schedule-note-title');
  const path = document.getElementById('schedule-note-path');
  const input = document.getElementById('schedule-note-input');
  const status = document.getElementById('schedule-note-status');
  if (!panel || !title || !path || !input || !status) return;
  const ctx = getScheduleNoteContext();
  if (!ctx) {
    panel.classList.add('hidden');
    input.value = '';
    path.textContent = '';
    status.textContent = '';
    status.className = 'note-status';
    return;
  }
  panel.classList.remove('hidden');
  title.textContent = `排程 Note：${ctx.label}`;
  path.textContent = `排程位置：${ctx.pathLabel || '（單一實驗）'}`;
  input.value = ctx.note || '';
  status.textContent = '';
  status.className = 'note-status';
}

async function saveCurrentScheduleNote() {
  const input = document.getElementById('schedule-note-input');
  const status = document.getElementById('schedule-note-status');
  const ctx = getScheduleNoteContext();
  if (!input || !status || !ctx) return;
  const text = input.value;
  status.textContent = '儲存中…';
  status.className = 'note-status saving';
  try {
    await fetchJSON('/api/schedules/note', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        group_path: ctx.groupPath,
        exp_id: ctx.expId,
        text,
      }),
    });
    if (ctx.group) {
      ctx.group.note = text;
      ctx.group.experiments.forEach((exp) => {
        exp.schedule_note = text;
      });
    }
    renderGroups();
    status.textContent = '已儲存';
    status.className = 'note-status saved';
  } catch (err) {
    console.error(err);
    status.textContent = '儲存失敗';
    status.className = 'note-status error';
  }
}

function queueScheduleNoteAutosave() {
  const status = document.getElementById('schedule-note-status');
  if (state.noteAutosaveTimer) {
    clearTimeout(state.noteAutosaveTimer);
  }
  if (status) {
    status.textContent = '編輯中…';
    status.className = 'note-status';
  }
  state.noteAutosaveTimer = setTimeout(() => {
    state.noteAutosaveTimer = null;
    saveCurrentScheduleNote();
  }, 700);
}

function applySearch(term) {
  state.searchTerm = term;
  const activeGroup = getSelectedGroup();
  const pool = activeGroup ? activeGroup.experiments : state.experiments;
  const t = (term || '').trim().toLowerCase();
  state.filtered = pool.filter((exp) => {
    if (!t) return true;
    return (
      exp.name.toLowerCase().includes(t) ||
      exp.models.join(' ').toLowerCase().includes(t) ||
      exp.preprocs.join(' ').toLowerCase().includes(t) ||
      (exp.relative_path || '').toLowerCase().includes(t)
    );
  });
  renderExperiments();
}

function normalizeGroupLabel(key) {
  if (!key || key === STANDALONE_GROUP_KEY) {
    return '單一實驗';
  }
  return key.replace(/\0/g, '').split('/').filter(Boolean).join(' / ');
}

function buildGroups() {
  const buckets = new Map();
  state.experiments.forEach((exp) => {
    const key = exp.group_path ? exp.group_path : STANDALONE_GROUP_KEY;
    if (!buckets.has(key)) {
      buckets.set(key, { key, label: normalizeGroupLabel(key), experiments: [], note: '' });
    }
    const bucket = buckets.get(key);
    bucket.experiments.push(exp);
    if (!bucket.note && exp.schedule_note) {
      bucket.note = exp.schedule_note;
    }
  });
  state.groups = Array.from(buckets.values()).sort((a, b) => {
    if (a.key === STANDALONE_GROUP_KEY && b.key !== STANDALONE_GROUP_KEY) return 1;
    if (b.key === STANDALONE_GROUP_KEY && a.key !== STANDALONE_GROUP_KEY) return -1;
    const aTime = a.experiments[0]?.created_at || '';
    const bTime = b.experiments[0]?.created_at || '';
    if (aTime !== bTime) return bTime.localeCompare(aTime);
    return a.label.localeCompare(b.label);
  });
  state.groups.forEach((group) => {
    group.experiments.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
  });
  if (!state.selectedGroup || !buckets.has(state.selectedGroup)) {
    state.selectedGroup = state.groups.length ? state.groups[0].key : null;
  }
}

function renderGroups() {
  const list = document.getElementById('group-list');
  if (!list) return;
  list.innerHTML = '';
  if (!state.groups.length) {
    const li = document.createElement('li');
    li.textContent = '沒有批次資料';
    li.classList.add('empty');
    list.appendChild(li);
  } else {
    state.groups.forEach((group) => {
      const li = document.createElement('li');
      const title = document.createElement('div');
      title.innerHTML = `<strong>${escapeHtml(group.label)}</strong> <small>(${group.experiments.length})</small>`;
      li.appendChild(title);
      const note = document.createElement('div');
      note.className = 'group-note-preview';
      note.textContent = group.note || '尚未填寫 note';
      li.appendChild(note);
      if (state.selectedGroup === group.key) {
        li.classList.add('active');
      }
      li.addEventListener('click', () => selectGroup(group.key));
      list.appendChild(li);
    });
  }
  const groupCount = document.getElementById('group-count');
  if (groupCount) {
    groupCount.textContent = `${state.groups.length} 組`;
  }
}

function selectGroup(key) {
  state.selectedGroup = key;
  state.current = null;
  renderGroups();
  renderScheduleNotePanel();
  renderGroupOverview();
  applySearch(state.searchTerm);
  showWelcome();
}

function showWelcome() {
  document.getElementById('welcome').classList.remove('hidden');
  document.getElementById('experiment-details').classList.add('hidden');
}

function hideWelcome() {
  document.getElementById('welcome').classList.add('hidden');
  document.getElementById('experiment-details').classList.remove('hidden');
}

function resetFoldControls() {
  const controls = document.getElementById('fold-controls');
  if (controls) controls.classList.add('hidden');
}

function hideLosoOverview() {
  const section = document.getElementById('loso-overview');
  if (section) section.classList.add('hidden');
  const subtitle = document.getElementById('loso-subtitle');
  if (subtitle) subtitle.textContent = '';
  const controls = document.getElementById('loso-chart-controls');
  if (controls) {
    controls.classList.add('hidden');
    controls.innerHTML = '';
  }
  const table = document.getElementById('loso-fold-table');
  if (table) table.innerHTML = '';
  renderGroupSummaryChart('loso-detection-chart', [], 'detection', DETECTION_SUMMARY_FIELDS, 'losoDetectionChart');
  renderGroupSummaryChart('loso-segmentation-chart', [], 'segmentation', SEGMENTATION_SUMMARY_FIELDS, 'losoSegmentationChart');
  renderGroupSummaryChart('loso-classification-chart', [], 'classification', CLASSIFICATION_SUMMARY_FIELDS, 'losoClassificationChart');
}

function showLosoOverview() {
  const section = document.getElementById('loso-overview');
  if (section) section.classList.remove('hidden');
}

function foldPreviewMetric(fold, section, key) {
  return fold?.preview?.[section]?.[key];
}

function renderLosoFoldTable(folds) {
  const table = document.getElementById('loso-fold-table');
  if (!table) return;
  if (!folds || !folds.length) {
    table.innerHTML = '<tr><td>無 fold 資料</td></tr>';
    return;
  }
  const columns = [
    { label: 'Fold', format: (f) => f.fold || f.exp_id },
    { label: 'Det IoU', format: (f) => formatPercentWithStd(foldPreviewMetric(f, 'detection', 'iou_mean'), foldPreviewMetric(f, 'detection', 'iou_std')) },
    { label: 'Det Center Err', format: (f) => formatWithStd(foldPreviewMetric(f, 'detection', 'ce_mean'), foldPreviewMetric(f, 'detection', 'ce_std')) },
    { label: 'Det SR@0.5', format: (f) => formatPercent(foldPreviewMetric(f, 'detection', 'success_rate_50')) },
    { label: 'Det ROI Fallback', format: (f) => formatPercent(foldPreviewMetric(f, 'detection', 'roi_fallback_rate')) },
    { label: 'Seg Dice', format: (f) => formatPercentWithStd(foldPreviewMetric(f, 'segmentation', 'dice_mean'), foldPreviewMetric(f, 'segmentation', 'dice_std')) },
    { label: 'Seg IoU', format: (f) => formatPercentWithStd(foldPreviewMetric(f, 'segmentation', 'iou_mean'), foldPreviewMetric(f, 'segmentation', 'iou_std')) },
    { label: 'Seg Centroid', format: (f) => formatWithStd(foldPreviewMetric(f, 'segmentation', 'centroid_mean'), foldPreviewMetric(f, 'segmentation', 'centroid_std')) },
    { label: 'Cls Acc', format: (f) => formatPercent(foldPreviewMetric(f, 'classification', 'accuracy')) },
    { label: 'Cls F1', format: (f) => formatPercent(foldPreviewMetric(f, 'classification', 'f1_positive')) },
    { label: 'Cls AUC', format: (f) => formatNumber(foldPreviewMetric(f, 'classification', 'roc_auc'), 4) },
    { label: 'Cls Brier', format: (f) => formatNumber(foldPreviewMetric(f, 'classification', 'brier_score'), 4) }
  ];
  const thead = '<tr>' + columns.map((c) => `<th>${c.label}</th>`).join('') + '</tr>';
  const tbody = folds
    .map((fold) => {
      const cells = columns.map((c) => `<td>${c.format(fold)}</td>`).join('');
      return `<tr data-exp-id="${escapeHtml(fold.exp_id)}">${cells}</tr>`;
    })
    .join('');
  table.innerHTML = thead + tbody;
}

function previewMetric(exp, section, key) {
  return exp?.preview?.[section]?.[key];
}

function initializeGroupChartSelection(group) {
  if (!group) return;
  const validIds = new Set(group.experiments.map((exp) => exp.id));
  const existing = state.groupChartSelections[group.key];
  if (!existing) {
    state.groupChartSelections[group.key] = new Set(validIds);
    return;
  }
  const filtered = new Set();
  existing.forEach((id) => {
    if (validIds.has(id)) {
      filtered.add(id);
    }
  });
  if (!filtered.size && existing.size) {
    state.groupChartSelections[group.key] = new Set(validIds);
  } else {
    state.groupChartSelections[group.key] = filtered;
  }
}

function getGroupChartSelection(group) {
  initializeGroupChartSelection(group);
  return state.groupChartSelections[group?.key] || new Set();
}

function setGroupChartSelection(groupKey, ids) {
  state.groupChartSelections[groupKey] = new Set(ids);
}

function initializeLosoChartSelection(expId, folds) {
  if (!expId) return;
  const validIds = new Set((folds || []).map((f) => f.exp_id).filter(Boolean));
  const existing = state.losoChartSelections[expId];
  if (!existing) {
    state.losoChartSelections[expId] = new Set(validIds);
    return;
  }
  const filtered = new Set();
  existing.forEach((id) => {
    if (validIds.has(id)) {
      filtered.add(id);
    }
  });
  if (!filtered.size && existing.size) {
    state.losoChartSelections[expId] = new Set(validIds);
  } else {
    state.losoChartSelections[expId] = filtered;
  }
}

function getLosoChartSelection(expId, folds) {
  initializeLosoChartSelection(expId, folds);
  return state.losoChartSelections[expId] || new Set();
}

function setLosoChartSelection(expId, ids) {
  state.losoChartSelections[expId] = new Set(ids);
}

function getSelectedLosoFolds(expId, folds) {
  if (!expId) return folds || [];
  const selection = getLosoChartSelection(expId, folds);
  return (folds || []).filter((f) => selection.has(f.exp_id));
}

function updateLosoCharts(expId, folds) {
  const selected = getSelectedLosoFolds(expId, folds);
  const foldAsExperiments = selected.map((f) => ({
    name: f.fold || f.exp_id,
    preview: f.preview || {},
    has_detection: !!f.has_detection,
    has_segmentation: !!f.has_segmentation,
    has_classification: !!f.has_classification
  }));
  renderGroupSummaryChart('loso-detection-chart', foldAsExperiments, 'detection', DETECTION_SUMMARY_FIELDS, 'losoDetectionChart');
  renderGroupSummaryChart('loso-segmentation-chart', foldAsExperiments, 'segmentation', SEGMENTATION_SUMMARY_FIELDS, 'losoSegmentationChart');
  renderGroupSummaryChart('loso-classification-chart', foldAsExperiments, 'classification', CLASSIFICATION_SUMMARY_FIELDS, 'losoClassificationChart');
}

function renderLosoChartControls(expId, folds) {
  const container = document.getElementById('loso-chart-controls');
  if (!container) return;
  if (!expId || !folds || folds.length <= 1) {
    container.classList.add('hidden');
    container.innerHTML = '';
    return;
  }
  const selection = getLosoChartSelection(expId, folds);
  container.classList.remove('hidden');
  container.innerHTML = '';
  const header = document.createElement('div');
  header.className = 'chart-controls-header';
  const title = document.createElement('h4');
  title.textContent = `比較圖顯示 folds (${selection.size}/${folds.length})`;
  header.appendChild(title);
  const actions = document.createElement('div');
  const selectAll = document.createElement('button');
  selectAll.type = 'button';
  selectAll.className = 'ghost-button compact';
  selectAll.textContent = '全選';
  selectAll.addEventListener('click', () => {
    setLosoChartSelection(expId, folds.map((f) => f.exp_id).filter(Boolean));
    renderLosoChartControls(expId, folds);
    updateLosoCharts(expId, folds);
  });
  const clearAll = document.createElement('button');
  clearAll.type = 'button';
  clearAll.className = 'ghost-button compact';
  clearAll.textContent = '清除';
  clearAll.addEventListener('click', () => {
    setLosoChartSelection(expId, []);
    renderLosoChartControls(expId, folds);
    updateLosoCharts(expId, folds);
  });
  actions.appendChild(selectAll);
  actions.appendChild(clearAll);
  header.appendChild(actions);
  container.appendChild(header);
  const toggleList = document.createElement('div');
  toggleList.className = 'chart-toggle-list';
  folds.forEach((fold) => {
    const foldId = fold.exp_id;
    if (!foldId) return;
    const label = document.createElement('label');
    label.className = 'chart-toggle';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = selection.has(foldId);
    input.addEventListener('change', () => {
      const currentSelection = getLosoChartSelection(expId, folds);
      if (input.checked) {
        currentSelection.add(foldId);
      } else {
        currentSelection.delete(foldId);
      }
      renderLosoChartControls(expId, folds);
      updateLosoCharts(expId, folds);
    });
    const span = document.createElement('span');
    span.textContent = fold.fold || foldId;
    label.appendChild(input);
    label.appendChild(span);
    toggleList.appendChild(label);
  });
  container.appendChild(toggleList);
}

function getSelectedGroupExperiments(group) {
  if (!group) return [];
  const selection = getGroupChartSelection(group);
  return group.experiments.filter((exp) => selection.has(exp.id));
}

function renderGroupChartControls(group) {
  const container = document.getElementById('group-chart-controls');
  if (!container) return;
  if (!group || group.experiments.length <= 1) {
    container.classList.add('hidden');
    container.innerHTML = '';
    return;
  }
  const selection = getGroupChartSelection(group);
  container.classList.remove('hidden');
  container.innerHTML = '';
  const header = document.createElement('div');
  header.className = 'chart-controls-header';
  const title = document.createElement('h4');
  title.textContent = `比較圖顯示實驗 (${selection.size}/${group.experiments.length})`;
  header.appendChild(title);
  const actions = document.createElement('div');
  const selectAll = document.createElement('button');
  selectAll.type = 'button';
  selectAll.className = 'ghost-button compact';
  selectAll.textContent = '全選';
  selectAll.addEventListener('click', () => {
    setGroupChartSelection(group.key, group.experiments.map((exp) => exp.id));
    renderGroupChartControls(group);
    updateGroupCharts(group);
  });
  const clearAll = document.createElement('button');
  clearAll.type = 'button';
  clearAll.className = 'ghost-button compact';
  clearAll.textContent = '清除';
  clearAll.addEventListener('click', () => {
    setGroupChartSelection(group.key, []);
    renderGroupChartControls(group);
    updateGroupCharts(group);
  });
  actions.appendChild(selectAll);
  actions.appendChild(clearAll);
  header.appendChild(actions);
  container.appendChild(header);
  const toggleList = document.createElement('div');
  toggleList.className = 'chart-toggle-list';
  group.experiments.forEach((exp) => {
    const label = document.createElement('label');
    label.className = 'chart-toggle';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = selection.has(exp.id);
    input.addEventListener('change', () => {
      const currentSelection = getGroupChartSelection(group);
      if (input.checked) {
        currentSelection.add(exp.id);
      } else {
        currentSelection.delete(exp.id);
      }
      renderGroupChartControls(group);
      updateGroupCharts(group);
    });
    const span = document.createElement('span');
    span.textContent = exp.name;
    label.appendChild(input);
    label.appendChild(span);
    toggleList.appendChild(label);
  });
  container.appendChild(toggleList);
}

function updateGroupCharts(group) {
  const wrapper = document.getElementById('group-chart-grid');
  if (!wrapper) return;
  if (!group || group.experiments.length <= 1) {
    wrapper.classList.add('hidden');
    return;
  }
  const selected = getSelectedGroupExperiments(group);
  const detectionExperiments = selected.filter((exp) => exp.has_detection);
  const segmentationExperiments = selected.filter((exp) => exp.has_segmentation);
  const classificationExperiments = selected.filter((exp) => exp.has_classification);
  renderGroupSummaryChart('group-detection-chart', detectionExperiments, 'detection', DETECTION_SUMMARY_FIELDS, 'groupDetectionChart');
  renderGroupSummaryChart('group-segmentation-chart', segmentationExperiments, 'segmentation', SEGMENTATION_SUMMARY_FIELDS, 'groupSegmentationChart');
  renderGroupSummaryChart('group-classification-chart', classificationExperiments, 'classification', CLASSIFICATION_SUMMARY_FIELDS, 'groupClassificationChart');
  if (!detectionExperiments.length && !segmentationExperiments.length && !classificationExperiments.length) {
    wrapper.classList.add('hidden');
  } else {
    wrapper.classList.remove('hidden');
  }
}

function renderGroupOverview() {
  const section = document.getElementById('group-overview');
  const grid = document.getElementById('compare-grid');
  const table = document.getElementById('compare-table');
  const charts = document.getElementById('group-chart-grid');
  const controls = document.getElementById('group-chart-controls');
  const title = document.getElementById('group-title');
  const subtitle = document.getElementById('group-subtitle');
  if (!section) return;
  const group = state.groups.find((g) => g.key === state.selectedGroup);
  if (!group || group.experiments.length <= 1) {
    section.classList.add('hidden');
    if (grid) grid.innerHTML = '';
    if (table) table.innerHTML = '';
    if (charts) charts.classList.add('hidden');
    if (controls) {
      controls.classList.add('hidden');
      controls.innerHTML = '';
    }
    renderGroupSummaryChart('group-detection-chart', [], 'detection', DETECTION_SUMMARY_FIELDS, 'groupDetectionChart');
    renderGroupSummaryChart('group-segmentation-chart', [], 'segmentation', SEGMENTATION_SUMMARY_FIELDS, 'groupSegmentationChart');
    renderGroupSummaryChart('group-classification-chart', [], 'classification', CLASSIFICATION_SUMMARY_FIELDS, 'groupClassificationChart');
    if (title) title.textContent = '批次比較';
    if (subtitle) subtitle.textContent = '';
    return;
  }
  section.classList.remove('hidden');
  if (title) title.textContent = `${group.label} - 批次比較`;
  if (subtitle) subtitle.textContent = `${group.experiments.length} 個實驗 (點選卡片可切換詳情)`;
  initializeGroupChartSelection(group);
  if (grid) {
    grid.innerHTML = '';
    group.experiments.forEach((exp) => {
      const card = document.createElement('div');
      card.className = 'compare-card';
      card.innerHTML = `
        <h4>${escapeHtml(exp.name)}</h4>
        <p>${escapeHtml(exp.relative_path)}</p>
        <div class="compare-metrics">
          <div class="compare-metric"><label>Det IoU</label><span>${formatPercent(previewMetric(exp, 'detection', 'iou_mean'))}</span></div>
          <div class="compare-metric"><label>Det SR@0.5</label><span>${formatPercent(previewMetric(exp, 'detection', 'success_rate_50'))}</span></div>
          <div class="compare-metric"><label>Seg Dice</label><span>${formatPercent(previewMetric(exp, 'segmentation', 'dice_mean'))}</span></div>
          <div class="compare-metric"><label>Seg IoU</label><span>${formatPercent(previewMetric(exp, 'segmentation', 'iou_mean'))}</span></div>
          <div class="compare-metric"><label>Cls Acc</label><span>${formatPercent(previewMetric(exp, 'classification', 'accuracy'))}</span></div>
          <div class="compare-metric"><label>Cls F1</label><span>${formatPercent(previewMetric(exp, 'classification', 'f1_positive'))}</span></div>
          <div class="compare-metric"><label>Cls Brier</label><span>${formatNumber(previewMetric(exp, 'classification', 'brier_score'), 4)}</span></div>
        </div>`;
      card.addEventListener('click', () => selectExperiment(exp.id));
      if (state.current && state.current.id === exp.id) {
        card.classList.add('active');
      }
      grid.appendChild(card);
    });
  }
  if (table) {
    const columns = [
      { label: 'Experiment', key: 'name', format: (exp) => exp.name },
      {
        label: 'Det IoU',
        key: 'det_iou',
        format: (exp) => formatPercentWithStd(previewMetric(exp, 'detection', 'iou_mean'), previewMetric(exp, 'detection', 'iou_std'))
      },
      {
        label: 'Det Center Err',
        key: 'det_ce',
        format: (exp) => formatWithStd(previewMetric(exp, 'detection', 'ce_mean'), previewMetric(exp, 'detection', 'ce_std'))
      },
      {
        label: 'Det SR@0.5',
        key: 'det_sr50',
        format: (exp) => formatPercentWithStd(previewMetric(exp, 'detection', 'success_rate_50'), previewMetric(exp, 'detection', 'success_rate_50_std'))
      },
      {
        label: 'Det FPS',
        key: 'det_fps',
        format: (exp) => formatWithStd(
          previewMetric(exp, 'detection', 'fps_mean') ?? previewMetric(exp, 'detection', 'fps'),
          previewMetric(exp, 'detection', 'fps_std')
        )
      },
      {
        label: 'Seg Dice',
        key: 'seg_dice',
        format: (exp) => formatPercentWithStd(previewMetric(exp, 'segmentation', 'dice_mean'), previewMetric(exp, 'segmentation', 'dice_std'))
      },
      {
        label: 'Seg IoU',
        key: 'seg_iou',
        format: (exp) => formatPercentWithStd(previewMetric(exp, 'segmentation', 'iou_mean'), previewMetric(exp, 'segmentation', 'iou_std'))
      },
      {
        label: 'Seg Centroid',
        key: 'seg_centroid',
        format: (exp) => formatWithStd(previewMetric(exp, 'segmentation', 'centroid_mean'), previewMetric(exp, 'segmentation', 'centroid_std'))
      },
      {
        label: 'Seg FPS',
        key: 'seg_fps',
        format: (exp) => formatWithStd(previewMetric(exp, 'segmentation', 'fps_mean'), previewMetric(exp, 'segmentation', 'fps_std'))
      },
      {
        label: 'Cls Acc',
        key: 'cls_acc',
        format: (exp) => formatPercent(previewMetric(exp, 'classification', 'accuracy'))
      },
      {
        label: 'Cls F1',
        key: 'cls_f1',
        format: (exp) => formatPercent(previewMetric(exp, 'classification', 'f1_positive'))
      },
      {
        label: 'Cls AUC',
        key: 'cls_auc',
        format: (exp) => formatNumber(previewMetric(exp, 'classification', 'roc_auc'), 4)
      },
      {
        label: 'Cls Brier',
        key: 'cls_brier',
        format: (exp) => formatNumber(previewMetric(exp, 'classification', 'brier_score'), 4)
      }
    ];
    const thead = '<tr>' + columns.map((c) => `<th>${c.label}</th>`).join('') + '</tr>';
    const tbody = group.experiments
      .map((exp) => '<tr>' + columns.map((col) => `<td>${col.format(exp)}</td>`).join('') + '</tr>')
      .join('');
    table.innerHTML = thead + tbody;
  }
  renderGroupChartControls(group);
  updateGroupCharts(group);
}

async function loadExperiments(options = {}) {
  const data = await fetchJSON('/api/experiments', { cacheBust: !!options.cacheBust });
  state.experiments = data.experiments;
  state.groupChartSelections = {};
  state.current = null;
  buildGroups();
  renderGroups();
  renderScheduleNotePanel();
  document.getElementById('root-path').textContent = `Results root: ${data.root}`;
  renderGroupOverview();
  const searchInput = document.getElementById('search');
  applySearch(searchInput ? searchInput.value : '');
  showWelcome();
}

async function forceRefreshViewer() {
  const currentExpId = state.currentView?.expId || state.current?.id || null;
  await fetchJSON('/api/experiments/refresh', { method: 'POST', cacheBust: true });
  await loadExperiments({ cacheBust: true });
  if (currentExpId) {
    await selectExperiment(currentExpId, { cacheBust: true });
  }
}

function buildCards(metrics) {
  const cards = document.getElementById('metric-cards');
  cards.innerHTML = '';
  const detection = metrics.detection?.summary;
  const segmentation = metrics.segmentation?.summary;
  const classification = metrics.classification?.summary;
  const tf = metrics.trajectory_filter?.summary;
  const entries = [];
  if (detection) {
    entries.push({ label: 'Detection IoU', value: formatPercentWithStd(detection.iou_mean, detection.iou_std) });
    entries.push({ label: 'Detection FPS', value: formatWithStd(detection.fps || detection.fps_mean, detection.fps_std) });
  }
  if (SHOW_LOCALIZATION_DEBUG && tf) {
    const beforeJ = tf.before?.jitter_cx;
    const afterJ = tf.after?.jitter_cx;
    if (beforeJ != null && afterJ != null) {
      const reduction = ((1 - afterJ / (beforeJ || 1)) * 100).toFixed(1);
      entries.push({ label: 'Jitter 降低', value: `${reduction}%` });
    }
    const afterStab = tf.after?.area_stability;
    if (afterStab != null) {
      entries.push({ label: 'Area Stability', value: formatPercent(afterStab) });
    }
  }
  if (segmentation) {
    entries.push({ label: 'Seg Dice', value: formatPercentWithStd(segmentation.dice_mean, segmentation.dice_std) });
    entries.push({ label: 'Seg FPS', value: formatWithStd(segmentation.fps_mean, segmentation.fps_std) });
  }
  if (classification) {
    entries.push({ label: 'Cls Accuracy', value: formatPercent(classification.accuracy) });
    entries.push({ label: 'Cls F1', value: formatPercent(classification.f1_positive) });
    const aucVal = classification.roc_auc;
    entries.push({ label: 'Cls ROC AUC', value: (aucVal !== undefined && aucVal !== null) ? formatNumber(aucVal, 4) : 'N/A' });
    entries.push({ label: 'Cls Brier', value: formatNumber(classification.brier_score, 4) });
  }
  entries.forEach((item) => {
    const div = document.createElement('div');
    div.className = 'card';
    div.innerHTML = `<h4>${item.label}</h4><span>${item.value}</span>`;
    cards.appendChild(div);
  });
}

function clearVisuals() {
  const det = document.getElementById('detection-gallery');
  const seg = document.getElementById('segmentation-gallery');
  if (det) det.innerHTML = '<p>聚合視圖不提供可視化，請選擇 fold 查看。</p>';
  if (seg) seg.innerHTML = '<p>聚合視圖不提供可視化，請選擇 fold 查看。</p>';
}

function clearPerVideoTablesAndCharts() {
  buildTable('detection-table', [], [{ label: 'Video', format: (row) => row.video }]);
  buildTable('segmentation-table', [], [{ label: 'Video', format: (row) => row.video }]);
  state.detectionChart = buildChart('detection-chart', state.detectionChart, [], []);
  state.segmentationChart = buildChart('segmentation-chart', state.segmentationChart, [], []);
}

async function renderDetailSections(payload, expIdForAssets, options = {}) {
  const cacheBust = !!options.cacheBust;
  // Keep track of which experiment id owns the currently displayed visuals.
  state.assetExpId = expIdForAssets || null;

  const detectionRows = payload.detection?.per_video || [];
  const segmentationRows = payload.segmentation?.per_video || [];

  // Diagnostic logging — helps debug "無資料" issues in detection/segmentation tables
  if (detectionRows.length === 0) {
    console.warn('[Viewer] detection per_video is empty.', 'payload.detection:', payload.detection);
  } else {
    console.log(`[Viewer] Rendering ${detectionRows.length} detection per_video rows.`);
  }

  buildTable('detection-table', detectionRows, [
    { label: 'Video', format: (row) => row.video },
    { label: 'IoU (μ ± σ)', format: (row) => formatPercentWithStd(row.metrics?.iou_mean, row.metrics?.iou_std) },
    { label: 'Center Err (px)', format: (row) => formatWithStd(row.metrics?.ce_mean, row.metrics?.ce_std) },
    { label: 'SR@0.5', format: (row) => formatPercent(row.metrics?.success_rate_50) },
    { label: 'ROI Fallback', format: (row) => formatPercent(row.metrics?.roi_fallback_rate) }
  ]);

  buildTable('segmentation-table', segmentationRows, [
    { label: 'Video', format: (row) => row.video },
    { label: 'Dice (μ ± σ)', format: (row) => formatPercentWithStd(row.metrics?.dice_mean, row.metrics?.dice_std) },
    { label: 'IoU (μ ± σ)', format: (row) => formatPercentWithStd(row.metrics?.iou_mean, row.metrics?.iou_std) },
    { label: 'Centroid (px)', format: (row) => formatWithStd(row.metrics?.centroid_mean, row.metrics?.centroid_std) }
  ]);

  state.detectionChart = buildChart('detection-chart', state.detectionChart, detectionRows, [
    { key: 'iou_mean', label: 'IoU', color: '#6aa5ff', percent: true },
    { key: 'success_rate_50', label: 'SR@0.5', color: '#f8c146', percent: true },
    { key: 'success_rate_75', label: 'SR@0.75', color: '#bd93f9', percent: true },
    { key: 'ce_mean', label: 'Center Err (pix)', color: '#66dfc5', axisId: 'y1' }
  ], {
    axes: {
      y: { percent: true, percentDigits: 2 },
      y1: { position: 'right', beginAtZero: true, grid: { drawOnChartArea: false } }
    }
  });

  state.segmentationChart = buildChart('segmentation-chart', state.segmentationChart, segmentationRows, [
    { key: 'dice_mean', label: 'Dice', color: '#66dfc5', percent: true },
    { key: 'iou_mean', label: 'IoU', color: '#bd93f9', percent: true },
    { key: 'centroid_mean', label: 'Centroid', color: '#ff8ba7', axisId: 'y1' }
  ], {
    axes: {
      y: { percent: true, percentDigits: 2 },
      y1: { position: 'right', beginAtZero: true, grid: { drawOnChartArea: false } }
    }
  });

  if (!expIdForAssets) {
    clearVisuals();
    return;
  }

  await loadGallery('detection-gallery', expIdForAssets, state.detectionCategory, { cacheBust });
  await loadGallery('segmentation-gallery', expIdForAssets, state.segmentationCategory, { cacheBust });
  renderVisualTabs('detection-gallery', 'detectionCategory');
  renderVisualTabs('segmentation-gallery', 'segmentationCategory');
}

async function renderExperimentPayload(payload, expIdForAssets, viewLabel, options = {}) {
  document.getElementById('exp-name').textContent = payload.experiment?.name || viewLabel || expIdForAssets || '';
  document.getElementById('exp-path').textContent = payload.experiment?.output_dir || '';

  // Each section rendering is wrapped in try-catch so that a failure in one
  // section (e.g. classification) does NOT prevent subsequent sections
  // (e.g. detection per-video tables) from rendering.
  try { buildCards(payload); } catch (e) { console.error('[Viewer] buildCards error:', e); }
  try {
    const detTitle = document.querySelector('#detection-summary-table')?.closest('div')?.querySelector('h3');
    if (detTitle) detTitle.textContent = 'Detection 彙總';
  } catch (e) { console.error('[Viewer] detection title error:', e); }
  try { renderSummaryTable('detection-summary-table', payload.detection?.summary, DETECTION_SUMMARY_FIELDS); } catch (e) { console.error('[Viewer] detection summary error:', e); }
  try { renderSummaryTable('segmentation-summary-table', payload.segmentation?.summary, SEGMENTATION_SUMMARY_FIELDS); } catch (e) { console.error('[Viewer] segmentation summary error:', e); }
  try {
    const _fdSec = document.getElementById('filtered-detection-section');
    const _fdData = payload.filtered_detection?.summary;
    if (SHOW_LOCALIZATION_DEBUG && _fdData) {
      renderSummaryTable('filtered-detection-summary-table', _fdData, DETECTION_SUMMARY_FIELDS);
      if (_fdSec) _fdSec.classList.remove('hidden');
    } else {
      if (_fdSec) _fdSec.classList.add('hidden');
    }
  } catch (e) { console.error('[Viewer] filtered detection summary error:', e); }
  try { renderTrajectoryFilterSection(SHOW_LOCALIZATION_DEBUG ? payload.trajectory_filter : null); } catch (e) { console.error('[Viewer] trajectory filter section error:', e); }
  try { renderClassificationSection(payload.classification); } catch (e) { console.error('[Viewer] classification section error:', e); }

  try {
    await renderDetailSections(payload, expIdForAssets, { cacheBust: !!options.cacheBust });
  } catch (e) {
    console.error('[Viewer] renderDetailSections error:', e);
  }
}

function normalizeClassificationPredictions(predictions) {
  const grouped = new Map();
  (predictions || []).forEach((p, index) => {
    const rawVideoId = p.video_id || p.video || p.subject_id || p.entity_id || `video_${index + 1}`;
    const videoId = String(rawVideoId);
    if (!grouped.has(videoId)) {
      grouped.set(videoId, {
        video: videoId,
        label_true: null,
        probs: [],
        preds: [],
        rows: [],
      });
    }
    const row = grouped.get(videoId);
    if (p.label_true !== null && p.label_true !== undefined && row.label_true === null) {
      row.label_true = Number(p.label_true);
    }
    const prob = clamp01(p.prob_positive);
    if (prob !== null) {
      row.probs.push(prob);
    }
    if (p.label_pred !== null && p.label_pred !== undefined) {
      row.preds.push(Number(p.label_pred));
    }
    row.rows.push(p);
  });

  return Array.from(grouped.values())
    .sort((a, b) => a.video.localeCompare(b.video))
    .map((row) => {
      const avgProb = row.probs.length
        ? row.probs.reduce((sum, value) => sum + value, 0) / row.probs.length
        : null;
      const vote = row.preds.length
        ? (row.preds.reduce((acc, value) => acc + value, 0) >= (row.preds.length / 2) ? 1 : 0)
        : (avgProb !== null ? (avgProb >= 0.5 ? 1 : 0) : null);
      const labelTrue = row.label_true;
      const correct = (vote !== null && labelTrue !== null) ? vote === labelTrue : null;
      return {
        id: row.video,
        video: row.video,
        label_true: labelTrue,
        label_pred: vote,
        prob_positive: avgProb,
        correct,
        sample_count: row.rows.length,
      };
    });
}

function classificationMetricsAtThreshold(rows, threshold) {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  (rows || []).forEach((row) => {
    const pred = (row.prob_positive ?? 0) >= threshold ? 1 : 0;
    const truth = Number(row.label_true);
    if (pred === 1 && truth === 1) tp += 1;
    else if (pred === 1 && truth === 0) fp += 1;
    else if (pred === 0 && truth === 1) fn += 1;
    else tn += 1;
  });
  const total = tp + fp + fn + tn;
  const accuracy = total ? (tp + tn) / total : null;
  const precision = (tp + fp) ? tp / (tp + fp) : 0;
  const recall = (tp + fn) ? tp / (tp + fn) : 0;
  const f1 = (precision + recall) ? (2 * precision * recall) / (precision + recall) : 0;
  return { threshold, accuracy, precision, recall, f1, tp, fp, fn, tn, total };
}

function suggestClassificationThreshold(predictions) {
  const rows = normalizeClassificationPredictions(predictions);
  if (!rows.length) return null;
  const uniqueProbs = Array.from(new Set(rows.map((row) => Number(row.prob_positive).toFixed(6)))).map(Number).sort((a, b) => a - b);
  const candidates = new Set([0, 0.5, 1]);
  uniqueProbs.forEach((p) => candidates.add(p));
  for (let i = 0; i < uniqueProbs.length - 1; i += 1) {
    candidates.add((uniqueProbs[i] + uniqueProbs[i + 1]) / 2);
  }
  const evaluated = Array.from(candidates)
    .map((threshold) => classificationMetricsAtThreshold(rows, threshold))
    .sort((a, b) => {
      if ((b.f1 ?? -1) !== (a.f1 ?? -1)) return (b.f1 ?? -1) - (a.f1 ?? -1);
      if ((b.accuracy ?? -1) !== (a.accuracy ?? -1)) return (b.accuracy ?? -1) - (a.accuracy ?? -1);
      return Math.abs(a.threshold - 0.5) - Math.abs(b.threshold - 0.5);
    });
  return {
    best: evaluated[0],
    baseline: classificationMetricsAtThreshold(rows, 0.5),
    rowCount: rows.length,
  };
}

function renderClassificationThresholdSuggestion(predictions, summary) {
  const panel = document.getElementById('classification-threshold-panel');
  const basis = document.getElementById('classification-threshold-basis');
  const value = document.getElementById('classification-threshold-value');
  const sample = document.getElementById('classification-threshold-sample');
  const acc = document.getElementById('classification-threshold-acc');
  const precision = document.getElementById('classification-threshold-precision');
  const recall = document.getElementById('classification-threshold-recall');
  const f1 = document.getElementById('classification-threshold-f1');
  const globalAcc = document.getElementById('classification-threshold-global-acc');
  const globalPrecision = document.getElementById('classification-threshold-global-precision');
  const globalRecall = document.getElementById('classification-threshold-global-recall');
  const globalF1 = document.getElementById('classification-threshold-global-f1');
  const note = document.getElementById('classification-threshold-note');
  if (
    !panel || !basis || !value || !sample || !acc || !precision || !recall || !f1 || !note
    || !globalAcc || !globalPrecision || !globalRecall || !globalF1
  ) return;

  const rows = normalizeClassificationPredictions(predictions);
  const suggestion = suggestClassificationThreshold(predictions);
  if (!suggestion && !summary) {
    panel.classList.add('hidden');
    return;
  }
  panel.classList.remove('hidden');

  // ── 訓練時使用的 Youden / fixed 閾值 ────────────────────────────────────
  const actualRow = document.getElementById('classification-threshold-actual-row');
  const actualBadge = document.getElementById('classification-threshold-actual-method-badge');
  const actualValue = document.getElementById('classification-threshold-actual-value');
  const actualJ = document.getElementById('classification-threshold-actual-j');
  const actualAcc = document.getElementById('classification-threshold-actual-acc');
  const actualPrec = document.getElementById('classification-threshold-actual-precision');
  const actualRec = document.getElementById('classification-threshold-actual-recall');
  const actualF1 = document.getElementById('classification-threshold-actual-f1');

  const trainThreshold = summary?.threshold_used ?? null;
  const trainMethod = summary?.threshold_method ?? null;
  const youdenJ = summary?.youden_j ?? null;
  const nLoo = summary?.threshold_n_loo_predictions ?? null;

  if (actualRow && trainThreshold != null) {
    actualRow.classList.remove('hidden');
    if (actualBadge) {
      actualBadge.textContent = trainMethod === 'youden' ? "Youden's Index (LOO)" : (trainMethod === 'fixed' ? 'Fixed' : trainMethod || '');
      actualBadge.className = 'threshold-method-badge threshold-method-' + (trainMethod || 'fixed');
    }
    if (actualValue) actualValue.textContent = Number(trainThreshold).toFixed(3);
    if (actualJ) {
      const jStr = youdenJ != null ? `J = ${Number(youdenJ).toFixed(3)}` : '';
      const looStr = nLoo != null ? `n_loo = ${nLoo}` : '';
      actualJ.textContent = [jStr, looStr].filter(Boolean).join('  ·  ');
    }
    if (rows.length) {
      const m = classificationMetricsAtThreshold(rows, trainThreshold);
      if (actualAcc) actualAcc.textContent = formatPercent(m.accuracy);
      if (actualPrec) actualPrec.textContent = formatPercent(m.precision);
      if (actualRec) actualRec.textContent = formatPercent(m.recall);
      if (actualF1) actualF1.textContent = formatPercent(m.f1);
    }
  } else if (actualRow) {
    actualRow.classList.add('hidden');
  }

  // ── 事後 Max-F1 建議 ────────────────────────────────────────────────────
  if (!suggestion) {
    basis.textContent = '';
    return;
  }
  basis.textContent = `${suggestion.rowCount} predictions`;
  value.textContent = Number(suggestion.best.threshold).toFixed(3);
  sample.textContent = `Max-F1`;
  acc.textContent = formatPercent(suggestion.best.accuracy);
  precision.textContent = formatPercent(suggestion.best.precision);
  recall.textContent = formatPercent(suggestion.best.recall);
  f1.textContent = formatPercent(suggestion.best.f1);

  // Compare note
  const compareThresh = trainThreshold ?? 0.5;
  const baseline = classificationMetricsAtThreshold(rows, compareThresh);
  const globalBaseline = classificationMetricsAtThreshold(rows, 0.5);
  globalAcc.textContent = formatPercent(globalBaseline.accuracy);
  globalPrecision.textContent = formatPercent(globalBaseline.precision);
  globalRecall.textContent = formatPercent(globalBaseline.recall);
  globalF1.textContent = formatPercent(globalBaseline.f1);
  const baseLabel = trainThreshold != null ? `訓練閾值=${Number(trainThreshold).toFixed(3)}` : 'threshold=0.500';
  note.textContent = `${baseLabel} 時：Acc ${formatPercent(baseline.accuracy)}，Precision ${formatPercent(baseline.precision)}，Recall ${formatPercent(baseline.recall)}，F1 ${formatPercent(baseline.f1)}。以上皆為 pooled predictions 重新切閾值後的結果。`;
}

function renderTrajectoryFilterSection(tf) {
  const section = document.getElementById('trajectory-filter-section');
  if (!section) return;
  const summary = tf?.summary;
  if (!summary || (!summary.before && !summary.after)) {
    section.classList.add('hidden');
    return;
  }
  section.classList.remove('hidden');

  // Render before / after tables
  renderSummaryTable('trajectory-filter-before-table', summary.before, TRAJECTORY_FILTER_FIELDS);
  renderSummaryTable('trajectory-filter-after-table', summary.after, TRAJECTORY_FILTER_FIELDS);

  // Show config details
  const configEl = document.getElementById('trajectory-filter-config');
  if (configEl && summary.config) {
    const strategy = summary.config.bbox_strategy || 'N/A';
    configEl.innerHTML = `<small>BBox Strategy: <strong>${escapeHtml(strategy)}</strong></small>`;
  }

  // Per-video table
  const perVideo = tf?.per_video || [];
  buildTable('trajectory-filter-table', perVideo, [
    { label: 'Model', format: (r) => r.model || '' },
    { label: 'Video', format: (r) => r.video || '' },
    { label: 'Frames', format: (r) => r.frames ?? '' },
    { label: 'Jitter CX (B→A)', format: (r) => `${formatNumber(r.before?.jitter_cx, 2)} → ${formatNumber(r.after?.jitter_cx, 2)}` },
    { label: 'Jitter CY (B→A)', format: (r) => `${formatNumber(r.before?.jitter_cy, 2)} → ${formatNumber(r.after?.jitter_cy, 2)}` },
    { label: 'Area Stab (B→A)', format: (r) => `${formatPercent(r.before?.area_stability)} → ${formatPercent(r.after?.area_stability)}` },
  ]);
}

function renderClassificationSection(classification) {
  const section = document.getElementById('classification-section');
  const thresholdPanel = document.getElementById('classification-threshold-panel');
  const explain = document.getElementById('classification-summary-explain');
  const stats = document.getElementById('classification-video-stats');
  if (!section) return;
  // For LOSO aggregate, use combined (summed) data which has integer TP/FP/FN/TN
  // and overall accuracy computed from all predictions. Fall back to summary for
  // single experiments.
  const combined = classification?.combined;
  const summary = combined || classification?.summary;
  const predictions = classification?.predictions || [];
  state.currentClassificationPredictions = predictions;

  if (!summary) {
    section.classList.add('hidden');
    renderFeatureImportanceSection(null);
    if (thresholdPanel) thresholdPanel.classList.add('hidden');
    if (explain) explain.textContent = '';
    if (stats) stats.textContent = '';
    // Destroy any lingering chart
    if (state.classificationChart && state.classificationChart.destroy) {
      state.classificationChart.destroy();
      state.classificationChart = null;
    }
    return;
  }
  section.classList.remove('hidden');

  const videoRows = normalizeClassificationPredictions(predictions);
  const correctCount = videoRows.filter((row) => row.correct === true).length;
  const wrongCount = videoRows.filter((row) => row.correct === false).length;
  if (stats) {
    const accuracy = videoRows.length ? (correctCount / videoRows.length) : null;
    stats.textContent = videoRows.length
      ? `影片數：${videoRows.length}，正確：${correctCount}，錯誤：${wrongCount}，影片層級準確率：${formatPercent(accuracy)}`
      : '沒有可顯示的影片層級預測。';
  }

  // Ensure specificity is always available for the summary table, even when
  // old summary.json files do not explicitly store it.
  const summaryForDisplay = { ...(summary || {}) };
  if (summaryForDisplay.specificity == null) {
    const tn = Number(summaryForDisplay.tn);
    const fp = Number(summaryForDisplay.fp);
    if (Number.isFinite(tn) && Number.isFinite(fp) && (tn + fp) > 0) {
      summaryForDisplay.specificity = tn / (tn + fp);
    }
  }

  if (explain) {
    const isCombined = Boolean(combined);
    const th = summary?.threshold_used;
    const method = summary?.threshold_method;
    const methodText = method ? String(method) : 'unknown';
    const thText = (th !== null && th !== undefined) ? Number(th).toFixed(3) : 'N/A';
    if (isCombined) {
      explain.textContent = `分類彙總使用 LOSO 各 fold 的原始影片預測結果加總（主報告值），不是用單一閾值重算。Threshold 分析區塊則是把 pooled video predictions 以單一閾值重算（例如訓練閾值平均 ${thText}，method=${methodText}，或固定 0.500）。`;
    } else {
      explain.textContent = `本實驗主報告值來自 summary.json（訓練/推論流程輸出）。Threshold 分析區塊為同一批影片預測的事後單一閾值重算（訓練閾值 ${thText}，method=${methodText}，或固定 0.500）。`;
    }
  }

  // Summary table
  renderSummaryTable('classification-summary-table', summaryForDisplay, CLASSIFICATION_SUMMARY_FIELDS);
  renderFeatureImportanceSection(classification?.feature_importance);
  renderClassificationThresholdSuggestion(predictions, summary);

  // Confusion matrix
  renderConfusionMatrix('confusion-matrix', summaryForDisplay);

  // Predictions table
  renderPredictionsTable('classification-predictions-table', predictions);

  // Probability distribution chart
  state.classificationChart = buildClassificationProbChart(
    'classification-chart',
    state.classificationChart,
    predictions
  );
}

function renderConfusionMatrix(containerId, summary) {
  const container = document.getElementById(containerId);
  if (!container) return;
  // Round to integer (guards against LOSO aggregate float averages)
  const tp = Math.round(summary.tp ?? 0);
  const fp = Math.round(summary.fp ?? 0);
  const fn = Math.round(summary.fn ?? 0);
  const tn = Math.round(summary.tn ?? 0);
  const total = tp + fp + fn + tn || 1;
  container.innerHTML = `
    <table class="cm-table">
      <thead>
        <tr><th></th><th>預測 Positive</th><th>預測 Negative</th></tr>
      </thead>
      <tbody>
        <tr>
          <th>實際 Positive</th>
          <td class="cm-tp">${tp} <small>(${(tp / total * 100).toFixed(1)}%)</small></td>
          <td class="cm-fn">${fn} <small>(${(fn / total * 100).toFixed(1)}%)</small></td>
        </tr>
        <tr>
          <th>實際 Negative</th>
          <td class="cm-fp">${fp} <small>(${(fp / total * 100).toFixed(1)}%)</small></td>
          <td class="cm-tn">${tn} <small>(${(tn / total * 100).toFixed(1)}%)</small></td>
        </tr>
      </tbody>
    </table>`;
}

function renderPredictionsTable(tableId, predictions) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const rows = normalizeClassificationPredictions(predictions);
  if (!rows.length) {
    table.innerHTML = '<tr><td>無預測資料</td></tr>';
    return;
  }
  const header = '<tr><th>影片</th><th>真實標籤</th><th>預測標籤</th><th>P(Positive)</th><th>是否正確</th><th>樣本數</th></tr>';
  const labelMap = { 0: '健康 (0)', 1: '患病 (1)' };
  const body = rows.map((row) => {
    const correct = row.correct === true;
    const predLabel = row.label_pred === null || row.label_pred === undefined ? '—' : (labelMap[row.label_pred] ?? row.label_pred);
    const trueLabel = row.label_true === null || row.label_true === undefined ? '—' : (labelMap[row.label_true] ?? row.label_true);
    const probText = row.prob_positive === null || row.prob_positive === undefined ? '—' : `${(row.prob_positive * 100).toFixed(1)}%`;
    const correctText = row.correct === null ? '—' : (correct ? '✓' : '✗');
    return `<tr class="${correct ? 'pred-correct' : 'pred-wrong'}">
      <td>${escapeHtml(row.video)}</td>
      <td>${trueLabel}</td>
      <td>${predLabel}</td>
      <td>${probText}</td>
      <td>${correctText}</td>
      <td>${row.sample_count}</td>
    </tr>`;
  }).join('');
  table.innerHTML = header + body;
}

function renderFeatureImportanceSection(featureImportance) {
  const panel = document.getElementById('classification-feature-importance');
  const topTable = document.getElementById('classification-fi-top-table');
  const bottomTable = document.getElementById('classification-fi-bottom-table');
  const meta = document.getElementById('classification-fi-meta');
  if (!panel || !topTable || !bottomTable || !meta) return;

  const topRows = featureImportance?.top || [];
  const bottomRows = featureImportance?.bottom || [];
  if (!topRows.length && !bottomRows.length) {
    panel.classList.add('hidden');
    topTable.innerHTML = '';
    bottomTable.innerHTML = '';
    meta.textContent = '';
    return;
  }

  panel.classList.remove('hidden');
  const metaInfo = featureImportance?.meta || {};
  if (metaInfo.mode === 'loso_mean') {
    const used = metaInfo.folds_used ?? '—';
    const total = metaInfo.folds_total ?? '—';
    meta.textContent = `LOSO 平均特徵重要度（folds used: ${used}/${total}）`;
  } else {
    meta.textContent = '單次實驗特徵重要度';
  }

  const renderRows = (rows) => {
    const header = '<tr><th>#</th><th>Feature</th><th>Importance</th></tr>';
    const body = rows.slice(0, 10).map((row, idx) => {
      const feature = escapeHtml(row?.feature ?? '');
      const value = Number(row?.importance ?? 0);
      return `<tr><td>${idx + 1}</td><td>${feature}</td><td>${formatNumber(value, 6)}</td></tr>`;
    }).join('');
    return header + body;
  };

  topTable.innerHTML = topRows.length ? renderRows(topRows) : '<tr><td>無資料</td></tr>';
  bottomTable.innerHTML = bottomRows.length ? renderRows(bottomRows) : '<tr><td>無資料</td></tr>';
}

function buildClassificationProbChart(canvasId, chartRef, predictions) {
  const canvas = document.getElementById(canvasId);
  if (chartRef && chartRef.destroy) {
    chartRef.destroy();
  }
  const rows = normalizeClassificationPredictions(predictions);
  if (!rows.length) return null;
  if (!hasChartJs()) {
    console.warn('[Viewer] Chart.js not available; skip classification probability chart.');
    return null;
  }

  const labels = rows.map((row) => row.video);
  const probs = rows.map((row) => (row.prob_positive === null ? null : row.prob_positive * 100));
  const colors = rows.map((row) => row.correct === false ? '#ff6b6b' : '#51cf66');

  return new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'P(Positive) %',
        data: probs,
        backgroundColor: colors,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `P(Positive): ${ctx.parsed.y.toFixed(1)}%`,
            afterLabel: (ctx) => {
              const row = rows[ctx.dataIndex];
              const trueLabel = row.label_true === 1 ? '患病' : '健康';
              const predLabel = row.label_pred === 1 ? '患病' : '健康';
              const correct = row.correct ? '正確' : '錯誤';
              return `影片：${row.video}\nTrue: ${trueLabel} | Pred: ${predLabel} | ${correct}`;
            }
          }
        },
        annotation: undefined,
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { callback: (v) => `${v}%` },
          title: { display: true, text: '預測機率 (%)' }
        },
        x: {
          title: { display: true, text: '影片' }
        }
      }
    }
  });
}

function buildTable(el, rows, columns) {
  const table = document.getElementById(el);
  if (!rows || rows.length === 0) {
    table.innerHTML = '<tr><td>無資料</td></tr>';
    return;
  }
  const thead = '<tr>' + columns.map((c) => `<th>${c.label}</th>`).join('') + '</tr>';
  const tbody = rows
    .map((row) => {
      const cells = columns.map((col) => `<td>${col.format(row)}</td>`).join('');
      return `<tr>${cells}</tr>`;
    })
    .join('');
  table.innerHTML = thead + tbody;
}

function renderSummaryTable(tableId, summary, fields) {
  const table = document.getElementById(tableId);
  if (!table) return;
  if (!summary) {
    table.innerHTML = '<tr><td>無資料</td></tr>';
    return;
  }
  const rows = fields
    .map((field) => {
      const value = summary[field.key];
      if (value == null && !field.allowNull) return '';
      if (value == null) {
        return `<tr><th>${field.label}</th><td>N/A</td></tr>`;
      }
      const std = field.std ? summary[field.std] : undefined;
      const formatted = field.percent
        ? formatPercentWithStd(value, std)
        : formatWithStd(value, std, field.decimals ?? 3);
      return `<tr><th>${field.label}</th><td>${formatted}</td></tr>`;
    })
    .filter(Boolean)
    .join('');
  table.innerHTML = rows;
}

function renderGroupSummaryChart(canvasId, experiments, sectionKey, fields, chartRefKey) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  if (state[chartRefKey] && state[chartRefKey].destroy) {
    state[chartRefKey].destroy();
    state[chartRefKey] = null;
  }
  const validExperiments = experiments.filter((exp) => exp.preview?.[sectionKey]);
  if (!validExperiments.length) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  if (!hasChartJs()) {
    console.warn('[Viewer] Chart.js not available; skip group summary chart render.');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  const labels = validExperiments.map((exp) => exp.name);
  const chartFields = fields.filter((f) => !f.chartExclude);
  const datasets = chartFields.map((field, idx) => {
    const color = SUMMARY_COLORS[idx % SUMMARY_COLORS.length];
    const stdValues = labels.map((_, i) => {
      const exp = validExperiments[i];
      const rawStd = field.std ? exp.preview?.[sectionKey]?.[field.std] ?? null : null;
      if (rawStd === null || rawStd === undefined) return rawStd;
      return field.percent ? rawStd * 100 : rawStd;
    });
    return {
      label: field.label,
      data: labels.map((_, i) => {
        const val = validExperiments[i].preview?.[sectionKey]?.[field.key] ?? null;
        if (val === null || val === undefined) return val;
        return field.percent ? val * 100 : val;
      }),
      backgroundColor: color,
      borderRadius: 4,
      stdValues,
      percent: !!field.percent,
    };
  });
  state[chartRefKey] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: (context) => {
              const value = context.parsed.y;
              const stdArray = context.dataset.stdValues;
              const std = stdArray ? stdArray[context.dataIndex] : null;
              if (context.dataset.percent) {
                if (std !== null && std !== undefined) {
                  return `${context.dataset.label}: ${formatPercentValue(value, 2)} ± ${formatPercentValue(std, 2)}`;
                }
                return `${context.dataset.label}: ${formatPercentValue(value, 2)}`;
              }
              if (std !== null && std !== undefined) {
                return `${context.dataset.label}: ${formatNumber(value)} ± ${formatNumber(std)}`;
              }
              return `${context.dataset.label}: ${formatNumber(value)}`;
            }
          }
        }
      },
      scales: {
        y: { beginAtZero: true }
      }
    }
  });
}

function buildChart(ctxId, chartRef, rows, metricKeys, options = {}) {
  const ctx = document.getElementById(ctxId);
  if (chartRef && chartRef.destroy) {
    chartRef.destroy();
  }
  if (!rows || rows.length === 0) {
    return null;
  }
  if (!hasChartJs()) {
    console.warn('[Viewer] Chart.js not available; skip chart render.');
    return null;
  }
  const labels = rows.map((r) => r.video);
  const datasets = metricKeys.map((metric) => {
    const values = rows.map((row) => {
      const raw = row.metrics?.[metric.key];
      if (raw === null || raw === undefined) return null;
      return metric.percent ? raw * 100 : raw;
    });
    return {
      label: metric.label,
      data: values,
      borderColor: metric.color,
      backgroundColor: metric.color,
      tension: 0.3,
      fill: false,
      yAxisID: metric.axisId || 'y',
      percent: !!metric.percent
    };
  });
  const axisOptions = options.axes || {};
  const axisIds = new Set([
    ...datasets.map((d) => d.yAxisID || 'y'),
    ...Object.keys(axisOptions)
  ]);
  if (!axisIds.size) {
    axisIds.add('y');
  }
  const scales = {};
  axisIds.forEach((axisId) => {
    const axisConfig = axisOptions[axisId] || {};
    const config = {
      beginAtZero: axisConfig.beginAtZero ?? true,
      position: axisConfig.position,
      grid: axisConfig.grid,
      ticks: axisConfig.ticks ? { ...axisConfig.ticks } : undefined
    };
    if (axisConfig.percent) {
      const digits = axisConfig.percentDigits ?? 0;
      config.ticks = config.ticks || {};
      const originalCallback = config.ticks.callback;
      config.ticks.callback = (val) => {
        const formatted = formatPercentValue(val, digits);
        return originalCallback ? originalCallback(val) : formatted;
      };
    }
    scales[axisId] = config;
  });
  if (!scales.y) {
    scales.y = { beginAtZero: true };
  }
  return new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: (context) => {
              const value = context.parsed.y;
              if (value === null || value === undefined) {
                return `${context.dataset.label}: —`;
              }
              if (context.dataset.percent) {
                return `${context.dataset.label}: ${formatPercentValue(value, 2)}`;
              }
              return `${context.dataset.label}: ${formatNumber(value)}`;
            }
          }
        }
      },
      interaction: { mode: 'index', intersect: false },
      scales
    }
  });
}

function renderVisualTabs(galleryId, categoryStateKey) {
  const gallery = document.getElementById(galleryId);
  if (!gallery) return;
  const tabsContainer = gallery.parentElement?.querySelector('.tabs');
  if (!tabsContainer) return;
  tabsContainer.querySelectorAll('button').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.category === state[categoryStateKey]);
    btn.onclick = async () => {
      // In LOSO aggregate mode we don't have per-fold assets; visuals are only
      // available when a concrete experiment (single or fold) is selected.
      if (state.currentView?.mode === 'aggregate') {
        return;
      }

      state[categoryStateKey] = btn.dataset.category;
      tabsContainer.querySelectorAll('button').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');

      const expIdForAssets = state.assetExpId || state.currentView?.expId || state.current?.id;
      if (!expIdForAssets) {
        return;
      }
      await loadGallery(galleryId, expIdForAssets, state[categoryStateKey]);
    };
  });
}

function slugify(value) {
  if (!value) return 'group';
  return (
    value
      .toString()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'group'
  );
}

const RESERVED_GALLERY_SEGMENTS = [
  'visualizations',
  'visualizations_roi',
  'visualization',
  'overlay',
  'overlays',
  'error',
  'errors',
  'metrics',
  'predictions',
  'detection',
  'segmentation'
];

function deriveGalleryGroup(label) {
  if (!label) return '未命名';
  const parts = label.split('/');
  if (parts.length <= 1) return label;
  const folders = parts.slice(0, -1);
  for (let i = folders.length - 1; i >= 0; i -= 1) {
    const segment = folders[i];
    const normalized = segment.toLowerCase();
    const isReserved = RESERVED_GALLERY_SEGMENTS.some((kw) => normalized.includes(kw));
    if (!isReserved) {
      return segment;
    }
  }
  return folders[folders.length - 1] || '未命名';
}

async function loadGallery(galleryId, expId, category, options = {}) {
  const gallery = document.getElementById(galleryId);
  gallery.innerHTML = '讀取中…';
  try {
    const params = new URLSearchParams({ exp_id: expId, category });
    const data = await fetchJSON(`/api/experiments/visuals?${params.toString()}`, { cacheBust: !!options.cacheBust });
    if (!data.items.length) {
      gallery.innerHTML = '<p>沒有可用的影像。</p>';
      return;
    }
    const grouped = new Map();
    data.items.forEach((item) => {
      const groupName = deriveGalleryGroup(item.label);
      if (!grouped.has(groupName)) {
        grouped.set(groupName, []);
      }
      grouped.get(groupName).push(item);
    });
    const entries = Array.from(grouped.entries());
    if (!entries.length) {
      gallery.innerHTML = '<p>沒有可用的影像。</p>';
      return;
    }
    gallery.innerHTML = '';
    const nav = document.createElement('div');
    nav.className = 'gallery-nav';
    const groupsContainer = document.createElement('div');
    groupsContainer.className = 'gallery-groups';
    gallery.appendChild(nav);
    gallery.appendChild(groupsContainer);
    entries.forEach(([groupName, items], idx) => {
      const groupId = `${galleryId}-${slugify(groupName)}`;
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = groupName;
      if (idx === 0) {
        button.classList.add('active');
      }
      button.addEventListener('click', () => {
        nav.querySelectorAll('button').forEach((btn) => btn.classList.remove('active'));
        button.classList.add('active');
        const target = groupsContainer.querySelector(`[data-group-id="${groupId}"]`);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
      nav.appendChild(button);

      const block = document.createElement('div');
      block.className = 'gallery-group';
      block.dataset.groupId = groupId;
      const heading = document.createElement('h4');
      heading.textContent = groupName;
      block.appendChild(heading);
      const grid = document.createElement('div');
      grid.className = 'gallery-group-grid';
      items.forEach((item) => {
        const figure = document.createElement('figure');
        const link = document.createElement('a');
        link.href = item.url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.title = '點擊於新分頁檢視原始圖';
        const img = document.createElement('img');
        img.src = item.url;
        img.alt = 'visual';
        img.loading = 'lazy';
        link.appendChild(img);
        figure.appendChild(link);
        const caption = document.createElement('figcaption');
        const lines = [item.label];
        if (item.ce_px !== null && item.ce_px !== undefined) {
          lines.push(`CE: ${formatNumber(item.ce_px, 2)} px`);
        }
        caption.textContent = lines.join('\n');
        figure.appendChild(caption);
        grid.appendChild(figure);
      });
      block.appendChild(grid);
      groupsContainer.appendChild(block);
    });
    if (entries.length <= 1) {
      nav.classList.add('hidden');
    }
  } catch (err) {
    gallery.innerHTML = `<p class="error">載入失敗：${err.message}</p>`;
  }
}

// ---------------------------------------------------------------------------
// Voting / modality analysis
// ---------------------------------------------------------------------------

const MODAL_DISPLAY = {
  doppler:   'Doppler',
  grasp:     'Grasp / R-G',
  relax:     'Relax / G-R',
  rest:      'Rest / R1',
  rest_post: 'Rest post / R2',
};
const MODAL_SHORT = {
  doppler: 'D', grasp: 'G', relax: 'Rel', rest: 'R', rest_post: 'Rp',
};
const VOTING_MODAL_MAP = {
  doppler: 'doppler',
  Grasp: 'grasp',
  Relax: 'relax',
  Rest: 'rest',
  'Rest post': 'rest_post',
  D: 'doppler',
  'G-R': 'relax',
  'R-G': 'grasp',
  R1: 'rest',
  R2: 'rest_post',
};
const VOTING_MODALITIES = ['doppler', 'grasp', 'relax', 'rest', 'rest_post'];

function buildSoftVotingAnalysisFromPredictions(predictions) {
  const subjects = new Map();
  (predictions || []).forEach((item) => {
    // Fall back to identity so datasets with arbitrary modality names work too
    const canonical = VOTING_MODAL_MAP[String(item.entity_id || '')] || String(item.entity_id || '');
    const subjectId = item.subject_id || '';
    const prob = clamp01(item.prob_positive);
    if (!canonical || !subjectId || prob === null) return;
    if (!subjects.has(subjectId)) {
      subjects.set(subjectId, {});
    }
    subjects.get(subjectId)[canonical] = {
      prob,
      true: Number(item.label_true),
    };
  });

  // Derive effective modalities from actual data (known ones in order, then rest sorted)
  const seenModalities = new Set();
  Array.from(subjects.values()).forEach((subj) => Object.keys(subj).forEach((m) => seenModalities.add(m)));
  const effectiveModalities = [
    ...VOTING_MODALITIES.filter((m) => seenModalities.has(m)),
    ...[...seenModalities].filter((m) => !VOTING_MODALITIES.includes(m)).sort(),
  ];

  function softVoteResult(modalities) {
    let correct = 0;
    let total = 0;
    const details = [];
    Array.from(subjects.entries()).sort((a, b) => a[0].localeCompare(b[0])).forEach(([subjectId, subj]) => {
      const avail = modalities.filter((m) => subj[m]);
      if (!avail.length) return;
      const probs = avail.map((m) => subj[m].prob).filter((v) => v !== null && v !== undefined);
      if (!probs.length) return;
      const trueLabel = subj[avail[0]].true;
      const meanProbPositive = probs.reduce((a, b) => a + b, 0) / probs.length;
      const vote = meanProbPositive >= 0.5 ? 1 : 0;
      const isCorrect = vote === trueLabel;
      if (isCorrect) correct += 1;
      total += 1;
      details.push({
        subject_id: subjectId,
        true_label: trueLabel,
        vote,
        mean_prob_positive: meanProbPositive,
        total_votes: probs.length,
        correct: isCorrect,
      });
    });
    return {
      correct,
      total,
      accuracy: total ? correct / total : null,
      details,
    };
  }

  const softFive = softVoteResult(effectiveModalities);
  softFive.modalities = [...effectiveModalities];
  // C(N, k) combos where k = min(3, N); skip if N <= combo size
  const comboSize = Math.min(3, effectiveModalities.length);
  const softTopCombos = [];
  if (comboSize >= 2 && effectiveModalities.length > comboSize) {
    for (let i = 0; i < effectiveModalities.length; i += 1) {
      if (comboSize === 2) {
        for (let j = i + 1; j < effectiveModalities.length; j += 1) {
          const combo = [effectiveModalities[i], effectiveModalities[j]];
          const result = softVoteResult(combo);
          result.modalities = combo;
          softTopCombos.push(result);
        }
      } else {
        for (let j = i + 1; j < effectiveModalities.length; j += 1) {
          for (let k = j + 1; k < effectiveModalities.length; k += 1) {
            const combo = [effectiveModalities[i], effectiveModalities[j], effectiveModalities[k]];
            const result = softVoteResult(combo);
            result.modalities = combo;
            softTopCombos.push(result);
          }
        }
      }
    }
  }
  softTopCombos.sort((a, b) => ((b.accuracy ?? 0) - (a.accuracy ?? 0)) || (b.correct - a.correct));
  const softTop10 = softTopCombos.slice(0, 10);
  return {
    soft_five_voting: softFive,
    soft_top_combos: softTop10,
  };
}

function renderVotingResultBlock(container, voting, options = {}) {
  if (!container) return;
  if (!voting) {
    container.innerHTML = '<p>無資料</p>';
    return;
  }
  const pct = voting.accuracy !== null ? `${(voting.accuracy * 100).toFixed(0)}%` : '—';
  const statColor = voting.accuracy === 1 ? '#22c55e' : voting.accuracy === 0 ? '#ef4444' : '#f59e0b';
  const detailRows = (voting.details || []).map((d) => {
    const icon = d.correct ? '✅' : '❌';
    const trueLabel = d.true_label === 1 ? '病患' : '正常';
    const voteLabel = d.vote === 1 ? '病患' : '正常';
    const scoreCell = options.soft
      ? `${((d.mean_prob_positive ?? 0) * 100).toFixed(1)}%`
      : `${d.votes_for_positive} / ${d.total_votes}`;
    const scoreHeader = options.soft ? '平均 P(Positive)' : '正票 / 總票';
    return { row: `<tr>
        <td>${d.subject_id}</td>
        <td>${trueLabel}</td>
        <td style="color:#9ea6c6">${scoreCell}</td>
        <td>${voteLabel}</td>
        <td>${icon}</td>
      </tr>`, scoreHeader };
  });
  const scoreHeader = detailRows[0]?.scoreHeader || (options.soft ? '平均 P(Positive)' : '正票 / 總票');
  container.innerHTML =
    `<p class="voting-stat" style="color:${statColor}">${pct}
      <small>(${voting.correct} / ${voting.total} subjects)</small>
    </p>
    <table class="voting-subject-table">
      <thead><tr><th>Subject</th><th>真實</th><th>${scoreHeader}</th><th>投票結果</th><th></th></tr></thead>
      <tbody>${detailRows.map((d) => d.row).join('')}</tbody>
    </table>`;
}

function renderVotingCombosTable(table, combos, options = {}) {
  if (!table) return;
  const topCombos = (combos || []).slice(0, 10);
  if (!topCombos.length) {
    table.innerHTML = '<tr><td>無資料</td></tr>';
    return;
  }
  const maxComboAcc = Math.max(0.001, ...topCombos.map((c) => c.accuracy ?? 0));
  const detailHeader = options.soft ? '各受試者平均機率' : '各受試者詳情';
  const thead = `<tr><th></th><th>組合</th><th>準確率</th><th>正確 / 總數</th><th>${detailHeader}</th></tr>`;
  const tbody = topCombos.map((c, idx) => {
    const rank = idx + 1;
    const rankClass = rank === 1 ? 'rank-1' : rank === 2 ? 'rank-2' : rank === 3 ? 'rank-3' : 'rank-other';
    const badge = `<span class="rank-badge ${rankClass}">${rank}</span>`;
    const combo = c.modalities.map((m) => MODAL_SHORT[m] || m).join(' + ');
    const acc = c.accuracy !== null ? `${(c.accuracy * 100).toFixed(0)}%` : '—';
    const detail = (c.details || []).map((d) => {
      const icon = d.correct ? '✅' : '❌';
      if (options.soft) {
        return `<span style="margin-right:0.6rem;white-space:nowrap">${d.subject_id}: ${((d.mean_prob_positive ?? 0) * 100).toFixed(1)}% ${icon}</span>`;
      }
      return `<span style="margin-right:0.6rem;white-space:nowrap">${d.subject_id}: ${d.votes_for_positive}/${d.total_votes} ${icon}</span>`;
    }).join('');
    const rowCls = c.accuracy === maxComboAcc ? 'top-rank' : c.accuracy !== null && c.accuracy < 0.6 ? 'mid-rank' : '';
    return `<tr class="${rowCls}">
      <td style="width:2.2rem;padding-right:0.25rem">${badge}</td>
      <td style="font-weight:${rank <= 3 ? 600 : 400}">${combo}</td>
      <td>${acc}</td>
      <td style="color:#9ea6c6">${c.correct} / ${c.total}</td>
      <td style="font-size:0.82rem;color:#7e86a8">${detail}</td>
    </tr>`;
  }).join('');
  table.innerHTML = thead + tbody;
}

function renderVotingSection(data) {
  const section = document.getElementById('voting-section');
  if (!section) return;
  if (!data || !data.per_modality || !data.per_modality.length) {
    section.classList.add('hidden');
    return;
  }
  section.classList.remove('hidden');

  // Hint: which subjects were tested
  const subjects = new Set();
  (data.five_voting?.details || []).forEach((d) => subjects.add(d.subject_id));
  const hint = document.getElementById('voting-subjects-hint');
  if (hint) hint.textContent = `測試受試者：${[...subjects].sort().join('、')}`;

  // ── Per-modality accuracy table ──
  const modTable = document.getElementById('voting-modality-table');
  if (modTable) {
    const maxAcc = Math.max(0.001, ...data.per_modality.map((m) => m.accuracy ?? 0));
    const thead = '<tr><th>模態</th><th>正確 / 總數</th><th>準確率</th></tr>';
    const tbody = data.per_modality.map((m) => {
      const acc = m.accuracy !== null && m.accuracy !== undefined ? m.accuracy : null;
      const pct = acc !== null ? `${(acc * 100).toFixed(0)}%` : '—';
      const barW = acc !== null ? Math.round((acc / maxAcc) * 72) : 0;
      const barHtml = `<div class="voting-bar-cell"><span>${pct}</span><div class="voting-bar" style="width:${barW}px"></div></div>`;
      return `<tr>
        <td>${MODAL_DISPLAY[m.modality] || m.modality}</td>
        <td style="color:#9ea6c6">${m.correct} / ${m.total}</td>
        <td>${barHtml}</td>
      </tr>`;
    }).join('');
    modTable.innerHTML = thead + tbody;
  }

  // ── 5-voting result ──
  const fiveEl = document.getElementById('voting-five-result');
  renderVotingResultBlock(fiveEl, data.five_voting, { soft: false });
  const softFiveEl = document.getElementById('voting-five-soft-result');
  const softFallback = (!data.soft_five_voting || !data.soft_top_combos)
    ? buildSoftVotingAnalysisFromPredictions(state.currentClassificationPredictions)
    : null;
  renderVotingResultBlock(softFiveEl, data.soft_five_voting || softFallback?.soft_five_voting, { soft: true });

  // ── C(5,3) combos table ──
  const combosTable = document.getElementById('voting-combos-table');
  renderVotingCombosTable(combosTable, data.top_combos, { soft: false });
  const softCombosTable = document.getElementById('voting-soft-combos-table');
  renderVotingCombosTable(softCombosTable, data.soft_top_combos || softFallback?.soft_top_combos, { soft: true });
}

async function selectExperiment(expId, options = {}) {
  const target = state.experiments.find((exp) => exp.id === expId);
  if (!target) {
    console.warn('Experiment not found in current index', expId);
    return;
  }
  state.current = target;
  state.currentView = { mode: 'single', expId: expId, label: '' };
  renderScheduleNotePanel();
  renderExperiments();
  hideWelcome();
  resetFoldControls();
  hideLosoOverview();

  // Clear voting section while loading
  const votingSection = document.getElementById('voting-section');
  if (votingSection) votingSection.classList.add('hidden');

  const params = new URLSearchParams({ exp_id: expId });
  const res = await fetchJSON(`/api/experiments/metrics?${params.toString()}`, { cacheBust: !!options.cacheBust });

  // LOSO aggregated experiment: show folds big table + aggregate charts; click fold to drill down.
  if (res && res.mode === 'aggregate' && Array.isArray(res.folds) && res.folds.length) {
    state.currentView = { mode: 'aggregate', expId: expId, label: '' };
    await renderExperimentPayload(res, null, `${res.experiment?.name || expId}`, { cacheBust: !!options.cacheBust });
    clearPerVideoTablesAndCharts();
    showLosoOverview();
    const subtitle = document.getElementById('loso-subtitle');
    if (subtitle) subtitle.textContent = `${res.folds.length} folds | 點選 fold 可載入細節`;
    renderLosoFoldTable(res.folds);

    renderLosoChartControls(expId, res.folds);
    updateLosoCharts(expId, res.folds);

    // Show combined LOSO voting analysis
    if (res.loso_voting) {
      renderVotingSection(res.loso_voting);
    } else if (target.has_classification) {
      // Fallback: fetch voting from the aggregate voting endpoint
      try {
        const vp = new URLSearchParams({ exp_id: expId });
        const vr = await fetchJSON(`/api/experiments/voting?${vp.toString()}`, { cacheBust: !!options.cacheBust });
        renderVotingSection(vr);
      } catch (e) {
        console.warn('Aggregate voting unavailable:', e);
      }
    }

    const table = document.getElementById('loso-fold-table');
    if (table) {
      table.querySelectorAll('tr[data-exp-id]').forEach((row) => {
        row.addEventListener('click', async () => {
          const foldId = row.getAttribute('data-exp-id');
          if (!foldId) return;
          state.currentView = { mode: 'fold', expId: foldId, label: '' };
          const foldParams = new URLSearchParams({ exp_id: foldId });
          const foldRes = await fetchJSON(`/api/experiments/metrics?${foldParams.toString()}`, { cacheBust: !!options.cacheBust });
          // Update per-video tables/charts + visuals for the selected fold
          await renderDetailSections(foldRes, foldId, { cacheBust: !!options.cacheBust });
          // Also update the classification section (confusion matrix / predictions) for this fold
          if (foldRes.classification) {
            renderClassificationSection(foldRes.classification);
          }
          // Load voting for this specific fold
          try {
            const fvp = new URLSearchParams({ exp_id: foldId });
            const fvr = await fetchJSON(`/api/experiments/voting?${fvp.toString()}`, { cacheBust: !!options.cacheBust });
            renderVotingSection(fvr);
          } catch (e) {
            console.warn('Fold voting unavailable:', e);
          }
          const subtitle = document.getElementById('loso-subtitle');
          if (subtitle) subtitle.textContent = `${res.folds.length} folds | 目前檢視：${row.querySelector('td')?.textContent || foldId}`;
        });
      });
    }
  } else {
    await renderExperimentPayload(res, expId, `${res.experiment?.name || expId}`, { cacheBust: !!options.cacheBust });

    // Fetch and render voting analysis for single (non-aggregate) experiments
    if (target.has_classification) {
      try {
        const votingParams2 = new URLSearchParams({ exp_id: expId });
        const votingRes = await fetchJSON(`/api/experiments/voting?${votingParams2.toString()}`, { cacheBust: !!options.cacheBust });
        renderVotingSection(votingRes);
      } catch (e) {
        console.warn('Voting analysis unavailable:', e);
        if (votingSection) votingSection.classList.add('hidden');
      }
    }
  }

  renderGroupOverview();
}

async function init() {
  ensureScheduleNotePanel();
  setupDownloadButtons();
  setupDataExportButtons();
  await loadExperiments();
  const search = document.getElementById('search');
  if (search) {
    search.addEventListener('input', (e) => applySearch(e.target.value));
  }
  const noteInput = document.getElementById('schedule-note-input');
  if (noteInput) {
    noteInput.addEventListener('input', () => {
      queueScheduleNoteAutosave();
    });
    noteInput.addEventListener('blur', () => {
      if (state.noteAutosaveTimer) {
        clearTimeout(state.noteAutosaveTimer);
        state.noteAutosaveTimer = null;
      }
      saveCurrentScheduleNote();
    });
    noteInput.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        saveCurrentScheduleNote();
      }
    });
  }
  const refreshBtn = document.getElementById('refresh-btn');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', async () => {
      await fetchJSON('/api/experiments/refresh', { method: 'POST' });
      await loadExperiments({ cacheBust: true });
    });
  }

  const forceRefreshBtn = document.getElementById('force-refresh-btn');
  if (forceRefreshBtn) {
    forceRefreshBtn.addEventListener('click', async () => {
      forceRefreshBtn.disabled = true;
      const originalText = forceRefreshBtn.textContent;
      forceRefreshBtn.textContent = '刷新中…';
      try {
        await forceRefreshViewer();
      } finally {
        forceRefreshBtn.disabled = false;
        forceRefreshBtn.textContent = originalText;
      }
    });
  }
}

init().catch((err) => {
  console.error(err);
  alert('初始化失敗：' + err.message);
});
