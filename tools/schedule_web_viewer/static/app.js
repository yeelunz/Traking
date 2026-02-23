const STANDALONE_GROUP_KEY = '__standalone__';

const state = {
  experiments: [],
  filtered: [],
  groups: [],
  selectedGroup: null,
  searchTerm: '',
  current: null,
  currentView: { mode: 'single', expId: null, label: '' },
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
  { label: 'F1', key: 'f1_positive', percent: true },
  { label: 'ROC AUC', key: 'roc_auc' }
];

const SUMMARY_COLORS = ['#6aa5ff', '#f8c146', '#66dfc5', '#ff8ba7', '#bd93f9', '#94f7c5'];

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json();
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
    li.innerHTML = `<strong>${exp.name}${foldSuffix}</strong><br /><small>${exp.relative_path}</small>`;
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

function applySearch(term) {
  state.searchTerm = term;
  const activeGroup = state.groups.find((g) => g.key === state.selectedGroup);
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
  return key.replace(/ /g, '').split('/').filter(Boolean).join(' / ');
}

function buildGroups() {
  const buckets = new Map();
  state.experiments.forEach((exp) => {
    const key = exp.group_path ? exp.group_path : STANDALONE_GROUP_KEY;
    if (!buckets.has(key)) {
      buckets.set(key, { key, label: normalizeGroupLabel(key), experiments: [] });
    }
    buckets.get(key).experiments.push(exp);
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
      li.textContent = `${group.label} (${group.experiments.length})`;
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
    { label: 'Cls AUC', format: (f) => formatNumber(foldPreviewMetric(f, 'classification', 'roc_auc'), 4) }
  ];
  const thead = '<tr>' + columns.map((c) => `<th>${c.label}</th>`).join('') + '</tr>';
  const tbody = folds
    .map((fold) => {
      const cells = columns.map((c) => `<td>${c.format(fold)}</td>`).join('');
      return `<tr data-exp-id="${fold.exp_id}">${cells}</tr>`;
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
        <h4>${exp.name}</h4>
        <p>${exp.relative_path}</p>
        <div class="compare-metrics">
          <div class="compare-metric"><label>Det IoU</label><span>${formatPercent(previewMetric(exp, 'detection', 'iou_mean'))}</span></div>
          <div class="compare-metric"><label>Det SR@0.5</label><span>${formatPercent(previewMetric(exp, 'detection', 'success_rate_50'))}</span></div>
          <div class="compare-metric"><label>Seg Dice</label><span>${formatPercent(previewMetric(exp, 'segmentation', 'dice_mean'))}</span></div>
          <div class="compare-metric"><label>Seg IoU</label><span>${formatPercent(previewMetric(exp, 'segmentation', 'iou_mean'))}</span></div>
          <div class="compare-metric"><label>Cls Acc</label><span>${formatPercent(previewMetric(exp, 'classification', 'accuracy'))}</span></div>
          <div class="compare-metric"><label>Cls F1</label><span>${formatPercent(previewMetric(exp, 'classification', 'f1_positive'))}</span></div>
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

async function loadExperiments() {
  const data = await fetchJSON('/api/experiments');
  state.experiments = data.experiments;
  state.groupChartSelections = {};
  state.current = null;
  buildGroups();
  renderGroups();
  document.getElementById('root-path').textContent = `Results root: ${data.root}`;
  renderGroupOverview();
  const searchInput = document.getElementById('search');
  applySearch(searchInput ? searchInput.value : '');
  showWelcome();
}

function buildCards(metrics) {
  const cards = document.getElementById('metric-cards');
  cards.innerHTML = '';
  const detection = metrics.detection?.summary;
  const segmentation = metrics.segmentation?.summary;
  const classification = metrics.classification?.summary;
  const entries = [];
  if (detection) {
    entries.push({ label: 'Detection IoU', value: formatPercentWithStd(detection.iou_mean, detection.iou_std) });
    entries.push({ label: 'Detection FPS', value: formatWithStd(detection.fps || detection.fps_mean, detection.fps_std) });
  }
  if (segmentation) {
    entries.push({ label: 'Seg Dice', value: formatPercentWithStd(segmentation.dice_mean, segmentation.dice_std) });
    entries.push({ label: 'Seg FPS', value: formatWithStd(segmentation.fps_mean, segmentation.fps_std) });
  }
  if (classification) {
    entries.push({ label: 'Cls Accuracy', value: formatPercent(classification.accuracy) });
    entries.push({ label: 'Cls F1', value: formatPercent(classification.f1_positive) });
    if (classification.roc_auc !== undefined && classification.roc_auc !== null) {
      entries.push({ label: 'Cls ROC AUC', value: formatNumber(classification.roc_auc, 4) });
    }
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

async function renderDetailSections(payload, expIdForAssets) {
  // Keep track of which experiment id owns the currently displayed visuals.
  state.assetExpId = expIdForAssets || null;

  const detectionRows = payload.detection?.per_video || [];
  const segmentationRows = payload.segmentation?.per_video || [];

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

  await loadGallery('detection-gallery', expIdForAssets, state.detectionCategory);
  await loadGallery('segmentation-gallery', expIdForAssets, state.segmentationCategory);
  renderVisualTabs('detection-gallery', 'detectionCategory');
  renderVisualTabs('segmentation-gallery', 'segmentationCategory');
}

async function renderExperimentPayload(payload, expIdForAssets, viewLabel) {
  document.getElementById('exp-name').textContent = payload.experiment?.name || viewLabel || expIdForAssets || '';
  document.getElementById('exp-path').textContent = payload.experiment?.output_dir || '';
  buildCards(payload);
  renderSummaryTable('detection-summary-table', payload.detection?.summary, DETECTION_SUMMARY_FIELDS);
  renderSummaryTable('segmentation-summary-table', payload.segmentation?.summary, SEGMENTATION_SUMMARY_FIELDS);
  renderClassificationSection(payload.classification);
  await renderDetailSections(payload, expIdForAssets);
}

function renderClassificationSection(classification) {
  const section = document.getElementById('classification-section');
  if (!section) return;
  const summary = classification?.summary;
  const predictions = classification?.predictions || [];

  if (!summary) {
    section.classList.add('hidden');
    // Destroy any lingering chart
    if (state.classificationChart && state.classificationChart.destroy) {
      state.classificationChart.destroy();
      state.classificationChart = null;
    }
    return;
  }
  section.classList.remove('hidden');

  // Summary table
  renderSummaryTable('classification-summary-table', summary, CLASSIFICATION_SUMMARY_FIELDS);

  // Confusion matrix
  renderConfusionMatrix('confusion-matrix', summary);

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
  const tp = summary.tp ?? 0;
  const fp = summary.fp ?? 0;
  const fn = summary.fn ?? 0;
  const tn = summary.tn ?? 0;
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
  if (!predictions || predictions.length === 0) {
    table.innerHTML = '<tr><td>無預測資料</td></tr>';
    return;
  }
  // Group predictions by subject
  const bySubject = new Map();
  predictions.forEach((p) => {
    const sid = p.subject_id || p.entity_id;
    if (!bySubject.has(sid)) {
      bySubject.set(sid, { subject_id: sid, label_true: p.label_true, preds: [] });
    }
    bySubject.get(sid).preds.push(p);
  });
  const subjects = Array.from(bySubject.values());
  subjects.sort((a, b) => (a.subject_id || '').localeCompare(b.subject_id || ''));
  const header = '<tr><th>Subject</th><th>True Label</th><th>Pred Label</th><th>Prob (Positive)</th><th>Correct</th></tr>';
  const rows = subjects.map((s) => {
    // For each subject use majority vote or first prediction
    const majorityPred = s.preds.length === 1
      ? s.preds[0].label_pred
      : (() => {
          const counts = {};
          s.preds.forEach((p) => { counts[p.label_pred] = (counts[p.label_pred] || 0) + 1; });
          return Number(Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]);
        })();
    const avgProb = s.preds.reduce((sum, p) => sum + (p.prob_positive || 0), 0) / s.preds.length;
    const correct = majorityPred === s.label_true;
    const labelMap = { 0: '健康 (0)', 1: '患病 (1)' };
    return `<tr class="${correct ? 'pred-correct' : 'pred-wrong'}">
      <td>${s.subject_id}</td>
      <td>${labelMap[s.label_true] ?? s.label_true}</td>
      <td>${labelMap[majorityPred] ?? majorityPred}</td>
      <td>${(avgProb * 100).toFixed(1)}%</td>
      <td>${correct ? '✓' : '✗'}</td>
    </tr>`;
  }).join('');
  table.innerHTML = header + rows;
}

function buildClassificationProbChart(canvasId, chartRef, predictions) {
  const canvas = document.getElementById(canvasId);
  if (chartRef && chartRef.destroy) {
    chartRef.destroy();
  }
  if (!predictions || predictions.length === 0) return null;

  // Group by subject and compute avg probability
  const bySubject = new Map();
  predictions.forEach((p) => {
    const sid = p.subject_id || p.entity_id;
    if (!bySubject.has(sid)) {
      bySubject.set(sid, { subject_id: sid, label_true: p.label_true, probs: [] });
    }
    bySubject.get(sid).probs.push(p.prob_positive || 0);
  });
  const subjects = Array.from(bySubject.values()).sort((a, b) =>
    (a.subject_id || '').localeCompare(b.subject_id || '')
  );
  const labels = subjects.map((s) => s.subject_id);
  const probs = subjects.map((s) => {
    const avg = s.probs.reduce((a, b) => a + b, 0) / s.probs.length;
    return avg * 100;
  });
  const colors = subjects.map((s) => s.label_true === 1 ? '#ff6b6b' : '#51cf66');

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
            afterLabel: (ctx) => `True: ${subjects[ctx.dataIndex].label_true === 1 ? '患病' : '健康'}`
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
          title: { display: true, text: 'Subject' }
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
      const std = field.std ? summary[field.std] : undefined;
      const formatted = field.percent ? formatPercentWithStd(value, std) : formatWithStd(value, std);
      return `<tr><th>${field.label}</th><td>${formatted}</td></tr>`;
    })
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
  const labels = validExperiments.map((exp) => exp.name);
  const datasets = fields.map((field, idx) => {
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
    btn.addEventListener('click', async () => {
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
    });
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

async function loadGallery(galleryId, expId, category) {
  const gallery = document.getElementById(galleryId);
  gallery.innerHTML = '讀取中…';
  try {
    const params = new URLSearchParams({ exp_id: expId, category });
    const data = await fetchJSON(`/api/experiments/visuals?${params.toString()}`);
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
        link.innerHTML = `<img src="${item.url}" alt="visual" loading="lazy" />`;
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

async function selectExperiment(expId) {
  const target = state.experiments.find((exp) => exp.id === expId);
  if (!target) {
    console.warn('Experiment not found in current index', expId);
    return;
  }
  state.current = target;
  state.currentView = { mode: 'single', expId: expId, label: '' };
  renderExperiments();
  hideWelcome();
  resetFoldControls();
  hideLosoOverview();

  const params = new URLSearchParams({ exp_id: expId });
  const res = await fetchJSON(`/api/experiments/metrics?${params.toString()}`);

  // LOSO aggregated experiment: show folds big table + aggregate charts; click fold to drill down.
  if (res && res.mode === 'aggregate' && Array.isArray(res.folds) && res.folds.length) {
    state.currentView = { mode: 'aggregate', expId: expId, label: '' };
    await renderExperimentPayload(res, null, `${res.experiment?.name || expId}`);
    clearPerVideoTablesAndCharts();
    showLosoOverview();
    const subtitle = document.getElementById('loso-subtitle');
    if (subtitle) subtitle.textContent = `${res.folds.length} folds | 點選 fold 可載入細節`;
    renderLosoFoldTable(res.folds);

    renderLosoChartControls(expId, res.folds);
    updateLosoCharts(expId, res.folds);

    const table = document.getElementById('loso-fold-table');
    if (table) {
      table.querySelectorAll('tr[data-exp-id]').forEach((row) => {
        row.addEventListener('click', async () => {
          const foldId = row.getAttribute('data-exp-id');
          if (!foldId) return;
          state.currentView = { mode: 'fold', expId: foldId, label: '' };
          const foldParams = new URLSearchParams({ exp_id: foldId });
          const foldRes = await fetchJSON(`/api/experiments/metrics?${foldParams.toString()}`);
          // Keep aggregate summary visible; only update per-video tables/charts + visuals for this fold.
          await renderDetailSections(foldRes, foldId);
          const subtitle = document.getElementById('loso-subtitle');
          if (subtitle) subtitle.textContent = `${res.folds.length} folds | 目前檢視：${row.querySelector('td')?.textContent || foldId}`;
        });
      });
    }
  } else {
    await renderExperimentPayload(res, expId, `${res.experiment?.name || expId}`);
  }

  renderGroupOverview();
}

async function init() {
  setupDownloadButtons();
  setupDataExportButtons();
  await loadExperiments();
  document.getElementById('search').addEventListener('input', (e) => applySearch(e.target.value));
  document.getElementById('refresh-btn').addEventListener('click', async () => {
    await fetchJSON('/api/experiments/refresh', { method: 'POST' });
    await loadExperiments();
  });
}

init().catch((err) => {
  console.error(err);
  alert('初始化失敗：' + err.message);
});
