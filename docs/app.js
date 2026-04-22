'use strict'

// ─── Category → color mapping ────────────────────────────────────────────────
// Labels sourced from YOLO models (see labels_t3.json and labels_t1.json)
// See: https://huggingface.co/NMFS-OSI/yolo11m-cls-noaa-pacific-benthic-cover-t3

const T3_DESCRIPTIONS = {
  ACAS:'Acanthastrea sp', ACBR:'Acropora (branching)', ACTA:'Acropora (tabular)',
  ASPP:'Asparagopsis sp', ASSP:'Astreopora sp', ASTS:'Astrea spp',
  BGMA:'Blue-green macroalga', BRMA:'Brown macroalgae', CAUL:'Caulerpa sp',
  CCAH:'Crustose Coralline Algae (Healthy)', CCAR:'Crustose Coralline Algae (Rubble)',
  CMOR:'Corallimorph', COSP:'Coscinaraea sp', CYPS:'Cyphastrea sp',
  DICO:'Dictyota spp.', DICT:'Dictyosphaeria sp', DISP:'Diploastrea sp',
  ECHP:'Echinopora sp', EMA:'Encrusting Macroalgae', ENC:'Encrusting hard coral',
  FASP:'Favites spp.', FAVS:'Favites sp', FINE:'Fine sediment',
  FOL:'Foliose hard coral', FREE:'Free-living hard coral', FUSP:'Fungia spp.',
  GASP:'Galaxea sp', GOAL:'Goniopora/Alveopora sp', GONS:'Goniastrea sp',
  GRMA:'Green macroalgae', HALI:'Halimeda spp.', HCOE:'Heliopora sp',
  HYSP:'Hydnophora sp', ISSP:'Isopora sp', LEPT:'Leptastrea spp.',
  LOBO:'Lobophora spp.', LOBS:'Lobophyllia spp.', LPHY:'Leptoria sp',
  MICR:'Microdictyon sp', MISP:'Millepora sp', MOBF:'Mobile fauna',
  MOBR:'Montipora (branching)', MOEN:'Montipora (encrusting)', MOFO:'Montipora foliose',
  OCTO:'Octocoral', PADI:'Padina sp', PAEN:'Pavona encrusting',
  PAMA:'Pavona massive', PESP:'Porites (encrusting)', PHSP:'Phymastrea sp',
  PLSP:'Platygyra spp.', POBR:'Porites (branching)', POCS:'Pocillopora spp.',
  POEN:'Porites encrusting', POFO:'Porites foliose', POMA:'Porites (massive)',
  PSSP:'Psammocora sp', RDMA:'Red macroalgae', SAND:'Sand',
  SP:'Sponge', STYS:'Stylophora sp', TUN:'Tunicate',
  TURFH:'Turf Algae (High)', TURFR:'Turf Algae (Rubble)', TURS:'Turbinaria sp',
  UPMA:'Upright macroalga', ZO:'Zoanthid',
}

const T3_CATEGORY = {
  // Hard Corals (38 codes)
  ACAS:'coral', ACBR:'coral', ACTA:'coral', ASSP:'coral', ASTS:'coral',
  COSP:'coral', CYPS:'coral', DISP:'coral', ECHP:'coral', ENC:'coral',
  FASP:'coral', FAVS:'coral', FOL:'coral', FREE:'coral', FUSP:'coral',
  GASP:'coral', GOAL:'coral', GONS:'coral', HYSP:'coral', ISSP:'coral',
  LEPT:'coral', LOBS:'coral', LPHY:'coral', MOBR:'coral', MOEN:'coral',
  MOFO:'coral', PAEN:'coral', PAMA:'coral', PESP:'coral', PHSP:'coral',
  PLSP:'coral', POBR:'coral', POCS:'coral', POEN:'coral', POFO:'coral',
  POMA:'coral', PSSP:'coral', STYS:'coral',
  // Crustose Coralline Algae (2 codes)
  CCAH:'cca', CCAR:'cca',
  // Turf Algae (2 codes)
  TURFH:'turf', TURFR:'turf',
  // Macroalgae (15 codes)
  ASPP:'macro', BGMA:'macro', BRMA:'macro', CAUL:'macro', DICO:'macro',
  DICT:'macro', EMA:'macro', GRMA:'macro', HALI:'macro', LOBO:'macro',
  PADI:'macro', RDMA:'macro', TURS:'macro', UPMA:'macro', MICR:'macro',
  // Soft Corals & Cnidarians (5 codes)
  CMOR:'soft', HCOE:'soft', MISP:'soft', OCTO:'soft', ZO:'soft',
  // Sediment (2 codes)
  FINE:'sed', SAND:'sed',
  // Other (5 codes)
  MOBF:'other', SP:'other', TUN:'other',
}

const CAT_COLOR = {
  coral: '#f97316', cca: '#a855f7', turf: '#a3e635', macro: '#22c55e',
  soft:  '#ec4899', sed: '#f3f707', other: '#60a5fa',
}

// T1 (8 broad functional groups) map to the same color categories as T3
const T1_CATEGORY = {
  CORAL:'coral', CCA:'cca', TURF:'turf', MA:'macro',
  SC:'soft', SED:'sed', MF:'other', I:'other',
}

// Human-readable names for T1 codes (used in label picker)
const T1_DESCRIPTIONS = {
  CCA:'Crustose Coralline Algae', CORAL:'Hard Coral', I:'Sessile Invertebrate',
  MA:'Macroalgae', MF:'Mobile Fauna', SC:'Soft Coral', SED:'Sediment', TURF:'Turf Algae',
}

function getPointColor(point) {
  const ann = point.annotations?.[0]
  if (!ann) return '#6b7280'
  if (ann.is_confirmed) return '#ffffff'
  const cat = T3_CATEGORY[ann.code] ?? T1_CATEGORY[ann.code]
  return CAT_COLOR[cat] ?? '#60a5fa'
}

function isConfirmed(p)    { return !!p.annotations?.[0]?.is_confirmed }
function isUnclassified(p) { return !p.annotations?.length }

// ─── ONNX Inference ──────────────────────────────────────────────────────────

const PATCH_SIZE = 112   // pixels to crop around each grid point
const IMGSZ      = 224   // ONNX model input resolution

const MODEL_URLS = {
  t3: './models/yolo11m_cls_noaa-pacific-benthic-t3.onnx',
  t1: './models/yolo11m_cls_noaa-pacific-benthic-t1.onnx',
}
const LABEL_URLS = {
  t3: './labels_t3.json',
  t1: './labels_t1.json',
}

// Stores in-flight / resolved promises — prevents duplicate concurrent loads
const _sessionPromises = {}
const _labelsCache = {}
let   _ortReady = false
let   _gpuBackend = 'cpu'   // 'gpu' | 'cpu' — set after first session loads

// Probe whether the browser supports WebGPU
async function _webGpuAvailable() {
  try {
    if (!navigator.gpu) return false
    const adapter = await navigator.gpu.requestAdapter()
    return !!adapter
  } catch { return false }
}

const MODEL_CACHE_NAME = 'kitcat-models-v1'

async function getSession(key) {
  if (_sessionPromises[key]) return _sessionPromises[key]
  // Store promise immediately so concurrent callers share the same load
  _sessionPromises[key] = _loadSession(key).catch(err => {
    delete _sessionPromises[key]  // allow retry if load fails
    throw err
  })
  return _sessionPromises[key]
}

async function _loadSession(key) {
  const url = MODEL_URLS[key]
  let arrayBuffer

  // Try browser Cache API — avoids re-downloading ~42 MB on every visit
  // Probe WebGPU once (first call only) and select execution provider
  const useGpu = await _webGpuAvailable()
  if (_gpuBackend !== 'gpu' && useGpu) _gpuBackend = 'gpu'

  const fromCache = await (async () => {
    try {
      const cache = await caches.open(MODEL_CACHE_NAME)
      return (await cache.match(url)) || null
    } catch { return null }
  })()

  if (fromCache) {
    setModelStatus('loading', `Loading ${key.toUpperCase()} model from browser cache…`)
    arrayBuffer = await fromCache.arrayBuffer()
  } else {
    setModelStatus('loading', `Fetching ${key.toUpperCase()} model from server (~42 MB) — cached after first load…`)
    const res = await fetch(url)
    if (!res.ok) throw new Error(`HTTP ${res.status} — model not found at ${url} (has the GitHub Actions deploy run?)`)
    const clone = res.clone()
    arrayBuffer = await res.arrayBuffer()
    // Store in cache in background — don't block inference startup
    caches.open(MODEL_CACHE_NAME)
      .then(c => c.put(url, clone))
      .catch(() => {})  // silently skip if Cache API unavailable (e.g. private browsing)
  }

  const backendLabel = useGpu ? 'GPU (WebGPU)' : 'CPU (WASM)'
  setModelStatus('loading', `Initialising ${key.toUpperCase()} model on ${backendLabel}…`)
  const providers = useGpu ? ['webgpu', 'wasm'] : ['wasm']
  const session = await ort.InferenceSession.create(arrayBuffer, {
    executionProviders: providers,
    graphOptimizationLevel: 'all',
  })
  // First-inference GPU warm-up notice
  if (useGpu) {
    setModelStatus('loading', `Warming up GPU shaders for ${key.toUpperCase()}…`)
    // Run a tiny dummy inference to trigger shader compilation before real images arrive
    try {
      const dummy = new ort.Tensor('float32', new Float32Array(3 * IMGSZ * IMGSZ), [1, 3, IMGSZ, IMGSZ])
      await session.run({ [session.inputNames[0]]: dummy })
    } catch { /* warm-up is best-effort */ }
  }
  return session
}

async function getLabelCodes(key) {
  if (_labelsCache[key]) return _labelsCache[key]
  const res = await fetch(LABEL_URLS[key])
  if (!res.ok) throw new Error(`labels_${key}.json not found — run scripts/export_onnx.py first`)
  _labelsCache[key] = await res.json()
  return _labelsCache[key]
}

// Crop patch, resize to 224×224, apply ImageNet normalisation → NCHW float32
function preprocessPatch(imgEl, cx, cy, patchSize) {
  const half = patchSize >> 1
  const sx = Math.max(0, cx - half)
  const sy = Math.max(0, cy - half)
  const sw = Math.min(imgEl.naturalWidth  - sx, patchSize)
  const sh = Math.min(imgEl.naturalHeight - sy, patchSize)

  const cv  = document.createElement('canvas')
  cv.width  = cv.height = IMGSZ
  const ctx = cv.getContext('2d')
  ctx.drawImage(imgEl, sx, sy, sw, sh, 0, 0, IMGSZ, IMGSZ)

  const { data } = ctx.getImageData(0, 0, IMGSZ, IMGSZ)
  const np  = IMGSZ * IMGSZ
  const buf = new Float32Array(3 * np)
  for (let i = 0; i < np; i++) {
    buf[i]        = data[i * 4]     / 255  // R
    buf[np  + i]  = data[i * 4 + 1] / 255  // G
    buf[np*2 + i] = data[i * 4 + 2] / 255  // B
  }
  return new ort.Tensor('float32', buf, [1, 3, IMGSZ, IMGSZ])
}

async function classifyPatch(imgEl, cx, cy, key, patchSize) {
  const [session, labels] = await Promise.all([getSession(key), getLabelCodes(key)])
  const tensor = preprocessPatch(imgEl, cx, cy, patchSize)
  const feeds  = { [session.inputNames[0]]: tensor }
  const output = await session.run(feeds)
  const probs  = Array.from(output[session.outputNames[0]].data)
  return probs
    .map((v, i) => ({ code: labels[i], score: v }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
}

// ─── Grid generation ─────────────────────────────────────────────────────────

function generateGrid(imgW, imgH, nRows, nCols, patchSize) {
  const margin = patchSize >> 1
  const xs = nCols > 1
    ? Array.from({ length: nCols }, (_, c) => Math.round(margin + c * (imgW - 2 * margin) / (nCols - 1)))
    : [Math.round(imgW / 2)]
  const ys = nRows > 1
    ? Array.from({ length: nRows }, (_, r) => Math.round(margin + r * (imgH - 2 * margin) / (nRows - 1)))
    : [Math.round(imgH / 2)]
  const pts = []
  for (const y of ys) for (const x of xs)
    pts.push({ id: crypto.randomUUID(), row: y, column: x, annotations: [] })
  return pts
}

// NOAA stratified random: divide into cellRows x cellCols cells, 1 random point per cell
function generateStratifiedRandom(imgW, imgH, cellRows, cellCols, patchSize) {
  const margin  = patchSize >> 1
  const usableW = imgW - 2 * margin
  const usableH = imgH - 2 * margin
  const cellW   = usableW / cellCols
  const cellH   = usableH / cellRows
  const pts = []
  for (let r = 0; r < cellRows; r++) {
    for (let c = 0; c < cellCols; c++) {
      const x = Math.round(margin + c * cellW + Math.random() * cellW)
      const y = Math.round(margin + r * cellH + Math.random() * cellH)
      pts.push({ id: crypto.randomUUID(), row: y, column: x, annotations: [] })
    }
  }
  return pts
}

// ─── Thumbnail ───────────────────────────────────────────────────────────────

function makeThumbnail(imgEl, size = 80) {
  const r   = imgEl.naturalWidth / imgEl.naturalHeight
  const cv  = document.createElement('canvas')
  cv.width  = r >= 1 ? size : Math.round(size * r)
  cv.height = r >= 1 ? Math.round(size / r) : size
  cv.getContext('2d').drawImage(imgEl, 0, 0, cv.width, cv.height)
  return cv.toDataURL('image/jpeg', 0.75)
}

// ─── State ───────────────────────────────────────────────────────────────────

const state = {
  images:        [],
  currentId:     null,
  record:        null,
  selectedIdx:   -1,
  hoverIdx:      -1,
  isDirty:       false,
  autoAdvance:   true,
  uploadSettings: { model: 't1', rows: 10, cols: 10, gridMethod: 'noaa', pointPlacement: 'uniform', patchSize: 112 },
  filterState: {
    visibility: 'all',
    labelCodes: new Set(),
    minConf:    0,
    sortBy:     'position',
  },
  allLabels:    [],
  customLabels: [],
  zoom:  1.0,
  panX:  0,
  panY:  0,
  loadedImg:  null,
  _panning:   false,
  _panStart:  null,
  _zoomLocked: false,
  overlayMode: 'overview',
  classifyingIds: new Set(),
}

// ─── DOM refs ────────────────────────────────────────────────────────────────

const $sidebar        = document.getElementById('sidebar')
const $imageList      = document.getElementById('image-list')
const $imageCanvas    = document.getElementById('image-canvas')
const $overlayCanvas  = document.getElementById('overlay-canvas')
const $container      = document.getElementById('canvas-container')
const $placeholder    = document.getElementById('canvas-placeholder')
const $detail         = document.getElementById('point-detail')
const $progressText   = document.getElementById('progress-text')
const $statusDot          = document.getElementById('model-status-dot')
const $statusText         = document.getElementById('model-status-text')
const $classifyLoading    = document.getElementById('classify-loading')
const $classifyLoadingTxt = document.getElementById('classify-loading-text')
const $autoAdvanceChk = document.getElementById('auto-advance')
const $batchConfirm     = document.getElementById('btn-batch-confirm')
const $batchConfInput   = document.getElementById('batch-conf-input')
const $confHistogram    = document.getElementById('conf-histogram')
const $coverSummary     = document.getElementById('cover-summary')
const $btnReclassify    = document.getElementById('btn-reclassify')
const $btnExport      = document.getElementById('btn-export')
const $btnExportAll   = document.getElementById('btn-export-all')
const $settingModel   = document.getElementById('setting-model')
const $sortSelect     = document.getElementById('sort-select')
const $confSlider     = document.getElementById('conf-slider')
const $confValue      = document.getElementById('conf-value')
const $labelFilterBtn      = document.getElementById('label-filter-btn')
const $labelFilterDropdown = document.getElementById('label-filter-dropdown')
const $labelFilterSearch   = document.getElementById('label-filter-search')
const $labelFilterList     = document.getElementById('label-filter-list')

const $patchPreview       = document.getElementById('patch-preview')
const $patchPreviewCanvas = document.getElementById('patch-preview-canvas')
const $patchPreviewLabel  = document.getElementById('patch-preview-label')

const imgCtx     = $imageCanvas.getContext('2d')
const overlayCtx = $overlayCanvas.getContext('2d')

function setModelStatus(cls, text) {
  $statusDot.className  = `status-dot ${cls}`
  $statusText.textContent = text
}

// Returns a consistent "ready" label including backend info
function readyLabel(modelKey) {
  const backend = _gpuBackend === 'gpu' ? ' · WebGPU ⚡' : ' · CPU'
  return `${modelKey.toUpperCase()} ready${backend}`
}

function startClassifyAnimation() {
  $classifyLoading?.classList.add('visible')
}

// ─── Upload ──────────────────────────────────────────────────────────────────

async function uploadFiles(files) {
  if (!_ortReady) {
    alert('ONNX Runtime is not ready. Check your internet connection and reload.')
    return
  }
  const { model, rows, cols, patchSize } = state.uploadSettings

  for (const file of files) {
    const placeholder = createUploadingItem(file.name)
    $imageList.appendChild(placeholder)

    // Show processing status on canvas/placeholder immediately
    state._processingFile = file.name
    _ensureCanvasSized()
    drawOverlay()

    try {
      // 1. Read file
      const dataUrl = await new Promise((res, rej) => {
        const r = new FileReader()
        r.onload  = e => res(e.target.result)
        r.onerror = rej
        r.readAsDataURL(file)
      })
      // 2. Load as image element
      const img = await new Promise((res, rej) => {
        const im = new Image()
        im.onload  = () => res(im)
        im.onerror = rej
        im.src = dataUrl
      })
      // 3. Build record (all-client state)
      const { gridMethod, pointPlacement } = state.uploadSettings
      const points = gridMethod === 'noaa'
        ? generateStratifiedRandom(img.naturalWidth, img.naturalHeight, 2, 5, patchSize)
        : (pointPlacement === 'stratified'
          ? generateStratifiedRandom(img.naturalWidth, img.naturalHeight, rows, cols, patchSize)
          : generateGrid(img.naturalWidth, img.naturalHeight, rows, cols, patchSize))
      const record = {
        id:   crypto.randomUUID(),
        name: file.name,
        image: dataUrl,
        thumbnail: makeThumbnail(img),
        original_image_width:  img.naturalWidth,
        original_image_height: img.naturalHeight,
        patch_size:  patchSize,
        model_used:  model,
        grid_rows:   gridMethod === 'noaa' ? 2 : rows,
        grid_cols:   gridMethod === 'noaa' ? 5 : cols,
        points,
        num_confirmed: 0,
      }
      state.images.push(record)
      placeholder.replaceWith(buildImageItem(record))
      const isFirst = !state.currentId
      if (isFirst) {
        await loadImage(record.id)
        delete state._processingFile  // classifyRecord pill takes over
        // 4. Classify the active image immediately (non-blocking)
        classifyRecord(record, img).catch(err =>
          console.error('Classification error:', err))
      } else {
        // Defer classification until the user clicks this image
        record._pendingClassify = true
        delete state._processingFile
        refreshImageListItem()
      }
    } catch (err) {
      delete state._processingFile
      placeholder.replaceWith(buildErrorItem(file.name, err?.message ?? String(err)))
    }
  }
}

// Ensure overlay canvas has usable dimensions even before first image loads
function _ensureCanvasSized() {
  if ($overlayCanvas.width === 0) {
    const rect = $container.getBoundingClientRect()
    if (rect.width > 0) {
      $imageCanvas.width  = $overlayCanvas.width  = rect.width
      $imageCanvas.height = $overlayCanvas.height = rect.height
      $placeholder.classList.add('hidden')
    }
  }
}

async function classifyRecord(record, imgEl) {
  const key    = record.model_used ?? 't3'
  const points = record.points
  let done = 0
  let inferenceError = null
  state.classifyingIds.add(record.id)
  record._classifyDone = 0   // read by drawOverlay for canvas progress pill
  startClassifyAnimation()
  refreshImageListItem()  // show classifying spinner in image list immediately

  // Redraw canvas periodically so the progress pill shows during model download
  const _progressTimer = setInterval(() => {
    if (record.id === state.currentId) drawOverlay()
  }, 400)

  for (const point of points) {
    try {
      const top5 = await classifyPatch(imgEl, point.column, point.row, key, record.patch_size ?? PATCH_SIZE)
      point.annotations = top5.map(t => ({
        id: crypto.randomUUID(),
        benthic_attribute: t.code,
        ba_gr:             t.code,
        ba_gr_label:       T3_DESCRIPTIONS[t.code] ?? t.code,
        code:              t.code,
        is_confirmed:      false,
        is_machine_created: true,
        score: parseFloat(t.score.toFixed(4)),
      }))
      inferenceError = null   // clear on first success
    } catch (err) {
      if (!inferenceError) {
        inferenceError = err
        const msg = err?.message ?? String(err)
        setModelStatus('error', `Inference error: ${msg.slice(0, 80)}`)
        console.error('classifyPatch error (first occurrence):', err)
      }
    }

    done++
    record._classifyDone = done
    const isActive = record.id === state.currentId
    if (!inferenceError) {
      const pct = Math.round(done / points.length * 100)
      if ($classifyLoadingTxt) $classifyLoadingTxt.textContent = `Classifying ${record.name} — ${done}/${points.length} (${pct}%)`
    }
    if (isActive && (done % 5 === 0 || done === points.length)) {
      drawOverlay()
      if (state.selectedIdx >= 0) renderDetail()
      renderProgress()
      refreshImageListItem()
    }
  }

  clearInterval(_progressTimer)
  delete record._classifyDone
  state.classifyingIds.delete(record.id)
  if (state.classifyingIds.size === 0) $classifyLoading?.classList.remove('visible')
  record.num_confirmed = record.points.filter(isConfirmed).length
  if (record.id === state.currentId) {
    if (state.selectedIdx < 0 && record.points.length > 0) state.selectedIdx = 0
    drawOverlay(); renderDetail(); renderProgress(); refreshImageListItem(); updateStatusText()
  }
  if (!inferenceError) {
    setModelStatus('ready', readyLabel(key))
  }
}

// ─── Image list ──────────────────────────────────────────────────────────────

function buildImageItem(record) {
  const confirmed = (record.points ?? []).filter(isConfirmed).length
  const total     = record.points?.length ?? 0
  const pct       = total > 0 ? Math.round(confirmed / total * 100) : 0
  const modelTag  = record.model_used ? record.model_used.toUpperCase() : ''
  const gridTag   = record.grid_rows  ? `${record.grid_rows}x${record.grid_cols}` : ''

  const el = document.createElement('div')
  const pendingCls = record._pendingClassify ? ' pending-classify' : ''
  el.className = 'image-item' + (record.id === state.currentId ? ' active' : '') + pendingCls
  el.dataset.id = record.id
  el.innerHTML = `
    <div class="image-thumb">
      <img src="${record.thumbnail}" alt="${record.name}" loading="lazy">
    </div>
    <div class="image-meta">
      <div class="image-name" title="${record.name}">${record.name}</div>
      <div class="image-tags">${modelTag ? `<span class="tag-model">${modelTag}</span>` : ''}${gridTag ? `<span class="tag-grid">${gridTag}</span>` : ''}</div>
      <div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
      <div class="image-progress">
        <span>${confirmed} confirmed</span>
        <span>${confirmed}/${total}</span>
      </div>
    </div>
    <button class="btn-delete" title="Remove image" aria-label="Remove">x</button>
  `
  el.querySelector('.btn-delete').addEventListener('click', e => {
    e.stopPropagation()
    if (!confirm(`Remove "${record.name}"?`)) return
    state.images = state.images.filter(i => i.id !== record.id)
    el.remove()
    if (state.currentId === record.id) {
      state.currentId = null; state.record = null; state.loadedImg = null
      clearCanvas(); renderDetail(); renderProgress()
    }
  })
  el.addEventListener('click', () => loadImage(record.id))
  return el
}

function createUploadingItem(name) {
  const el = document.createElement('div')
  el.className = 'image-item uploading'
  el.innerHTML = `<div class="upload-spinner"></div><div class="image-meta"><div class="image-name">${name}</div><div class="image-tags">Uploading…</div></div>`
  return el
}

function buildErrorItem(name, msg) {
  const el = document.createElement('div')
  el.className = 'image-item error'
  el.innerHTML = `<span class="error-icon">!</span><div class="image-meta"><div class="image-name" title="${msg}">${name}</div><div class="image-tags error-msg">${String(msg).slice(0, 70)}</div></div>`
  return el
}

function refreshImageListItem() {
  const el = $imageList.querySelector(`[data-id="${state.currentId}"]`)
  if (!el || !state.record) return
  const pts       = state.record.points
  const confirmed = pts.filter(isConfirmed).length
  const total     = pts.length
  const pct       = Math.round(confirmed / total * 100)
  el.querySelector('.progress-fill').style.width = `${pct}%`
  const spans = el.querySelectorAll('.image-progress span')
  if (spans[0]) spans[0].textContent = `${confirmed} confirmed`
  if (spans[1]) spans[1].textContent = `${confirmed}/${total}`
  // Show pulsing progress bar while classifying
  el.classList.toggle('classifying', state.classifyingIds.has(state.currentId))
}

// ─── Load image ──────────────────────────────────────────────────────────────

async function loadImage(id) {
  state.currentId   = id
  state.selectedIdx = -1
  state.hoverIdx    = -1
  state.zoom = 1.0; state.panX = 0; state.panY = 0

  document.querySelectorAll('.image-item').forEach(el => {
    el.classList.toggle('active', el.dataset.id === id)
    if (el.dataset.id === id) el.classList.remove('pending-classify')
  })

  // Direct reference — mutations to points are visible immediately
  const record = state.images.find(r => r.id === id)
  if (!record) return
  state.record  = record
  state.isDirty = false

  $placeholder.classList.add('hidden')

  const img = new Image()
  img.onload = () => {
    state.loadedImg = img
    resizeAndDraw()
    // Start classification now if this image was deferred at upload time
    if (record._pendingClassify) {
      record._pendingClassify = false
      classifyRecord(record, img).catch(err =>
        console.error('Classification error:', err))
    }
  }
  img.src = record.image

  renderProgress()
  // Auto-select first point if the record is already classified
  if (state.selectedIdx < 0 && record.points.some(p => p.annotations?.length)) {
    state.selectedIdx = 0
  }
  renderDetail()
  updateStatusText()
  if ($btnReclassify) $btnReclassify.disabled = false
}

// ─── Canvas ──────────────────────────────────────────────────────────────────

function getTransform() {
  if (!state.record || !state.loadedImg) return null
  const cw = $imageCanvas.width, ch = $imageCanvas.height
  const iw = state.record.original_image_width, ih = state.record.original_image_height
  const baseScale = Math.min(cw / iw, ch / ih)
  const scale     = baseScale * state.zoom
  const offsetX   = (cw - iw * scale) / 2 + state.panX
  const offsetY   = (ch - ih * scale) / 2 + state.panY
  return { scale, offsetX, offsetY }
}

// Returns the exact screen-space half-width of the patch crop at the current transform.
// Single source of truth shared by drawOverlay and hitTest.
function getPointScreenHalf(t) {
  return ((state.record?.patch_size ?? PATCH_SIZE) / 2) * t.scale
}

function resetView() {
  state.zoom = 1.0; state.panX = 0; state.panY = 0
  resizeAndDraw()
}

function resizeAndDraw() {
  const rect = $container.getBoundingClientRect()
  $imageCanvas.width  = $overlayCanvas.width  = rect.width
  $imageCanvas.height = $overlayCanvas.height = rect.height
  drawImage()
  drawOverlay()
}

function clearCanvas() {
  imgCtx.clearRect(0, 0, $imageCanvas.width, $imageCanvas.height)
  overlayCtx.clearRect(0, 0, $overlayCanvas.width, $overlayCanvas.height)
  $placeholder.classList.remove('hidden')
}

function drawImage() {
  imgCtx.clearRect(0, 0, $imageCanvas.width, $imageCanvas.height)
  const t = getTransform()
  if (!t || !state.loadedImg) return
  imgCtx.drawImage(state.loadedImg,
    t.offsetX, t.offsetY,
    state.record.original_image_width  * t.scale,
    state.record.original_image_height * t.scale)
}

function drawOverlay() {
  overlayCtx.clearRect(0, 0, $overlayCanvas.width, $overlayCanvas.height)
  const t = getTransform()

  if (t && state.record?.points.length) {
    const patchSize = state.record.patch_size ?? PATCH_SIZE
    const visible   = new Set(filteredPoints().map(p => p.id))
    if (state.overlayMode === 'focus') {
      _drawOverlayFocus(t, patchSize, visible)
    } else {
      _drawOverlayOverview(t, patchSize, visible)
    }
    _drawPatchPreview(state.selectedIdx >= 0 ? state.record?.points[state.selectedIdx] : null)
    _drawZoomHud(overlayCtx, $overlayCanvas.width, $overlayCanvas.height, patchSize)
  }

  // ── Canvas status pill — shown even while image is loading ─────────────────
  const _showProcessingPill  = state._processingFile && !state.classifyingIds.has(state.record?.id)
  const _showClassifyingPill = state.record && state.classifyingIds.has(state.record.id)
  const W = $overlayCanvas.width
  const H = $overlayCanvas.height

  if (_showProcessingPill) {
    _drawCanvasPill(overlayCtx, W, H, `Processing  ${state._processingFile}…`, null)
  } else if (_showClassifyingPill) {
    const total = state.record.points.length
    const done  = state.record._classifyDone ?? 0
    const pct   = total > 0 ? Math.round(done / total * 100) : 0
    const modelLabel = (state.record.model_used ?? 't3').toUpperCase()
    const txt = done === 0
      ? `Loading ${modelLabel} model…  (first load may take 30–60 s)`
      : `Classifying  ${done} / ${total}  (${pct}%)`
    _drawCanvasPill(overlayCtx, W, H, txt, done > 0 ? pct : null)
  }
}

function _drawCanvasPill(ctx, W, H, txt, pct) {
  const barH = 4
  ctx.save()
  ctx.font = 'bold 14px sans-serif'
  const tw   = ctx.measureText(txt).width
  const padX = 18, padY = 10
  const bw   = Math.min(tw + padX * 2, W - 24)
  const bh   = 14 + padY * 2 + (pct != null ? barH + 6 : 0)
  const bx   = Math.round((W - bw) / 2)
  const by   = Math.round(H * 0.42)

  ctx.fillStyle = 'rgba(0,0,0,0.75)'
  ctx.beginPath()
  if (ctx.roundRect) ctx.roundRect(bx, by, bw, bh, 8)
  else ctx.rect(bx, by, bw, bh)
  ctx.fill()

  if (pct != null) {
    const barY  = by + bh - barH - 6
    const barXs = bx + 8, barXe = bw - 16
    ctx.fillStyle = 'rgba(255,255,255,0.18)'
    ctx.fillRect(barXs, barY, barXe, barH)
    ctx.fillStyle = '#3b82f6'
    ctx.fillRect(barXs, barY, Math.round(barXe * pct / 100), barH)
  }

  ctx.fillStyle    = '#fff'
  ctx.textAlign    = 'center'
  ctx.textBaseline = 'top'
  ctx.fillText(txt, W / 2, by + padY, bw - padX * 2)
  ctx.restore()
}

// ── Overview mode: compact clamped-radius squares — original appearance ──────
function _drawOverlayOverview(t, patchSize, visible) {
  const sparse = state.record.points.length <= 12
  const r = Math.max(sparse ? 7 : 4, Math.min(sparse ? 16 : 11, (patchSize / 2) * t.scale * (sparse ? 0.55 : 0.35)))
  state.record.points.forEach((point, idx) => {
    if (!visible.has(point.id)) return
    const x     = t.offsetX + point.column * t.scale
    const y     = t.offsetY + point.row    * t.scale
    const color  = getPointColor(point)
    const isSel  = idx === state.selectedIdx
    const isHov  = idx === state.hoverIdx
    const half   = isSel ? r * 1.7 : r + (isHov && !isSel ? 2 : 0)
    overlayCtx.shadowBlur  = isSel ? 18 : 0
    overlayCtx.shadowColor = color
    overlayCtx.fillStyle   = color
    overlayCtx.globalAlpha = isConfirmed(point) ? 0.18 : 0.12
    overlayCtx.fillRect(x - half, y - half, half * 2, half * 2)
    overlayCtx.globalAlpha = 1.0
    overlayCtx.shadowBlur  = 0
    overlayCtx.strokeStyle = color
    overlayCtx.lineWidth   = isSel ? 2.5 : 1.5
    overlayCtx.strokeRect(x - half, y - half, half * 2, half * 2)
    overlayCtx.beginPath()
    overlayCtx.arc(x, y, 2, 0, Math.PI * 2)
    overlayCtx.fillStyle   = color
    overlayCtx.globalAlpha = isConfirmed(point) ? 1.0 : 0.85
    overlayCtx.fill()
    overlayCtx.globalAlpha = 1.0
    if (isSel) {
      overlayCtx.strokeStyle = 'rgba(255,255,255,0.75)'
      overlayCtx.lineWidth   = 2
      overlayCtx.strokeRect(x - half - 5, y - half - 5, (half + 5) * 2, (half + 5) * 2)
      overlayCtx.strokeStyle = 'rgba(255,255,255,0.25)'
      overlayCtx.lineWidth   = 1.5
      overlayCtx.strokeRect(x - half - 11, y - half - 11, (half + 11) * 2, (half + 11) * 2)
    }
    if (isSel || isHov) {
      const ann  = point.annotations?.[0]
      const code = ann?.code ?? '?'
      const name = isSel ? (ann?.ba_gr_label ?? '') : ''
      _drawTooltipPill(overlayCtx, x, y - half - 5, code, name)
    }
  })
}

// ── Focus mode: exact patch boxes, double-stroke halo, dim mask ───────────────
function _drawOverlayFocus(t, patchSize, visible) {
  const patchHalf = getPointScreenHalf(t)
  const dotR    = Math.max(3.5, Math.min(14, patchHalf * 0.18))
  const borderW = Math.max(1, Math.min(3, 2 / t.scale))
  let selX = null, selY = null, selHalf = null, selCode = null, selName = null
  state.record.points.forEach((point, idx) => {
    if (!visible.has(point.id)) return
    const x     = t.offsetX + point.column * t.scale
    const y     = t.offsetY + point.row    * t.scale
    const color  = getPointColor(point)
    const isSel  = idx === state.selectedIdx
    const isHov  = idx === state.hoverIdx
    const half   = isSel ? patchHalf * 1.04 : patchHalf
    overlayCtx.shadowBlur  = isSel ? 18 : 0
    overlayCtx.shadowColor = color
    overlayCtx.fillStyle   = color
    overlayCtx.globalAlpha = isConfirmed(point) ? 0.15 : 0.09
    overlayCtx.fillRect(x - half, y - half, half * 2, half * 2)
    overlayCtx.globalAlpha = 1.0
    overlayCtx.shadowBlur  = 0
    overlayCtx.lineWidth   = borderW + 2
    overlayCtx.strokeStyle = 'rgba(0,0,0,0.65)'
    overlayCtx.strokeRect(x - half, y - half, half * 2, half * 2)
    overlayCtx.lineWidth   = borderW
    overlayCtx.strokeStyle = color
    overlayCtx.strokeRect(x - half, y - half, half * 2, half * 2)
    overlayCtx.beginPath()
    overlayCtx.arc(x, y, Math.max(2.5, dotR), 0, Math.PI * 2)
    overlayCtx.fillStyle   = color
    overlayCtx.globalAlpha = isConfirmed(point) ? 1.0 : 0.85
    overlayCtx.fill()
    overlayCtx.globalAlpha = 1.0
    if (isSel) {
      overlayCtx.strokeStyle = 'rgba(255,255,255,0.80)'
      overlayCtx.lineWidth   = 2
      overlayCtx.strokeRect(x - half - 4, y - half - 4, (half + 4) * 2, (half + 4) * 2)
      selX = x; selY = y; selHalf = half
      const ann = point.annotations?.[0]
      selCode = ann?.code ?? '?'
      selName = ann?.ba_gr_label ?? ''
    } else if (isHov) {
      overlayCtx.strokeStyle = 'rgba(255,255,255,0.45)'
      overlayCtx.lineWidth   = 1.5
      overlayCtx.strokeRect(x - half - 3, y - half - 3, (half + 3) * 2, (half + 3) * 2)
      const ann  = point.annotations?.[0]
      _drawTooltipPill(overlayCtx, x, y - half - 5, ann?.code ?? '?', '')
    }
  })
  if (selX !== null) {
    const gap = 6
    overlayCtx.save()
    overlayCtx.beginPath()
    overlayCtx.rect(0, 0, $overlayCanvas.width, $overlayCanvas.height)
    overlayCtx.rect(selX - selHalf - gap, selY - selHalf - gap,
                    (selHalf + gap) * 2, (selHalf + gap) * 2)
    overlayCtx.fillStyle = 'rgba(0,0,0,0.38)'
    overlayCtx.fill('evenodd')
    overlayCtx.restore()
    _drawTooltipPill(overlayCtx, selX, selY - selHalf - 5, selCode, selName)
  }
}

// Bottom-left HUD showing current zoom level, patch size, and key hints.
function _drawZoomHud(ctx, W, H, patchSize) {
  const txt = `${state.zoom.toFixed(1)}×  ·  patch ${patchSize}px  ·  Z=zoom  R=reset`
  ctx.save()
  ctx.font = '11px sans-serif'
  const tw   = ctx.measureText(txt).width
  const padX = 9, padY = 4
  const bw   = tw + padX * 2
  const bh   = 11 + padY * 2
  const bx   = 8
  const by   = H - bh - 8
  ctx.fillStyle = 'rgba(0,0,0,0.55)'
  ctx.beginPath()
  if (ctx.roundRect) ctx.roundRect(bx, by, bw, bh, 4)
  else ctx.rect(bx, by, bw, bh)
  ctx.fill()
  ctx.fillStyle   = 'rgba(200,210,230,0.80)'
  ctx.textAlign    = 'left'
  ctx.textBaseline = 'middle'
  ctx.fillText(txt, bx + padX, by + bh / 2)
  ctx.restore()
}

function _drawTooltipPill(ctx, cx, bottomY, code, name) {
  const fs   = 11
  const padX = 7, padY = 3
  ctx.font   = `700 ${fs}px monospace`
  const codeW    = ctx.measureText(code).width
  const sep      = name ? ' · ' : ''
  const nameSnip = name.slice(0, 26)
  ctx.font   = `400 ${fs}px sans-serif`
  const nameW = name ? ctx.measureText(sep + nameSnip).width : 0
  const boxW  = codeW + nameW + padX * 2
  const boxH  = fs + padY * 2
  const bx    = Math.max(2, Math.min(cx - boxW / 2, ctx.canvas.width - boxW - 2))
  const by    = Math.max(2, bottomY - boxH)
  ctx.shadowBlur  = 6
  ctx.shadowColor = 'rgba(0,0,0,0.8)'
  ctx.fillStyle   = 'rgba(8,10,22,0.93)'
  ctx.beginPath()
  ctx.roundRect(bx, by, boxW, boxH, 4)
  ctx.fill()
  ctx.shadowBlur  = 0
  ctx.beginPath()
  ctx.moveTo(cx, by + boxH)
  ctx.lineTo(cx, bottomY + 5)
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'
  ctx.lineWidth   = 1
  ctx.stroke()
  ctx.textAlign    = 'left'
  ctx.textBaseline = 'middle'
  const midY = by + boxH / 2
  ctx.font      = `700 ${fs}px monospace`
  ctx.fillStyle = '#fbbf24'
  ctx.fillText(code, bx + padX, midY)
  if (name) {
    ctx.font      = `400 ${fs}px sans-serif`
    ctx.fillStyle = '#cbd5e1'
    ctx.fillText(sep + nameSnip, bx + padX + codeW, midY)
  }
}

function _drawPatchPreview(point) {
  if (!point || !state.loadedImg) {
    $patchPreview.classList.add('hidden')
    return
  }
  const patchSize = state.record.patch_size ?? PATCH_SIZE
  const half      = patchSize >> 1
  const sx = Math.max(0, point.column - half)
  const sy = Math.max(0, point.row    - half)
  const sw = Math.min(state.record.original_image_width  - sx, patchSize)
  const sh = Math.min(state.record.original_image_height - sy, patchSize)

  const sz = $patchPreview.clientWidth - 16
  $patchPreviewCanvas.width  = sz
  $patchPreviewCanvas.height = sz
  const ctx = $patchPreviewCanvas.getContext('2d')
  ctx.drawImage(state.loadedImg, sx, sy, sw, sh, 0, 0, sz, sz)

  const color = getPointColor(point)
  ctx.strokeStyle = color
  ctx.lineWidth   = 3
  ctx.strokeRect(1, 1, sz - 2, sz - 2)

  const ann   = point.annotations?.[0]
  const code  = ann?.code ?? '?'
  const score = ann?.score != null ? `  ${(ann.score * 100).toFixed(0)}%` : ''
  $patchPreviewLabel.textContent = code + score
  $patchPreviewLabel.style.color = color
  $patchPreview.classList.remove('hidden')
}

// ─── Filtering ───────────────────────────────────────────────────────────────

function filteredPoints() {
  if (!state.record) return []
  let pts = state.record.points
  const { visibility, labelCodes, minConf, sortBy } = state.filterState

  if (visibility === 'confirmed') {
    pts = pts.filter(isConfirmed)
  } else if (visibility === 'unclassified') {
    pts = pts.filter(isUnclassified)
  } else if (visibility === 'unconfirmed') {
    pts = pts.filter(p => !isConfirmed(p))
  }

  if (labelCodes.size > 0) {
    pts = pts.filter(p => {
      const code = p.annotations?.[0]?.code
      return code && labelCodes.has(code)
    })
  }

  if (minConf > 0) {
    pts = pts.filter(p => {
      if (isUnclassified(p)) return false
      const score = p.annotations?.[0]?.score
      return score != null && score * 100 >= minConf
    })
  }

  if (sortBy === 'label') {
    pts = [...pts].sort((a, b) => {
      const ca = a.annotations?.[0]?.code ?? 'zzz'
      const cb = b.annotations?.[0]?.code ?? 'zzz'
      return ca.localeCompare(cb)
    })
  } else if (sortBy === 'confidence') {
    pts = [...pts].sort((a, b) => {
      const sa = a.annotations?.[0]?.score ?? -1
      const sb = b.annotations?.[0]?.score ?? -1
      return sb - sa
    })
  }
  return pts
}

// ─── Hit-test ────────────────────────────────────────────────────────────────

function hitTest(cx, cy) {
  const t = getTransform()
  if (!t) return -1
  // Use the true patch half-width; guarantee a minimum 12 px touch target
  const hitRadius  = Math.max(12, getPointScreenHalf(t))
  const visibleIds = new Set(filteredPoints().map(p => p.id))
  let bestIdx = -1, bestDist = hitRadius * hitRadius

  state.record.points.forEach((point, idx) => {
    if (!visibleIds.has(point.id)) return
    const dx = cx - (t.offsetX + point.column * t.scale)
    const dy = cy - (t.offsetY + point.row    * t.scale)
    const d2 = dx * dx + dy * dy
    if (d2 < bestDist) { bestDist = d2; bestIdx = idx }
  })
  return bestIdx
}

// ─── Canvas interaction ───────────────────────────────────────────────────────

$overlayCanvas.addEventListener('wheel', e => {
  e.preventDefault()
  const t = getTransform()
  if (!t) return
  const rect   = $overlayCanvas.getBoundingClientRect()
  const mx     = e.clientX - rect.left
  const my     = e.clientY - rect.top
  const wx     = (mx - t.offsetX) / t.scale
  const wy     = (my - t.offsetY) / t.scale
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
  state.zoom   = Math.min(15, Math.max(0.5, state.zoom * factor))
  const cw = $imageCanvas.width, ch = $imageCanvas.height
  const iw = state.record.original_image_width, ih = state.record.original_image_height
  const newScale = Math.min(cw / iw, ch / ih) * state.zoom
  state.panX = mx - wx * newScale - (cw - iw * newScale) / 2
  state.panY = my - wy * newScale - (ch - ih * newScale) / 2
  drawImage(); drawOverlay()
}, { passive: false })

$overlayCanvas.addEventListener('dblclick', resetView)

let _dragMoved = false
$overlayCanvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return
  _dragMoved = false
  state._panning  = true
  state._panStart = { x: e.clientX - state.panX, y: e.clientY - state.panY }
  $overlayCanvas.style.cursor = 'grabbing'
})

window.addEventListener('mousemove', e => {
  if (!state._panning) {
    const rect = $overlayCanvas.getBoundingClientRect()
    if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) return
    const idx = hitTest(e.clientX - rect.left, e.clientY - rect.top)
    if (idx !== state.hoverIdx) {
      state.hoverIdx = idx
      drawOverlay()
      $overlayCanvas.style.cursor = idx >= 0 ? 'pointer' : 'default'
    }
    return
  }
  const newPanX = e.clientX - state._panStart.x
  const newPanY = e.clientY - state._panStart.y
  if (Math.abs(newPanX - state.panX) > 3 || Math.abs(newPanY - state.panY) > 3) _dragMoved = true
  state.panX = newPanX; state.panY = newPanY
  drawImage(); drawOverlay()
})

window.addEventListener('mouseup', e => {
  if (!state._panning) return
  state._panning = false
  if (!_dragMoved) {
    const rect = $overlayCanvas.getBoundingClientRect()
    selectPoint(hitTest(e.clientX - rect.left, e.clientY - rect.top))
  }
  $overlayCanvas.style.cursor = state.hoverIdx >= 0 ? 'pointer' : 'default'
})

$overlayCanvas.addEventListener('mouseleave', () => {
  if (!state._panning) {
    state.hoverIdx = -1; drawOverlay(); $overlayCanvas.style.cursor = 'default'
  }
})

function selectPoint(idx) {
  state.selectedIdx = idx
  state._zoomLocked = false  // reset zoom-lock when selection changes
  drawOverlay()
  renderDetail()
}

// Zoom and center the viewport on point idx so its patch fills ~35 % of the short viewport edge.
function zoomToPoint(idx) {
  const point = state.record?.points[idx]
  if (!point || !state.loadedImg) return
  const cw = $imageCanvas.width, ch = $imageCanvas.height
  const iw = state.record.original_image_width, ih = state.record.original_image_height
  const patchSize = state.record.patch_size ?? PATCH_SIZE
  const baseScale = Math.min(cw / iw, ch / ih)
  // Target: patch fills ~35 % of the viewport's shorter dimension
  const targetScreenSize = Math.min(cw, ch) * 0.35
  state.zoom = Math.min(15, Math.max(0.5, (targetScreenSize / patchSize) / baseScale))
  const newScale = baseScale * state.zoom
  // Center selected point in viewport
  state.panX = cw / 2 - point.column * newScale - (cw - iw * newScale) / 2
  state.panY = ch / 2 - point.row    * newScale - (ch - ih * newScale) / 2
  drawImage(); drawOverlay()
}

function scrollToPointIfNeeded(idx) {
  const point = state.record?.points[idx]
  if (!point || !state.loadedImg) return
  const cw = $imageCanvas.width, ch = $imageCanvas.height
  const iw = state.record.original_image_width, ih = state.record.original_image_height
  const scale = Math.min(cw / iw, ch / ih) * state.zoom
  const x = (cw - iw * scale) / 2 + state.panX + point.column * scale
  const y = (ch - ih * scale) / 2 + state.panY + point.row    * scale
  const margin = 80
  if (x < margin || x > cw - margin || y < margin || y > ch - margin) {
    state.panX += cw / 2 - x
    state.panY += ch / 2 - y
    drawImage()
  }
}

// ─── Detail panel ─────────────────────────────────────────────────────────────

function renderDetail() {
  const idx   = state.selectedIdx
  const point = state.record?.points[idx]

  if (!point) {
    $detail.innerHTML = '<p class="hint">Click a point on the image to annotate it</p>'
    return
  }

  const anns      = point.annotations ?? []
  const confirmed = isConfirmed(point)
  const total     = state.record.points.length
  const statusCls = confirmed ? 'status-confirmed' : anns.length ? 'status-unconfirmed' : 'status-unclassified'
  const statusLbl = confirmed ? 'Confirmed'        : anns.length ? 'Unconfirmed'         : 'Unclassified'

  const predRows = anns.slice(0, 5).map((a, i) => {
    const scoreStr = a.score != null ? (a.score * 100).toFixed(1) + '%' : (a.is_machine_created === false ? 'manual' : '')
    return `<div class="prediction-row ${i === 0 ? 'is-top' : ''} ${a.is_confirmed ? 'is-confirmed' : ''}" data-ann-idx="${i}">
      <span class="pred-rank">${i + 1}</span>
      <span class="pred-code">${a.code ?? '?'}</span>
      <span class="pred-label" title="${a.ba_gr_label}">${a.ba_gr_label}</span>
      <span class="pred-score">${scoreStr}</span>
      <button class="btn-use" data-ann-idx="${i}">Use</button>
    </div>`
  }).join('')

  $detail.innerHTML = `
    <div class="detail-header">
      <span class="point-num">Point ${idx + 1} / ${total}</span>
      <span class="point-coords">${point.column}, ${point.row}</span>
    </div>
    <div class="detail-status ${statusCls}">${statusLbl}</div>
    <div class="predictions-section">
      <div class="predictions-header">Top Predictions</div>
      ${anns.length ? predRows : '<div class="no-predictions">No predictions yet</div>'}
    </div>
    <div class="detail-actions">
      ${!confirmed && anns.length ? '<button class="btn-primary" id="btn-confirm-top">Confirm #1 <kbd>Enter</kbd></button>' : ''}
      ${confirmed                 ? '<button class="btn-secondary" id="btn-unconfirm">Undo confirm</button>' : ''}
      <button class="btn-ghost" id="btn-unclassify">Unclassify <kbd>U</kbd></button>
    </div>
    <div class="custom-label-section">
      <div class="custom-label-header">Apply Label</div>
      <div class="label-search-wrap">
        <input type="text" id="label-search-input" placeholder="Search labels…" autocomplete="off">
      </div>
      <div class="label-results" id="label-results"></div>
      <div class="custom-label-create">
        <div class="custom-label-create-title">+ Create new label</div>
        <div class="custom-label-create-row">
          <input type="text" id="new-label-code" placeholder="CODE" maxlength="10">
          <input type="text" id="new-label-name" placeholder="Full name">
          <button id="btn-add-label" class="btn-tool">Add</button>
        </div>
        <div id="new-label-error" class="label-error"></div>
      </div>
    </div>
  `

  $detail.querySelector('#btn-confirm-top')?.addEventListener('click', () => confirmPoint(idx, 0))
  $detail.querySelector('#btn-unconfirm')?.addEventListener('click',   () => unconfirmPoint(idx))
  $detail.querySelector('#btn-unclassify')?.addEventListener('click',  () => unclassifyPoint(idx))
  $detail.querySelectorAll('.btn-use').forEach(btn =>
    btn.addEventListener('click', () => useAnnotation(idx, parseInt(btn.dataset.annIdx))))

  const $lsi = $detail.querySelector('#label-search-input')
  const $lr  = $detail.querySelector('#label-results')

  function renderLabelResults(query) {
    const q = query.trim().toLowerCase()
    const combined = [...state.allLabels, ...state.customLabels.filter(c => !state.allLabels.find(l => l.code === c.code))]
    const filtered = q
      ? combined.filter(l => l.code.toLowerCase().includes(q) || l.name.toLowerCase().includes(q))
      : combined.slice(0, 40)
    if (!filtered.length) {
      $lr.innerHTML = '<div class="label-result-empty">No labels found</div>'
      return
    }
    $lr.innerHTML = filtered.map(l =>
      `<div class="label-result-row" data-code="${l.code}" data-name="${l.name}" data-custom="${!!l.is_custom}">
        <span class="lr-code">${l.code}</span>
        <span class="lr-name">${l.name}</span>
        ${l.is_custom ? '<span class="lr-tag">custom</span>' : ''}
      </div>`
    ).join('')
    $lr.querySelectorAll('.label-result-row').forEach(row => {
      row.addEventListener('click', () => {
        applyLabel(idx, row.dataset.code, row.dataset.name, row.dataset.custom === 'true')
        $lsi.value = ''; renderLabelResults('')
      })
    })
  }

  renderLabelResults('')
  $lsi.addEventListener('input', () => renderLabelResults($lsi.value))

  const $btnAdd = $detail.querySelector('#btn-add-label')
  const $errEl  = $detail.querySelector('#new-label-error')
  $btnAdd?.addEventListener('click', () => {
    const code = $detail.querySelector('#new-label-code').value.trim().toUpperCase()
    const name = $detail.querySelector('#new-label-name').value.trim()
    $errEl.textContent = ''
    if (!code || !name) { $errEl.textContent = 'Code and name are required.'; return }
    const existing = [...state.allLabels, ...state.customLabels].find(l => l.code === code)
    if (existing) { $errEl.textContent = `"${code}" already exists.`; return }
    const created = { code, name, is_custom: true }
    state.customLabels.push(created)
    state.allLabels.push(created)
    applyLabel(idx, created.code, created.name, true)
    $detail.querySelector('#new-label-code').value = ''
    $detail.querySelector('#new-label-name').value = ''
    renderLabelResults('')
    refreshLabelFilterList()
  })
}

// ─── Apply label ─────────────────────────────────────────────────────────────

function applyLabel(pointIdx, code, name, isCustom) {
  const point = state.record?.points[pointIdx]
  if (!point) return
  const ann = {
    id: crypto.randomUUID(),
    benthic_attribute: code, ba_gr: code, ba_gr_label: name, code,
    is_confirmed: true, is_machine_created: false, score: null,
  }
  const _prePos = filteredPoints().findIndex(p => p.id === point.id)
  point.annotations.forEach(a => { a.is_confirmed = false })
  point.annotations.unshift(ann)
  saveDebounced()
  if (state.autoAdvance) _nextUnconfirmedAfterMutation(_prePos, point.id)
  else { drawOverlay(); renderDetail() }
}

// ─── Annotation actions ───────────────────────────────────────────────────────

function confirmPoint(idx, annIdx = 0) {
  const point = state.record?.points[idx]
  if (!point?.annotations?.length) return
  const _prePos = filteredPoints().findIndex(p => p.id === point.id)
  const anns = point.annotations
  const [ann] = anns.splice(annIdx, 1)
  ann.is_confirmed = true
  anns.forEach(a => { a.is_confirmed = false })
  anns.unshift(ann)
  saveDebounced()
  if (state.autoAdvance) _nextUnconfirmedAfterMutation(_prePos, point.id)
}

function useAnnotation(idx, annIdx) {
  const point = state.record?.points[idx]
  if (!point?.annotations?.length) return
  const [ann] = point.annotations.splice(annIdx, 1)
  point.annotations.unshift(ann)
  renderDetail(); drawOverlay()
}

function unconfirmPoint(idx) {
  const point = state.record?.points[idx]
  if (!point) return
  point.annotations?.forEach(a => { a.is_confirmed = false })
  saveDebounced()
}

function unclassifyPoint(idx) {
  const point = state.record?.points[idx]
  if (!point) return
  point.annotations = []
  saveDebounced()
}

function nextUnconfirmed() {
  const pts = state.record?.points
  if (!pts?.length) return
  const ordered  = filteredPoints()
  const selPoint = state.selectedIdx >= 0 ? pts[state.selectedIdx] : null
  const curPos   = selPoint ? ordered.findIndex(p => p.id === selPoint.id) : -1
  for (let i = 1; i <= ordered.length; i++) {
    const p = ordered[(curPos + i) % ordered.length]
    if (!isConfirmed(p)) { selectPoint(pts.findIndex(q => q.id === p.id)); return }
  }
  selectPoint(-1)
}

function _nextUnconfirmedAfterMutation(prevFilteredPos, mutatedId) {
  const pts = state.record?.points
  if (!pts?.length) return
  const ordered       = filteredPoints()
  const stillPresent  = ordered.findIndex(p => p.id === mutatedId) !== -1
  const startPos      = stillPresent ? prevFilteredPos + 1 : prevFilteredPos
  for (let i = 0; i < ordered.length; i++) {
    const p = ordered[(startPos + i) % ordered.length]
    if (p.id !== mutatedId && !isConfirmed(p)) {
      selectPoint(pts.findIndex(q => q.id === p.id)); return
    }
  }
  selectPoint(-1)
}

// ─── Save (client-side only) ──────────────────────────────────────────────────

let _saveTimer = null
function saveDebounced() {
  state.isDirty = true
  clearTimeout(_saveTimer)
  _saveTimer = setTimeout(() => {
    state.isDirty = false
    // Sync num_confirmed on the images array entry
    const rec = state.images.find(r => r.id === state.currentId)
    if (rec) rec.num_confirmed = rec.points.filter(isConfirmed).length
    drawOverlay(); renderDetail(); renderProgress(); refreshImageListItem()
  }, 150)
}

// ─── Progress ────────────────────────────────────────────────────────────────

function drawHistogram() {
  if (!$confHistogram) return
  const panel = document.getElementById('histogram-panel')
  const pts = state.record?.points ?? []
  const scored = pts.filter(p => p.annotations?.[0]?.score != null)
  if (!scored.length) { panel?.classList.remove('has-data'); return }
  panel?.classList.add('has-data')

  // Size canvas to its rendered CSS width (fills sidebar)
  const cssW = $confHistogram.parentElement?.clientWidth || 268
  $confHistogram.width  = cssW
  $confHistogram.height = 40

  const ctx = $confHistogram.getContext('2d')
  const W = cssW, H = 40
  ctx.clearRect(0, 0, W, H)

  // 10 bins: [0,.1)…[.9,1.0]
  const bins = new Array(10).fill(0)
  scored.forEach(p => {
    const s = p.annotations[0].score
    bins[Math.min(9, Math.floor(s * 10))]++
  })
  const maxCount = Math.max(1, ...bins)
  const barW = W / 10
  bins.forEach((count, i) => {
    if (!count) return
    const bH = Math.max(2, Math.round((count / maxCount) * (H - 4)))
    ctx.fillStyle = `hsl(${Math.round((i / 9) * 120)}, 65%, 52%)`
    ctx.fillRect(i * barW + 1, H - bH, barW - 2, bH)
  })

  // 10% tick labels below bars
  ctx.fillStyle = 'rgba(148,163,184,0.6)'
  ctx.font = '8px system-ui,sans-serif'
  ctx.textAlign = 'center'
  ;['0', '', '', '', '', '50', '', '', '', '100'].forEach((lbl, i) => {
    if (lbl) ctx.fillText(lbl + '%', i * barW + barW / 2, H - 1)
  })

  // Threshold line
  const lx = Math.round(((parseInt($batchConfInput?.value) || 80) / 100) * W)
  ctx.strokeStyle = 'rgba(255,255,255,0.75)'
  ctx.lineWidth = 1.5; ctx.setLineDash([2, 3])
  ctx.beginPath(); ctx.moveTo(lx, 0); ctx.lineTo(lx, H - 10); ctx.stroke()
  ctx.setLineDash([])
}

function renderCoverSummary() {
  if (!$coverSummary) return
  const pts = state.record?.points ?? []
  const counts = {}
  pts.forEach(p => {
    const code = p.annotations?.[0]?.code
    if (code) counts[code] = (counts[code] ?? 0) + 1
  })
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1])
  if (!sorted.length) { $coverSummary.innerHTML = ''; $coverSummary.classList.add('hidden'); return }
  const total = pts.length
  $coverSummary.classList.remove('hidden')

  const activeCodes = state.filterState.labelCodes
  const chips = sorted.map(([code, n]) => {
    const pct      = Math.round(n / total * 100)
    const cat      = T3_CATEGORY[code] ?? T1_CATEGORY[code]
    const dotColor = CAT_COLOR[cat] ?? '#60a5fa'
    const isActive = activeCodes.size === 1 && activeCodes.has(code)
    const name     = T3_DESCRIPTIONS[code] ?? T1_DESCRIPTIONS[code] ?? code
    return `<button class="cover-chip${isActive ? ' active' : ''}" data-code="${code}" title="${name} — click to filter">
      <span class="cover-dot" style="background:${dotColor}"></span>
      <span class="cover-code">${code}</span>
      <span class="cover-pct">${pct}%</span>
    </button>`
  }).join('')
  // Inline confirm button shown when a single chip is active
  const activeCode = activeCodes.size === 1 ? [...activeCodes][0] : null
  let confirmBtn = ''
  if (activeCode) {
    const unconfirmedCount = (state.record?.points ?? []).filter(p =>
      p.annotations?.[0]?.code === activeCode && !isConfirmed(p)
    ).length
    if (unconfirmedCount > 0) {
      confirmBtn = `<button class="cover-confirm-btn" data-code="${activeCode}" title="Confirm all unconfirmed ${activeCode} points">Confirm ${activeCode} (${unconfirmedCount})</button>`
    }
  }

  $coverSummary.innerHTML = '<span class="cover-label">Benthic Cover:</span>' + chips + confirmBtn

  $coverSummary.querySelectorAll('.cover-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const code = btn.dataset.code
      const alreadyActive = state.filterState.labelCodes.size === 1 && state.filterState.labelCodes.has(code)
      if (alreadyActive) {
        state.filterState.labelCodes.clear()
      } else {
        state.filterState.labelCodes.clear()
        state.filterState.labelCodes.add(code)
      }
      updateLabelFilterBadge()
      refreshLabelFilterList()
      drawOverlay()
      renderCoverSummary()
    })
  })

  $coverSummary.querySelector('.cover-confirm-btn')?.addEventListener('click', () => {
    const code = activeCode
    let count = 0
    state.record?.points.forEach(point => {
      if (point.annotations?.[0]?.code === code && !isConfirmed(point)) {
        point.annotations[0].is_confirmed = true
        count++
      }
    })
    if (count) {
      drawOverlay(); renderProgress(); refreshImageListItem(); saveDebounced()
    }
    renderCoverSummary()
  })
}

function renderProgress() {
  const pts = state.record?.points
  if (!pts?.length) { $progressText.textContent = ''; drawHistogram(); renderCoverSummary(); return }
  const confirmed    = pts.filter(isConfirmed).length
  const unclassified = pts.filter(isUnclassified).length
  const unconfirmed  = pts.length - confirmed - unclassified
  $progressText.textContent =
    `${confirmed}/${pts.length} confirmed  ${unconfirmed} unconfirmed  ${unclassified} unclassified`
  drawHistogram()
  renderCoverSummary()
}

function updateStatusText() {
  if (!state.record) return
  const m = (state.record.model_used ?? 't3').toUpperCase()
  const r = state.record.grid_rows ?? 10
  const c = state.record.grid_cols ?? 10
  const base = $statusText.textContent.replace(/ — .+$/, '')
  $statusText.textContent = `${base} — ${m} ${r}x${c}`
}

// ─── CSV Export (client-side blob) ───────────────────────────────────────────

function exportCsv(imageIds) {
  const header = 'image_name,point_id,row,column,confirmed,code,label,score,machine_created,is_custom_label,model_used,grid_rows,grid_cols'
  const rows   = [header]
  for (const id of imageIds) {
    const rec = state.images.find(r => r.id === id)
    if (!rec) continue
    for (const pt of rec.points ?? []) {
      const ann  = pt.annotations?.[0]
      const conf = ann?.is_confirmed ? 'true' : 'false'
      const code = ann?.code ?? ''
      const lbl  = ann?.ba_gr_label ?? ''
      const sc   = ann?.score != null ? ann.score.toFixed(4) : ''
      const mach = ann ? String(!!ann.is_machine_created) : ''
      const cust = ann ? String(!ann.is_machine_created)  : ''
      rows.push([
        `"${rec.name}"`, `"${pt.id}"`, pt.row, pt.column,
        conf, `"${code}"`, `"${lbl}"`, sc, mach, cust,
        rec.model_used ?? '', rec.grid_rows ?? '', rec.grid_cols ?? '',
      ].join(','))
    }
  }
  const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
  const url  = URL.createObjectURL(blob)
  const base = imageIds.length === 1
    ? (state.images.find(r => r.id === imageIds[0])?.name.replace(/\.[^.]+$/, '') ?? 'annotations') + '.csv'
    : 'kitcat_all_annotations.csv'
  const a = document.createElement('a')
  a.href = url; a.download = base
  document.body.appendChild(a); a.click(); document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}

// ─── Keyboard shortcuts ───────────────────────────────────────────────────────

document.addEventListener('keydown', e => {
  const tag = e.target.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
  const idx = state.selectedIdx
  const pts = state.record?.points

  switch (e.key) {
    case 'Enter':
      if (idx >= 0) confirmPoint(idx, 0)
      break
    case '1': case '2': case '3': case '4': case '5':
      if (idx >= 0) confirmPoint(idx, parseInt(e.key) - 1)
      break
    case 'u': case 'U':
      if (idx >= 0) unclassifyPoint(idx)
      break
    case 'n': case 'N':
      nextUnconfirmed()
      break
    case 'r': case 'R':
      resetView()
      break
    case 'z': case 'Z':
      if (state.selectedIdx >= 0) {
        if (state._zoomLocked) {
          state._zoomLocked = false
          resetView()
        } else {
          state._zoomLocked = true
          zoomToPoint(state.selectedIdx)
        }
      }
      break
    case 'ArrowRight': case 'ArrowDown': {
      e.preventDefault()
      if (!pts?.length) break
      const ordered = filteredPoints()
      if (!ordered.length) break
      const pos  = ordered.findIndex(p => pts[idx] && p.id === pts[idx].id)
      const next = ordered[(pos + 1) % ordered.length]
      selectPoint(pts.findIndex(p => p.id === next.id))
      break
    }
    case 'ArrowLeft': case 'ArrowUp': {
      e.preventDefault()
      if (!pts?.length) break
      const ordered = filteredPoints()
      if (!ordered.length) break
      const pos  = ordered.findIndex(p => pts[idx] && p.id === pts[idx].id)
      const prev = ordered[(pos - 1 + ordered.length) % ordered.length]
      selectPoint(pts.findIndex(p => p.id === prev.id))
      break
    }
    case 'Escape':
      selectPoint(-1)
      break
  }
})

// ─── Overlay mode toggle ─────────────────────────────────────────────────────

document.querySelectorAll('.overlay-mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    state.overlayMode = btn.dataset.mode
    document.querySelectorAll('.overlay-mode-btn').forEach(b =>
      b.classList.toggle('active', b === btn))
    drawOverlay()
  })
})

// ─── Visibility filter buttons ────────────────────────────────────────────────

document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (!btn.dataset.filter) return
    state.filterState.visibility = btn.dataset.filter
    document.querySelectorAll('.filter-btn[data-filter]').forEach(b =>
      b.classList.toggle('active', b === btn))
    drawOverlay()
  })
})

// ─── Confidence slider ────────────────────────────────────────────────────────

$confSlider?.addEventListener('input', () => {
  state.filterState.minConf = parseInt($confSlider.value)
  if ($confValue) $confValue.textContent = `${$confSlider.value}%`
  drawOverlay()
})

$sortSelect?.addEventListener('change', () => {
  state.filterState.sortBy = $sortSelect.value
  drawOverlay()
})

// ─── Label filter dropdown ───────────────────────────────────────────────────

function refreshLabelFilterList() {
  if (!$labelFilterList) return
  const q = ($labelFilterSearch?.value ?? '').trim().toLowerCase()
  const combined = [...state.allLabels, ...state.customLabels.filter(c => !state.allLabels.find(l => l.code === c.code))]
  const filtered = q
    ? combined.filter(l => l.code.toLowerCase().includes(q) || l.name.toLowerCase().includes(q))
    : combined
  $labelFilterList.innerHTML = filtered.map(l => {
    const checked = state.filterState.labelCodes.has(l.code) ? 'checked' : ''
    return `<label class="label-filter-item"><input type="checkbox" value="${l.code}" ${checked}><span class="lr-code">${l.code}</span><span class="lr-name">${l.name}</span></label>`
  }).join('')
  $labelFilterList.querySelectorAll('input[type=checkbox]').forEach(chk => {
    chk.addEventListener('change', () => {
      if (chk.checked) state.filterState.labelCodes.add(chk.value)
      else state.filterState.labelCodes.delete(chk.value)
      updateLabelFilterBadge(); drawOverlay(); renderCoverSummary()
    })
  })
}

function updateLabelFilterBadge() {
  if (!$labelFilterBtn) return
  const n = state.filterState.labelCodes.size
  $labelFilterBtn.textContent = n > 0 ? `Labels (${n}) ▾` : 'Any label ▾'
  $labelFilterBtn.classList.toggle('active', n > 0)
}

$labelFilterBtn?.addEventListener('click', e => {
  e.stopPropagation()
  $labelFilterDropdown?.classList.toggle('open')
  if ($labelFilterDropdown?.classList.contains('open')) refreshLabelFilterList()
})

$labelFilterSearch?.addEventListener('input', refreshLabelFilterList)

document.addEventListener('click', e => {
  if ($labelFilterDropdown && !$labelFilterDropdown.contains(e.target) && e.target !== $labelFilterBtn)
    $labelFilterDropdown.classList.remove('open')
})

// ─── Reclassify current image ─────────────────────────────────────────────

$btnReclassify?.addEventListener('click', async () => {
  const record = state.record
  if (!record || !state.loadedImg) return
  if (state.classifyingIds.has(record.id)) return

  const confirmedCount = record.points.filter(isConfirmed).length
  if (confirmedCount > 0) {
    if (!confirm(`This will discard ${confirmedCount} confirmed annotation${confirmedCount > 1 ? 's' : ''} and reclassify from scratch. Continue?`)) return
  }

  const { gridMethod, pointPlacement, rows, cols, patchSize } = state.uploadSettings
  const model = state.uploadSettings.model ?? 't1'
  const W = record.original_image_width
  const H = record.original_image_height

  const newPoints = gridMethod === 'noaa'
    ? generateStratifiedRandom(W, H, 2, 5, patchSize)
    : (pointPlacement === 'stratified'
      ? generateStratifiedRandom(W, H, rows, cols, patchSize)
      : generateGrid(W, H, rows, cols, patchSize))

  record.points       = newPoints
  record.model_used   = model
  record.grid_rows    = gridMethod === 'noaa' ? 2 : rows
  record.grid_cols    = gridMethod === 'noaa' ? 5 : cols
  record.patch_size   = patchSize
  record.num_confirmed = 0
  delete record._classifyDone

  state.selectedIdx = -1
  state.hoverIdx    = -1

  // Rebuild image list item so model/grid tags update
  const el = $imageList.querySelector(`[data-id="${record.id}"]`)
  if (el) el.replaceWith(buildImageItem(record))

  drawOverlay(); renderDetail(); renderProgress(); renderCoverSummary(); updateStatusText()
  $btnReclassify.disabled = true
  await classifyRecord(record, state.loadedImg).catch(err =>
    console.error('Reclassify error:', err))
  $btnReclassify.disabled = false
})

// ─── Auto-advance / Batch confirm ────────────────────────────────────────────

$autoAdvanceChk.addEventListener('change', () => { state.autoAdvance = $autoAdvanceChk.checked })

$batchConfInput?.addEventListener('input', drawHistogram)

$batchConfirm.addEventListener('click', () => {
  const pct = Math.min(100, Math.max(1, parseInt($batchConfInput?.value) || 80))
  const threshold = pct / 100
  let count = 0
  state.record?.points.forEach(point => {
    if (isConfirmed(point)) return
    const top = point.annotations?.[0]
    if (top?.score >= threshold) { top.is_confirmed = true; count++ }
  })
  if (!count) { alert(`No unconfirmed points with ≥${pct}% confidence found.`); return }
  drawOverlay(); renderProgress(); refreshImageListItem()
  saveDebounced()
  alert(`Auto-confirmed ${count} point${count > 1 ? 's' : ''} (confidence ≥ ${pct}%)`)
})

// ─── Export buttons ──────────────────────────────────────────────────────────

$btnExport.addEventListener('click', () => {
  if (!state.currentId) return
  exportCsv([state.currentId])
})

$btnExportAll.addEventListener('click', () => {
  if (!state.images.length) return
  exportCsv(state.images.map(r => r.id))
})

// ─── Upload settings wiring ──────────────────────────────────────────────────

$settingModel?.addEventListener('change', () => {
  const newKey = $settingModel.value
  state.uploadSettings.model = newKey
  // Preload the chosen model so it's ready before the first upload
  if (!_sessionPromises[newKey]) {
    getSession(newKey)
      .then(() => setModelStatus('ready', readyLabel(newKey)))
      .catch(err => setModelStatus('error', `Model load failed: ${err?.message ?? String(err)}`))
  } else {
    // Already loading or loaded — just sync the status text when settled
    _sessionPromises[newKey]
      .then(() => setModelStatus('ready', readyLabel(newKey)))
      .catch(() => {})
  }
})

document.querySelector('.preset-btn[data-grid="noaa"]')?.addEventListener('click', function() {
  state.uploadSettings.gridMethod = 'noaa'
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'))
  this.classList.add('active')
  document.getElementById('custom-grid-row')?.classList.add('hidden')
  document.getElementById('placement-row')?.classList.add('hidden')
})

document.querySelectorAll('.preset-btn[data-rows]').forEach(btn => {
  if (btn.dataset.rows === 'custom') return
  btn.addEventListener('click', () => {
    const r = parseInt(btn.dataset.rows), c = parseInt(btn.dataset.cols)
    state.uploadSettings.rows = r; state.uploadSettings.cols = c
    state.uploadSettings.gridMethod = 'uniform'
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'))
    btn.classList.add('active')
    document.getElementById('custom-grid-row')?.classList.add('hidden')
    document.getElementById('placement-row')?.classList.remove('hidden')
    const $ri = document.getElementById('grid-rows-input')
    const $ci = document.getElementById('grid-cols-input')
    if ($ri) $ri.value = r; if ($ci) $ci.value = c
  })
})

document.querySelector('.preset-btn[data-rows="custom"]')?.addEventListener('click', function() {
  state.uploadSettings.gridMethod = 'uniform'
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'))
  this.classList.add('active')
  document.getElementById('custom-grid-row')?.classList.remove('hidden')
  document.getElementById('placement-row')?.classList.remove('hidden')
})

const $gridRowsInput = document.getElementById('grid-rows-input')
const $gridColsInput = document.getElementById('grid-cols-input')
$gridRowsInput?.addEventListener('change', () => {
  const v = Math.min(50, Math.max(2, parseInt($gridRowsInput.value) || 10))
  state.uploadSettings.rows = v; $gridRowsInput.value = v
})
$gridColsInput?.addEventListener('change', () => {
  const v = Math.min(50, Math.max(2, parseInt($gridColsInput.value) || 10))
  state.uploadSettings.cols = v; $gridColsInput.value = v
})

document.querySelectorAll('.placement-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    state.uploadSettings.pointPlacement = btn.dataset.placement
    document.querySelectorAll('.placement-btn').forEach(b => b.classList.remove('active'))
    btn.classList.add('active')
  })
})

document.getElementById('setting-patch-size')?.addEventListener('change', function() {
  state.uploadSettings.patchSize = parseInt(this.value) || 112
})

// ─── Upload zone ──────────────────────────────────────────────────────────────

const $dropZone  = document.getElementById('drop-zone')
const $fileInput = document.getElementById('file-input')

$dropZone.addEventListener('click', () => $fileInput.click())
$fileInput.addEventListener('change', () => { uploadFiles([...$fileInput.files]); $fileInput.value = '' })

$dropZone.addEventListener('dragover',  e => { e.preventDefault(); $dropZone.classList.add('drag-over') })
$dropZone.addEventListener('dragleave', ()=> $dropZone.classList.remove('drag-over'))
$dropZone.addEventListener('drop', e => {
  e.preventDefault(); $dropZone.classList.remove('drag-over')
  const files = [...e.dataTransfer.files].filter(f => f.type.startsWith('image/'))
  if (files.length) uploadFiles(files)
})

// ─── Resize observer ─────────────────────────────────────────────────────────

new ResizeObserver(() => { if (state.loadedImg) resizeAndDraw() }).observe($container)

// ─── Init ─────────────────────────────────────────────────────────────────────

async function init() {
  // Verify ONNX Runtime loaded from CDN
  if (typeof ort === 'undefined') {
    setModelStatus('error', 'ONNX Runtime failed to load — check internet')
    return
  }

  // Configure ONNX WASM path (CDN-served wasm files)
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/'
  _ortReady = true
  // Status stays 'loading' (from HTML) while we preload the model below

  // Load labels for label picker — fall back to T3_DESCRIPTIONS if JSON not yet exported
  try {
    const t3codes = await getLabelCodes('t3')
    state.allLabels = t3codes.map(code => ({ code, name: T3_DESCRIPTIONS[code] ?? code, is_custom: false }))
    const t1codes = await getLabelCodes('t1').catch(() => [])
    t1codes.forEach(code => {
      if (!state.allLabels.find(l => l.code === code))
        state.allLabels.push({ code, name: T1_DESCRIPTIONS[code] ?? T3_DESCRIPTIONS[code] ?? code, is_custom: false })
    })
  } catch {
    // Models not yet exported — use hardcoded descriptions as fallback
    const t3entries = Object.entries(T3_DESCRIPTIONS).map(([code, name]) => ({ code, name, is_custom: false }))
    const t1entries = Object.entries(T1_DESCRIPTIONS)
      .filter(([code]) => !T3_DESCRIPTIONS[code])
      .map(([code, name]) => ({ code, name, is_custom: false }))
    state.allLabels = [...t3entries, ...t1entries]
    setModelStatus('ready', 'Browser mode — run export_onnx.py to enable inference')
    return
  }

  // Preload both models in parallel so switching between T3/T1 is instant
  const defaultKey = state.uploadSettings.model ?? 't3'
  const otherKey   = defaultKey === 't3' ? 't1' : 't3'
  const loadDefault = getSession(defaultKey)
    .then(() => setModelStatus('ready', readyLabel(defaultKey)))
    .catch(err => setModelStatus('error', `Model load failed: ${err?.message ?? String(err)}`))
  // Load other model silently in background (don't change status text on completion)
  getSession(otherKey).catch(() => {})  // failure handled when actually used
}

init()
