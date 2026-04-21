'use strict'

// ─── Category → color mapping ────────────────────────────────────────────────

const T3_DESCRIPTIONS = {
  TURFH:'Turf Algae (High)', TURFR:'Turf Algae (Rubble)', TURF:'Turf Alga',
  EMA:'Encrusting Macroalgae', HALI:'Halimeda spp.', DICO:'Dictyota spp.',
  LOBO:'Lobophora spp.', MA:'Macroalga',
  CCAH:'Crustose Coralline Algae (Healthy)', CCAR:'Crustose Coralline Algae (Rubble)', CCA:'Coralline Alga',
  POCS:'Pocillopora spp.', MOEN:'Montipora (encrusting)', MOSP:'Montipora (submassive)',
  MOBR:'Montipora (branching)', PESP:'Porites (encrusting)', POMA:'Porites (massive)',
  POBR:'Porites (branching)', ACBR:'Acropora (branching)', ACTA:'Acropora (tabular)',
  ACEN:'Acropora (encrusting)', ACDI:'Acropora (digitate)', PASP:'Pavona spp.',
  FUSP:'Fungia spp.', LEPT:'Leptastrea spp.', FASP:'Favites spp.',
  PLSP:'Platygyra spp.', PLER:'Platygyra erosa', LOBS:'Lobophyllia spp.',
  GAST:'Galaxea spp.', HCON:'Hard Coral (unidentified)', CORAL:'Coral',
  OCTO:'Octocoral', ZOAN:'Zoanthids', SC:'Soft Coral',
  SAND:'Sand', RUB:'Rubble', HBON:'Hard Bottom (bare)', HBOA:'Hard Bottom (algae)', SED:'Sediment',
  SPON:'Sponge', MF:'Mobile Fauna', I:'Sessile Invertebrate', INVERT:'Sessile Invertebrate', UNK:'Unknown',
}

const T3_CATEGORY = {
  POCS:'coral', MOEN:'coral', MOSP:'coral', MOBR:'coral', PESP:'coral', POMA:'coral',
  POBR:'coral', ACBR:'coral', ACTA:'coral', ACEN:'coral', ACDI:'coral', PASP:'coral',
  FUSP:'coral', LEPT:'coral', FASP:'coral', PLSP:'coral', PLER:'coral', LOBS:'coral',
  GAST:'coral', HCON:'coral', CORAL:'coral',
  CCAH:'cca', CCAR:'cca', CCA:'cca',
  TURFH:'turf', TURFR:'turf', TURF:'turf',
  EMA:'macro', HALI:'macro', DICO:'macro', LOBO:'macro', MA:'macro',
  OCTO:'soft', ZOAN:'soft', SC:'soft',
  SAND:'sed', RUB:'sed', SED:'sed', HBON:'sed', HBOA:'sed',
  SPON:'other', MF:'other', I:'other', INVERT:'other', UNK:'other',
}

const CAT_COLOR = {
  coral: '#f97316', cca: '#a855f7', turf: '#a3e635', macro: '#22c55e',
  soft:  '#ec4899', sed: '#d97706', other: '#60a5fa',
}

function getPointColor(point) {
  const ann = point.annotations?.[0]
  if (!ann) return '#6b7280'
  if (ann.is_confirmed) return '#ffffff'
  return CAT_COLOR[T3_CATEGORY[ann.code]] ?? '#60a5fa'
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

const _sessions = {}
const _labelsCache = {}
let   _ortReady = false

async function getSession(key) {
  if (_sessions[key]) return _sessions[key]
  setModelStatus('loading', `Loading ${key.toUpperCase()} model (~20 MB)…`)
  const session = await ort.InferenceSession.create(MODEL_URLS[key], {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  _sessions[key] = session
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
function preprocessPatch(imgEl, cx, cy) {
  const half = PATCH_SIZE >> 1
  const sx = Math.max(0, cx - half)
  const sy = Math.max(0, cy - half)
  const sw = Math.min(imgEl.naturalWidth  - sx, PATCH_SIZE)
  const sh = Math.min(imgEl.naturalHeight - sy, PATCH_SIZE)

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

async function classifyPatch(imgEl, cx, cy, key) {
  const [session, labels] = await Promise.all([getSession(key), getLabelCodes(key)])
  const tensor = preprocessPatch(imgEl, cx, cy)
  const feeds  = { [session.inputNames[0]]: tensor }
  const output = await session.run(feeds)
  const probs  = Array.from(output[session.outputNames[0]].data)
  return probs
    .map((v, i) => ({ code: labels[i], score: v }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
}

// ─── Grid generation ─────────────────────────────────────────────────────────

function generateGrid(imgW, imgH, nRows, nCols) {
  const margin = PATCH_SIZE >> 1
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
  uploadSettings: { model: 't3', rows: 10, cols: 10 },
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
const $batchConfirm   = document.getElementById('btn-batch-confirm')
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

function startClassifyAnimation() {
  $classifyLoading?.classList.add('visible')
}

// ─── Upload ──────────────────────────────────────────────────────────────────

async function uploadFiles(files) {
  if (!_ortReady) {
    alert('ONNX Runtime is not ready. Check your internet connection and reload.')
    return
  }
  const { model, rows, cols } = state.uploadSettings

  for (const file of files) {
    const placeholder = createUploadingItem(file.name)
    $imageList.appendChild(placeholder)
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
      const record = {
        id:   crypto.randomUUID(),
        name: file.name,
        image: dataUrl,
        thumbnail: makeThumbnail(img),
        original_image_width:  img.naturalWidth,
        original_image_height: img.naturalHeight,
        patch_size:  PATCH_SIZE,
        model_used:  model,
        grid_rows:   rows,
        grid_cols:   cols,
        points:      generateGrid(img.naturalWidth, img.naturalHeight, rows, cols),
        num_confirmed: 0,
      }
      state.images.push(record)
      placeholder.replaceWith(buildImageItem(record))
      if (!state.currentId) await loadImage(record.id)
      // 4. Classify in background (non-blocking)
      classifyRecord(record, img).catch(err =>
        console.error('Classification error:', err))
    } catch (err) {
      placeholder.replaceWith(buildErrorItem(file.name, err?.message ?? String(err)))
    }
  }
}

async function classifyRecord(record, imgEl) {
  const key    = record.model_used ?? 't3'
  const points = record.points
  let done = 0
  let inferenceError = null
  state.classifyingIds.add(record.id)
  startClassifyAnimation()

  for (const point of points) {
    try {
      const top5 = await classifyPatch(imgEl, point.column, point.row, key)
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

  state.classifyingIds.delete(record.id)
  if (state.classifyingIds.size === 0) $classifyLoading?.classList.remove('visible')
  record.num_confirmed = record.points.filter(isConfirmed).length
  if (record.id === state.currentId) {
    drawOverlay(); renderDetail(); renderProgress(); refreshImageListItem(); updateStatusText()
  }
  setModelStatus('ready', `Browser mode — ${key.toUpperCase()} ready`)
}

// ─── Image list ──────────────────────────────────────────────────────────────

function buildImageItem(record) {
  const confirmed = (record.points ?? []).filter(isConfirmed).length
  const total     = record.points?.length ?? 0
  const pct       = total > 0 ? Math.round(confirmed / total * 100) : 0
  const modelTag  = record.model_used ? record.model_used.toUpperCase() : ''
  const gridTag   = record.grid_rows  ? `${record.grid_rows}x${record.grid_cols}` : ''

  const el = document.createElement('div')
  el.className = 'image-item' + (record.id === state.currentId ? ' active' : '')
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
}

// ─── Load image ──────────────────────────────────────────────────────────────

async function loadImage(id) {
  state.currentId   = id
  state.selectedIdx = -1
  state.hoverIdx    = -1
  state.zoom = 1.0; state.panX = 0; state.panY = 0

  document.querySelectorAll('.image-item').forEach(el =>
    el.classList.toggle('active', el.dataset.id === id))

  // Direct reference — mutations to points are visible immediately
  const record = state.images.find(r => r.id === id)
  if (!record) return
  state.record  = record
  state.isDirty = false

  $placeholder.classList.add('hidden')

  const img = new Image()
  img.onload = () => { state.loadedImg = img; resizeAndDraw() }
  img.src = record.image

  renderProgress()
  renderDetail()
  updateStatusText()
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
  if (!t || !state.record?.points.length) return

  const patchSize = state.record.patch_size ?? PATCH_SIZE
  const r = Math.max(4, Math.min(11, (patchSize / 2) * t.scale * 0.35))
  const visible = new Set(filteredPoints().map(p => p.id))

  state.record.points.forEach((point, idx) => {
    if (!visible.has(point.id)) return
    const x     = t.offsetX + point.column * t.scale
    const y     = t.offsetY + point.row    * t.scale
    const color  = getPointColor(point)
    const isSel  = idx === state.selectedIdx
    const isHov  = idx === state.hoverIdx
    const half   = isSel ? r * 1.7 : r + (isHov && !isSel ? 2 : 0)

    // Square: subtle fill so points are visible when zoomed out
    overlayCtx.shadowBlur  = isSel ? 18 : 0
    overlayCtx.shadowColor = color
    overlayCtx.fillStyle   = color
    overlayCtx.globalAlpha = isConfirmed(point) ? 0.18 : 0.12
    overlayCtx.fillRect(x - half, y - half, half * 2, half * 2)
    overlayCtx.globalAlpha = 1.0
    overlayCtx.shadowBlur  = 0

    // Square outline
    overlayCtx.strokeStyle = color
    overlayCtx.lineWidth   = isSel ? 2.5 : 1.5
    overlayCtx.strokeRect(x - half, y - half, half * 2, half * 2)

    // Tiny center dot for precise location
    overlayCtx.beginPath()
    overlayCtx.arc(x, y, 2, 0, Math.PI * 2)
    overlayCtx.fillStyle   = color
    overlayCtx.globalAlpha = isConfirmed(point) ? 1.0 : 0.85
    overlayCtx.fill()
    overlayCtx.globalAlpha = 1.0

    if (isSel) {
      // Inner selection square ring
      overlayCtx.strokeStyle = 'rgba(255,255,255,0.75)'
      overlayCtx.lineWidth   = 2
      overlayCtx.strokeRect(x - half - 5, y - half - 5, (half + 5) * 2, (half + 5) * 2)
      // Outer dim square ring
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

  // Sidebar patch preview
  _drawPatchPreview(state.selectedIdx >= 0 ? state.record?.points[state.selectedIdx] : null)
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

  if (visibility === 'unclassified') {
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
  const hitRadius  = Math.max(7, Math.min(16, (PATCH_SIZE / 2) * t.scale * 0.6))
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
  drawOverlay()
  renderDetail()
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

function renderProgress() {
  const pts = state.record?.points
  if (!pts?.length) { $progressText.textContent = ''; return }
  const confirmed    = pts.filter(isConfirmed).length
  const unclassified = pts.filter(isUnclassified).length
  const unconfirmed  = pts.length - confirmed - unclassified
  $progressText.textContent =
    `${confirmed}/${pts.length} confirmed  ${unconfirmed} unconfirmed  ${unclassified} unclassified`
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
      updateLabelFilterBadge(); drawOverlay()
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

// ─── Auto-advance / Batch confirm ────────────────────────────────────────────

$autoAdvanceChk.addEventListener('change', () => { state.autoAdvance = $autoAdvanceChk.checked })

$batchConfirm.addEventListener('click', () => {
  const threshold = 0.90
  let count = 0
  state.record?.points.forEach(point => {
    if (isConfirmed(point)) return
    const top = point.annotations?.[0]
    if (top?.score >= threshold) { top.is_confirmed = true; count++ }
  })
  if (!count) { alert('No unconfirmed points with ≥90% confidence found.'); return }
  saveDebounced()
  alert(`Auto-confirmed ${count} point${count > 1 ? 's' : ''} (confidence ≥ 90%)`)
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
  state.uploadSettings.model = $settingModel.value
})

document.querySelectorAll('.preset-btn[data-rows]').forEach(btn => {
  if (btn.dataset.rows === 'custom') return
  btn.addEventListener('click', () => {
    const r = parseInt(btn.dataset.rows), c = parseInt(btn.dataset.cols)
    state.uploadSettings.rows = r; state.uploadSettings.cols = c
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'))
    btn.classList.add('active')
    document.getElementById('custom-grid-row')?.classList.add('hidden')
    const $ri = document.getElementById('grid-rows-input')
    const $ci = document.getElementById('grid-cols-input')
    if ($ri) $ri.value = r; if ($ci) $ci.value = c
  })
})

document.querySelector('.preset-btn[data-rows="custom"]')?.addEventListener('click', function() {
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'))
  this.classList.add('active')
  document.getElementById('custom-grid-row')?.classList.remove('hidden')
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
  setModelStatus('ready', 'Browser mode — ready')

  // Load labels for label picker — fall back to T3_DESCRIPTIONS if JSON not yet exported
  try {
    const t3codes = await getLabelCodes('t3')
    state.allLabels = t3codes.map(code => ({ code, name: T3_DESCRIPTIONS[code] ?? code, is_custom: false }))
    const t1codes = await getLabelCodes('t1').catch(() => [])
    t1codes.forEach(code => {
      if (!state.allLabels.find(l => l.code === code))
        state.allLabels.push({ code, name: T3_DESCRIPTIONS[code] ?? code, is_custom: false })
    })
  } catch {
    // Models not yet exported — use hardcoded descriptions as fallback
    state.allLabels = Object.entries(T3_DESCRIPTIONS).map(([code, name]) => ({ code, name, is_custom: false }))
    setModelStatus('ready', 'Browser mode — run export_onnx.py to enable inference')
  }
}

init()
