'use strict'

// ─── Category → color mapping ────────────────────────────────────────────────

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

function isConfirmed(p) { return !!p.annotations?.[0]?.is_confirmed }
function isUnclassified(p) { return !p.annotations?.length }

// ─── State ───────────────────────────────────────────────────────────────────

const state = {
  images: [],
  currentId: null,
  record: null,
  selectedIdx: -1,
  hoverIdx: -1,
  isDirty: false,
  autoAdvance: true,
  uploadSettings: { model: 't1', rows: 10, cols: 10, gridMethod: 'noaa', pointPlacement: 'uniform' },
  filterState: {
    visibility: 'all',
    labelCodes: new Set(),
    minConf: 0,
    sortBy: 'position',
  },
  allLabels: [],
  customLabels: [],
  zoom: 1.0,
  panX: 0,
  panY: 0,
  loadedImg: null,
  _panning: false,
  _panStart: null,
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
const $statusDot      = document.getElementById('model-status-dot')
const $statusText     = document.getElementById('model-status-text')
const $autoAdvanceChk = document.getElementById('auto-advance')
const $batchConfirm   = document.getElementById('btn-batch-confirm')
const $batchConfInput = document.getElementById('batch-conf-input')
const $confHistogram  = document.getElementById('conf-histogram')
const $coverSummary   = document.getElementById('cover-summary')
const $btnReclassify  = document.getElementById('btn-reclassify')
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

// ─── API ─────────────────────────────────────────────────────────────────────

const api = {
  health:      ()          => fetch('/api/health').then(r => r.json()),
  list:        ()          => fetch('/api/images').then(r => r.json()),
  get:         (id)        => fetch(`/api/images/${id}`).then(r => r.json()),
  del:         (id)        => fetch(`/api/images/${id}`, { method: 'DELETE' }),
  save:        (id, pts)   => fetch(`/api/images/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ points: pts }),
  }).then(r => r.json()),
  exportUrl:   (id)        => `/api/images/${id}/export.csv`,
  labels:      ()          => fetch('/api/labels').then(r => r.json()),
  createLabel: (code, name) => fetch('/api/labels', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, name }),
  }).then(r => { if (!r.ok) return r.json().then(e => Promise.reject(e)); return r.json() }),
}

// ─── Upload ──────────────────────────────────────────────────────────────────

async function uploadFiles(files) {
  const { model, rows, cols, gridMethod, pointPlacement } = state.uploadSettings
      const apiGridMethod = gridMethod === 'noaa' ? 'noaa' : (pointPlacement === 'stratified' ? 'stratified' : 'uniform')
  for (const file of files) {
    const placeholder = createUploadingItem(file.name)
    $imageList.appendChild(placeholder)
    try {
      const fd = new FormData()
      fd.append('image', file)
      fd.append('model_name', model)
      fd.append('grid_method', apiGridMethod)
      fd.append('grid_rows',   String(gridMethod === 'noaa' ? 2 : rows))
      fd.append('grid_cols',   String(gridMethod === 'noaa' ? 5 : cols))
      const res = await fetch('/api/images', { method: 'POST', body: fd })
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
      const record = await res.json()
      state.images.push(record)
      placeholder.replaceWith(buildImageItem(record))
      if (!state.currentId) await loadImage(record.id)
    } catch (err) {
      placeholder.replaceWith(buildErrorItem(file.name, err.message ?? String(err)))
    }
  }
}

// ─── Image list ──────────────────────────────────────────────────────────────

function buildImageItem(record) {
  const confirmed = (record.points ?? []).filter(isConfirmed).length
  const total     = record.points?.length ?? 100
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
        <span>${record.num_confirmed ?? confirmed} confirmed</span>
        <span>${confirmed}/${total}</span>
      </div>
    </div>
    <button class="btn-delete" title="Remove image" aria-label="Remove">x</button>
  `
  el.querySelector('.btn-delete').addEventListener('click', async (e) => {
    e.stopPropagation()
    if (!confirm(`Remove "${record.name}"?`)) return
    await api.del(record.id)
    state.images = state.images.filter(i => i.id !== record.id)
    el.remove()
    if (state.currentId === record.id) {
      state.currentId = null; state.record = null
      clearCanvas(); renderDetail(); renderProgress()
    }
  })
  el.addEventListener('click', () => loadImage(record.id))
  return el
}

function createUploadingItem(name) {
  const el = document.createElement('div')
  el.className = 'image-item uploading'
  el.innerHTML = `<div class="upload-spinner"></div><div class="image-meta"><div class="image-name">${name}</div><div class="image-tags">Uploading...</div></div>`
  return el
}

function buildErrorItem(name, msg) {
  const el = document.createElement('div')
  el.className = 'image-item error'
  el.innerHTML = `<span class="error-icon">!</span><div class="image-meta"><div class="image-name" title="${msg}">${name}</div><div class="image-tags error-msg">${String(msg).slice(0,60)}</div></div>`
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

async function refreshImageList() {
  const list = await api.list()
  state.images = list
  $imageList.innerHTML = ''
  list.forEach(r => $imageList.appendChild(buildImageItem(r)))
}

// ─── Load image ──────────────────────────────────────────────────────────────

async function loadImage(id) {
  state.currentId   = id
  state.selectedIdx = -1
  state.hoverIdx    = -1
  state.zoom = 1.0; state.panX = 0; state.panY = 0

  document.querySelectorAll('.image-item').forEach(el =>
    el.classList.toggle('active', el.dataset.id === id))

  const record = await api.get(id)
  state.record = { ...record, points: record.points.map(p => ({ ...p, annotations: [...(p.annotations ?? [])] })) }
  state.isDirty = false

  $placeholder.classList.add('hidden')

  const img = new Image()
  img.onload = () => { state.loadedImg = img; resizeAndDraw() }
  img.src = record.image

  renderProgress()
  renderDetail()
  updateStatusText()
  // Auto-select first point if already classified
  if (state.selectedIdx < 0 && state.record.points.some(p => p.annotations?.length)) {
    state.selectedIdx = 0
  }
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

  const patchSize = state.record.patch_size ?? 112
  const sparse = state.record.points.length <= 12
  const r = Math.max(sparse ? 7 : 4, Math.min(sparse ? 16 : 11, (patchSize / 2) * t.scale * (sparse ? 0.55 : 0.35)))
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
  const codeW = ctx.measureText(code).width
  const sep   = name ? ' · ' : ''
  const nameSnip = name.slice(0, 26)
  ctx.font   = `400 ${fs}px sans-serif`
  const nameW = name ? ctx.measureText(sep + nameSnip).width : 0
  const boxW  = codeW + nameW + padX * 2
  const boxH  = fs + padY * 2
  const bx    = Math.max(2, Math.min(cx - boxW / 2, ctx.canvas.width - boxW - 2))
  const by    = Math.max(2, bottomY - boxH)
  // pill
  ctx.shadowBlur  = 6
  ctx.shadowColor = 'rgba(0,0,0,0.8)'
  ctx.fillStyle   = 'rgba(8,10,22,0.93)'
  ctx.beginPath()
  ctx.roundRect(bx, by, boxW, boxH, 4)
  ctx.fill()
  ctx.shadowBlur  = 0
  // connector
  ctx.beginPath()
  ctx.moveTo(cx, by + boxH)
  ctx.lineTo(cx, bottomY + 5)
  ctx.strokeStyle = 'rgba(255,255,255,0.35)'
  ctx.lineWidth   = 1
  ctx.stroke()
  // code (yellow)
  ctx.textAlign    = 'left'
  ctx.textBaseline = 'middle'
  const midY       = by + boxH / 2
  ctx.font         = `700 ${fs}px monospace`
  ctx.fillStyle    = '#fbbf24'
  ctx.fillText(code, bx + padX, midY)
  // name (muted)
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
  const patchSize = state.record.patch_size ?? 112
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
  const patchSize  = state.record?.patch_size ?? 112
  const hitRadius  = Math.max(7, Math.min(16, (patchSize / 2) * t.scale * 0.6))
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
  const rect = $overlayCanvas.getBoundingClientRect()
  const mx = e.clientX - rect.left
  const my = e.clientY - rect.top
  const wx = (mx - t.offsetX) / t.scale
  const wy = (my - t.offsetY) / t.scale
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
  state.zoom = Math.min(15, Math.max(0.5, state.zoom * factor))
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
  state._panning = true
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
  state.panX = newPanX
  state.panY = newPanY
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
    state.hoverIdx = -1
    drawOverlay()
    $overlayCanvas.style.cursor = 'default'
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
  const y = (ch - ih * scale) / 2 + state.panY + point.row * scale
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
      ${anns.length ? predRows : '<div class="no-predictions">No ML predictions</div>'}
    </div>
    <div class="detail-actions">
      ${!confirmed && anns.length ? '<button class="btn-primary" id="btn-confirm-top">Confirm #1 <kbd>Enter</kbd></button>' : ''}
      ${confirmed                 ? '<button class="btn-secondary" id="btn-unconfirm">Undo confirm</button>' : ''}
      <button class="btn-ghost" id="btn-unclassify">Unclassify <kbd>U</kbd></button>
    </div>
    <div class="custom-label-section">
      <div class="custom-label-header">Apply Label</div>
      <div class="label-search-wrap">
        <input type="text" id="label-search-input" placeholder="Search labels..." autocomplete="off">
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
        $lsi.value = ''
        renderLabelResults('')
      })
    })
  }

  renderLabelResults('')
  $lsi.addEventListener('input', () => renderLabelResults($lsi.value))

  const $btnAdd = $detail.querySelector('#btn-add-label')
  const $errEl  = $detail.querySelector('#new-label-error')
  $btnAdd?.addEventListener('click', async () => {
    const code = $detail.querySelector('#new-label-code').value.trim().toUpperCase()
    const name = $detail.querySelector('#new-label-name').value.trim()
    $errEl.textContent = ''
    if (!code || !name) { $errEl.textContent = 'Code and name are required.'; return }
    try {
      const created = await api.createLabel(code, name)
      state.customLabels.push(created)
      state.allLabels.push(created)
      applyLabel(idx, created.code, created.name, true)
      $detail.querySelector('#new-label-code').value = ''
      $detail.querySelector('#new-label-name').value = ''
      renderLabelResults('')
      refreshLabelFilterList()
    } catch (err) {
      $errEl.textContent = err.detail ?? String(err)
    }
  })
}

// ─── Apply label ─────────────────────────────────────────────────────────────

function applyLabel(pointIdx, code, name, isCustom) {
  const point = state.record?.points[pointIdx]
  if (!point) return
  const ann = {
    id: typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2),
    benthic_attribute: code,
    ba_gr: code,
    ba_gr_label: name,
    code,
    is_confirmed: true,
    is_machine_created: false,
    score: null,
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
  renderDetail()
  drawOverlay()
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
    if (!isConfirmed(p)) {
      selectPoint(pts.findIndex(q => q.id === p.id))
      return
    }
  }
  selectPoint(-1)
}

// Called after confirming/labeling a point so we advance correctly even when
// the just-acted-on point disappears from the filtered list (e.g. filter=unconfirmed).
function _nextUnconfirmedAfterMutation(prevFilteredPos, mutatedId) {
  const pts = state.record?.points
  if (!pts?.length) return
  const ordered = filteredPoints() // re-evaluated after the mutation
  // If the mutated point is still visible (filter=all), start after its position.
  // If it was removed from the list (filter=unconfirmed), prevFilteredPos now
  // points to what was the next item, so start there.
  const stillPresent = ordered.findIndex(p => p.id === mutatedId) !== -1
  const startPos = stillPresent ? prevFilteredPos + 1 : prevFilteredPos
  for (let i = 0; i < ordered.length; i++) {
    const p = ordered[(startPos + i) % ordered.length]
    if (p.id !== mutatedId && !isConfirmed(p)) {
      selectPoint(pts.findIndex(q => q.id === p.id))
      return
    }
  }
  selectPoint(-1)
}

// ─── Save ────────────────────────────────────────────────────────────────────

let _saveTimer = null
function saveDebounced() {
  state.isDirty = true
  clearTimeout(_saveTimer)
  _saveTimer = setTimeout(async () => {
    if (!state.currentId || !state.isDirty) return
    await api.save(state.currentId, state.record.points)
    state.isDirty = false
    drawOverlay()
    renderDetail()
    renderProgress()
    refreshImageListItem()
  }, 250)
}

// ─── Progress ────────────────────────────────────────────────────────────────

function drawHistogram() {
  if (!$confHistogram) return
  const panel = document.getElementById('histogram-panel')
  const pts = state.record?.points ?? []
  const scored = pts.filter(p => p.annotations?.[0]?.score != null)
  if (!scored.length) { panel?.classList.remove('has-data'); return }
  panel?.classList.add('has-data')
  const cssW = $confHistogram.parentElement?.clientWidth || 268
  $confHistogram.width  = cssW
  $confHistogram.height = 40
  const ctx = $confHistogram.getContext('2d')
  const W = cssW, H = 40
  ctx.clearRect(0, 0, W, H)
  const bins = new Array(10).fill(0)
  scored.forEach(p => { const s = p.annotations[0].score; bins[Math.min(9, Math.floor(s * 10))]++ })
  const maxCount = Math.max(1, ...bins)
  const barW = W / 10
  bins.forEach((count, i) => {
    if (!count) return
    const bH = Math.max(2, Math.round((count / maxCount) * (H - 4)))
    ctx.fillStyle = `hsl(${Math.round((i / 9) * 120)}, 65%, 52%)`
    ctx.fillRect(i * barW + 1, H - bH, barW - 2, bH)
  })
  ctx.fillStyle = 'rgba(148,163,184,0.6)'
  ctx.font = '8px system-ui,sans-serif'; ctx.textAlign = 'center'
  ;['0', '', '', '', '', '50', '', '', '', '100'].forEach((lbl, i) => {
    if (lbl) ctx.fillText(lbl + '%', i * barW + barW / 2, H - 1)
  })
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
  pts.forEach(p => { const code = p.annotations?.[0]?.code; if (code) counts[code] = (counts[code] ?? 0) + 1 })
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1])
  if (!sorted.length) { $coverSummary.innerHTML = ''; $coverSummary.classList.add('hidden'); return }
  const total = pts.length
  $coverSummary.classList.remove('hidden')
  const chips = sorted.slice(0, 12).map(([code, n]) =>
    `<span class="cover-chip"><span class="cover-code">${code}</span><span class="cover-pct">${Math.round(n / total * 100)}%</span></span>`
  ).join('')
  const extra = sorted.length > 12 ? `<span class="cover-more">+${sorted.length - 12} more</span>` : ''
  $coverSummary.innerHTML = '<span class="cover-label">Cover:</span>' + chips + extra
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
    state.filterState.visibility = btn.dataset.filter
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.toggle('active', b === btn))
    drawOverlay()
  })
})

// ─── Confidence slider ────────────────────────────────────────────────────────

if ($confSlider) {
  $confSlider.addEventListener('input', () => {
    state.filterState.minConf = parseInt($confSlider.value)
    if ($confValue) $confValue.textContent = `${$confSlider.value}%`
    drawOverlay()
  })
}

// ─── Sort select ─────────────────────────────────────────────────────────────

if ($sortSelect) {
  $sortSelect.addEventListener('change', () => {
    state.filterState.sortBy = $sortSelect.value
    drawOverlay()
  })
}

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
      updateLabelFilterBadge()
      drawOverlay()
    })
  })
}

function updateLabelFilterBadge() {
  if (!$labelFilterBtn) return
  const n = state.filterState.labelCodes.size
  $labelFilterBtn.textContent = n > 0 ? `Labels (${n}) v` : 'Any label v'
  $labelFilterBtn.classList.toggle('active', n > 0)
}

if ($labelFilterBtn) {
  $labelFilterBtn.addEventListener('click', e => {
    e.stopPropagation()
    $labelFilterDropdown?.classList.toggle('open')
    if ($labelFilterDropdown?.classList.contains('open')) refreshLabelFilterList()
  })
}

if ($labelFilterSearch) {
  $labelFilterSearch.addEventListener('input', refreshLabelFilterList)
}

document.addEventListener('click', e => {
  if ($labelFilterDropdown && !$labelFilterDropdown.contains(e.target) && e.target !== $labelFilterBtn) {
    $labelFilterDropdown.classList.remove('open')
  }
})

// ─── Auto-advance toggle ──────────────────────────────────────────────────────

$autoAdvanceChk.addEventListener('change', () => { state.autoAdvance = $autoAdvanceChk.checked })

$batchConfInput?.addEventListener('input', drawHistogram)

// ─── Batch confirm ────────────────────────────────────────────────────────────

$batchConfirm.addEventListener('click', async () => {
  const pct = Math.min(100, Math.max(1, parseInt($batchConfInput?.value) || 80))
  const threshold = pct / 100
  let count = 0
  state.record?.points.forEach(point => {
    if (isConfirmed(point)) return
    const top = point.annotations?.[0]
    if (top?.score >= threshold) { top.is_confirmed = true; count++ }
  })
  if (!count) { alert(`No unconfirmed points with ≥${pct}% confidence found.`); return }
  await api.save(state.currentId, state.record.points)
  drawOverlay(); renderDetail(); renderProgress(); refreshImageListItem()
  alert(`Auto-confirmed ${count} point${count > 1 ? 's' : ''} (confidence ≥ ${pct}%)`)
})

// ─── Reclassify current image ─────────────────────────────────────────────────

$btnReclassify?.addEventListener('click', async () => {
  const record = state.record
  if (!record) return
  const confirmedCount = record.points.filter(isConfirmed).length
  if (confirmedCount > 0) {
    if (!confirm(`This will discard ${confirmedCount} confirmed annotation${confirmedCount > 1 ? 's' : ''} and reclassify from scratch. Continue?`)) return
  }
  const { gridMethod, pointPlacement, rows, cols } = state.uploadSettings
  const model = state.uploadSettings.model ?? 't1'
  const apiGridMethod = gridMethod === 'noaa' ? 'noaa' : (pointPlacement === 'stratified' ? 'stratified' : 'uniform')
  $btnReclassify.disabled = true
  $btnReclassify.textContent = '↺ Reclassifying…'
  try {
    const res = await fetch(`/api/images/${record.id}/reclassify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_name:  model,
        grid_method: apiGridMethod,
        grid_rows:   gridMethod === 'noaa' ? 2 : rows,
        grid_cols:   gridMethod === 'noaa' ? 5 : cols,
      }),
    })
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
    state.selectedIdx = -1
    await loadImage(record.id)
    const el = $imageList.querySelector(`[data-id="${record.id}"]`)
    if (el) el.replaceWith(buildImageItem(state.record))
  } catch (err) {
    alert(`Reclassify failed: ${err.message ?? String(err)}`)
  } finally {
    $btnReclassify.disabled = false
    $btnReclassify.textContent = '↺ Reclassify current image'
  }
})

// ─── Export ───────────────────────────────────────────────────────────────────

$btnExport.addEventListener('click', () => {
  if (!state.currentId) return
  triggerDownload(api.exportUrl(state.currentId))
})

$btnExportAll.addEventListener('click', async () => {
  for (const img of state.images) {
    triggerDownload(api.exportUrl(img.id))
    await new Promise(r => setTimeout(r, 200))
  }
})

function triggerDownload(url) {
  const a = document.createElement('a')
  a.href = url
  a.download = ''
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
}

// ─── Upload settings wiring ──────────────────────────────────────────────────

if ($settingModel) {
  $settingModel.addEventListener('change', () => {
    state.uploadSettings.model = $settingModel.value
  })
}

document.querySelector('.preset-btn[data-grid="noaa"]')?.addEventListener('click', function() {
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'))
  this.classList.add('active')
  state.uploadSettings.gridMethod = 'noaa'
  document.getElementById('custom-grid-row')?.classList.add('hidden')
  document.getElementById('placement-row')?.classList.add('hidden')
})

document.querySelectorAll('.preset-btn[data-rows]').forEach(btn => {
  if (btn.dataset.rows === 'custom') return
  btn.addEventListener('click', () => {
    const r = parseInt(btn.dataset.rows)
    const c = parseInt(btn.dataset.cols)
    state.uploadSettings.rows = r
    state.uploadSettings.cols = c
    state.uploadSettings.gridMethod = 'uniform'
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'))
    btn.classList.add('active')
    document.getElementById('custom-grid-row')?.classList.add('hidden')
    document.getElementById('placement-row')?.classList.remove('hidden')
    const $ri = document.getElementById('grid-rows-input')
    const $ci = document.getElementById('grid-cols-input')
    if ($ri) $ri.value = r
    if ($ci) $ci.value = c
  })
})

document.querySelector('.preset-btn[data-rows="custom"]')?.addEventListener('click', function() {
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'))
  this.classList.add('active')
  state.uploadSettings.gridMethod = 'uniform'
  document.getElementById('custom-grid-row')?.classList.remove('hidden')
  document.getElementById('placement-row')?.classList.remove('hidden')
})

const $gridRowsInput = document.getElementById('grid-rows-input')
const $gridColsInput = document.getElementById('grid-cols-input')
if ($gridRowsInput) {
  $gridRowsInput.addEventListener('change', () => {
    const v = Math.min(50, Math.max(2, parseInt($gridRowsInput.value) || 10))
    state.uploadSettings.rows = v
    $gridRowsInput.value = v
  })
}
if ($gridColsInput) {
  $gridColsInput.addEventListener('change', () => {
    const v = Math.min(50, Math.max(2, parseInt($gridColsInput.value) || 10))
    state.uploadSettings.cols = v
    $gridColsInput.value = v
  })
}

document.querySelectorAll('.placement-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    state.uploadSettings.pointPlacement = btn.dataset.placement
    document.querySelectorAll('.placement-btn').forEach(b => b.classList.remove('active'))
    btn.classList.add('active')
  })
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

// ─── Health polling ───────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const h = await api.health()
    $statusDot.className = 'status-dot ' + (h.models_ready ? 'ready' : 'loading')
    const parts = []
    if (h.t3_loaded) parts.push('T3')
    if (h.t1_loaded) parts.push('T1')
    $statusText.textContent = h.models_ready
      ? `Models ready (${parts.join(', ')})`
      : 'Loading models...'
    if (!h.models_ready) setTimeout(checkHealth, 2500)
    if ($settingModel) {
      if (!h.t3_loaded) $settingModel.querySelector('option[value="t3"]')?.setAttribute('disabled', '')
      if (!h.t1_loaded) $settingModel.querySelector('option[value="t1"]')?.setAttribute('disabled', '')
    }
  } catch {
    $statusDot.className = 'status-dot error'
    $statusText.textContent = 'Service unavailable'
    setTimeout(checkHealth, 4000)
  }
}

// ─── Resize ───────────────────────────────────────────────────────────────────

const resizeObserver = new ResizeObserver(() => {
  if (state.loadedImg) resizeAndDraw()
})
resizeObserver.observe($container)

// ─── Init ─────────────────────────────────────────────────────────────────────

async function init() {
  checkHealth()
  try {
    const data = await api.labels()
    state.allLabels    = data.labels ?? []
    state.customLabels = data.custom ?? []
  } catch { /* labels not critical on startup */ }
  await refreshImageList()
  if (state.images.length > 0) await loadImage(state.images[0].id)
}

init()