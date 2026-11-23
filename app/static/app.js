const state = {
  mode: "image",
  file: null,
  previewUrl: null,
  result: null,
  config: null,
  abort: null,
  silentStop: false,
};

const $ = (id) => document.getElementById(id);
const els = {
  apiStatus: $("apiStatus"),
  modeButtons: [...document.querySelectorAll(".mode-btn")],
  fileInput: $("fileInput"),
  dropzone: $("dropzone"),
  dropPlaceholder: $("dropPlaceholder"),
  dropTitle: $("dropTitle"),
  dropSubtitle: $("dropSubtitle"),
  mediaPreview: $("mediaPreview"),
  previewImage: $("previewImage"),
  previewVideo: $("previewVideo"),
  fileChip: $("fileChip"),
  clearFile: $("clearFile"),
  promptInput: $("promptInput"),
  settingsToggle: $("settingsToggle"),
  settingsGrid: $("settingsGrid"),
  boxThreshold: $("boxThreshold"),
  boxThresholdValue: $("boxThresholdValue"),
  textThreshold: $("textThreshold"),
  textThresholdValue: $("textThresholdValue"),
  maxObjects: $("maxObjects"),
  maxObjectsValue: $("maxObjectsValue"),
  processFps: $("processFps"),
  processFpsValue: $("processFpsValue"),
  videoFpsField: $("videoFpsField"),
  analyzeButton: $("analyzeButton"),
  analyzeButtonText: $("analyzeButtonText"),
  stopButton: $("stopButton"),
  clearButton: $("clearButton"),
  limitHint: $("limitHint"),
  emptyState: $("emptyState"),
  resultImage: $("resultImage"),
  resultVideo: $("resultVideo"),
  loadingOverlay: $("loadingOverlay"),
  loadingTitle: $("loadingTitle"),
  loadingText: $("loadingText"),
  downloadButton: $("downloadButton"),
  metricObjects: $("metricObjects"),
  metricMode: $("metricMode"),
  metricDevice: $("metricDevice"),
  metricTime: $("metricTime"),
  metricAvgConf: $("metricAvgConf"),
  metricSemantic: $("metricSemantic"),
  metricFrames: $("metricFrames"),
  metricDuration: $("metricDuration"),
  metricTrackFps: $("metricTrackFps"),
  metricThroughput: $("metricThroughput"),
  barConf: $("barConf"),
  barSem: $("barSem"),
  barConfLabel: $("barConfLabel"),
  barSemLabel: $("barSemLabel"),
  objectsTitle: $("objectsTitle"),
  objectList: $("objectList"),
  jsonToggle: $("jsonToggle"),
  jsonView: $("jsonView"),
  toast: $("toast"),
};

function init() {
  bindEvents();
  bindRange(els.boxThreshold, els.boxThresholdValue, (v) => Number(v).toFixed(2));
  bindRange(els.textThreshold, els.textThresholdValue, (v) => Number(v).toFixed(2));
  bindRange(els.maxObjects, els.maxObjectsValue, (v) => String(v));
  bindRange(els.processFps, els.processFpsValue, (v) => String(v));
  bindVideoSync();
  checkApi();
}

function bindEvents() {
  els.modeButtons.forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  els.fileInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) selectFile(file);
  });

  ["dragenter", "dragover"].forEach((name) => {
    els.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      els.dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    els.dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      els.dropzone.classList.remove("dragover");
    });
  });
  els.dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) selectFile(file);
  });

  els.dropzone.addEventListener("click", (event) => {
    if (event.target.closest("#clearFile")) return;
    if (event.target.closest("video")) return;
    openFilePicker();
  });
  els.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFilePicker();
    }
  });
  els.clearFile.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    clearSelectedFile();
  });
  els.fileChip.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    openFilePicker();
  });

  document.querySelectorAll(".prompt-presets button").forEach((button) => {
    button.addEventListener("click", () => {
      const existing = els.promptInput.value.trim();
      const prompt = button.dataset.prompt;
      els.promptInput.value = existing ? `${existing}\n${prompt}` : prompt;
      els.promptInput.focus();
    });
  });

  els.settingsToggle.addEventListener("click", () => {
    const expanded = els.settingsToggle.getAttribute("aria-expanded") === "true";
    els.settingsToggle.setAttribute("aria-expanded", String(!expanded));
    els.settingsGrid.classList.toggle("collapsed", expanded);
  });

  els.downloadButton.addEventListener("click", openResultsFolder);
  els.analyzeButton.addEventListener("click", analyze);
  els.stopButton.addEventListener("click", () => stopRun());
  els.clearButton.addEventListener("click", clearAll);
  els.jsonToggle.addEventListener("click", () => {
    els.jsonView.classList.toggle("hidden");
    els.jsonToggle.textContent = els.jsonView.classList.contains("hidden") ? "JSON" : "Hide";
  });
}

let videoSyncLock = false;

function bindVideoSync() {
  const link = (master, follower) => {
    master.addEventListener("play", () => followVideo(master, follower));
    master.addEventListener("pause", () => followVideo(master, follower));
    master.addEventListener("seeked", () => followVideo(master, follower, true));
    master.addEventListener("ratechange", () => followVideo(master, follower, true));
    master.addEventListener("timeupdate", () => followVideo(master, follower, true));
  };
  link(els.resultVideo, els.previewVideo);
  link(els.previewVideo, els.resultVideo);
}

function followVideo(master, follower, fromTick = false) {
  if (videoSyncLock) return;
  if (!master.getAttribute("src") || !follower.getAttribute("src")) return;
  if (follower.classList.contains("hidden") || master.classList.contains("hidden")) return;
  videoSyncLock = true;
  const masterDuration = Number.isFinite(master.duration) ? master.duration : 0;
  const followerDuration = Number.isFinite(follower.duration) ? follower.duration : 0;
  const scale = masterDuration > 0 && followerDuration > 0 ? followerDuration / masterDuration : 1;
  const target = master.currentTime * scale;
  if (Number.isFinite(target) && (!fromTick || Math.abs(follower.currentTime - target) > 0.25)) {
    const limit = followerDuration > 0 ? Math.min(target, Math.max(0, followerDuration - 0.05)) : target;
    follower.currentTime = Math.max(0, limit);
  }
  if (follower.playbackRate !== master.playbackRate) follower.playbackRate = master.playbackRate || 1;
  if (master.paused) follower.pause();
  else if (follower.paused) follower.play().catch(() => {});
  videoSyncLock = false;
}

function bindRange(input, output, format) {
  const update = () => { output.textContent = format(input.value); };
  input.addEventListener("input", update);
  update();
}

async function checkApi() {
  try {
    const [healthResponse, configResponse] = await Promise.all([
      fetch("/api/health"),
      fetch("/api/models"),
    ]);
    if (!healthResponse.ok || !configResponse.ok) throw new Error("API unavailable");
    const health = await healthResponse.json();
    state.config = await configResponse.json();
    els.apiStatus.classList.remove("offline");
    els.apiStatus.classList.add("online");
    els.apiStatus.querySelector("span:last-child").textContent = `${health.device.toUpperCase()} ready`;
    els.metricDevice.textContent = health.device.toUpperCase();
    els.limitHint.textContent = `Limit ${state.config.max_video_seconds}s · ${state.config.max_upload_mb} MB · first run downloads weights`;
    els.processFps.value = String(Math.round(state.config.default_video_process_fps || 4));
    els.processFpsValue.textContent = els.processFps.value;
  } catch (error) {
    els.apiStatus.classList.remove("online");
    els.apiStatus.classList.add("offline");
    els.apiStatus.querySelector("span:last-child").textContent = "API offline";
  }
}

function openFilePicker() {
  els.fileInput.value = "";
  els.fileInput.click();
}

function setMode(mode, options = {}) {
  state.mode = mode;
  if (!options.keepFile) clearSelectedFile();

  els.modeButtons.forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });

  const isVideo = mode === "video";
  els.dropTitle.textContent = isVideo ? "Drop a video here" : "Drop an image here";
  els.dropSubtitle.textContent = isVideo ? "MP4, MOV, AVI, MKV, WEBM" : "PNG, JPG, WEBP — or switch to Video";
  els.videoFpsField.classList.toggle("hidden", !isVideo);
  els.analyzeButtonText.textContent = isVideo ? "Track" : "Discover";
  if (!options.keepFile) resetResults();
}

function clearSelectedFile() {
  state.file = null;
  els.fileInput.value = "";
  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
  }
  els.previewVideo.pause();
  els.previewVideo.removeAttribute("src");
  els.previewImage.removeAttribute("src");
  els.previewVideo.classList.add("hidden");
  els.previewImage.classList.add("hidden");
  els.mediaPreview.classList.add("hidden");
  els.dropPlaceholder.classList.remove("hidden");
  els.dropzone.classList.remove("has-preview");
  els.fileChip.textContent = "";
}

function selectFile(file) {
  const video = file.type.startsWith("video/") || looksLikeVideo(file.name);
  const image = file.type.startsWith("image/") || looksLikeImage(file.name);
  if (!video && !image) {
    showToast("Choose an image or a video file.");
    return;
  }
  const isVideo = video && !image;
  if (isVideo !== (state.mode === "video")) {
    setMode(isVideo ? "video" : "image", { keepFile: true });
  }

  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);

  els.dropPlaceholder.classList.add("hidden");
  els.mediaPreview.classList.remove("hidden");
  els.dropzone.classList.add("has-preview");
  els.fileChip.textContent = `${file.name} · ${formatBytes(file.size)}`;

  if (isVideo) {
    els.previewImage.classList.add("hidden");
    els.previewImage.removeAttribute("src");
    els.previewVideo.src = state.previewUrl;
    els.previewVideo.classList.remove("hidden");
    els.previewVideo.load();
  } else {
    els.previewVideo.pause();
    els.previewVideo.classList.add("hidden");
    els.previewVideo.removeAttribute("src");
    els.previewImage.src = state.previewUrl;
    els.previewImage.classList.remove("hidden");
  }
}

function looksLikeVideo(name) {
  return /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(name || "");
}

function looksLikeImage(name) {
  return /\.(png|jpe?g|webp|gif|bmp)$/i.test(name || "");
}

async function analyze() {
  if (!state.file) {
    showToast(`Choose an ${state.mode === "image" ? "image" : "video"} first.`);
    return;
  }
  const prompt = els.promptInput.value.trim();
  if (!prompt) {
    showToast("Describe at least one target to discover.");
    els.promptInput.focus();
    return;
  }

  const form = new FormData();
  form.append("file", state.file);
  form.append("prompt", prompt);
  form.append("box_threshold", els.boxThreshold.value);
  form.append("text_threshold", els.textThreshold.value);
  form.append("max_objects", els.maxObjects.value);
  if (state.mode === "video") form.append("process_fps", els.processFps.value);

  resetResults();

  const controller = new AbortController();
  state.abort = controller;
  setLoading(true);
  try {
    const response = await fetch(`/api/${state.mode}`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    state.result = payload;
    renderResult(payload);
  } catch (error) {
    if (error.name === "AbortError") {
      if (!state.silentStop) showToast("Stopped.");
      return;
    }
    showToast(error.message || "Inference failed.");
  } finally {
    state.abort = null;
    state.silentStop = false;
    setLoading(false);
  }
}

function stopRun(options = {}) {
  state.silentStop = Boolean(options.silent);
  if (state.abort) state.abort.abort();
  fetch("/api/stop", { method: "POST" }).catch(() => {});
}

function clearAll() {
  const running = Boolean(state.abort);
  stopRun({ silent: true });
  if (!running) state.silentStop = false;
  els.promptInput.value = "";
  els.boxThreshold.value = "0.35";
  els.boxThresholdValue.textContent = "0.35";
  els.textThreshold.value = "0.25";
  els.textThresholdValue.textContent = "0.25";
  els.maxObjects.value = "12";
  els.maxObjectsValue.textContent = "12";
  const fps = String(Math.round(state.config?.default_video_process_fps || 4));
  els.processFps.value = fps;
  els.processFpsValue.textContent = fps;
  setMode("image");
}

function setLoading(loading) {
  els.analyzeButton.disabled = loading;
  els.stopButton.disabled = !loading;
  els.loadingOverlay.classList.toggle("hidden", !loading);
  if (loading) {
    els.emptyState.classList.add("hidden");
    els.loadingTitle.textContent = state.mode === "video" ? "Tracking objects" : "Running inference";
    els.loadingText.textContent = state.mode === "video"
      ? "Frames → grounding → SAM → BoT-SORT"
      : "Grounding DINO → SAM → CLIP";
  }
}

function renderResult(result) {
  els.emptyState.classList.add("hidden");
  els.downloadButton.dataset.filename = result.download_name || "";
  els.downloadButton.classList.remove("hidden");
  els.jsonView.textContent = JSON.stringify(result, null, 2);
  els.jsonView.classList.add("hidden");
  els.jsonToggle.textContent = "JSON";

  const elapsed = Number(result.elapsed_ms || 0);
  els.metricDevice.textContent = String(result.device || "—").toUpperCase();
  els.metricTime.textContent = formatDuration(elapsed);

  if (result.media_type === "video") {
    els.resultImage.classList.add("hidden");
    els.resultVideo.src = cacheBust(result.result_url);
    els.resultVideo.classList.remove("hidden");
    els.resultVideo.load();
    if (els.previewVideo.src) {
      els.previewVideo.pause();
      els.previewVideo.currentTime = 0;
    }

    const tracks = result.tracks || [];
    const confs = tracks.map((t) => Number(t.max_confidence || 0));
    const sems = tracks.map((t) => Number(t.best_semantic_similarity || 0));
    const avgConf = average(confs);
    const bestSem = sems.length ? Math.max(...sems) : 0;
    const frames = Number(result.processed_frames || 0);
    const duration = Number(result.source_duration_s || 0);
    const processFps = Number(result.process_fps || 0);

    els.metricObjects.textContent = String(result.unique_tracks ?? tracks.length);
    els.metricMode.textContent = "Tracking";
    els.metricAvgConf.textContent = pct(avgConf);
    els.metricSemantic.textContent = fmt(bestSem);
    els.metricFrames.textContent = String(frames || "—");
    els.metricDuration.textContent = duration ? `${duration.toFixed(1)} s` : "—";
    els.metricTrackFps.textContent = processFps ? processFps.toFixed(1) : "—";
    els.metricThroughput.textContent = frames && elapsed
      ? `${(frames / (elapsed / 1000)).toFixed(1)} f/s`
      : "—";
    setBars(avgConf, bestSem);
    els.objectsTitle.textContent = "Tracked identities";
    renderTracks(tracks);
  } else {
    els.resultVideo.pause();
    els.resultVideo.removeAttribute("src");
    els.resultVideo.classList.add("hidden");
    els.resultImage.src = cacheBust(result.result_url);
    els.resultImage.classList.remove("hidden");

    const detections = result.detections || [];
    const confs = detections.map((d) => Number(d.score || 0));
    const sems = detections.map((d) => Number(d.semantic_similarity || 0));
    const avgConf = average(confs);
    const bestSem = sems.length ? Math.max(...sems) : 0;

    els.metricObjects.textContent = String(result.object_count ?? detections.length);
    els.metricMode.textContent = "Discovery";
    els.metricAvgConf.textContent = pct(avgConf);
    els.metricSemantic.textContent = fmt(bestSem);
    els.metricFrames.textContent = "1";
    els.metricDuration.textContent = "—";
    els.metricTrackFps.textContent = "—";
    els.metricThroughput.textContent = elapsed
      ? `${(1000 / elapsed).toFixed(2)} img/s`
      : "—";
    setBars(avgConf, bestSem);
    els.objectsTitle.textContent = "Detected objects";
    renderDetections(detections);
  }
}

function setBars(avgConf, bestSem) {
  const confPct = Math.max(0, Math.min(100, Math.round(avgConf * 100)));
  const semPct = Math.max(0, Math.min(100, Math.round(bestSem * 100)));
  els.barConf.style.width = `${confPct}%`;
  els.barSem.style.width = `${semPct}%`;
  els.barConfLabel.textContent = confPct ? `${confPct}%` : "—";
  els.barSemLabel.textContent = semPct ? `${semPct}%` : "—";
}

function renderDetections(detections) {
  els.objectList.replaceChildren();
  if (!detections.length) {
    els.objectList.append(makePlaceholder("No objects passed thresholds. Lower detection threshold."));
    return;
  }
  detections.forEach((det, index) => {
    const row = document.createElement("div");
    row.className = "object-row";

    const badge = document.createElement("div");
    badge.className = "object-badge";
    badge.textContent = String(index + 1).padStart(2, "0");

    const copy = document.createElement("div");
    copy.className = "object-copy";
    const title = document.createElement("strong");
    title.textContent = det.label;
    const meta = document.createElement("span");
    meta.textContent = `sem ${fmt(det.semantic_similarity)} · ${formatBox(det.box)}`;
    copy.append(title, meta);

    const score = document.createElement("span");
    score.className = "score-pill";
    score.textContent = pct(det.score);
    row.append(badge, copy, score);
    els.objectList.append(row);
  });
}

function renderTracks(tracks) {
  els.objectList.replaceChildren();
  if (!tracks.length) {
    els.objectList.append(makePlaceholder("No stable tracks. Lower threshold or raise process FPS."));
    return;
  }
  tracks.forEach((track) => {
    const row = document.createElement("div");
    row.className = "object-row";

    const badge = document.createElement("div");
    badge.className = "object-badge";
    badge.textContent = `#${track.track_id}`;

    const copy = document.createElement("div");
    copy.className = "object-copy";
    const title = document.createElement("strong");
    title.textContent = track.label;
    const meta = document.createElement("span");
    meta.textContent = `${Number(track.first_seen_s).toFixed(1)}→${Number(track.last_seen_s).toFixed(1)}s · ${track.observations} obs · sem ${fmt(track.best_semantic_similarity)}`;
    copy.append(title, meta);

    const score = document.createElement("span");
    score.className = "score-pill";
    score.textContent = pct(track.max_confidence);
    row.append(badge, copy, score);
    els.objectList.append(row);
  });
}

function makePlaceholder(text) {
  const node = document.createElement("div");
  node.className = "list-placeholder";
  node.textContent = text;
  return node;
}

function resetResults() {
  state.result = null;
  els.resultImage.removeAttribute("src");
  els.resultVideo.pause();
  els.resultVideo.removeAttribute("src");
  els.resultImage.classList.add("hidden");
  els.resultVideo.classList.add("hidden");
  els.emptyState.classList.remove("hidden");
  els.downloadButton.classList.add("hidden");
  delete els.downloadButton.dataset.filename;
  els.metricObjects.textContent = "—";
  els.metricMode.textContent = "—";
  els.metricTime.textContent = "—";
  els.metricAvgConf.textContent = "—";
  els.metricSemantic.textContent = "—";
  els.metricFrames.textContent = "—";
  els.metricDuration.textContent = "—";
  els.metricTrackFps.textContent = "—";
  els.metricThroughput.textContent = "—";
  setBars(0, 0);
  els.objectsTitle.textContent = "Detected objects";
  els.objectList.replaceChildren(makePlaceholder("Results appear here."));
  els.jsonView.classList.add("hidden");
  els.jsonToggle.textContent = "JSON";
}

async function openResultsFolder() {
  const filename = els.downloadButton.dataset.filename;
  if (!filename) return;
  const body = new FormData();
  body.append("filename", filename);
  try {
    const response = await fetch("/api/reveal", { method: "POST", body });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "Could not open the results folder.");
    }
    showToast("Saved in the results folder.");
  } catch (error) {
    showToast(error.message || "Could not open the results folder.");
  }
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.remove("hidden");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => els.toast.classList.add("hidden"), 4800);
}

function cacheBust(url) {
  return `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
}

function formatBox(box = []) {
  return box.map((value) => Math.round(value)).join(", ");
}

function fmt(value) {
  return Number(value ?? 0).toFixed(2);
}

function pct(value) {
  return `${Math.round(Number(value ?? 0) * 100)}%`;
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

function formatDuration(ms) {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} s`;
  return `${(ms / 60000).toFixed(1)} min`;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (let i = 0; i < units.length; i += 1) {
    unit = units[i];
    if (value < 1024 || i === units.length - 1) break;
    value /= 1024;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${unit}`;
}

init();
