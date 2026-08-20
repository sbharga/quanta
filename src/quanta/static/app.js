const state = {
  sessionId: null,
  session: null,
  source: null,
  cellNodes: {},
  draftBuffers: {},
  pendingDraftId: null,
  pendingPrompt: null,
  pinned: true,
  submitting: false,
  toastTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

document.addEventListener("DOMContentLoaded", () => {
  localStorage.removeItem("quanta-sidebar-collapsed");
  $("#dataset-form").addEventListener("submit", createSession);
  $("#composer-form").addEventListener("submit", submitComposer);
  $("#composer-input").addEventListener("input", resizeComposerInput);
  $("#composer-input").addEventListener("keydown", composerKeydown);
  $("#new-session").addEventListener("click", () => closeSession());
  $("#cancel-button").addEventListener("click", cancelSession);
  $("#jump-latest").addEventListener("click", () => scrollToBottom(true));
  $("#timeline-host").addEventListener("scroll", handleTimelineScroll);
  $("#settings-button").addEventListener("click", openSettings);
  $("#settings-close").addEventListener("click", () => $("#settings-dialog").close());
  $("#settings-form").addEventListener("submit", saveSettings);
  $("#clear-llm").addEventListener("change", syncClearControls);
  $("#clear-kaggle").addEventListener("change", syncClearControls);
  document.addEventListener("click", closeAllMenus);

  loadSettings();
  loadSessions();
  const previous = localStorage.getItem("quanta-session");
  if (previous) openSession(previous);
});

/* ---------------------------------------------------------------------- */
/* Generic helpers                                                        */
/* ---------------------------------------------------------------------- */

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.remove("hidden");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => node.classList.add("hidden"), 5000);
}

function formatDate(value) {
  try {
    return new Date(value).toLocaleDateString();
  } catch {
    return "";
  }
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/* ---------------------------------------------------------------------- */
/* Settings                                                                */
/* ---------------------------------------------------------------------- */

async function loadSettings() {
  try {
    const settings = await api("/api/settings");
    renderSettings(settings);
    return settings;
  } catch (error) {
    toast(error.message);
    return null;
  }
}

function renderSettings(settings) {
  $("#sidebar-model").textContent = settings.llm_model;
  $("#key-warning").classList.toggle("hidden", settings.llm_configured);
  $("#llm-model").value = settings.llm_model;
  $("#llm-base-url").value = settings.llm_base_url;
  setCredentialStatus("#llm-status", settings.llm_configured, settings.llm_source);
  setCredentialStatus("#kaggle-status", settings.kaggle_configured, settings.kaggle_source);
}

function setCredentialStatus(selector, configured, source) {
  const status = $(selector);
  status.textContent = configured ? `Configured · ${source}` : "Not configured";
  status.classList.toggle("bg-accent-dim", configured);
  status.classList.toggle("text-accent", configured);
}

async function openSettings() {
  $("#llm-key").value = "";
  $("#kaggle-token").value = "";
  $("#clear-llm").checked = false;
  $("#clear-kaggle").checked = false;
  $("#settings-message").textContent = "";
  syncClearControls();
  await loadSettings();
  $("#settings-dialog").showModal();
}

function syncClearControls() {
  $("#llm-key").disabled = $("#clear-llm").checked;
  $("#kaggle-token").disabled = $("#clear-kaggle").checked;
}

async function saveSettings(event) {
  event.preventDefault();
  const body = {
    llm_model: $("#llm-model").value.trim(),
    llm_base_url: $("#llm-base-url").value.trim(),
  };
  const llmKey = $("#llm-key").value.trim();
  const kaggleToken = $("#kaggle-token").value.trim();
  if (llmKey || $("#clear-llm").checked) {
    body.llm_api_key = $("#clear-llm").checked ? "" : llmKey;
  }
  if (kaggleToken || $("#clear-kaggle").checked) {
    body.kaggle_api_token = $("#clear-kaggle").checked ? "" : kaggleToken;
  }
  const button = event.currentTarget.querySelector("button[type='submit']");
  button.disabled = true;
  try {
    const settings = await api("/api/settings", {method: "PUT", body: JSON.stringify(body)});
    renderSettings(settings);
    $("#llm-key").value = "";
    $("#kaggle-token").value = "";
    $("#clear-llm").checked = false;
    $("#clear-kaggle").checked = false;
    syncClearControls();
    $("#settings-message").textContent = "Saved";
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
  }
}

/* ---------------------------------------------------------------------- */
/* Sidebar / session list                                                 */
/* ---------------------------------------------------------------------- */

async function loadSessions() {
  try {
    const data = await api("/api/sessions");
    const list = $("#session-list");
    list.replaceChildren();
    data.sessions.forEach((session) => list.appendChild(buildSessionRow(session)));
    highlightSidebarActive();
  } catch (error) {
    toast(error.message);
  }
}

function highlightSidebarActive() {
  document.querySelectorAll(".session-row").forEach((row) => {
    row.classList.toggle("bg-panel-raised", row.dataset.sessionId === state.sessionId);
  });
}

function statusDotClass(status) {
  if (["running", "downloading", "inspecting", "cancelling"].includes(status)) return "bg-accent animate-pulse";
  if (status === "error") return "bg-err";
  if (status === "complete") return "bg-accent";
  return "bg-faint";
}

function buildSessionRow(session) {
  const row = el("div", "session-row group relative flex items-center rounded hover:bg-panel-raised");
  row.dataset.sessionId = session.id;

  const openButton = el("button", "open flex-1 min-w-0 flex items-center gap-2 px-2 py-1.5 text-left");
  openButton.type = "button";
  const dot = el("span", `session-status-dot w-1.5 h-1.5 rounded-full shrink-0 ${statusDotClass(session.status)}`);
  const labelWrap = el("span", "sidebar-label min-w-0 flex-1");
  const titleEl = el("span", "title block truncate text-[12.5px]", session.title || session.dataset_ref);
  const metaEl = el("span", "session-meta block truncate text-[10px] text-dim", `${session.status} · ${formatDate(session.updated_at)}`);
  labelWrap.append(titleEl, metaEl);
  openButton.append(dot, labelWrap);
  openButton.addEventListener("click", () => openSession(session.id));

  const menuWrap = el("div", "menu-wrap sidebar-label relative shrink-0");
  const menuToggle = el("button", "menu-toggle opacity-0 group-hover:opacity-100 px-1.5 text-dim hover:text-fg", "⋮");
  menuToggle.type = "button";
  menuToggle.title = "More";
  const menu = el("div", "menu hidden absolute right-0 top-full z-20 mt-1 w-32 bg-panel-raised border border-line rounded shadow-lg text-[12px] py-1");
  const renameButton = el("button", "rename block w-full text-left px-2.5 py-1.5 hover:bg-line", "Rename");
  renameButton.type = "button";
  const deleteButton = el("button", "delete block w-full text-left px-2.5 py-1.5 hover:bg-line text-err", "Delete");
  deleteButton.type = "button";
  menu.append(renameButton, deleteButton);
  menuWrap.append(menuToggle, menu);

  menuToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    closeAllMenus();
    menu.classList.toggle("hidden");
  });
  renameButton.addEventListener("click", (event) => {
    event.stopPropagation();
    menu.classList.add("hidden");
    beginRenameSession(session, titleEl);
  });
  let deleteArmed = false;
  let disarmTimer = null;
  deleteButton.addEventListener("click", async (event) => {
    event.stopPropagation();
    if (!deleteArmed) {
      deleteArmed = true;
      deleteButton.textContent = "Confirm delete";
      disarmTimer = setTimeout(() => {
        deleteArmed = false;
        deleteButton.textContent = "Delete";
      }, 3000);
      return;
    }
    clearTimeout(disarmTimer);
    menu.classList.add("hidden");
    await deleteSession(session.id);
  });

  row.append(openButton, menuWrap);
  return row;
}

function closeAllMenus() {
  document.querySelectorAll(".menu:not(.hidden)").forEach((menu) => menu.classList.add("hidden"));
}

function beginRenameSession(session, titleEl) {
  const current = session.title || session.dataset_ref;
  const input = el("input", "title block w-full bg-bg border border-accent rounded px-1 py-0.5 text-[12.5px]");
  input.value = current;
  titleEl.replaceWith(input);
  input.focus();
  input.select();

  let settled = false;
  const restore = () => {
    if (settled) return;
    settled = true;
    input.replaceWith(titleEl);
  };
  const commit = async () => {
    if (settled) return;
    const value = input.value.trim();
    if (!value || value === current) {
      restore();
      return;
    }
    settled = true;
    try {
      await api(`/api/sessions/${session.id}`, {method: "PATCH", body: JSON.stringify({title: value})});
      session.title = value;
      titleEl.textContent = value;
      input.replaceWith(titleEl);
      if (session.id === state.sessionId) $("#header-title").textContent = value;
    } catch (error) {
      toast(error.message);
      input.replaceWith(titleEl);
    }
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      commit();
    } else if (event.key === "Escape") {
      restore();
    }
  });
  input.addEventListener("blur", commit);
}

async function deleteSession(id) {
  try {
    await api(`/api/sessions/${id}`, {method: "DELETE"});
    if (id === state.sessionId) closeSession();
    loadSessions();
  } catch (error) {
    toast(error.message);
  }
}

/* ---------------------------------------------------------------------- */
/* Session lifecycle                                                      */
/* ---------------------------------------------------------------------- */

async function createSession(event) {
  event.preventDefault();
  const input = $("#dataset-input");
  const value = input.value.trim();
  if (!value) return;
  input.disabled = true;
  try {
    const session = await api("/api/sessions", {method: "POST", body: JSON.stringify({dataset: value})});
    input.value = "";
    input.disabled = false;
    await openSession(session.id);
    loadSessions();
  } catch (error) {
    input.disabled = false;
    toast(error.message);
  }
}

async function openSession(sessionId) {
  if (state.source) state.source.close();
  state.sessionId = sessionId;
  state.session = null;
  state.cellNodes = {};
  state.draftBuffers = {};
  state.pendingDraftId = null;
  state.pendingPrompt = null;
  state.submitting = false;
  state.pinned = true;
  $("#composer-input").value = "";
  resizeComposerInput();
  localStorage.setItem("quanta-session", sessionId);

  $("#empty-state").classList.add("hidden");
  $("#timeline-host").classList.remove("hidden");
  $("#timeline").replaceChildren();
  $("#jump-latest").classList.add("hidden");

  try {
    const session = await api(`/api/sessions/${sessionId}`);
    if (session.id !== state.sessionId) return;
    state.session = session;
    $("#session-header").classList.remove("hidden");
    renderSessionHeader(session);
    renderTimelineSeed(session.timeline || []);
    if (session.active_draft) {
      startDraft(session.active_draft);
      appendDraftDelta({draft_id: session.active_draft.draft_id, delta: session.active_draft.text || ""});
    }
    $("#composer").classList.remove("hidden");
    if (session.pending_prompt) {
      showQuestion(session.pending_prompt);
    } else {
      state.pendingPrompt = null;
      $("#suggestions").classList.add("hidden");
      $("#suggestions").replaceChildren();
      $("#prompt-text").classList.add("hidden");
      $("#composer-input").placeholder = "Ask a follow-up or request a different approach...";
    }
    updateComposerAvailability();
    connectEvents(session.event_cursor || 0);
    scrollToBottom(true);
    highlightSidebarActive();
  } catch (error) {
    localStorage.removeItem("quanta-session");
    toast(error.message);
    closeSession();
  }
}

function closeSession() {
  if (state.source) state.source.close();
  state.source = null;
  state.sessionId = null;
  state.session = null;
  state.cellNodes = {};
  state.draftBuffers = {};
  state.pendingDraftId = null;
  state.pendingPrompt = null;
  state.submitting = false;
  localStorage.removeItem("quanta-session");
  $("#timeline").replaceChildren();
  $("#timeline-host").classList.add("hidden");
  $("#composer").classList.add("hidden");
  $("#session-header").classList.add("hidden");
  $("#empty-state").classList.remove("hidden");
  highlightSidebarActive();
}

function renderSessionHeader(session) {
  $("#header-title").textContent = session.title || session.dataset_ref;
  $("#header-dataset").textContent = session.dataset_ref;
  $("#download-button").href = `/api/sessions/${session.id}/notebook.ipynb`;
  applySessionStatus(session.status, session.error || session.status, session.running);
}

function setStatus(status, message) {
  $("#status-text").textContent = message || status;
  $("#status-dot").className = `w-2 h-2 rounded-full ${statusDotClass(status)}`;
}

function applySessionStatus(status, message, running = null) {
  if (!state.session) return;
  state.session.status = status;
  state.session.running = running === null
    ? ["running", "downloading", "inspecting", "cancelling"].includes(status)
    : running;
  setStatus(status, message);
  $("#cancel-button").classList.toggle("hidden", !state.session.running || status === "cancelling");
  const row = document.querySelector(`.session-row[data-session-id="${CSS.escape(state.sessionId)}"]`);
  if (row) {
    const dot = row.querySelector(".session-status-dot");
    const meta = row.querySelector(".session-meta");
    if (dot) dot.className = `session-status-dot w-1.5 h-1.5 rounded-full shrink-0 ${statusDotClass(status)}`;
    if (meta) meta.textContent = `${status} · ${formatDate(state.session.updated_at)}`;
  }
  updateComposerAvailability();
}

async function cancelSession() {
  if (!state.sessionId) return;
  try {
    applySessionStatus("cancelling", "Stopping current work", true);
    await api(`/api/sessions/${state.sessionId}/cancel`, {method: "POST", body: "{}"});
  } catch (error) {
    toast(error.message);
  }
}

/* ---------------------------------------------------------------------- */
/* Composer (initial question + follow-ups)                               */
/* ---------------------------------------------------------------------- */

function resizeComposerInput() {
  const input = $("#composer-input");
  input.style.height = "";
  if (!input.value) return;
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
}

function composerKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#composer-form").requestSubmit();
  }
}

function showQuestion(prompt) {
  state.pendingPrompt = prompt;
  const input = $("#composer-input");
  const promptText = $("#prompt-text");
  if (prompt.kind === "followup") {
    $("#suggestions").classList.add("hidden");
    $("#suggestions").replaceChildren();
    input.placeholder = prompt.question || "Reply...";
    promptText.textContent = prompt.question || "Reply to continue.";
  } else {
    input.placeholder = "Add your own request, or pick a suggestion above...";
    renderSuggestions(prompt.suggestions || []);
    $("#suggestions").classList.toggle("hidden", !(prompt.suggestions || []).length);
    promptText.textContent = prompt.summary || "Choose a starting point or enter your own request.";
  }
  promptText.classList.toggle("hidden", !promptText.textContent);
  $("#composer").classList.remove("hidden");
  updateComposerAvailability();
}

function renderSuggestions(suggestions) {
  const container = $("#suggestions");
  container.replaceChildren();
  suggestions.forEach((text) => {
    const chip = el("button", "px-2.5 py-1 rounded-full border border-line text-[11.5px] text-dim hover:text-fg hover:border-dim", text);
    chip.type = "button";
    chip.setAttribute("aria-pressed", "false");
    chip.addEventListener("click", () => {
      $("#composer-input").value = text;
      resizeComposerInput();
      container.querySelectorAll("button").forEach((button) => {
        const active = button === chip;
        button.setAttribute("aria-pressed", String(active));
        button.classList.toggle("border-accent", active);
        button.classList.toggle("text-accent", active);
      });
      $("#composer-input").focus();
    });
    container.append(chip);
  });
}

async function submitComposer(event) {
  event.preventDefault();
  const input = $("#composer-input");
  const custom = input.value.trim();
  const content = custom;
  if (!content) return toast("Choose a suggestion or enter a request.");
  state.submitting = true;
  updateComposerAvailability();
  try {
    await api(`/api/sessions/${state.sessionId}/responses`, {method: "POST", body: JSON.stringify({content})});
  } catch (error) {
    state.submitting = false;
    updateComposerAvailability();
    toast(error.message);
    return;
  }
  input.value = "";
  resizeComposerInput();
  state.pendingPrompt = null;
  $("#suggestions").classList.add("hidden");
  $("#suggestions").replaceChildren();
  $("#prompt-text").classList.add("hidden");
  input.placeholder = "Ask a follow-up or request a different approach...";
  applySessionStatus("running", "Starting the next analysis step", true);
}

function updateComposerAvailability() {
  const session = state.session;
  if (!session) return;
  const busy = state.submitting || session.running || ["running", "downloading", "inspecting", "cancelling"].includes(session.status);
  const submit = $("#composer-form").querySelector("button[type='submit']");
  submit.disabled = busy;
  submit.classList.toggle("opacity-40", busy);
  $("#composer-input").disabled = busy;
}

/* ---------------------------------------------------------------------- */
/* Live event stream                                                      */
/* ---------------------------------------------------------------------- */

const KNOWN_EVENTS = [
  "status", "progress", "kernel_replay_started", "kernel_replay_progress", "kernel_replay_finished",
  "cells_stale", "note", "user_request", "cell_created", "cell_revised",
  "execution_started", "execution_output", "execution_finished",
  "draft_started", "draft_delta", "draft_finished", "draft_discarded", "model_retry",
  "error", "complete", "question",
];

function connectEvents(cursor) {
  const source = new EventSource(`/api/sessions/${state.sessionId}/events?after=${cursor}`);
  state.source = source;
  KNOWN_EVENTS.forEach((name) => {
    source.addEventListener(name, (event) => {
      let data = {};
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }
      applyEvent(name, data);
    });
  });
  source.onerror = () => setStatus("reconnecting", "Reconnecting to event stream");
}

function applyEvent(type, data) {
  switch (type) {
    case "status":
      state.submitting = false;
      applySessionStatus(data.status, data.message);
      break;
    case "progress":
      applyTimelineEntry({type: "note", event: "progress", message: data.message});
      break;
    case "kernel_replay_started":
      applySessionStatus(
        "running",
        `Rebuilding execution state (${data.count} cell${data.count === 1 ? "" : "s"})`,
        true,
      );
      applyTimelineEntry({type: "note", event: "kernel_replay_started", count: data.count});
      break;
    case "kernel_replay_progress":
      setStatus("running", `Rebuilding execution state ${data.current}/${data.total}`);
      applyTimelineEntry({type: "note", event: "kernel_replay_progress", current: data.current, total: data.total});
      break;
    case "kernel_replay_finished":
      setStatus("running", "Execution state restored");
      applyTimelineEntry({type: "note", event: "kernel_replay_finished"});
      break;
    case "cells_stale":
      applyTimelineEntry({type: "note", event: "cells_stale", message: data.message});
      break;
    case "note":
      clearPendingDraft();
      applyTimelineEntry({type: "note", event: "note", message: data.message});
      break;
    case "user_request":
      applyTimelineEntry({type: "request", content: data.content});
      break;
    case "cell_created":
      clearPendingDraft();
      applyTimelineEntry({type: "cell", cell: {...data.cell, executions: []}});
      break;
    case "cell_revised":
      clearPendingDraft();
      updateCellSource(data.cell);
      break;
    case "execution_started":
      applySessionStatus("running", "Executing code", true);
      startExecutionGroup(data.cell_id, data.execution_id, data.execution_count);
      break;
    case "execution_output":
      appendLiveOutput(data.cell_id, data.execution_id, data.output);
      break;
    case "execution_finished":
      finishExecutionGroup(data.cell_id, data.execution_id, data.status, data.outputs);
      setStatus("running", "Inspecting execution results");
      break;
    case "draft_started":
      startDraft(data);
      break;
    case "draft_delta":
      appendDraftDelta(data);
      break;
    case "draft_finished":
      finishDraft(data);
      setStatus("running", `Saving generated ${data.kind}`);
      break;
    case "draft_discarded":
      clearPendingDraft();
      break;
    case "model_retry":
      applySessionStatus("running", data.message || `Retrying model request in ${data.delay}s`, true);
      break;
    case "error":
      clearPendingDraft();
      applyTimelineEntry({type: "error", message: data.message});
      state.submitting = false;
      applySessionStatus("error", data.message, false);
      break;
    case "complete":
      clearPendingDraft();
      applyTimelineEntry({type: "complete", message: data.message});
      state.submitting = false;
      applySessionStatus("complete", data.message || "Analysis complete", false);
      break;
    case "question":
      clearPendingDraft();
      state.submitting = false;
      applySessionStatus("waiting", "Waiting for your response", false);
      showQuestion(data.prompt);
      break;
    default:
      break;
  }
  if (state.pinned) scrollToBottom(false);
  else showJumpPill();
}

/* ---------------------------------------------------------------------- */
/* Timeline: narrative entries (request / note / error / done)            */
/* ---------------------------------------------------------------------- */

function renderTimelineSeed(entries) {
  entries.forEach((entry) => applyTimelineEntry(entry));
}

function applyTimelineEntry(entry) {
  const timeline = $("#timeline");
  if (entry.type === "request") {
    timeline.appendChild(buildRequestNode(entry));
  } else if (entry.type === "cell") {
    const created = ensureCellNode(entry.cell);
    if (created) timeline.appendChild(created);
  } else if (entry.type === "note") {
    timeline.appendChild(buildNoteNode(entry));
  } else if (entry.type === "error") {
    timeline.appendChild(buildErrorNode(entry));
  } else if (entry.type === "complete") {
    timeline.appendChild(buildCompleteNode(entry));
  }
}

function buildRequestNode(entry) {
  const wrap = el("div", "border-l-2 border-accent/60 pl-3 py-1");
  wrap.append(el("div", "text-[10px] uppercase tracking-wide text-dim mb-0.5", "request"));
  wrap.append(el("div", "whitespace-pre-wrap text-[13px]", entry.content || ""));
  return wrap;
}

function buildNoteNode(entry) {
  const wrap = el("div", "flex gap-2 px-1 text-[12px] text-dim");
  wrap.append(el("span", "text-faint shrink-0", "→"));
  wrap.append(el("span", "", noteMessage(entry)));
  return wrap;
}

function noteMessage(entry) {
  switch (entry.event) {
    case "kernel_replay_started":
      return `Rebuilding kernel state (${entry.count} cell${entry.count === 1 ? "" : "s"})`;
    case "kernel_replay_progress":
      return `Rebuilding kernel ${entry.current}/${entry.total}`;
    case "kernel_replay_finished":
      return "Kernel state restored";
    default:
      return entry.message || "";
  }
}

function buildErrorNode(entry) {
  const wrap = el("div", "border border-err/40 bg-err-dim/20 text-err rounded px-3 py-2 text-[12px] whitespace-pre-wrap", entry.message || "Error");
  return wrap;
}

function buildCompleteNode(entry) {
  const wrap = el("div", "flex gap-2 px-1 text-[12px] text-accent");
  wrap.append(el("span", "shrink-0", "✓"), el("span", "", entry.message || "Analysis complete"));
  return wrap;
}

/* ---------------------------------------------------------------------- */
/* Timeline: cells                                                        */
/* ---------------------------------------------------------------------- */

function ensureCellNode(cell) {
  if (state.cellNodes[cell.id]) {
    updateCellSource(cell);
    return null;
  }
  const article = buildCellNode(cell);
  state.cellNodes[cell.id] = article._refs;
  (cell.executions || []).forEach((execution) => {
    startExecutionGroup(cell.id, execution.id, execution.execution_count);
    finishExecutionGroup(cell.id, execution.id, execution.status, execution.outputs);
  });
  return article;
}

function updateCellSource(cell) {
  const refs = state.cellNodes[cell.id];
  if (!refs) return;
  refs.cell = {...refs.cell, ...cell};
  clearCellExecutions(refs);
  renderCellBody(refs, refs.cell);
}

function clearCellExecutions(refs) {
  refs.executionGroups.clear();
  refs.outputs.replaceChildren();
  refs.outputs.classList.add("hidden");
  refs.runStatus.classList.add("hidden");
}

function buildCellNode(cell) {
  const article = el("article", "cell rounded-lg border border-line bg-panel overflow-hidden group");
  article.dataset.cellId = cell.id;

  const toolbar = el("div", "flex items-center justify-between gap-2 px-3 py-1.5 border-b border-line text-[10.5px] uppercase tracking-wide text-dim");
  const labelWrap = el("div", "flex items-center gap-2 min-w-0");
  const posEl = el("span", "text-faint shrink-0", String(cell.position).padStart(2, "0"));
  const kindEl = el("span", "truncate");
  labelWrap.append(posEl, kindEl);
  const runStatus = el("span", "hidden text-accent normal-case tracking-normal", "running");
  const actionsClass = "flex gap-1 opacity-0 group-hover:opacity-100 shrink-0";
  const actions = el("div", actionsClass);
  const editButton = el("button", "px-1.5 py-0.5 border border-line rounded hover:bg-panel-raised normal-case tracking-normal", "Edit");
  editButton.type = "button";
  actions.append(editButton);
  let runButton = null;
  if (cell.kind === "code") {
    runButton = el("button", "px-1.5 py-0.5 border border-line rounded hover:bg-panel-raised normal-case tracking-normal", "Run");
    runButton.type = "button";
    actions.append(runButton);
  }
  toolbar.append(labelWrap, runStatus, actions);

  const body = el("div", "cell-body");
  const outputs = el("div", "outputs divide-y divide-line-soft border-t border-line hidden");
  article.append(toolbar, body, outputs);

  const refs = {
    article, toolbar, kindEl, runStatus, body, outputs, actions, actionsClass,
    executionGroups: new Map(), cell,
  };
  article._refs = refs;
  renderCellBody(refs, cell);

  editButton.addEventListener("click", () => beginEditCell(refs));
  if (runButton) runButton.addEventListener("click", () => runCell(cell.id));

  return article;
}

function renderCellBody(refs, cell) {
  refs.kindEl.textContent = `${cell.kind} · rev ${cell.revision}`;
  refs.body.replaceChildren();
  if (cell.kind === "code") {
    const pre = el("pre", "font-mono text-[12.5px] leading-relaxed p-4 overflow-x-auto whitespace-pre");
    const code = document.createElement("code");
    code.innerHTML = highlightPython(cell.source || "");
    pre.appendChild(code);
    refs.body.appendChild(pre);
  } else {
    const div = el("div", "markdown-body p-4 text-[13px]");
    div.innerHTML = renderMarkdown(cell.source || "");
    refs.body.appendChild(div);
  }
}

function beginEditCell(refs) {
  refs.body.replaceChildren();
  const textarea = document.createElement("textarea");
  textarea.value = refs.cell.source;
  textarea.className = "w-full min-h-[140px] bg-bg border-0 p-4 text-[12.5px] font-mono focus:outline-none resize-y";
  refs.body.appendChild(textarea);
  textarea.focus();

  refs.actions.className = "flex gap-1 shrink-0";
  const originalActions = Array.from(refs.actions.children);
  refs.actions.replaceChildren();
  const saveButton = el("button", "px-1.5 py-0.5 border border-accent text-accent rounded hover:bg-panel-raised normal-case tracking-normal", "Save");
  saveButton.type = "button";
  const cancelButton = el("button", "px-1.5 py-0.5 border border-line rounded hover:bg-panel-raised normal-case tracking-normal", "Cancel");
  cancelButton.type = "button";
  refs.actions.append(saveButton, cancelButton);

  const finishEditing = () => {
    refs.actions.className = refs.actionsClass;
    refs.actions.replaceChildren(...originalActions);
  };
  cancelButton.addEventListener("click", () => {
    finishEditing();
    renderCellBody(refs, refs.cell);
  });
  saveButton.addEventListener("click", async () => {
    try {
      const updated = await api(`/api/sessions/${state.sessionId}/cells/${refs.cell.id}`, {
        method: "PATCH", body: JSON.stringify({source: textarea.value}),
      });
      finishEditing();
      updateCellSource(updated);
    } catch (error) {
      toast(error.message);
    }
  });
}

async function runCell(cellId) {
  try {
    await api(`/api/sessions/${state.sessionId}/cells/${cellId}/run`, {method: "POST", body: "{}"});
  } catch (error) {
    toast(error.message);
  }
}

/* ---------------------------------------------------------------------- */
/* Timeline: executions and outputs                                       */
/* ---------------------------------------------------------------------- */

function startExecutionGroup(cellId, executionId, executionCount) {
  const refs = state.cellNodes[cellId];
  if (!refs) return null;
  if (!refs.executionGroups.has(executionId)) clearCellExecutions(refs);
  refs.outputs.classList.remove("hidden");
  let group = refs.executionGroups.get(executionId);
  if (!group) {
    const wrap = el("div", "execution-group");
    const label = el("div", "px-4 py-1 text-[10px] text-dim", `In [${executionCount}] · running`);
    const list = el("div", "output-list");
    wrap.append(label, list);
    refs.outputs.appendChild(wrap);
    group = {wrap, label, list};
    refs.executionGroups.set(executionId, group);
  }
  refs.runStatus.classList.remove("hidden");
  return group;
}

function appendLiveOutput(cellId, executionId, output) {
  const refs = state.cellNodes[cellId];
  if (!refs) return;
  const group = refs.executionGroups.get(executionId) || startExecutionGroup(cellId, executionId, "?");
  if (group) group.list.appendChild(buildOutputNode(output));
}

function finishExecutionGroup(cellId, executionId, status, outputs) {
  const refs = state.cellNodes[cellId];
  if (!refs) return;
  refs.runStatus.classList.add("hidden");
  const group = refs.executionGroups.get(executionId) || startExecutionGroup(cellId, executionId, "?");
  if (!group) return;
  group.label.textContent = `${status} · ${outputs.length} output${outputs.length === 1 ? "" : "s"}`;
  group.label.classList.toggle("text-err", status !== "ok");
  // This is the authoritative final output list for this one execution
  // group, so replace just its contents rather than trusting whatever
  // arrived incrementally — this is what closes the duplicate-output
  // window the old debounced full-timeline rebuild used to open.
  group.list.replaceChildren(...outputs.map(buildOutputNode));
}

function buildOutputNode(output) {
  const type = output.type || "result";
  let tone = "";
  if (type === "stderr" || type === "error") tone = "text-err bg-err-dim/20";
  else if (type === "result") tone = "text-accent";
  const wrapper = el("div", `px-4 py-2 text-[12.5px] ${tone}`.trim());

  if (["stdout", "stderr", "error", "result"].includes(type)) {
    wrapper.appendChild(el("pre", "whitespace-pre-wrap font-mono text-[12px]", output.text || output.message || ""));
  } else if (type === "image") {
    const img = document.createElement("img");
    img.src = `/api/sessions/${state.sessionId}/artifacts/${encodeURIComponent(output.path)}`;
    img.alt = output.alt || "Generated plot";
    img.className = "max-w-full rounded border border-line";
    wrapper.appendChild(img);
  } else if (type === "table") {
    const scroll = el("div", "overflow-x-auto max-h-96");
    const table = document.createElement("table");
    table.className = "w-full text-[11px] border-collapse";
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    (output.columns || []).forEach((column) => {
      headRow.appendChild(el("th", "sticky top-0 bg-panel-raised border border-line px-2 py-1 text-left", String(column)));
    });
    thead.appendChild(headRow);
    const tbody = document.createElement("tbody");
    (output.rows || []).forEach((row) => {
      const tr = document.createElement("tr");
      row.forEach((value) => tr.appendChild(el("td", "border border-line px-2 py-1", value === null ? "∅" : String(value))));
      tbody.appendChild(tr);
    });
    table.append(thead, tbody);
    scroll.appendChild(table);
    wrapper.appendChild(scroll);
  }
  return wrapper;
}

/* ---------------------------------------------------------------------- */
/* Streaming drafts                                                       */
/* ---------------------------------------------------------------------- */

function startDraft(data) {
  clearPendingDraft();
  const refs = data.target_cell_id ? state.cellNodes[data.target_cell_id] : null;
  let article = null;
  let host;
  let dot;
  if (refs) {
    refs.body.replaceChildren();
    refs.runStatus.textContent = "editing · streaming";
    refs.runStatus.classList.remove("hidden");
    host = refs.body;
    dot = refs.runStatus;
  } else {
    article = el("article", "cell rounded-lg border border-accent/50 bg-panel overflow-hidden");
    article.dataset.draftId = data.draft_id;
    const toolbar = el("div", "flex items-center gap-2 px-3 py-1.5 border-b border-line text-[10.5px] uppercase tracking-wide text-dim");
    dot = el("span", "text-accent blink", "●");
    const label = data.action === "edit" ? "revising cell" : `writing ${data.kind}`;
    toolbar.append(dot, el("span", "", label));
    article.appendChild(toolbar);
    host = article;
  }

  let target;
  if (data.kind === "code") {
    const pre = el("pre", "font-mono text-[12.5px] leading-relaxed p-4 overflow-x-auto whitespace-pre-wrap");
    target = document.createElement("code");
    pre.appendChild(target);
    host.appendChild(pre);
  } else {
    target = el("div", "markdown-body p-4 text-[13px] whitespace-pre-wrap");
    host.appendChild(target);
  }

  if (article) $("#timeline").appendChild(article);
  state.pendingDraftId = data.draft_id;
  state.draftBuffers[data.draft_id] = {kind: data.kind, text: "", target, dot, article, refs};
}

function appendDraftDelta(data) {
  const buffer = state.draftBuffers[data.draft_id];
  if (!buffer) return;
  buffer.text += data.delta;
  buffer.target.textContent = buffer.text;
}

function finishDraft(data) {
  const buffer = state.draftBuffers[data.draft_id];
  if (!buffer) return;
  if (buffer.kind === "code") buffer.target.innerHTML = highlightPython(buffer.text);
  else buffer.target.innerHTML = renderMarkdown(buffer.text);
  if (buffer.refs) buffer.dot.textContent = "saving revision";
  else {
    buffer.dot.classList.remove("blink", "text-accent");
    buffer.dot.classList.add("text-faint");
  }
  // The node itself stays until the next non-draft event (normally the
  // cell_created/cell_revised that follows) removes it via
  // clearPendingDraft — or, if the model produced nothing, the empty-cell
  // "note" event does the same.
}

function clearPendingDraft() {
  if (!state.pendingDraftId) return;
  const buffer = state.draftBuffers[state.pendingDraftId];
  if (buffer?.article) buffer.article.remove();
  if (buffer?.refs) {
    buffer.refs.runStatus.textContent = "running";
    buffer.refs.runStatus.classList.add("hidden");
    renderCellBody(buffer.refs, buffer.refs.cell);
  }
  delete state.draftBuffers[state.pendingDraftId];
  state.pendingDraftId = null;
}

/* ---------------------------------------------------------------------- */
/* Scroll pinning                                                         */
/* ---------------------------------------------------------------------- */

const PIN_SLACK = 80;

function isPinned() {
  const host = $("#timeline-host");
  return host.scrollHeight - host.scrollTop - host.clientHeight < PIN_SLACK;
}

function handleTimelineScroll() {
  state.pinned = isPinned();
  if (state.pinned) $("#jump-latest").classList.add("hidden");
}

function scrollToBottom(force) {
  if (force) state.pinned = true;
  if (!state.pinned) return;
  const host = $("#timeline-host");
  host.scrollTop = host.scrollHeight;
  $("#jump-latest").classList.add("hidden");
}

function showJumpPill() {
  $("#jump-latest").classList.remove("hidden");
}

/* ---------------------------------------------------------------------- */
/* Dependency-free markdown renderer                                      */
/* ---------------------------------------------------------------------- */
/* Block-level structure is detected on the raw (unescaped) source, since
 * none of the markers we look for (#, -, *, digits, >, |, backticks)
 * collide with our escaping. Every piece of user text that ends up in the
 * output HTML passes through inline(), which escapes first and only then
 * wraps the already-safe text in tags we author ourselves. */

function inline(text) {
  let out = escapeHtml(text);
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, text2, href) => {
    const safeHref = /^https?:\/\//.test(href) ? href : "#";
    return `<a href="${safeHref}" target="_blank" rel="noopener noreferrer">${text2}</a>`;
  });
  return out;
}

function isTableSeparator(line) {
  return /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line);
}

function tableCells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
}

function renderMarkdown(source) {
  const lines = (source || "").split("\n");
  let html = "";
  let listType = null;
  let inQuote = false;
  let inCode = false;
  let codeBuffer = [];
  let tableRows = [];
  let i = 0;

  const closeList = () => {
    if (listType) {
      html += `</${listType}>`;
      listType = null;
    }
  };
  const closeQuote = () => {
    if (inQuote) {
      html += "</blockquote>";
      inQuote = false;
    }
  };
  const flushTable = () => {
    if (tableRows.length < 2) {
      tableRows = [];
      return;
    }
    const [headerLine, separator, ...bodyLines] = tableRows;
    if (!isTableSeparator(separator)) {
      tableRows = [];
      return;
    }
    const headerCells = tableCells(headerLine);
    html += `<table><thead><tr>${headerCells.map((cell) => `<th>${inline(cell)}</th>`).join("")}</tr></thead><tbody>`;
    bodyLines.forEach((line) => {
      html += `<tr>${tableCells(line).map((cell) => `<td>${inline(cell)}</td>`).join("")}</tr>`;
    });
    html += "</tbody></table>";
    tableRows = [];
  };

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim().startsWith("```")) {
      closeList();
      closeQuote();
      flushTable();
      if (!inCode) {
        inCode = true;
        codeBuffer = [];
      } else {
        html += `<pre class="md-code"><code>${codeBuffer.join("\n")}</code></pre>`;
        inCode = false;
      }
      i += 1;
      continue;
    }
    if (inCode) {
      codeBuffer.push(escapeHtml(line));
      i += 1;
      continue;
    }

    if (line.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      closeList();
      closeQuote();
      tableRows = [line, lines[i + 1]];
      i += 2;
      while (i < lines.length && lines[i].includes("|") && lines[i].trim() !== "") {
        tableRows.push(lines[i]);
        i += 1;
      }
      flushTable();
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeList();
      closeQuote();
      const level = heading[1].length;
      html += `<h${level}>${inline(heading[2])}</h${level}>`;
      i += 1;
      continue;
    }

    if (/^>\s?/.test(line)) {
      closeList();
      if (!inQuote) {
        html += "<blockquote>";
        inQuote = true;
      }
      html += `<p>${inline(line.replace(/^>\s?/, ""))}</p>`;
      i += 1;
      continue;
    }
    closeQuote();

    const unordered = line.match(/^\s*[-*]\s+(.*)$/);
    const ordered = line.match(/^\s*\d+\.\s+(.*)$/);
    if (unordered || ordered) {
      const wantType = unordered ? "ul" : "ol";
      if (listType !== wantType) {
        closeList();
        html += `<${wantType}>`;
        listType = wantType;
      }
      html += `<li>${inline((unordered || ordered)[1])}</li>`;
      i += 1;
      continue;
    }
    closeList();

    if (line.trim() === "") {
      i += 1;
      continue;
    }

    html += `<p>${inline(line)}</p>`;
    i += 1;
  }

  closeList();
  closeQuote();
  flushTable();
  if (inCode) html += `<pre class="md-code"><code>${codeBuffer.join("\n")}</code></pre>`;
  return html;
}

/* ---------------------------------------------------------------------- */
/* Dependency-free Python syntax highlighting                             */
/* ---------------------------------------------------------------------- */

const PY_KEYWORDS = new Set([
  "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
  "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
  "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
  "try", "while", "with", "yield", "match", "case",
]);
const PY_BUILTINS = new Set([
  "print", "len", "range", "list", "dict", "set", "tuple", "str", "int", "float", "bool",
  "enumerate", "zip", "map", "filter", "sorted", "sum", "min", "max", "open", "type",
  "isinstance", "super", "self", "abs", "round", "any", "all",
]);
const PY_TOKEN_RE = /(#.*$)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|(\b\d+\.?\d*\b)|(@[A-Za-z_][\w.]*)|([A-Za-z_]\w*)/gm;

function highlightPython(source) {
  let out = "";
  let last = 0;
  PY_TOKEN_RE.lastIndex = 0;
  let match = PY_TOKEN_RE.exec(source);
  while (match) {
    const [full, comment, string, number, decorator, word] = match;
    out += escapeHtml(source.slice(last, match.index));
    last = match.index + full.length;
    if (comment) out += `<span class="tok-comment">${escapeHtml(comment)}</span>`;
    else if (string) out += `<span class="tok-string">${escapeHtml(string)}</span>`;
    else if (number) out += `<span class="tok-number">${escapeHtml(number)}</span>`;
    else if (decorator) out += `<span class="tok-decorator">${escapeHtml(decorator)}</span>`;
    else if (word) {
      if (PY_KEYWORDS.has(word)) out += `<span class="tok-keyword">${word}</span>`;
      else if (PY_BUILTINS.has(word)) out += `<span class="tok-builtin">${word}</span>`;
      else out += word;
    }
    match = PY_TOKEN_RE.exec(source);
  }
  out += escapeHtml(source.slice(last));
  return out;
}
