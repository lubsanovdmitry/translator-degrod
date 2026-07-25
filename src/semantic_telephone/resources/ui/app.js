const state = {
  bootstrap: null,
  yaml: "",
  view: null,
  profile: null,
  pipeline: [],
  plan: null,
  planHash: null,
  selectedRun: null,
  draftId: null,
  dirty: false,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escaped = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");

async function api(path, options = {}) {
  const request = { headers: {}, ...options };
  if (request.body && typeof request.body !== "string") {
    request.headers["Content-Type"] = "application/json";
    request.body = JSON.stringify(request.body);
  }
  const response = await fetch(path, request);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    throw new Error(payload?.error || payload || `Request failed (${response.status})`);
  }
  return payload;
}

function toast(message, type = "") {
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.textContent = message;
  $("#toasts").append(item);
  setTimeout(() => item.remove(), 4800);
}

function setDirty(value = true) {
  state.dirty = value;
  $("#save-state").textContent = value ? "Unsaved changes" : "Draft synchronized";
  if (value) {
    state.plan = null;
    state.planHash = null;
  }
}

function selectView(name) {
  const headings = {
    studio: ["Experiment workspace", "Configure a run"],
    plan: ["Offline resolution", "Review the plan"],
    queue: ["Controlled execution", "Run queue"],
    results: ["Artifact archive", "Inspect results"],
  };
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  $("#view-eyebrow").textContent = headings[name][0];
  $("#view-title").textContent = headings[name][1];
  if (name === "plan" && !state.plan) buildPlan();
  if (name === "results") refreshRuns();
}

function lineNumbers() {
  const editor = $("#yaml-editor");
  $("#yaml-lines").textContent = Array.from(
    { length: editor.value.split("\n").length },
    (_, index) => index + 1,
  ).join("\n");
}

function markValidation(kind, text) {
  const element = $("#validation-state");
  element.className = `validation-state ${kind}`;
  element.innerHTML = `<i></i> ${escaped(text)}`;
}

function profileMeta(id) {
  return state.bootstrap?.profiles.find((item) => item.id === id);
}

async function loadProfile(id) {
  try {
    const result = await api(`/api/profiles/${encodeURIComponent(id)}`);
    state.profile = profileMeta(id) || result;
    state.draftId = null;
    state.yaml = result.yaml;
    state.view = result.view;
    state.pipeline = structuredClone(result.view.pipeline || []);
    $("#yaml-editor").value = state.yaml;
    populateGuided(result.view);
    updateProfileBadge();
    lineNumbers();
    markValidation("valid", "Valid");
    setDirty(false);
  } catch (error) {
    toast(error.message, "error");
    markValidation("invalid", "Invalid");
  }
}

async function loadDraft(id) {
  try {
    const draft = await api(`/api/drafts/${encodeURIComponent(id)}`);
    const result = await api("/api/config/validate", {
      method: "POST", body: { yaml: draft.yaml },
    });
    state.profile = {
      id: `draft:${id}`,
      name: draft.name,
      kind: "draft",
      runnable: true,
      note: `Editable UI draft · last saved ${formatTime(draft.updated_at)}`,
    };
    state.draftId = id;
    state.yaml = result.yaml;
    state.view = result.view;
    $("#yaml-editor").value = result.yaml;
    populateGuided(result.view);
    updateProfileBadge();
    lineNumbers();
    markValidation("valid", "Valid draft");
    setDirty(false);
  } catch (error) {
    toast(error.message, "error");
  }
}

function updateProfileBadge() {
  const meta = state.profile || {};
  const badge = $("#profile-badge");
  badge.className = `profile-badge ${meta.kind || "error"}`;
  badge.textContent = meta.kind === "mock" ? "Mock smoke test" :
    meta.kind === "real" ? "Real experiment" :
      meta.kind === "draft" ? "UI draft" : "Unavailable";
  $("#profile-note").textContent = meta.note || "";
  $("#plan-top").disabled = meta.runnable === false;
  $("#delete-draft").hidden = !state.draftId;
}

function numberOrNull(id, integer = true) {
  const value = $(id).value.trim();
  if (!value) return null;
  return integer ? Number.parseInt(value, 10) : Number.parseFloat(value);
}

function populateGuided(view) {
  $("#run-name").value = view.run.name ?? "";
  $("#run-seed").value = view.run.seed ?? 0;
  $("#source-language").value = view.run.source_language ?? "auto";
  $("#target-language").value = view.run.target_language ?? "en";
  $("#route-mode").value = view.translation.route_mode ?? "fixed";
  $("#route-languages").value = (view.translation.languages || []).join(", ");
  $("#chunk-target").value = view.chunking.target_chars ?? 1100;
  $("#chunk-max").value = view.chunking.max_chars ?? 2200;
  $("#concurrency").value = view.runtime.concurrency ?? 1;
  $("#requests-minute").value = view.runtime.requests_per_minute ?? "";
  $("#retries").value = view.runtime.retries ?? 4;
  $("#budget-requests").value = view.runtime.budgets.max_requests ?? "";
  $("#budget-tokens").value = view.runtime.budgets.max_total_tokens ?? "";
  $("#budget-cost").value = view.runtime.budgets.max_cost_usd ?? "";
  $("#context-enabled").checked = view.context.enabled;
  $("#memory-enabled").checked = view.memory.enabled;
  state.pipeline = structuredClone(view.pipeline || []);
  renderPipeline();
}

function renderPipeline() {
  const list = $("#pipeline-list");
  list.replaceChildren();
  state.pipeline.forEach((stage, index) => {
    const row = document.createElement("div");
    row.className = "pipeline-stage";
    const enabled = stage.enabled !== false;
    row.innerHTML = `
      <label class="toggle stage-grip" title="Enable stage">
        <input type="checkbox" data-stage-enabled="${index}" ${enabled ? "checked" : ""}><span></span>
      </label>
      <select data-stage-type="${index}">
        ${["translation_cycle", "conservative_repair", "reconstruction",
          "contextual_reconstruction", "memory_extraction", "final_translation"]
          .map((type) => `<option ${type === stage.type ? "selected" : ""}>${type}</option>`).join("")}
      </select>
      <input class="stage-repeat" data-stage-repeat="${index}" type="number" min="1"
        value="${escaped(stage.repeat ?? 1)}" title="Repeat">
      <input class="stage-probability" data-stage-probability="${index}" type="number"
        min="0" max="1" step="0.1" value="${escaped(stage.probability ?? 1)}" title="Probability">
      <button class="stage-remove" data-stage-remove="${index}" title="Remove">×</button>`;
    list.append(row);
  });
}

function collectPipeline() {
  return state.pipeline.map((original, index) => {
    const stage = structuredClone(original);
    stage.type = $(`[data-stage-type="${index}"]`).value;
    stage.enabled = $(`[data-stage-enabled="${index}"]`).checked;
    stage.repeat = Number.parseInt($(`[data-stage-repeat="${index}"]`).value, 10);
    stage.probability = Number.parseFloat($(`[data-stage-probability="${index}"]`).value);
    if (stage.enabled) delete stage.enabled;
    if (stage.repeat === 1) delete stage.repeat;
    if (stage.probability === 1) delete stage.probability;
    return stage;
  });
}

async function applyGuided() {
  const languages = $("#route-languages").value.split(",").map((item) => item.trim()).filter(Boolean);
  const fields = {
    "run.name": $("#run-name").value.trim(),
    "run.seed": numberOrNull("#run-seed"),
    "run.source_language": $("#source-language").value.trim(),
    "run.target_language": $("#target-language").value.trim(),
    "translation.route_mode": $("#route-mode").value,
    "translation.languages": languages,
    "chunking.target_chars": numberOrNull("#chunk-target"),
    "chunking.max_chars": numberOrNull("#chunk-max"),
    "runtime.concurrency": numberOrNull("#concurrency"),
    "runtime.requests_per_minute": numberOrNull("#requests-minute"),
    "runtime.retries": numberOrNull("#retries"),
    "runtime.budgets.max_requests": numberOrNull("#budget-requests"),
    "runtime.budgets.max_total_tokens": numberOrNull("#budget-tokens"),
    "runtime.budgets.max_cost_usd": numberOrNull("#budget-cost", false),
    "context.enabled": $("#context-enabled").checked,
    "memory.enabled": $("#memory-enabled").checked,
    pipeline: collectPipeline(),
  };
  try {
    const result = await api("/api/config/patch", {
      method: "POST", body: { yaml: $("#yaml-editor").value, fields },
    });
    acceptValidation(result);
    toast("Guided controls applied to YAML.");
  } catch (error) {
    markValidation("invalid", "Needs attention");
    toast(error.message, "error");
  }
}

function acceptValidation(result) {
  state.yaml = result.yaml;
  state.view = result.view;
  $("#yaml-editor").value = result.yaml;
  populateGuided(result.view);
  lineNumbers();
  markValidation("valid", "Valid");
  setDirty(true);
}

async function validateYaml() {
  try {
    const result = await api("/api/config/validate", {
      method: "POST", body: { yaml: $("#yaml-editor").value },
    });
    acceptValidation(result);
    toast("Configuration is valid and formatted.");
    return true;
  } catch (error) {
    markValidation("invalid", "Invalid YAML");
    toast(error.message, "error");
    return false;
  }
}

async function buildPlan() {
  if (state.profile?.runnable === false) {
    toast("This profile is an interface stub and cannot be run.", "error");
    return;
  }
  const valid = await validateYaml();
  if (!valid) return;
  try {
    const result = await api("/api/plan", {
      method: "POST",
      body: { yaml: $("#yaml-editor").value, source: $("#source-text").value },
    });
    state.plan = result.plan;
    state.planHash = result.plan_hash;
    renderPlan();
    selectView("plan");
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderPlan() {
  const plan = state.plan;
  $("#plan-empty").hidden = true;
  $("#plan-content").hidden = false;
  $("#plan-seed").textContent = `seed ${plan.seed}`;
  const bounds = plan.request_bounds || {};
  const metrics = [
    ["Chunks", plan.chunks],
    ["Provider calls", bounds.provider_calls ?? 0],
    ["Maximum attempts", bounds.maximum_attempts ?? 0],
    ["Concurrency", plan.effective_concurrency],
  ];
  $("#plan-metrics").innerHTML = metrics.map(([label, value]) =>
    `<div class="metric-card"><small>${escaped(label)}</small><strong>${escaped(value)}</strong></div>`
  ).join("");
  $("#route-list").innerHTML = (plan.planned_routes || []).map((route) => `
    <article class="route-card">
      <div class="route-meta"><span>Chunk ${escaped(route.chunk)} / Stage ${escaped(route.stage)}</span>
        <span>seed ${escaped(route.seed)}</span></div>
      <div class="route-track">${(route.languages || []).map((language) =>
        `<div class="route-node"><span>${escaped(language)}</span></div>`).join("")}</div>
      <div class="route-engines">Engine candidates: ${(route.engines || [])
        .map((engine) => escaped((engine || []).join(" → "))).join(" · ") || "provider default"}</div>
    </article>`).join("") || `<div class="route-card">No translation cycle is enabled.</div>`;
  $("#resource-list").innerHTML = (plan.resources || []).map((resource) => `
    <div class="resource"><strong>${escaped(resource.alias)} · ${escaped(resource.type)}</strong>
      <span>${escaped(resource.model || "provider default")}${resource.revision ? ` @ ${escaped(resource.revision)}` : ""}</span>
      ${resource.may_download ? "<em>May download</em>" : ""}</div>`).join("");
  const warnings = [...(plan.warnings || [])];
  if ((plan.remote_services || []).length) {
    warnings.push(`Remote services: ${plan.remote_services.join(", ")}.`);
  }
  const downloads = (plan.resources || []).filter((item) => item.may_download).length;
  if (downloads) warnings.push(`${downloads} configured resource${downloads === 1 ? "" : "s"} may download model data.`);
  if (!warnings.length) warnings.push("No additional planner warnings.");
  $("#plan-warnings").innerHTML = warnings.map((warning) => `<p>— ${escaped(warning)}</p>`).join("");
}

function showLaunchDialog() {
  const plan = state.plan;
  if (!plan || !state.planHash) return;
  const bounds = plan.request_bounds || {};
  $("#launch-summary").textContent = `${plan.run} will process ${plan.chunks} chunk${plan.chunks === 1 ? "" : "s"} using the reviewed seeded plan.`;
  const facts = [
    ["Provider calls", bounds.provider_calls ?? 0],
    ["Max attempts", bounds.maximum_attempts ?? 0],
    ["Remote services", (plan.remote_services || []).length],
  ];
  $("#launch-facts").innerHTML = facts.map(([label, value]) =>
    `<div class="confirm-fact"><span>${escaped(label)}</span><strong>${escaped(value)}</strong></div>`
  ).join("");
  const remote = (plan.remote_services || []).join(", ");
  $("#launch-warning").textContent = remote
    ? `This plan may contact ${remote}. Provider-side costs depend on your configuration.`
    : "Local providers may download configured model weights on first use.";
  $("#launch-dialog").showModal();
}

async function launchRun(event) {
  event.preventDefault();
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      body: {
        yaml: $("#yaml-editor").value,
        source: $("#source-text").value,
        plan_hash: state.planHash,
      },
    });
    $("#launch-dialog").close();
    toast(`Queued ${job.name}.`);
    await refreshJobs();
    selectView("queue");
  } catch (error) {
    toast(error.message, "error");
  }
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
}

async function refreshJobs() {
  try {
    const jobs = await api("/api/jobs");
    const active = jobs.filter((job) => ["queued", "running"].includes(job.status));
    $("#queue-count").textContent = active.length;
    $("#job-list").innerHTML = jobs.map((job) => {
      const total = Number(job.progress?.total || 0);
      const done = Number(job.progress?.processed || 0);
      const percent = total ? Math.min(100, Math.round(done / total * 100)) : 0;
      const cancellable = ["queued", "running"].includes(job.status);
      return `<article class="job-card">
        <span class="job-status ${escaped(job.status)}"><i></i>${escaped(job.status)}</span>
        <div class="job-name"><strong>${escaped(job.name)}</strong>
          <span>${escaped(job.id.slice(0, 12))} · ${escaped(formatTime(job.created_at))}</span>
          ${job.error ? `<small class="error-copy">${escaped(job.error)}</small>` : ""}</div>
        <div class="job-progress"><div class="progress-track"><i style="width:${percent}%"></i></div>
          <div class="progress-label"><span>${done} / ${total} chunks</span><span>${percent}%</span></div></div>
        ${cancellable ? `<button class="cancel-job" data-cancel-job="${escaped(job.id)}">Cancel</button>` : "<span></span>"}
      </article>`;
    }).join("") || `<div class="empty-state compact"><h2>The queue is clear.</h2><p>Build and confirm a plan to start an experiment.</p></div>`;
  } catch (error) {
    toast(error.message, "error");
  }
}

async function cancelJob(id) {
  try {
    await api(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST", body: {} });
    toast("Cancellation requested.");
    refreshJobs();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function runDoctor() {
  const valid = await validateYaml();
  if (!valid) return;
  $("#run-doctor").disabled = true;
  $("#run-doctor").textContent = "Checking…";
  try {
    const result = await api("/api/doctor", {
      method: "POST", body: { yaml: $("#yaml-editor").value, source: $("#source-text").value },
    });
    const checks = result.doctor?.checks || [];
    const failed = checks.filter((check) => check.status !== "pass");
    toast(failed.length ? `Doctor found ${failed.length} item${failed.length === 1 ? "" : "s"} to review.` : "Doctor checks passed.", failed.length ? "error" : "");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    $("#run-doctor").disabled = false;
    $("#run-doctor").textContent = "Run doctor";
  }
}

async function saveDraft(event) {
  event.preventDefault();
  const name = $("#draft-name").value.trim();
  if (!name) {
    toast("Give the draft a name.", "error");
    return;
  }
  try {
    const updating = Boolean(state.draftId);
    const saved = await api("/api/drafts", {
      method: "POST", body: {
        id: state.draftId,
        name,
        yaml: $("#yaml-editor").value,
      },
    });
    state.draftId = saved.id;
    $("#draft-dialog").close();
    setDirty(false);
    await refreshDrafts();
    toast(updating ? "Draft updated." : "Draft saved without changing the repository profile.");
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderProfileOptions() {
  const profiles = state.bootstrap.profiles || [];
  const drafts = state.bootstrap.drafts || [];
  const selected = $("#profile-select").value;
  $("#profile-select").innerHTML = `
    <optgroup label="Read-only profiles">${profiles.map((profile) =>
      `<option value="${escaped(profile.id)}">${escaped(profile.name)}${profile.kind === "mock" ? " — smoke test" : ""}</option>`
    ).join("")}</optgroup>
    ${drafts.length ? `<optgroup label="UI drafts">${drafts.map((draft) =>
      `<option value="draft:${escaped(draft.id)}">${escaped(draft.name)}</option>`).join("")}</optgroup>` : ""}`;
  if ($(`#profile-select option[value="${CSS.escape(selected)}"]`)) {
    $("#profile-select").value = selected;
  }
}

async function refreshDrafts() {
  state.bootstrap.drafts = await api("/api/drafts");
  renderProfileOptions();
  if (state.draftId) $("#profile-select").value = `draft:${state.draftId}`;
}

async function deleteDraft() {
  if (!state.draftId || !window.confirm("Delete this UI draft? Repository profiles and runs are unaffected.")) return;
  try {
    await api(`/api/drafts/${encodeURIComponent(state.draftId)}`, { method: "DELETE" });
    const fallback = state.bootstrap.profiles.find((item) => item.id === "mixed_local" && item.runnable)
      || state.bootstrap.profiles.find((item) => item.runnable);
    state.draftId = null;
    await refreshDrafts();
    if (fallback) {
      $("#profile-select").value = fallback.id;
      await loadProfile(fallback.id);
    }
    toast("Draft deleted.");
  } catch (error) {
    toast(error.message, "error");
  }
}

function flattenMetrics(metrics, prefix = "") {
  const values = [];
  Object.entries(metrics || {}).forEach(([key, value]) => {
    const label = prefix ? `${prefix}.${key}` : key;
    if (typeof value === "number" || typeof value === "string") values.push([label, value]);
    else if (value && typeof value === "object" && !Array.isArray(value)) values.push(...flattenMetrics(value, label));
  });
  return values;
}

async function refreshRuns() {
  try {
    const runs = await api("/api/runs");
    state.bootstrap.runs = runs;
    $("#run-list").innerHTML = runs.map((run) => `
      <button class="run-item ${state.selectedRun === run.id ? "active" : ""}" data-run-id="${escaped(run.id)}">
        <i class="${escaped(run.state)}"></i><span><strong>${escaped(run.name)}</strong>
        <small>${escaped(run.state)} · ${escaped(formatTime(run.started_at))}<br>${escaped(run.id)}</small></span>
      </button>`).join("") || `<div class="profile-note">No manifests found under the configured run root.</div>`;
    const completed = runs.filter((run) => run.state === "completed");
    for (const id of ["#compare-left", "#compare-right"]) {
      const selected = $(id).value;
      $(id).innerHTML = completed.map((run) =>
        `<option value="${escaped(run.id)}">${escaped(run.name)} · ${escaped(run.id)}</option>`
      ).join("");
      if (completed.some((run) => run.id === selected)) $(id).value = selected;
    }
    if (completed.length > 1 && !$("#compare-right").value) $("#compare-right").selectedIndex = 1;
  } catch (error) {
    toast(error.message, "error");
  }
}

async function selectRun(id) {
  state.selectedRun = id;
  try {
    const run = await api(`/api/runs/${encodeURIComponent(id)}`);
    renderRun(run);
    refreshRuns();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderRun(run) {
  const manifest = run.manifest || {};
  const config = manifest.resolved_config || {};
  const metrics = flattenMetrics(run.metrics).slice(0, 6);
  const stages = (run.checkpoints || []).flatMap((chunk) =>
    (chunk.stages || []).map((stage) => ({ ...stage, chunk: chunk.chunk })));
  $("#result-detail").innerHTML = `
    <div class="result-head"><div><p class="eyebrow">${escaped(manifest.state || "unknown")}</p>
      <h2>${escaped(config.name || run.id)}</h2></div>
      <div class="result-head-meta"><span>seed ${escaped(manifest.seed)}</span>
        <span>${escaped(manifest.processed_chunks?.length || 0)} chunks</span></div></div>
    ${manifest.last_error ? `<div class="confirm-warning"><strong>Run diagnostic</strong><span>${escaped(manifest.last_error)}</span></div>` : ""}
    <div class="result-metrics">${metrics.map(([name, value]) =>
      `<div class="result-metric"><span>${escaped(name)}</span><strong>${escaped(
        typeof value === "number" ? Number(value).toFixed(value % 1 ? 3 : 0) : value)}</strong></div>`
    ).join("")}</div>
    <div class="text-pair">
      <div class="text-pane"><header><span>Source</span><span>${run.source.length} chars</span></header><pre>${escaped(run.source || "Not written yet.")}</pre></div>
      <div class="text-pane"><header><span>Final</span><span>${run.final.length} chars</span></header><pre>${escaped(run.final || "Not written yet.")}</pre></div>
    </div>
    <section class="panel checkpoint-panel">
      <div class="section-head"><div><span>Provenance</span><h2>Stage checkpoints</h2></div><span class="counter">${stages.length} stages</span></div>
      ${stages.map((stage) => `<details class="checkpoint">
        <summary>Chunk ${escaped(stage.chunk)} · ${escaped(stage.stage_type || stage.file)} · ${escaped(stage.provider || "unknown provider")}</summary>
        <div class="checkpoint-body">
          <div class="route-engines">Languages: ${escaped((stage.route || []).join(" → "))}<br>
            Engines: ${escaped((stage.provider_route || []).join(" → "))}<br>
            Model: ${escaped(stage.model || "—")} · Duration: ${escaped(stage.duration_seconds ?? "—")}s</div>
          ${stage.error ? `<div class="confirm-warning">${escaped(stage.error)}</div>` : ""}
          <pre>${escaped(stage.output || "No output artifact.")}</pre>
        </div></details>`).join("") || `<div class="profile-note">No checkpoint metadata has been written.</div>`}
    </section>
    <section class="panel checkpoint-panel">
      <div class="section-head"><div><span>Report</span><h2>Run narrative</h2></div></div>
      <pre class="report-pre">${escaped(run.report || "No report has been written.")}</pre>
    </section>`;
}

async function compareRuns() {
  const first = $("#compare-left").value;
  const second = $("#compare-right").value;
  if (!first || !second) {
    toast("Two completed runs are required.", "error");
    return;
  }
  try {
    const comparison = await api("/api/compare", {
      method: "POST", body: { first, second },
    });
    $("#compare-output").innerHTML = `<div class="compare-texts">
      <div class="text-pane"><header><span>${escaped(comparison.left.manifest.resolved_config?.name || first)}</span>
        <span>${comparison.left.final.length} chars</span></header><pre>${escaped(comparison.left.final)}</pre></div>
      <div class="text-pane"><header><span>${escaped(comparison.right.manifest.resolved_config?.name || second)}</span>
        <span>${comparison.right.final.length} chars</span></header><pre>${escaped(comparison.right.final)}</pre></div>
    </div>`;
  } catch (error) {
    toast(error.message, "error");
  }
}

function bindEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => selectView(button.dataset.view)));
  $("#profile-select").addEventListener("change", (event) => {
    const value = event.target.value;
    if (value.startsWith("draft:")) loadDraft(value.slice(6));
    else loadProfile(value);
  });
  $("#source-text").addEventListener("input", () => {
    $("#source-count").textContent = `${$("#source-text").value.length.toLocaleString()} characters`;
    setDirty();
  });
  $("#yaml-editor").addEventListener("input", () => {
    state.yaml = $("#yaml-editor").value;
    lineNumbers();
    markValidation("", "Not validated");
    setDirty();
  });
  $("#yaml-editor").addEventListener("scroll", () => {
    $("#yaml-lines").scrollTop = $("#yaml-editor").scrollTop;
  });
  $("#yaml-editor").addEventListener("keydown", (event) => {
    if (event.key !== "Tab") return;
    event.preventDefault();
    const editor = event.target;
    editor.setRangeText("  ", editor.selectionStart, editor.selectionEnd, "end");
    editor.dispatchEvent(new Event("input"));
  });
  $("#apply-guided").addEventListener("click", applyGuided);
  $("#format-yaml").addEventListener("click", validateYaml);
  $("#run-doctor").addEventListener("click", runDoctor);
  $("#plan-top").addEventListener("click", buildPlan);
  $("#plan-empty-button").addEventListener("click", buildPlan);
  $("#review-launch").addEventListener("click", showLaunchDialog);
  $("#confirm-launch").addEventListener("click", launchRun);
  $("#save-draft").addEventListener("click", () => {
    $("#draft-name").value = $("#run-name").value || "Experiment draft";
    $("#draft-dialog").showModal();
  });
  $("#confirm-draft").addEventListener("click", saveDraft);
  $("#delete-draft").addEventListener("click", deleteDraft);
  $("#add-stage").addEventListener("click", () => {
    state.pipeline.push({ type: "translation_cycle" });
    renderPipeline();
    setDirty();
  });
  $("#pipeline-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-stage-remove]");
    if (!button) return;
    state.pipeline.splice(Number(button.dataset.stageRemove), 1);
    renderPipeline();
    setDirty();
  });
  $("#job-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-cancel-job]");
    if (button) cancelJob(button.dataset.cancelJob);
  });
  $("#run-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-run-id]");
    if (button) selectRun(button.dataset.runId);
  });
  $("#compare-button").addEventListener("click", compareRuns);
}

async function initialize() {
  bindEvents();
  try {
    state.bootstrap = await api("/api/bootstrap");
    const profiles = state.bootstrap.profiles || [];
    renderProfileOptions();
    const preferred = profiles.find((item) => item.id === "mixed_local" && item.runnable)
      || profiles.find((item) => item.runnable);
    if (preferred) {
      $("#profile-select").value = preferred.id;
      await loadProfile(preferred.id);
    }
    $("#source-text").value = state.bootstrap.source || "";
    $("#source-count").textContent = `${$("#source-text").value.length.toLocaleString()} characters`;
    await refreshJobs();
    await refreshRuns();
    setInterval(async () => {
      await refreshJobs();
      const queueVisible = $("#view-queue").classList.contains("active");
      const resultsVisible = $("#view-results").classList.contains("active");
      if (queueVisible || resultsVisible) await refreshRuns();
    }, 2000);
  } catch (error) {
    toast(error.message, "error");
  }
}

initialize();
