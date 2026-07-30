const app = document.getElementById("app");
let traces = [];
let sortKey = "created_at", sortDir = -1;

const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const short = id => (id || "").slice(0, 8);
// clip before escaping, so a cut never lands inside an entity
const clip = (s, n) => s.length > n ? s.slice(0, n) + "…" : s;

async function boot() {
  traces = await (await fetch("/api/traces")).json();
  renderList();
}

function distinct(key) { return [...new Set(traces.map(t => t[key]).filter(Boolean))].sort(); }

function renderList() {
  const f = window.__filters || {};
  const opt = (key, label) =>
    `<select data-f="${key}"><option value="">${label}</option>` +
    distinct(key).map(v => `<option ${f[key]===v?"selected":""}>${esc(v)}</option>`).join("") + `</select>`;
  let rows = traces.filter(t =>
    (!f.task || (t.task||"").toLowerCase().includes(f.task.toLowerCase())) &&
    (!f.model || t.model === f.model) &&
    (!f.thinking || t.thinking === f.thinking) &&
    (!f.status || t.status === f.status) &&
    (!f.fixture_id || t.fixture_id === f.fixture_id));
  rows.sort((a, b) => {
    const x = a[sortKey] ?? "", y = b[sortKey] ?? "";
    return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
  });
  const cols = [["created_at","created"],["status","status"],["steps_taken","steps"],
                ["model","model"],["thinking","think"],["fixture_id","fixture"],["task","task"]];
  app.innerHTML = `
    <h1>webagent traces <span class="muted">(${rows.length}/${traces.length})</span></h1>
    <div class="controls">
      <input data-f="task" placeholder="filter task…" value="${esc(f.task||"")}">
      ${opt("model","any model")} ${opt("thinking","any thinking")}
      ${opt("status","any status")} ${opt("fixture_id","any fixture")}
    </div>
    <table><thead><tr>${cols.map(([k,l]) =>
        `<th data-k="${k}">${l}${sortKey===k?(sortDir>0?" ▲":" ▼"):""}</th>`).join("")}
      <th>tok in/out</th></tr></thead>
    <tbody>${rows.map(t => `
      <tr data-id="${t.trace_id}">
        <td>${esc((t.created_at||"").replace("T"," ").slice(0,16))}</td>
        <td class="status-${esc(t.status)}">${esc(t.status)}</td>
        <td>${t.steps_taken}</td>
        <td>${esc(t.model)}</td>
        <td>${esc(t.thinking)}</td>
        <td>${esc(t.fixture_id||"-")}</td>
        <td class="task"><span title="${esc(t.task)}">${esc(t.task)}</span></td>
        <td class="muted">${t.total_input_tokens}/${t.total_output_tokens}</td>
      </tr>`).join("")}</tbody></table>`;
  app.querySelectorAll("[data-f]").forEach(el => {
    const ev = el.tagName === "SELECT" ? "change" : "input";
    el.addEventListener(ev, e => {
      window.__filters = {...(window.__filters||{}), [el.dataset.f]: e.target.value};
      renderList();
    });
  });
  app.querySelectorAll("th[data-k]").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.k;
    if (sortKey === k) sortDir = -sortDir; else { sortKey = k; sortDir = 1; }
    renderList();
  }));
  app.querySelectorAll("tr[data-id]").forEach(tr =>
    tr.addEventListener("click", () => showDetail(tr.dataset.id)));
}

function reasoningNote(g) {
  if (g.reasoning) return esc(g.reasoning);
  if (g.reasoning_encrypted || g.reasoning_tokens) {
    const tok = g.reasoning_tokens ? `${g.reasoning_tokens} reasoning tokens; ` : "";
    return `<span class="muted">&lt;withheld by provider — ${tok}no reasoning summary returned&gt;</span>`;
  }
  return null;
}
const asText = v => typeof v === "string" ? v : JSON.stringify(v, null, 2);

function obsCard(o) {
  if (o.type === "generation") {
    const rn = reasoningNote(o);
    return `<div class="obs gen">
      <div class="obs-hd"><span class="kind">◆ generation</span>
        <span>[${esc(o.name)}]</span>
        <span class="muted">→ ${esc(clip(asText(o.output), 90))}</span>
        <span class="muted" style="margin-left:auto">${o.duration_seconds.toFixed(1)}s</span></div>
      <div class="obs-body">
        ${rn !== null ? `<div class="field-label">reasoning</div><pre>${rn}</pre>` : ""}
        ${o.memory ? `<div class="field-label">memory</div><pre>${esc(o.memory)}</pre>` : ""}
        <div class="field-label">action / output</div><pre>${esc(asText(o.output))}</pre>
        <details class="fold"><summary class="field-label">input prompt
          <span class="muted">(${o.input_prompt.length} chars)</span></summary>
          <pre>${esc(o.input_prompt)}</pre></details>
        <div class="muted">model=${esc(o.model)} · tokens in=${o.input_tokens} out=${o.output_tokens}${
          o.reasoning_tokens ? " reasoning="+o.reasoning_tokens : ""} · finish=${esc(o.finish_reason||"-")}</div>
      </div></div>`;
  }
  const args = Object.entries(o.args||{}).map(([k,v]) => `${k}=${esc(asText(v))}`).join(", ");
  const bad = o.status !== "ok";
  return `<div class="obs tool">
    <div class="obs-hd"><span class="kind">▸ tool</span>
      <span>${esc(o.name)}(${args})</span>
      <span class="${bad?"err":"muted"}">→ ${bad ? "ERROR" : "ok"}</span>
      <span class="muted" style="margin-left:auto">${o.duration_seconds.toFixed(1)}s</span></div>
    <div class="obs-body">
      ${o.error ? `<div class="field-label err">error</div><pre class="err">${esc(o.error)}</pre>` : ""}
      ${o.result ? `<div class="field-label">result</div><pre>${esc(o.result)}</pre>` : ""}
      ${!o.error && !o.result ? '<div class="muted">no result returned to the model</div>' : ""}
    </div></div>`;
}

async function showDetail(id) {
  const t = await (await fetch("/api/traces/" + id)).json();
  const byStep = {};
  (t.observations||[]).forEach(o => (byStep[o.step] = byStep[o.step] || []).push(o));
  const steps = Object.keys(byStep).map(Number).sort((a,b)=>a-b);
  app.innerHTML = `
    <h1><a onclick="renderList()">&larr; all traces</a> &nbsp; ${short(t.trace_id)}</h1>
    <div class="meta">
      <div><b>task:</b> ${esc(t.task)}</div>
      <div><b>url:</b> ${esc(t.url)}</div>
      <div><b>model:</b> ${esc(t.model)} · <b>thinking:</b> ${esc(t.thinking)} · <b>output:</b> ${esc(t.output_mode)}</div>
      <div class="status-${esc(t.status)}"><b>status:</b> ${esc(t.status)} · steps ${t.steps_taken} ·
        ${t.duration_seconds.toFixed(1)}s · tokens ${t.total_input_tokens}/${t.total_output_tokens}</div>
      ${t.fixture_id ? `<div><b>fixture:</b> ${esc(t.fixture_id)} · run ${esc(t.run_id||"-")}</div>` : ""}
    </div>
    ${steps.map(s => `<div class="step"><div class="step-hd">step ${s}</div>
        ${byStep[s].map(obsCard).join("")}</div>`).join("")}
    <details class="step"><summary class="step-hd" style="cursor:pointer">system prompt</summary>
      <pre style="margin:.7rem">${esc(t.system_prompt)}</pre></details>`;
  app.querySelectorAll(".obs-hd").forEach(hd =>
    hd.addEventListener("click", () => hd.nextElementSibling.classList.toggle("open")));
}

boot();
