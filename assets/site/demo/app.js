(() => {
  'use strict';
  const DATA = JSON.parse(document.getElementById('replay-data').textContent);
  const CONFIG = JSON.parse(document.getElementById('replay-config').textContent);
  const RECORDING = JSON.parse(document.getElementById('recording-config').textContent);
  const BASE_PLAYBACK_RATE = 2;
  const HORIZON = DATA.chartDuration || DATA.duration;
  const $ = id => document.getElementById(id);
  const escapeHTML = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const formatScore = n => Number(n).toFixed(CONFIG.score.decimals);
  const formatTime = seconds => {
    const s = Math.max(0, Math.floor(seconds));
    return [Math.floor(s / 3600), Math.floor(s / 60) % 60, s % 60].map(n => String(n).padStart(2,'0')).join(':');
  };
  const formatReplay = seconds => {
    const elapsed = Math.floor(seconds / BASE_PLAYBACK_RATE);
    return `${String(Math.floor(elapsed / 60)).padStart(2,'0')}:${String(elapsed % 60).padStart(2,'0')}`;
  };
  const events = [...DATA.submissions, ...DATA.edits, ...DATA.writes].sort((a,b) => a.time - b.time);
  const fileEvents = events.filter(event => event.kind !== 'submission');
  const byId = new Map(events.map(e => [e.id,e]));
  const capture = new URLSearchParams(location.search).has('capture');
  const makeTimeScale = () => {
    const boundaries = [...new Set([0,3600,9000,12600,HORIZON])].sort((a,b) => a-b);
    let total = 0;
    const segments = boundaries.slice(1).map((end,index) => {
      const start = boundaries[index];
      const weight = start < 3600 ? .5 : start < 9000 ? 1/3 : start < 12600 ? .25 : 1;
      const segment = {start,end,offset:total,weight};
      total += (end-start)*weight;
      return segment;
    });
    return {
      length: total,
      forward(time) {
        const value = clamp(time,0,HORIZON);
        const segment = segments.find(part => value <= part.end) || segments.at(-1);
        return segment.offset + (value-segment.start)*segment.weight;
      },
      inverse(position) {
        const value = clamp(position,0,total);
        const segment = segments.find(part => value <= part.offset+(part.end-part.start)*part.weight) || segments.at(-1);
        return segment.start + (value-segment.offset)/segment.weight;
      },
    };
  };
  const chartScale = makeTimeScale();
  const playbackScale = chartScale;
  const START = CONFIG.timing.trajectoryStart;
  const END = START + (CONFIG.timing.trajectoryEnd-START)*playbackScale.length/HORIZON;
  const DURATION = END + CONFIG.timing.duration-CONFIG.timing.trajectoryEnd;
  const stageTimes = {statement:0, approach:CONFIG.timing.methodStart, trajectory:START};
  const state = {time:capture ? 0 : START, actual:0, mode:'playback', playing:false, speed:1, view:'replay',
    stage:capture ? 'statement' : 'trajectory', selected:null, sourceTab:'code', inspectorPanel:'code', eventFilter:'all', query:'', lastInspectorKey:'', external:false};
  const demoVideo = $('demoVideo');
  let raf = 0, lastFrame = 0, toastTimer = 0, positions = new Map();
  let embeddedPlaybackRequested = false;
  let reportedView = null;
  if (capture) document.body.classList.add('capture');
  const stageAt = time => time < CONFIG.timing.methodStart ? 'statement' : time < START ? 'approach' : 'trajectory';
  const actualAt = time => playbackScale.inverse(clamp((time - START) / (END - START), 0, 1) * playbackScale.length);
  const replayAt = actual => START + playbackScale.forward(actual) / playbackScale.length * (END - START);
  const latestSubmission = time => DATA.submissions.filter(e => e.time <= time + 0.001).at(-1) || null;
  const bestAt = time => DATA.submissions.reduce((best,s) => s.time <= time + 0.001 ? Math.max(best,s.score) : best, CONFIG.score.minimum);
  const baselineAt = time => (DATA.baseline?.points || []).reduce((best,s) => s.time <= time + 0.001 ? Math.max(best,s.score) : best, 0);
  const editOutcome = event => event.outcome || (event.failed ? 'failed' : 'applied');
  const editLabel = event => ({applied:'Patch applied',failed:'Not applied',unknown:'Outcome unknown'})[editOutcome(event)];
  const activeEvent = () => state.selected ? byId.get(state.selected) : latestSubmission(state.actual) || events.filter(e=>e.time<=state.actual+0.001).at(-1) || null;
  const eventLabel = event => `${event.kind.toUpperCase()} ${String(event.number).padStart(event.kind === 'submission' ? 2 : 3,'0')}`;
  const eventTitle = event => event.kind === 'submission' ? event.algorithm.title : event.filename;

  let exampleStep = 0;
  function renderBoxExample(step) {
    exampleStep = step;
    const stacks = step === 0 ? [[1,4,7],[6,2,8],[9,5,3]] : [[...(step === 1 ? [1] : [])],[6,2,8,4,7],[9,5,3]];
    let diagram = '<defs><marker id="boxArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#9ca3af"/></marker></defs>';
    stacks.forEach((stack, index) => {
      const x = 43 + index * 135;
      diagram += `<path d="M${x-8} 205v7h82v-7" fill="none" stroke="#9ca3af" stroke-width="2"/><text x="${x+33}" y="235" text-anchor="middle" fill="#6b7280" font-size="11">Stack ${index+1}</text>`;
      stack.forEach((box, level) => {
        const y = 210 - (level+1)*31;
        const target = box === 1, moved = box === 4 || box === 7;
        diagram += `<g><rect x="${x}" y="${y}" width="66" height="28" rx="4" fill="${target ? 'var(--accent)' : moved ? 'var(--accent-soft)' : '#f4f6f8'}" stroke="${target ? 'var(--accent)' : moved ? 'var(--accent-line)' : '#9ca3af'}"/><path d="M${x+33} ${y+1}v7" stroke="${target ? '#9ca3af' : moved ? 'var(--accent-line)' : '#9ca3af'}"/><text x="${x+33}" y="${y+20}" text-anchor="middle" font-size="16" font-weight="500" fill="${target ? '#ffffff' : moved ? 'var(--accent)' : '#6b7280'}">${box}</text></g>`;
      });
    });
    if (step === 0) diagram += '<path d="M82 106C99 54 163 27 202 42" fill="none" stroke="#9ca3af" stroke-width="1.8" stroke-dasharray="4 4" marker-end="url(#boxArrow)"/><text x="97" y="22" fill="var(--accent)" font-size="11">2 boxes · +3 energy</text>';
    if (step === 1) diagram += '<path d="M76 168V99" stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="4 4"/><text x="76" y="82" text-anchor="middle" fill="var(--accent)" font-size="12">1 is ready</text>';
    if (step === 2) diagram += '<rect x="43" y="179" width="66" height="28" rx="4" fill="none" stroke="#9ca3af" stroke-dasharray="4 4"/><text x="76" y="133" text-anchor="middle" fill="var(--accent)" font-size="13">1 → out</text><text x="76" y="154" text-anchor="middle" fill="#6b7280" font-size="11">+0 energy</text>';
    $('boxDiagram').innerHTML = diagram;
    const titles = ['Box 1 is blocked.', 'Box 1 is now on top.', 'First box carried out.'];
    const captions = ['Move boxes 4 and 7 together to stack 2, keeping their order.', 'Moving 2 boxes costs 2 + 1 = 3 energy. Now remove box 1.', 'Removing box 1 costs nothing. Next, expose and carry out box 2.'];
    $('boxExampleTitle').textContent = titles[step];
    $('boxExampleCaption').textContent = captions[step];
    $('boxDiagram').setAttribute('aria-label', titles[step] + ' ' + captions[step]);
    $('boxEnergy').textContent = step ? '3' : '0';
    $('boxStep').innerHTML = ['Move 2 boxes <span>→</span>', 'Carry out box 1 <span>→</span>', 'Try again <span>↺</span>'][step];
    $('boxReset').disabled = step === 0;
  }
  $('boxStep').addEventListener('click', () => {stopPlayback(); renderBoxExample((exampleStep+1)%3);});
  $('boxReset').addEventListener('click', () => {stopPlayback(); renderBoxExample(0);});
  renderBoxExample(0);

  function syntaxLine(text) {
    // Tokenize original text, then escape every token. Archived code is inert text.
    const pattern = /(\/\/.*$|#[^\n]*$|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:alignas|auto|bool|break|case|char|class|const|continue|default|double|else|enum|false|float|for|if|inline|int|long|namespace|nullptr|private|public|return|short|signed|sizeof|static|struct|switch|template|true|typedef|typename|unsigned|using|void|while|__int128)\b|\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?[uUlLfF]*\b)/g;
    let cursor = 0, out = '', match;
    while ((match = pattern.exec(text))) {
      out += escapeHTML(text.slice(cursor,match.index));
      const token = match[0];
      const cls = token.startsWith('//') ? 'comment' : token.startsWith('#') ? 'preproc' : /^["']/.test(token) ? 'string' : /^\d/.test(token) ? 'number' : 'keyword';
      out += `<span class="syn-${cls}">${escapeHTML(token)}</span>`;
      cursor = match.index + token.length;
    }
    return out + escapeHTML(text.slice(cursor));
  }
  function sourceHTML(event, tab = 'code') {
    if (tab === 'feedback') return `<pre class="source-code">${escapeHTML(event.feedback)}</pre>`;
    if (event.sourceKind === 'request') return `<pre class="source-code request-source">${escapeHTML(event.request)}</pre>`;
    if(event.kind==='submission' && event.codeAvailable===false)return `<pre class="source-code">${escapeHTML(event.codeSource || 'Source code was not recovered for this submission.')}</pre>`;
    const source = event.kind === 'edit' ? event.patch : event.code;
    return `<pre class="source-code">${source.split('\n').map((line,index) => {
      if (event.kind === 'edit') {
        const cls = /^(---|\+\+\+)/.test(line) ? 'patch-header' : line.startsWith('+') ? 'patch-add' : line.startsWith('-') ? 'patch-remove' : line.startsWith('@@') ? 'patch-hunk' : '';
        return `<span class="code-line ${cls}">${escapeHTML(line) || ' '}</span>`;
      }
      return `<span class="code-line"><span class="line-no" aria-hidden="true">${index + 1}</span><span class="line-text">${syntaxLine(line)}</span></span>`;
    }).join('')}</pre>`;
  }
  function renderInspector() {
    const event = activeEvent();
    const key = `${event?.id || 'empty'}:${state.sourceTab}`;
    if (key === state.lastInspectorKey) return;
    state.lastInspectorKey = key;
    $('expandButton').disabled = !event;
    $('previousEvent').disabled = !event || events.indexOf(event) <= 0;
    $('nextEvent').disabled = !events.length || (!!event && events.indexOf(event) === events.length - 1);
    if (!event) {
      $('inspectorKicker').textContent = 'INSIDE THE RUN';
      $('inspectorContent').innerHTML = '<div class="inspector-empty"><h3>The solution takes shape.</h3><p>Follow solution.cpp from its first write, then inspect each submitted version and the edits along the way.</p><div class="empty-hints"><span>● Submission or edit</span><span>■ Full-file write</span></div></div>';
      return;
    }
    const isEdit = event.kind === 'edit', isSubmission = event.kind === 'submission';
    $('inspectorKicker').textContent = 'INSIDE THE RUN';
    const description = isSubmission ? event.algorithm.summary : event.summary;
    const badges = isSubmission ? event.algorithm.tags.slice(0,3) : [];
    const statusLabel = typeof event.detail === 'string' && event.detail.length > 0 && event.detail.length < 48 ? event.detail : event.status || 'Recorded score';
    const score = isSubmission
      ? `<div class="event-score"><strong>${formatScore(event.score)}</strong><span>${escapeHTML(CONFIG.score.suffix)}</span>${event.improvement ? `<span class="improvement">${event.bestBefore !== null ? '+' + formatScore(event.score - event.bestBefore) : 'First score'}</span>` : ''}<span class="status-tag">${escapeHTML(statusLabel)}</span></div>`
      : `<div class="event-score"><span class="${event.outcome==='applied' ? 'improvement' : 'status-tag'}">${isEdit ? editLabel(event) : event.outcome==='applied' ? 'File written' : 'Write '+event.outcome}</span><span>${isEdit ? `+${event.added} / −${event.removed} lines` : `${event.code.split('\n').length} lines`}</span></div>`;
    const historyIds = [...new Set([event.firstWriteId,event.previousWriteId].filter(id=>byId.has(id)))];
    const history = historyIds.length ? `<div class="file-history">${historyIds.map(id=>`<button data-history-event="${id}">${id===event.firstWriteId ? 'First recorded write' : 'Previous write'} · ${formatTime(byId.get(id).time)} ↗</button>`).join('')}</div>` : '';
    const source = isEdit ? event.patchSource : event.codeSource;
    const sourceLabel = isEdit ? 'Patch' : event.sourceKind === 'request' ? 'Request' : 'Code';
    $('inspectorContent').innerHTML = `<div class="inspection-meta"><b class="${event.failed ? 'failed' : ''}">${eventLabel(event)}</b><span title="Elapsed since the original run start">${formatTime(event.time)}</span></div><div class="algorithm-info"><h3>${escapeHTML(eventTitle(event))}</h3><p>${escapeHTML(description)}</p>${isSubmission ? '<span class="analysis-label">Analysis from recorded source</span>' : ''}${badges.length ? `<div class="algorithm-tags">${badges.map(tag=>`<span>${escapeHTML(tag)}</span>`).join('')}</div>` : ''}${score}</div>${history}<div class="source-toolbar"><span title="${escapeHTML(event.sourcePath || event.filename)}">${escapeHTML(event.filename)}</span><div><button data-source-tab="code" class="${state.sourceTab === 'code' ? 'active' : ''}">${sourceLabel}</button><button data-source-tab="feedback" class="${state.sourceTab === 'feedback' ? 'active' : ''}">Feedback</button><button data-copy title="Copy source" aria-label="Copy source">Copy</button></div></div><div class="source-view ${isEdit && state.sourceTab === 'code' ? 'patch-view' : ''}" aria-label="${isEdit ? 'Recorded patch' : event.sourceKind === 'request' ? 'Rejected request' : 'Recorded source'}">${sourceHTML(event,state.sourceTab)}</div><div class="source-provenance"><span title="${escapeHTML(source+(event.submitCommand ? '\n'+event.submitCommand : ''))}">${escapeHTML(source)}</span><button data-download>↓ Save</button></div>`;
  }

  function renderChart() {
    if (state.stage !== 'trajectory') return;
    const container = $('chartContainer');
    const width = container.clientWidth;
    if (width < 50) return;
    const mobile = width < 540;
    const margin = {left: mobile ? 36 : 45, right: mobile ? 17 : 22, top:36, bottom:60};
    const pw = width - margin.left - margin.right;
    const x = time => margin.left + chartScale.forward(time) / chartScale.length * pw;
    // As in cancer-dna, nearby events step down into the next available row.
    // Use the complete timeline so rows remain stable while seeking or playing.
    const laneEnd = [], fileLanes = new Map(), spacing = 12;
    for (const event of fileEvents) {
      const px = x(event.time);
      let lane = laneEnd.findIndex(end => px-end >= spacing);
      if (lane < 0) lane = laneEnd.length;
      laneEnd[lane] = px;
      fileLanes.set(event.id,lane);
    }
    const laneDepth = Math.max(0,laneEnd.length-1)*spacing;
    margin.bottom += laneDepth;
    // Reserve the score plot independently of the event rows below it.
    const minimumHeight = margin.top + 280 + margin.bottom;
    const minimumHeightPx = `${minimumHeight}px`;
    if (container.style.getPropertyValue('--chart-min-height') !== minimumHeightPx) {
      container.style.setProperty('--chart-min-height', minimumHeightPx);
    }
    const height = Math.max(minimumHeight, container.clientHeight);
    const ph = height - margin.top - margin.bottom, bottom = margin.top + ph;
    const y = score => margin.top + (CONFIG.score.maximum - score) / (CONFIG.score.maximum - CONFIG.score.minimum) * ph;
    const shownTime = state.mode === 'explore' ? HORIZON : state.actual;
    const visible = DATA.submissions.filter(s => s.time <= shownTime + 0.001);
    const selected = activeEvent();
    const boundary = DATA.explorationEnd === null ? null : x(DATA.explorationEnd);
    const f = n => Number(n.toFixed(2));
    const labelBoxes = [];
    positions = new Map();
    let html = `<defs><linearGradient id="scoreArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="var(--accent)" stop-opacity=".16"/><stop offset="100%" stop-color="var(--accent)" stop-opacity=".015"/></linearGradient></defs>`;
    if(boundary !== null) {
      const phaseLabelX = mobile ? width-margin.right : boundary+14;
      html += `<text x="${margin.left}" y="13" fill="#6b7280" font-size="8" letter-spacing="1.5">EXPLORATION</text><text x="${phaseLabelX}" y="13" text-anchor="${mobile ? 'end' : 'start'}" fill="#6b7280" font-size="8" letter-spacing="1.5">EXPLOITATION</text>`;
      html += `<path d="M${margin.left} 22H${boundary}" stroke="#9ca3af" stroke-width="2"/><path d="M${boundary} 22H${width-margin.right}" stroke="#d9dee7" stroke-width="2"/>`;
    } else html += `<text x="${margin.left}" y="13" fill="#6b7280" font-size="8" letter-spacing="1.5">RECORDED ITERATIONS</text>`;
    for(let i=0;i<CONFIG.score.ticks;i++) {
      const score=CONFIG.score.minimum+i*(CONFIG.score.maximum-CONFIG.score.minimum)/(CONFIG.score.ticks-1);
      html += `<line x1="${margin.left}" y1="${y(score)}" x2="${width-margin.right}" y2="${y(score)}" stroke="#f4f6f8" ${i ? 'stroke-dasharray="3 5"' : ''}/><text x="${margin.left-12}" y="${y(score)+3}" text-anchor="end" fill="#9ca3af" font-family="var(--mono)" font-size="9">${Number(score.toFixed(CONFIG.score.decimals))}</text>`;
    }
    // Display zero until a first submission arrives; no synthetic score records.
    let line = `M${f(x(0))} ${f(y(0))}`, best = 0;
    for (const sub of visible) {
      line += `H${f(x(sub.time))}`;
      if (sub.score > best) {best = sub.score; line += `V${f(y(best))}`;}
    }
    const mainEnd = Math.min(shownTime, DATA.duration);
    line += `H${f(x(mainEnd))}`;
    html += `<path d="${line}L${f(x(mainEnd))} ${bottom}H${f(x(0))}Z" fill="url(#scoreArea)" pointer-events="none"/><path id="mainScoreLine" d="${line}" stroke="var(--accent)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" fill="none" pointer-events="none"/>`;
    const baseline = DATA.baseline.points.filter(point => point.time <= shownTime + 0.001);
    let baselineValue = 0, baselinePath = `M${f(x(0))} ${f(y(0))}`;
    for (const point of baseline) {
      if (point.score > baselineValue) {baselinePath += `H${f(x(point.time))}V${f(y(point.score))}`; baselineValue = point.score;}
    }
    baselinePath += `H${f(x(Math.min(shownTime, DATA.baseline.duration)))}`;
    html += `<path id="baselineScoreLine" d="${baselinePath}" stroke="#9ca3af" stroke-width="2.2" stroke-dasharray="7 5" stroke-linecap="round" fill="none" pointer-events="none" aria-label="${escapeHTML(CONFIG.baselineLabel)} best score ${formatScore(baselineValue)}"/>`;
    const cursorX = x(state.actual);
    if (state.actual > 0 && (visible.length || state.mode === 'explore')) {
      html += `<line x1="${cursorX}" y1="30" x2="${cursorX}" y2="${bottom+40+laneDepth}" stroke="#6b7280" stroke-width="1" stroke-dasharray="3 4" pointer-events="none"/>`;
    }
    for (const sub of visible) {
      const px = x(sub.time), py = y(sub.score), isSelected = selected?.id === sub.id;
      positions.set(sub.id,{x:px,y:py});
      const radius = isSelected ? 6 : sub.improvement ? 4.8 : 3.7;
      const fill = sub.improvement ? 'var(--accent)' : '#ffffff';
      const stroke = sub.improvement ? 'var(--accent)' : '#6b7280';
      const label = `Submission ${sub.number}, ${sub.filename}, score ${formatScore(sub.score)}, at ${formatTime(sub.time)}. ${sub.algorithm.title}.`;
      html += `<g class="plot-point" data-event="${sub.id}" tabindex="0" role="button" aria-label="${escapeHTML(label)}"><circle class="hit-ring" cx="${px}" cy="${py}" r="10" fill="transparent" stroke="${isSelected ? '#9ca3af' : 'transparent'}"/><circle cx="${px}" cy="${py}" r="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="1.4"/><title>${escapeHTML(label)}</title></g>`;
      if (sub.improvement) {
        const label = formatScore(sub.score), labelWidth = label.length * 7;
        const offsets = sub.id === DATA.submissions[0]?.id ? [22,-15,-30,37] : [-15,22,-30,37,-45];
        let chosen;
        for (const dy of offsets) {
          for (const side of [1,-1]) {
            const left = side === 1 ? px+8 : px-8-labelWidth;
            const box = {left, right:left+labelWidth, top:py+dy-12, bottom:py+dy+2};
            const overlaps = other => box.left < other.right+3 && box.right > other.left-3 && box.top < other.bottom+2 && box.bottom > other.top-2;
            if (box.left < margin.left || box.right > width-margin.right || box.top < margin.top-15 || box.bottom > bottom+12) continue;
            if (labelBoxes.some(overlaps) || visible.some(point => overlaps({left:x(point.time)-6,right:x(point.time)+6,top:y(point.score)-6,bottom:y(point.score)+6}))) continue;
            chosen = {box, x:side === 1 ? left : box.right, y:py+dy, side};break;
          }
          if (chosen) break;
        }
        if (chosen) {
          labelBoxes.push(chosen.box);
          html += `<text x="${chosen.x}" y="${chosen.y}" text-anchor="${chosen.side === 1 ? 'start' : 'end'}" fill="#6b7280" stroke="#f4f6f8" stroke-width="3" paint-order="stroke" stroke-linejoin="round" font-size="10" font-family="var(--mono)" pointer-events="none">${label}</text>`;
        }
      }
    }
    html += `<text x="${margin.left}" y="${bottom+22}" fill="#9ca3af" font-size="7" letter-spacing="1.2">FILE EVENTS</text><text class="chart-axis-label" x="${margin.left+pw/2}" y="${bottom+22}" text-anchor="middle" fill="#6b7280" font-size="9">Time</text>`;
    for (const event of fileEvents) {
      if (event.time > shownTime + 0.001) continue;
      const px = x(event.time), py = bottom+36+fileLanes.get(event.id)*spacing, isSelected = selected?.id === event.id;
      positions.set(event.id,{x:px,y:py});
      const isWrite = event.kind === 'write', radius = isSelected ? 4.6 : 3.4;
      const label = `${event.kind} ${event.number}, ${event.filename}, ${event.outcome}, at ${formatTime(event.time)}.`;
      const fill = event.outcome!=='applied' ? '#ffffff' : isWrite ? '#9ca3af' : '#6b7280';
      const shape = isWrite ? `<rect class="write-dot" x="${px-radius}" y="${py-radius}" width="${radius*2}" height="${radius*2}" rx="1" fill="${fill}" stroke="#6b7280" stroke-width="1"/>` : `<circle class="edit-dot" cx="${px}" cy="${py}" r="${radius}" fill="${fill}" stroke="${event.failed ? '#6b7280' : '#6b7280'}" stroke-width="1"/>`;
      html += `<g class="plot-point ${event.kind}-point" data-event="${event.id}" tabindex="0" role="button" aria-label="${escapeHTML(label)}"><circle class="hit-ring" cx="${px}" cy="${py}" r="7.5" fill="transparent" stroke="${isSelected ? '#6b7280' : 'transparent'}"/>${shape}<title>${escapeHTML(label)}</title></g>`;
    }
    $('chart').setAttribute('viewBox',`0 0 ${width} ${height}`);
    const fontScale = width > 1050 ? 1.25 : 1.08;
    $('chart').innerHTML = html.replace(/font-size="(\d+)"/g,(_,size)=>`font-size="${Number(size)*fontScale}"`);
    $('chart').dataset.left = String(margin.left);
    $('chart').dataset.plotWidth = String(pw);
  }

  function renderNarrative() {
    let label='THE TASK',caption=CONFIG.captions.statement;
    if(state.stage==='approach') {label='THE METHOD';caption=CONFIG.captions.method;}
    if(state.stage==='trajectory') {
      const note=CONFIG.narrative.filter(item=>item.time<=state.actual).at(-1);
      const latest=latestSubmission(state.actual);
      label=note?.label || 'ITERATING';
      caption=note?.text || (latest ? `Submission ${latest.number}: ${formatScore(latest.score)} ${CONFIG.score.suffix}. Best so far: ${formatScore(bestAt(state.actual))}.` : 'The run begins. Follow edits and evaluated candidates as they appear.');
    }
    $('captionIndex').textContent=label;$('caption').textContent=caption;
  }
  function renderControls() {
    $('playButton').innerHTML = state.playing ? '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 4h4v12H5zm7 0h4v12h-4z"/></svg><span>Pause</span>' : `<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6 3.5L16 10 6 16.5Z"/></svg><span>${state.time >= DURATION ? 'Replay' : state.time > 0 ? 'Play' : 'Play demo'}</span>`;
    $('playButton').setAttribute('aria-label',state.playing ? 'Pause demo' : 'Play demo');
    for (const mode of ['playback','explore']) {
      $(mode+'Mode').classList.toggle('active',state.mode === mode);
      $(mode+'Mode').setAttribute('aria-pressed',String(state.mode === mode));
    }
    const isExplore = state.mode === 'explore';
    const max = isExplore ? chartScale.length : DURATION;
    const value = isExplore ? chartScale.forward(state.actual) : state.time;
    $('timeline').max = String(max);
    $('timeline').value = String(value);
    $('timeline').style.setProperty('--progress',`${value / max * 100}%`);
    $('timeline').setAttribute('aria-label',isExplore ? 'Elapsed run time' : 'Replay position');
    $('timeline').setAttribute('aria-valuetext',isExplore ? formatTime(state.actual) : formatReplay(value));
    $('timeLabel').innerHTML = isExplore ? `${formatTime(state.actual)} <span>/ ${formatTime(HORIZON)}</span>` : `${formatReplay(state.time)} <span>/ ${formatReplay(DURATION)}</span>`;
    $('elapsedLabel').textContent = isExplore ? 'Click a point to inspect code or patch' : state.stage === 'trajectory' ? `Run time ${formatTime(state.actual)} · condensed replay` : `${Math.round(DURATION / BASE_PLAYBACK_RATE)}-second condensed replay`;
  }
  function render() {
    const videoVisible = state.view === 'video';
    for (const stage of ['statement','approach','trajectory']) {
      const active = !videoVisible && state.stage === stage;
      $(stage+'Scene').hidden = !active;
      const nav = document.querySelector(`[data-stage="${stage}"]`);
      nav.classList.toggle('active',active);
      if (active) nav.setAttribute('aria-current','step'); else nav.removeAttribute('aria-current');
    }
    $('videoScene').hidden = !videoVisible;
    $('interactiveNav').hidden = videoVisible;
    for (const [id,active] of [['interactiveTab',!videoVisible],['videoTab',videoVisible]]) {
      $(id).classList.toggle('active',active);
      if (active) $(id).setAttribute('aria-current','page'); else $(id).removeAttribute('aria-current');
    }
    document.querySelector('.narrative-line').hidden = videoVisible;
    document.querySelector('.playback-bar').hidden = videoVisible;
    document.querySelector('.app').classList.toggle('is-video-view',videoVisible);
    if (reportedView !== state.view) {
      reportedView = state.view;
      parent.postMessage({type:'arex-demo:view',view:state.view},location.origin === 'null' ? '*' : location.origin);
    }
    if (videoVisible) return;
    const best = bestAt(state.actual);
    $('bestScore').textContent = formatScore(best);
    const baseline = baselineAt(state.actual);
    $('baselineScore').textContent = formatScore(baseline);
    document.querySelector('.score-display>span').textContent = state.mode === 'explore' ? 'BEST AT CURSOR' : 'BEST SCORE';
    $('phaseDescription').innerHTML = `<i class="live-dot"></i>${DATA.explorationEnd === null ? 'Recorded iterations' : state.actual < DATA.explorationEnd ? 'Exploration' : 'Exploitation'}`;
    renderNarrative(); renderControls(); renderChart(); renderInspector();
  }
  function setTime(seconds, manual = false) {
    if (state.view === 'video') {
      demoVideo.pause();
      state.view = 'replay';
    }
    state.time = clamp(seconds,0,DURATION);
    state.actual = actualAt(state.time);
    state.stage = stageAt(state.time);
    if (state.stage === 'statement') renderBoxExample(Math.min(2, Math.floor(state.time / (CONFIG.timing.methodStart / 3))));
    if (manual) {state.selected = null; state.sourceTab = 'code'; state.inspectorPanel = 'code';}
    render();
  }
  function stopPlayback() {
    embeddedPlaybackRequested = false;
    state.playing = false;
    cancelAnimationFrame(raf);
    raf = 0;
    renderControls();
  }
  // The parent page owns navigation; the embedded viewer never scrolls it.
  function updateFocusButton() {}
  function focusWorkspace() {}
  function tick(now) {
    if (!state.playing || capture || state.external) return;
    const dt = lastFrame ? (now-lastFrame)/1000 : 0;
    lastFrame = now;
    setTime(state.time + dt*BASE_PLAYBACK_RATE*state.speed);
    if (state.time >= DURATION) stopPlayback(); else raf=requestAnimationFrame(tick);
  }
  function play() {
    if (capture || state.view === 'video') return;
    if (state.playing) {stopPlayback();return;}
    state.external = false;
    if (state.mode === 'explore') {state.mode='playback';state.time=replayAt(state.actual);}
    if (state.time >= END) state.time=START;
    state.selected=null; state.sourceTab='code'; state.inspectorPanel='code';
    state.playing=true; lastFrame=0;
    setTime(state.time);
    focusWorkspace();
    raf=requestAnimationFrame(tick);
  }
  function setMode(mode) {
    stopPlayback();
    const wasIntro = state.stage !== 'trajectory';
    state.mode = mode;
    if (mode === 'explore') {
      state.stage='trajectory';
      if (wasIntro) state.actual=HORIZON;
      state.time=replayAt(state.actual);
    } else {
      state.time=replayAt(state.actual);
      state.stage='trajectory';
    }
    state.selected=null;state.sourceTab='code';
    render();
    focusWorkspace();
  }
  function seekActual(time) {
    stopPlayback();state.mode='explore';state.stage='trajectory';state.actual=clamp(time,0,HORIZON);
    state.time=replayAt(state.actual);state.selected=null;state.sourceTab='code';render();
  }
  function selectEvent(id) {
    const event = byId.get(id);
    if (!event) return;
    stopPlayback();state.mode='explore';state.stage='trajectory';state.selected=id;
    state.actual=event.time;state.time=replayAt(event.time);state.sourceTab='code';state.inspectorPanel='code';
    $('eventBrowser').hidden=true;$('chartTooltip').hidden=true;
    render();
  }
  function moveEvent(delta) {
    if (!events.length) return;
    const event=activeEvent();
    let index=event ? events.indexOf(event) : -1;
    index=clamp(index+delta,0,events.length-1);
    selectEvent(events[index].id);
  }
  function renderEventList() {
    const filtered=events.filter(e => (state.eventFilter==='all' || e.kind===state.eventFilter) && `${e.filename} ${eventTitle(e)} ${e.number} ${e.kind}`.toLowerCase().includes(state.query));
    $('eventList').innerHTML=filtered.map(event => `<button class="event-row ${event.kind} ${activeEvent()?.id === event.id ? 'selected' : ''}" data-list-event="${event.id}"><i></i><span><strong>${escapeHTML(eventLabel(event)+' · '+eventTitle(event))}</strong><small>${formatTime(event.time)} · ${escapeHTML(event.filename)}${event.failed ? ' · NOT APPLIED' : ''}</small></span><span>${event.kind==='submission' ? formatScore(event.score) : event.kind==='write' ? '■' : '↗'}</span></button>`).join('') || '<p class="no-results">No matching events.</p>';
    document.querySelectorAll('[data-filter]').forEach(button=>button.classList.toggle('active',button.dataset.filter===state.eventFilter));
  }
  function openEvents() {
    stopPlayback();
    if (state.mode!=='explore') setMode('explore');
    $('eventBrowser').hidden=false;
    renderEventList();
  }
  function showDialog(kicker,title,html) {
    stopPlayback();
    $('dialogKicker').textContent=kicker;$('dialogTitle').textContent=title;$('dialogBody').innerHTML=html;
    $('detailDialog').showModal();
    $('dialogBody').scrollTop=0;
  }
  function showStatement() {
    const html = DOMPurify.sanitize(marked.parse(DATA.displayStatement, {async: false, gfm: true}), {USE_PROFILES: {html: true}});
    showDialog('ALGORITHMIC TASK', CONFIG.taskTitle, `<article class="statement-full">${html}</article>`);
    const statement = $('dialogBody').querySelector('.statement-full');
    statement.querySelectorAll('a[href]').forEach(link => {
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
    });
    const mathOptions = {
      delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}],
      throwOnError: false,
      trust: false,
    };
    renderMathInElement(statement, mathOptions);
    // The archived input/output formats use fenced blocks containing only TeX.
    statement.querySelectorAll('pre > code').forEach(code => {
      if (/^(?:\s*\$[^$\n]+\$\s*)+$/.test(code.textContent)) {
        code.parentElement.classList.add('statement-format');
        renderMathInElement(code, mathOptions);
      }
    });
  }
  function expandSource() {
    const event=activeEvent();if(!event)return;
    showDialog(`${eventLabel(event)} · ${formatTime(event.time)}`,
      event.kind==='edit' ? `${event.filename} · ${event.failed ? 'requested patch' : 'patch'}` : `${event.filename} · ${eventTitle(event)}`,
      `<div class="source-view ${event.kind==='edit' && state.sourceTab==='code' ? 'patch-view' : ''}">${sourceHTML(event,state.sourceTab)}</div>`);
  }
  function notify(message) {
    $('toast').textContent=message;$('toast').hidden=false;
    clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,2200);
  }
  const sourceText = event => state.sourceTab==='feedback' ? event.feedback : event.kind==='edit' ? event.patch : event.sourceKind==='request' ? event.request : event.code;
  async function copySource() {
    const event=activeEvent();if(!event)return;
    const text=sourceText(event);
    try {
      if(navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
      else {const field=document.createElement('textarea');field.value=text;field.style.cssText='position:fixed;left:-9999px';document.body.append(field);field.select();const ok=document.execCommand('copy');field.remove();if(!ok)throw Error('Copy unavailable');}
      notify('Copied to clipboard');
    } catch {notify('Clipboard unavailable. Use Save to download the source.');}
  }
  function downloadSource() {
    const event=activeEvent();if(!event)return;
    const blob=new Blob([sourceText(event)],{type:'text/plain;charset=utf-8'});
    const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;
    a.download=state.sourceTab==='feedback' ? `${event.id}-feedback.txt` : event.kind==='edit' ? `${event.id}-${event.filename}.patch` : `${event.id}-${event.downloadFilename || event.filename}`;
    a.click();setTimeout(()=>URL.revokeObjectURL(url),500);
  }

  $('playButton').addEventListener('click',play);
  $('restartButton').addEventListener('click',()=>{stopPlayback();state.mode='playback';setTime(START,true);updateFocusButton();});
  $('interactiveTab').addEventListener('click',()=>{
    if (state.view === 'replay') return;
    demoVideo.pause();
    state.view='replay';
    render();
  });
  $('videoTab').addEventListener('click',()=>{
    stopPlayback();
    state.view='video';
    if (!demoVideo.getAttribute('src')) {
      demoVideo.poster=demoVideo.dataset.poster;
      demoVideo.preload='metadata';
      demoVideo.src=demoVideo.dataset.src;
      demoVideo.load();
    }
    render();
  });
  $('playbackMode').addEventListener('click',()=>setMode('playback'));
  $('exploreMode').addEventListener('click',()=>setMode('explore'));
  $('speedSelect').addEventListener('change',event=>{state.speed=Number(event.target.value);});
  $('timeline').addEventListener('input',event=>{const value=Number(event.target.value);stopPlayback();if(state.mode==='explore')seekActual(chartScale.inverse(value));else setTime(value,true);});
  document.querySelectorAll('[data-stage], [data-go]').forEach(button=>button.addEventListener('click',()=>{
    stopPlayback();state.mode='playback';setTime(stageTimes[button.dataset.stage||button.dataset.go],true);
    focusWorkspace();
  }));
  $('eventsButton').addEventListener('click',openEvents);
  $('closeEvents').addEventListener('click',()=>$('eventBrowser').hidden=true);
  $('eventSearch').addEventListener('input',event=>{state.query=event.target.value.trim().toLowerCase();renderEventList();});
  document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{state.eventFilter=button.dataset.filter;renderEventList();}));
  $('eventList').addEventListener('click',event=>{const target=event.target.closest('[data-list-event]');if(target)selectEvent(target.dataset.listEvent);});
  $('previousEvent').addEventListener('click',()=>moveEvent(-1));
  $('nextEvent').addEventListener('click',()=>moveEvent(1));
  $('expandButton').addEventListener('click',expandSource);
  $('inspectorContent').addEventListener('click',event=>{
    const history=event.target.closest('[data-history-event]');if(history){selectEvent(history.dataset.historyEvent);return;}
    const tab=event.target.closest('[data-source-tab]');
    if(tab){state.sourceTab=tab.dataset.sourceTab;renderInspector();}
    if(event.target.closest('[data-copy]'))copySource();
    if(event.target.closest('[data-download]'))downloadSource();
  });
  function chartPointer(event) {
    const pointer=$('chart').createSVGPoint();pointer.x=event.clientX;pointer.y=event.clientY;
    return pointer.matrixTransform($('chart').getScreenCTM().inverse());
  }
  function nearestEvent(point) {
    let found = null, distance = Infinity;
    for (const [id, pos] of positions) {
      const radius = ['edit','write'].includes(byId.get(id).kind) ? 7.5 : 10;
      const delta = (point.x-pos.x)**2+(point.y-pos.y)**2;
      if (delta <= radius**2 && delta < distance) {found=id;distance=delta;}
    }
    return found;
  }
  $('chart').addEventListener('click',event=>{
    const point=chartPointer(event), id=nearestEvent(point);
    if(id){selectEvent(id);return;}
    const fraction=(point.x-Number($('chart').dataset.left))/Number($('chart').dataset.plotWidth);
    seekActual(chartScale.inverse(fraction*chartScale.length));
  });
  $('chart').addEventListener('keydown',event=>{
    if(event.key==='Enter'||event.key===' '){const point=event.target.closest('[data-event]');if(point){event.preventDefault();event.stopPropagation();selectEvent(point.dataset.event);}}
  });
  $('chart').addEventListener('pointermove',event=>{
    const id=nearestEvent(chartPointer(event));
    if(!id){$('chartTooltip').hidden=true;return;}
    const item=byId.get(id), pos=positions.get(item.id);
    if(!pos)return;
    $('chartTooltip').innerHTML=item.kind==='submission'
      ? `<strong>#${String(item.number).padStart(2,'0')} · ${formatScore(item.score)} / 100</strong>${escapeHTML(item.filename)} · ${escapeHTML(item.algorithm.title)}<br><small>${formatTime(item.time)} · Click to inspect code</small>`
      : `<strong>${eventLabel(item)} · ${escapeHTML(item.filename)}</strong><small>${formatTime(item.time)} · ${item.failed ? 'Not applied' : item.kind==='write' ? 'Click to inspect full-file write' : 'Click to inspect patch'}</small>`;
    $('chartTooltip').hidden=false;
    const tw=$('chartTooltip').offsetWidth,th=$('chartTooltip').offsetHeight;
    $('chartTooltip').style.left=clamp(pos.x-tw/2,0,$('chartContainer').clientWidth-tw)+'px';
    $('chartTooltip').style.top=Math.max(0,pos.y-th-14)+'px';
  });
  $('chart').addEventListener('pointerleave',()=>$('chartTooltip').hidden=true);
  $('statementButton').addEventListener('click',showStatement);
  $('closeDialog').addEventListener('click',()=>$('detailDialog').close());
  $('detailDialog').addEventListener('click',event=>{if(event.target===$('detailDialog')){const r=event.target.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)event.target.close();}});
  document.addEventListener('keydown',event=>{
    if(state.view==='video' || $('detailDialog').open || /INPUT|SELECT|TEXTAREA|BUTTON/.test(event.target.tagName) || event.ctrlKey || event.metaKey || event.altKey)return;
    if(event.code==='Space'){event.preventDefault();play();}
    if(event.key==='ArrowLeft'||event.key==='ArrowRight'){
      event.preventDefault();const delta=event.key==='ArrowRight'?1:-1;
      if(state.mode==='explore')seekActual(chartScale.inverse(chartScale.forward(state.actual)+delta*chartScale.length/60));else {stopPlayback();setTime(state.time+delta*DURATION/36,true);}
    }
  });
  const resize=new ResizeObserver(()=>requestAnimationFrame(renderChart));resize.observe($('chartContainer'));
  const recordingTime = seconds => {
    const time = clamp(seconds,0,RECORDING.duration);
    if (time <= START) return time;
    if (time < RECORDING.trajectoryEnd) return START + (time-START)/(RECORDING.trajectoryEnd-START)*(END-START);
    return END + (time-RECORDING.trajectoryEnd)/(RECORDING.duration-RECORDING.trajectoryEnd)*(DURATION-END);
  };
  window.demoMeta={...CONFIG.capture,duration:capture ? RECORDING.duration : DURATION,previewTimes:capture ? RECORDING.previewTimes : [0,(CONFIG.timing.methodStart+START)/2,START+(END-START)*.22,START+(END-START)*.64,DURATION-1],mode:'recorded reconstruction',audio:false};
  window.setDemoTime=seconds=>{stopPlayback();clearTimeout(toastTimer);$('toast').hidden=true;state.external=true;state.mode='playback';state.selected=null;state.sourceTab='code';state.inspectorPanel='code';$('eventBrowser').hidden=true;$('chartTooltip').hidden=true;if($('detailDialog').open)$('detailDialog').close();const time=Number(seconds)||0;setTime(capture ? recordingTime(time) : time,true);if(state.time>=CONFIG.timing.methodStart)focusWorkspace();else updateFocusButton();};
  // Read-only state is useful for verification and deterministic embedding.
  window.replayState=()=>({time:state.time,actualTime:state.actual,mode:state.mode,stage:state.stage,view:state.view,playing:state.playing,speed:state.speed,selected:activeEvent()?.id||null,bestScore:bestAt(state.actual),inspectorPanel:state.inspectorPanel});
  const imagesReady=Promise.all([...document.images].map(img=>img.decode()));
  window.demoReady=Promise.all([document.fonts.ready,imagesReady]).then(()=>{render();updateFocusButton();return true;});
  let reportedHeight = 0;
  const reportSize = () => {
    const height = Math.ceil(document.querySelector('.app').getBoundingClientRect().height);
    if (height !== reportedHeight) {
      reportedHeight = height;
      parent.postMessage({type: 'arex-demo:size', height}, location.origin === 'null' ? '*' : location.origin);
    }
  };
  new ResizeObserver(reportSize).observe(document.querySelector('.app'));
  window.addEventListener('message', event => {
    if (event.source !== parent || event.origin !== location.origin) return;
    if (event.data?.type === 'arex-demo:pause') {stopPlayback();demoVideo.pause();}
    if (event.data?.type === 'arex-demo:play') {
      embeddedPlaybackRequested = true;
      window.demoReady.then(() => {
        if (embeddedPlaybackRequested && state.view==='replay' && !state.playing && !document.hidden && !$('detailDialog').open) play();
      });
    }
    if (event.data?.type === 'arex-demo:viewport' && Number.isFinite(event.data.height)) {
      document.documentElement.style.setProperty('--embedded-height', `${Math.max(580, event.data.height)}px`);
    }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) {stopPlayback();demoVideo.pause();} });
  window.demoReady.then(reportSize);
  render();
})();
