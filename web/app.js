/**
 * app.js — UI logic, state management, pipeline orchestration
 */

import { detectHighlights, computeEnergy, findPeakInWindow } from "./audio-analyzer.js";
import {
    loadFFmpeg,
    processGoalClip,
    processHighlightClip,
} from "./video-processor.js";

// ── State ────────────────────────────────────────────────────────────────

const state = {
    file: null,
    mode: "goals",
    processing: false,
    results: [],
};

// ── DOM refs ─────────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dropZone = $("#drop-zone");
const fileInput = $("#file-input");
const fileInfo = $("#file-info");
const fileName = $("#file-name");
const clearFileBtn = $("#clear-file");
const modeSection = $("#mode-section");
const goalsConfig = $("#goals-config");
const highlightsConfig = $("#highlights-config");
const goalsList = $("#goals-list");
const addGoalBtn = $("#add-goal");
const importConfigBtn = $("#import-config");
const startBtn = $("#start-btn");
const progressSection = $("#progress-section");
const progressSteps = $("#progress-steps");
const progressBar = $("#progress-bar");
const progressStatus = $("#progress-status");
const resultsSection = $("#results-section");
const resultsList = $("#results-list");
const topPercentSlider = $("#top-percent");
const mergeGapSlider = $("#merge-gap");
const maxDurationSlider = $("#max-duration");

// ── File upload ──────────────────────────────────────────────────────────

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("video/")) {
        setFile(file);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
});

clearFileBtn.addEventListener("click", () => {
    state.file = null;
    fileInput.value = "";
    dropZone.classList.remove("hidden");
    fileInfo.classList.add("hidden");
    modeSection.classList.add("hidden");
    progressSection.classList.add("hidden");
    resultsSection.classList.add("hidden");
});

function setFile(file) {
    state.file = file;
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    fileName.textContent = `${file.name} (${sizeMB} MB)`;
    dropZone.classList.add("hidden");
    fileInfo.classList.remove("hidden");
    modeSection.classList.remove("hidden");
}

// ── Mode tabs ────────────────────────────────────────────────────────────

$$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
        $$(".tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        state.mode = tab.dataset.mode;

        if (state.mode === "goals") {
            goalsConfig.classList.remove("hidden");
            highlightsConfig.classList.add("hidden");
        } else {
            goalsConfig.classList.add("hidden");
            highlightsConfig.classList.remove("hidden");
        }
    });
});

// ── Slider labels ────────────────────────────────────────────────────────

topPercentSlider.addEventListener("input", () => {
    $("#top-percent-val").textContent = `${topPercentSlider.value}%`;
});
mergeGapSlider.addEventListener("input", () => {
    $("#merge-gap-val").textContent = `${mergeGapSlider.value}s`;
});
maxDurationSlider.addEventListener("input", () => {
    $("#max-duration-val").textContent = `${maxDurationSlider.value}min`;
});

// ── Goals list ───────────────────────────────────────────────────────────

let goalCounter = 0;

function addGoalEntry(goal = null) {
    goalCounter++;
    const div = document.createElement("div");
    div.className = "goal-entry";
    div.dataset.id = goalCounter;
    div.innerHTML = `
        <div>
            <label>Min</label>
            <input type="number" class="goal-minute" value="${goal?.minute || ""}" min="1" max="120" placeholder="45">
        </div>
        <div>
            <label>Scorer</label>
            <input type="text" class="goal-scorer" value="${goal?.scorer || ""}" placeholder="Player name">
        </div>
        <div>
            <label>Team</label>
            <select class="goal-team">
                <option value="home" ${goal?.team === "home" ? "selected" : ""}>Home</option>
                <option value="away" ${goal?.team === "away" ? "selected" : ""}>Away</option>
            </select>
        </div>
        <div>
            <label>Score</label>
            <input type="text" class="goal-score" value="${goal?.score_after || ""}" placeholder="1-0">
        </div>
        <button class="remove-goal" title="Remove">&times;</button>
    `;

    div.querySelector(".remove-goal").addEventListener("click", () => div.remove());
    goalsList.appendChild(div);
}

addGoalBtn.addEventListener("click", () => addGoalEntry());

// Start with one empty goal row
addGoalEntry();

// ── Config import ────────────────────────────────────────────────────────

importConfigBtn.addEventListener("click", () => {
    try {
        const config = JSON.parse($("#config-json").value);
        if (config.match) {
            $("#home-team").value = config.match.home || "Home";
            $("#away-team").value = config.match.away || "Away";
        }
        if (config.kickoff_offsets) {
            $("#kickoff-1st").value = config.kickoff_offsets.first_half || 0;
            $("#kickoff-2nd").value = config.kickoff_offsets.second_half || 2700;
        }
        if (config.goals) {
            goalsList.innerHTML = "";
            config.goals.forEach((g) => addGoalEntry(g));
        }
    } catch (e) {
        alert("Invalid JSON: " + e.message);
    }
});

// ── Collect config from UI ───────────────────────────────────────────────

function collectGoalsConfig() {
    const goals = [];
    goalsList.querySelectorAll(".goal-entry").forEach((entry) => {
        const minute = parseInt(entry.querySelector(".goal-minute").value);
        const scorer = entry.querySelector(".goal-scorer").value.trim();
        const team = entry.querySelector(".goal-team").value;
        const scoreAfter = entry.querySelector(".goal-score").value.trim();
        if (minute && scorer && scoreAfter) {
            goals.push({ minute, scorer, team, score_after: scoreAfter });
        }
    });

    return {
        home: $("#home-team").value.trim() || "Home",
        away: $("#away-team").value.trim() || "Away",
        kickoff1st: parseInt($("#kickoff-1st").value) || 0,
        kickoff2nd: parseInt($("#kickoff-2nd").value) || 2700,
        goals,
    };
}

// ── Goal window computation (port of goal_detector.py) ───────────────────

function matchMinuteToVideoSec(minute, kickoff1st, kickoff2nd) {
    if (minute <= 45) {
        return kickoff1st + minute * 60;
    }
    return kickoff2nd + (minute - 45) * 60;
}

function computeGoalWindows(config, energy) {
    const windows = [];
    let prevScore = "0-0";

    for (let i = 0; i < config.goals.length; i++) {
        const goal = config.goals[i];
        const expectedSec = matchMinuteToVideoSec(
            goal.minute, config.kickoff1st, config.kickoff2nd
        );

        // Refine with audio peak if we have energy data
        let peakSec = expectedSec;
        if (energy) {
            peakSec = findPeakInWindow(energy, expectedSec, 120);
        }

        const clipStart = Math.max(0, peakSec - 15);
        const clipEnd = peakSec + 90;

        windows.push({
            goal_num: i + 1,
            scorer: goal.scorer,
            team: goal.team,
            score_before: prevScore,
            score_after: goal.score_after,
            minute: goal.minute,
            peak_sec: peakSec,
            clip_start: clipStart,
            clip_end: clipEnd,
        });

        prevScore = goal.score_after;
    }

    return windows;
}

// ── Progress UI ──────────────────────────────────────────────────────────

function showProgress(steps) {
    progressSection.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    resultsList.innerHTML = "";
    progressSteps.innerHTML = "";

    steps.forEach((label, i) => {
        const div = document.createElement("div");
        div.className = "step";
        div.id = `step-${i}`;
        div.innerHTML = `<span class="step-icon">&#9711;</span><span>${label}</span>`;
        progressSteps.appendChild(div);
    });

    progressBar.style.width = "0%";
    progressStatus.textContent = "Starting...";
}

function updateStep(index, status) {
    const step = $(`#step-${index}`);
    if (!step) return;

    step.className = `step ${status}`;
    const icon = step.querySelector(".step-icon");
    if (status === "active") icon.innerHTML = "&#9203;"; // spinner
    else if (status === "done") icon.innerHTML = "&#10004;";
    else if (status === "error") icon.innerHTML = "&#10008;";
}

function setProgress(pct, statusText) {
    progressBar.style.width = `${pct}%`;
    if (statusText) progressStatus.textContent = statusText;
}

// ── Result display ───────────────────────────────────────────────────────

function addResult(blob, filename, meta = "") {
    const url = URL.createObjectURL(blob);
    const item = document.createElement("div");
    item.className = "result-item";
    const sizeMB = (blob.size / 1024 / 1024).toFixed(1);

    item.innerHTML = `
        <h3>${filename}</h3>
        ${meta ? `<p class="result-meta">${meta}</p>` : ""}
        <video src="${url}" controls playsinline></video>
        <p class="result-meta">${sizeMB} MB</p>
        <a href="${url}" download="${filename}" class="download-btn">Download</a>
    `;

    resultsList.appendChild(item);
    resultsSection.classList.remove("hidden");
    state.results.push({ blob, filename, url });
}

// ── Pipeline execution ───────────────────────────────────────────────────

startBtn.addEventListener("click", async () => {
    if (state.processing || !state.file) return;
    state.processing = true;
    state.results = [];
    startBtn.disabled = true;

    const aspect = $("#aspect-ratio").value;
    const colorGrade = $("#color-grade").value;

    try {
        if (state.mode === "goals") {
            await runGoalsPipeline(aspect, colorGrade);
        } else {
            await runHighlightsPipeline(aspect, colorGrade);
        }
    } catch (err) {
        progressStatus.textContent = `Error: ${err.message}`;
        console.error(err);
    } finally {
        state.processing = false;
        startBtn.disabled = false;
    }
});

async function runGoalsPipeline(aspect, colorGrade) {
    const config = collectGoalsConfig();
    if (config.goals.length === 0) {
        alert("Add at least one goal.");
        return;
    }

    const steps = [
        "Load ffmpeg.wasm",
        "Analyse audio for peak detection",
        ...config.goals.map((g, i) => `Process goal ${i + 1}: ${g.scorer} (${g.minute}')`),
    ];
    showProgress(steps);

    // Step 1: Load ffmpeg
    updateStep(0, "active");
    await loadFFmpeg((msg) => setProgress(5, msg));
    updateStep(0, "done");
    setProgress(10);

    // Step 2: Audio analysis for peak refinement
    updateStep(1, "active");
    setProgress(12, "Decoding audio...");
    const { energy } = await computeEnergy(state.file, (p) => {
        setProgress(10 + p * 15, "Analysing audio...");
    });
    updateStep(1, "done");
    setProgress(25);

    // Compute goal windows with audio refinement
    const goalWindows = computeGoalWindows(config, energy);

    // Step 3+: Process each goal
    const goalPct = 75 / goalWindows.length;
    for (let i = 0; i < goalWindows.length; i++) {
        const stepIdx = i + 2;
        updateStep(stepIdx, "active");

        try {
            const result = await processGoalClip(
                state.file,
                goalWindows[i],
                { aspect, colorGrade, home: config.home, away: config.away },
                (msg) => setProgress(25 + i * goalPct + goalPct * 0.5, msg)
            );

            const gw = goalWindows[i];
            const meta = `${gw.scorer} (${gw.minute}') | ${gw.score_before} → ${gw.score_after}`;
            addResult(result.blob, result.filename, meta);
            updateStep(stepIdx, "done");
        } catch (err) {
            updateStep(stepIdx, "error");
            console.error(`Goal ${i + 1} failed:`, err);
        }

        setProgress(25 + (i + 1) * goalPct);
    }

    setProgress(100, `Done! ${state.results.length} goal clips created.`);
}

async function runHighlightsPipeline(aspect, colorGrade) {
    const topPercent = parseInt(topPercentSlider.value);
    const mergeGap = parseInt(mergeGapSlider.value);
    const maxDuration = parseInt(maxDurationSlider.value) * 60;

    const steps = [
        "Load ffmpeg.wasm",
        "Detect highlights (audio analysis)",
        "Process clips",
    ];
    showProgress(steps);

    // Step 1: Load ffmpeg
    updateStep(0, "active");
    await loadFFmpeg((msg) => setProgress(5, msg));
    updateStep(0, "done");
    setProgress(10);

    // Step 2: Detect highlights
    updateStep(1, "active");
    let events = await detectHighlights(state.file, {
        topPercent,
        mergeGap,
        onProgress: (p) => setProgress(10 + p * 30, "Analysing audio..."),
    });
    updateStep(1, "done");

    if (events.length === 0) {
        setProgress(100, "No highlights detected. Try increasing sensitivity.");
        return;
    }

    // Trim to max duration
    let totalDur = events.reduce((s, e) => s + (e.end - e.start), 0);
    if (totalDur > maxDuration) {
        const trimmed = [];
        let running = 0;
        for (const ev of events) {
            const dur = ev.end - ev.start;
            if (running + dur <= maxDuration) {
                trimmed.push(ev);
                running += dur;
            }
        }
        events = trimmed;
    }

    setProgress(40, `Found ${events.length} highlight events`);

    // Step 3: Process each clip
    updateStep(2, "active");
    const clipPct = 60 / events.length;

    for (let i = 0; i < events.length; i++) {
        try {
            const result = await processHighlightClip(
                state.file,
                events[i],
                i,
                { aspect, colorGrade },
                (msg) => setProgress(40 + i * clipPct + clipPct * 0.5, msg)
            );

            const startMin = Math.floor(events[i].start / 60);
            const startSec = Math.floor(events[i].start % 60);
            const dur = Math.floor(events[i].end - events[i].start);
            const meta = `${startMin}:${String(startSec).padStart(2, "0")} | ${dur}s`;
            addResult(result.blob, result.filename, meta);
        } catch (err) {
            console.error(`Clip ${i + 1} failed:`, err);
        }

        setProgress(40 + (i + 1) * clipPct);
    }

    updateStep(2, "done");
    setProgress(100, `Done! ${state.results.length} highlight clips created.`);
}
