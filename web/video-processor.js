/**
 * video-processor.js — ffmpeg.wasm wrapper for video processing
 *
 * Handles: cutting clips, cropping to aspect ratio, score overlay,
 * and color grading. Uses ffmpeg.wasm 0.12.x from CDN.
 */

let ffmpeg = null;
let ffmpegLoaded = false;

const GRADE_FILTERS = {
    cinematic: "eq=contrast=1.1:brightness=0.02:saturation=1.2",
    dramatic: "eq=contrast=1.3:saturation=0.7",
    vibrant: "eq=contrast=1.05:saturation=1.5",
    none: "null",
};

/**
 * Load ffmpeg.wasm from CDN.
 * @param {function} onProgress - Loading progress callback
 */
export async function loadFFmpeg(onProgress) {
    if (ffmpegLoaded) return;

    onProgress?.("Loading ffmpeg.wasm...");

    const { FFmpeg } = await import(
        "https://unpkg.com/@ffmpeg/ffmpeg@0.12.10/dist/esm/index.js"
    );
    const { toBlobURL } = await import(
        "https://unpkg.com/@ffmpeg/util@0.12.1/dist/esm/index.js"
    );

    ffmpeg = new FFmpeg();

    const baseURL = "https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm";
    await ffmpeg.load({
        coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, "text/javascript"),
        wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, "application/wasm"),
    });

    ffmpegLoaded = true;
    onProgress?.("ffmpeg.wasm loaded");
}

/**
 * Write a File/Blob to ffmpeg's virtual filesystem.
 * @param {string} name - Filename in virtual FS
 * @param {File|Blob} file - File to write
 */
async function writeFile(name, file) {
    const data = new Uint8Array(await file.arrayBuffer());
    await ffmpeg.writeFile(name, data);
}

/**
 * Read a file from ffmpeg's virtual filesystem as a Blob.
 * @param {string} name - Filename to read
 * @param {string} mimeType - MIME type for the blob
 * @returns {Promise<Blob>}
 */
async function readFile(name, mimeType = "video/mp4") {
    const data = await ffmpeg.readFile(name);
    return new Blob([data.buffer], { type: mimeType });
}

/**
 * Delete a file from ffmpeg's virtual FS (ignore errors).
 */
async function deleteFile(name) {
    try { await ffmpeg.deleteFile(name); } catch {}
}

/**
 * Cut a clip from the source video.
 * @param {string} inputName - Input filename in virtual FS
 * @param {number} start - Start time in seconds
 * @param {number} duration - Duration in seconds
 * @param {string} outputName - Output filename
 */
export async function cutClip(inputName, start, duration, outputName) {
    await ffmpeg.exec([
        "-ss", String(start),
        "-i", inputName,
        "-t", String(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        outputName,
    ]);
}

/**
 * Center-crop a video to the target aspect ratio.
 * @param {string} inputName - Input filename
 * @param {string} outputName - Output filename
 * @param {string} aspect - "9:16" or "1:1"
 */
export async function cropVideo(inputName, outputName, aspect = "9:16") {
    let vf;
    if (aspect === "1:1") {
        vf = "crop=in_h:in_h,scale=1080:1080";
    } else {
        vf = "crop=in_h*9/16:in_h,scale=1080:1920";
    }

    await ffmpeg.exec([
        "-i", inputName,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast",
        "-c:a", "aac", "-b:a", "128k",
        outputName,
    ]);
}

/**
 * Apply score overlay + color grade in one pass.
 * @param {string} inputName - Input filename
 * @param {string} outputName - Output filename
 * @param {object} goalInfo - Goal info with score_before, score_after, etc.
 * @param {number} clipDuration - Clip duration in seconds
 * @param {string} colorGrade - Grade preset name
 */
export async function applyOverlayAndGrade(
    inputName, outputName, goalInfo, clipDuration, colorGrade = "cinematic"
) {
    const filters = buildScoreFilter(goalInfo, clipDuration);
    const gradeFilter = GRADE_FILTERS[colorGrade] || "null";

    let vf;
    if (filters && gradeFilter !== "null") {
        vf = `${filters},${gradeFilter}`;
    } else if (filters) {
        vf = filters;
    } else if (gradeFilter !== "null") {
        vf = gradeFilter;
    } else {
        vf = "null";
    }

    await ffmpeg.exec([
        "-i", inputName,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast",
        "-c:a", "copy",
        outputName,
    ]);
}

/**
 * Apply only color grading (no score overlay).
 * @param {string} inputName
 * @param {string} outputName
 * @param {string} colorGrade
 */
export async function applyGrade(inputName, outputName, colorGrade = "cinematic") {
    const vf = GRADE_FILTERS[colorGrade] || "null";

    await ffmpeg.exec([
        "-i", inputName,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast",
        "-c:a", "copy",
        outputName,
    ]);
}

/**
 * Concatenate multiple clips into one.
 * @param {string[]} inputNames - Array of filenames in virtual FS
 * @param {string} outputName - Output filename
 */
export async function concatClips(inputNames, outputName) {
    const listContent = inputNames.map(n => `file '${n}'`).join("\n");
    await ffmpeg.writeFile("concat_list.txt", new TextEncoder().encode(listContent));

    await ffmpeg.exec([
        "-f", "concat", "-safe", "0",
        "-i", "concat_list.txt",
        "-c", "copy",
        outputName,
    ]);

    await deleteFile("concat_list.txt");
}

/**
 * Build drawtext filter for score overlay.
 * Port of build_score_filter() from score_overlay.py.
 * Uses built-in sans font since custom fonts aren't available in wasm.
 */
function buildScoreFilter(goalInfo, clipDuration) {
    if (!goalInfo) return null;

    const scoreBefore = goalInfo.score_before;
    const scoreAfter = goalInfo.score_after;
    const home = goalInfo.home || "HOME";
    const away = goalInfo.away || "AWAY";
    const peakOffset = goalInfo.peak_sec - goalInfo.clip_start;

    const textBefore = escapeDrawtext(`${home}  ${scoreBefore}  ${away}`);
    const textAfter = escapeDrawtext(`${home}  ${scoreAfter}  ${away}`);
    const scorerText = escapeDrawtext(`GOAL! ${goalInfo.scorer}`);
    const flashEnd = Math.min(peakOffset + 5, clipDuration);

    const filters = [
        `drawtext=text='${textBefore}'`
        + `:fontcolor=white:fontsize=42`
        + `:box=1:boxcolor=black@0.6:boxborderw=12`
        + `:x=(w-text_w)/2:y=60`
        + `:enable='between(t,0,${peakOffset.toFixed(2)})'`,

        `drawtext=text='${textAfter}'`
        + `:fontcolor=white:fontsize=42`
        + `:box=1:boxcolor=black@0.6:boxborderw=12`
        + `:x=(w-text_w)/2:y=60`
        + `:enable='between(t,${peakOffset.toFixed(2)},${clipDuration.toFixed(2)})'`,

        `drawtext=text='${scorerText}'`
        + `:fontcolor=yellow:fontsize=36`
        + `:box=1:boxcolor=black@0.5:boxborderw=8`
        + `:x=(w-text_w)/2:y=120`
        + `:enable='between(t,${peakOffset.toFixed(2)},${flashEnd.toFixed(2)})'`,
    ];

    return filters.join(",");
}

function escapeDrawtext(text) {
    return text
        .replace(/\\/g, "\\\\")
        .replace(/'/g, "\\'")
        .replace(/:/g, "\\:")
        .replace(/%/g, "%%");
}

/**
 * Process a single goal clip end-to-end.
 * @param {File} videoFile - Source video file
 * @param {object} goalWindow - Goal window with clip_start, clip_end, etc.
 * @param {object} options - { aspect, colorGrade, home, away }
 * @param {function} onProgress - Status callback
 * @returns {Promise<{blob: Blob, filename: string}>}
 */
export async function processGoalClip(videoFile, goalWindow, options, onProgress) {
    const { aspect = "9:16", colorGrade = "cinematic" } = options;
    const clipDuration = goalWindow.clip_end - goalWindow.clip_start;
    const goalNum = goalWindow.goal_num;
    const scorer = goalWindow.scorer.toLowerCase().replace(/\s+/g, "_");
    const filename = `goal_${goalNum}_${goalWindow.score_after}_${scorer}.mp4`;

    onProgress?.(`Goal ${goalNum}: Writing source video...`);
    await writeFile("source.mp4", videoFile);

    onProgress?.(`Goal ${goalNum}: Cutting clip...`);
    await cutClip("source.mp4", goalWindow.clip_start, clipDuration, "raw.mp4");

    onProgress?.(`Goal ${goalNum}: Cropping to ${aspect}...`);
    await cropVideo("raw.mp4", "cropped.mp4", aspect);

    onProgress?.(`Goal ${goalNum}: Applying overlay + grade...`);
    const goalInfo = {
        ...goalWindow,
        home: options.home || "HOME",
        away: options.away || "AWAY",
    };
    await applyOverlayAndGrade("cropped.mp4", "final.mp4", goalInfo, clipDuration, colorGrade);

    onProgress?.(`Goal ${goalNum}: Reading output...`);
    const blob = await readFile("final.mp4");

    // Cleanup
    for (const f of ["source.mp4", "raw.mp4", "cropped.mp4", "final.mp4"]) {
        await deleteFile(f);
    }

    return { blob, filename };
}

/**
 * Process a highlight clip (no score overlay).
 * @param {File} videoFile - Source video file
 * @param {object} event - { start, end }
 * @param {number} index - Clip index
 * @param {object} options - { aspect, colorGrade }
 * @param {function} onProgress - Status callback
 * @returns {Promise<{blob: Blob, filename: string}>}
 */
export async function processHighlightClip(videoFile, event, index, options, onProgress) {
    const { aspect = "9:16", colorGrade = "cinematic" } = options;
    const duration = event.end - event.start;
    const filename = `highlight_${String(index + 1).padStart(3, "0")}.mp4`;

    onProgress?.(`Clip ${index + 1}: Writing source...`);
    await writeFile("source.mp4", videoFile);

    onProgress?.(`Clip ${index + 1}: Cutting...`);
    await cutClip("source.mp4", event.start, duration, "raw.mp4");

    onProgress?.(`Clip ${index + 1}: Cropping to ${aspect}...`);
    await cropVideo("raw.mp4", "cropped.mp4", aspect);

    onProgress?.(`Clip ${index + 1}: Color grading...`);
    await applyGrade("cropped.mp4", "final.mp4", colorGrade);

    onProgress?.(`Clip ${index + 1}: Reading output...`);
    const blob = await readFile("final.mp4");

    for (const f of ["source.mp4", "raw.mp4", "cropped.mp4", "final.mp4"]) {
        await deleteFile(f);
    }

    return { blob, filename };
}
