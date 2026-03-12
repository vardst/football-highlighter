/**
 * audio-analyzer.js — Web Audio API port of highlight_audio.py
 *
 * Computes RMS energy per second, applies median filtering,
 * thresholds by percentile, and merges nearby peaks into events.
 */

/**
 * Compute RMS energy per second from a video/audio file.
 * @param {File} file - Video or audio file
 * @param {function} onProgress - Progress callback (0-1)
 * @returns {Promise<{energy: Float32Array, sampleRate: number, duration: number}>}
 */
export async function computeEnergy(file, onProgress) {
    const audioCtx = new OfflineAudioContext(1, 1, 44100);

    onProgress?.(0.1);
    const arrayBuffer = await file.arrayBuffer();

    onProgress?.(0.3);
    const fullCtx = new OfflineAudioContext(1, 1, 44100);
    const audioBuffer = await fullCtx.decodeAudioData(arrayBuffer);

    const sampleRate = audioBuffer.sampleRate;
    const duration = audioBuffer.duration;
    const nSeconds = Math.floor(duration);
    const channelData = audioBuffer.getChannelData(0);

    onProgress?.(0.5);

    const energy = new Float32Array(nSeconds);
    for (let i = 0; i < nSeconds; i++) {
        const start = i * sampleRate;
        const end = Math.min((i + 1) * sampleRate, channelData.length);
        let sum = 0;
        for (let j = start; j < end; j++) {
            sum += channelData[j] * channelData[j];
        }
        energy[i] = Math.sqrt(sum / (end - start));

        if (i % 100 === 0) {
            onProgress?.(0.5 + 0.5 * (i / nSeconds));
        }
    }

    onProgress?.(1);
    return { energy, sampleRate, duration };
}

/**
 * Median filter (port of scipy.signal.medfilt).
 * @param {Float32Array} arr - Input array
 * @param {number} kernelSize - Must be odd
 * @returns {Float32Array}
 */
export function medianFilter(arr, kernelSize) {
    if (kernelSize % 2 === 0) kernelSize++;
    const half = Math.floor(kernelSize / 2);
    const result = new Float32Array(arr.length);
    const window = [];

    for (let i = 0; i < arr.length; i++) {
        window.length = 0;
        for (let j = i - half; j <= i + half; j++) {
            if (j >= 0 && j < arr.length) {
                window.push(arr[j]);
            }
        }
        window.sort((a, b) => a - b);
        result[i] = window[Math.floor(window.length / 2)];
    }

    return result;
}

/**
 * Compute a percentile value from an array.
 * @param {Float32Array} arr
 * @param {number} p - Percentile (0-100)
 * @returns {number}
 */
export function percentile(arr, p) {
    const sorted = Float32Array.from(arr).sort();
    const idx = (p / 100) * (sorted.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/**
 * Detect highlights from audio energy — full pipeline.
 * Port of detect_highlights_audio() from highlight_audio.py.
 *
 * @param {File} file - Video file
 * @param {object} options
 * @param {number} options.topPercent - Top N% loudest (default 10)
 * @param {number} options.windowSec - Median filter window (default 10)
 * @param {number} options.mergeGap - Merge events within N seconds (default 60)
 * @param {function} options.onProgress - Progress callback (0-1)
 * @returns {Promise<Array<{start: number, end: number}>>}
 */
export async function detectHighlights(file, {
    topPercent = 10,
    windowSec = 10,
    mergeGap = 60,
    onProgress,
} = {}) {
    // Step 1: Compute RMS energy per second
    const { energy, duration } = await computeEnergy(file, (p) => {
        onProgress?.(p * 0.6); // 60% of total progress
    });

    if (energy.length === 0) return [];

    // Step 2: Median filter smoothing
    const kernelSize = windowSec % 2 === 1 ? windowSec : windowSec + 1;
    const smoothed = medianFilter(energy, kernelSize);
    onProgress?.(0.7);

    // Step 3: Percentile threshold
    const threshold = percentile(smoothed, 100 - topPercent);
    onProgress?.(0.8);

    // Step 4: Find timestamps above threshold
    const peakTimes = [];
    for (let i = 0; i < smoothed.length; i++) {
        if (smoothed[i] >= threshold) {
            peakTimes.push(i);
        }
    }

    // Step 5: Merge nearby peaks
    const events = [];
    if (peakTimes.length > 0) {
        let currentStart = peakTimes[0];
        let currentEnd = peakTimes[0];

        for (let i = 1; i < peakTimes.length; i++) {
            if (peakTimes[i] - currentEnd <= mergeGap) {
                currentEnd = peakTimes[i];
            } else {
                events.push({
                    start: Math.max(0, currentStart - 5),
                    end: currentEnd + 10,
                });
                currentStart = peakTimes[i];
                currentEnd = peakTimes[i];
            }
        }
        events.push({
            start: Math.max(0, currentStart - 5),
            end: Math.min(currentEnd + 10, Math.floor(duration)),
        });
    }

    onProgress?.(1);
    return events;
}

/**
 * Find the loudest moment in a time window.
 * Port of find_peak_in_window() from highlight_audio.py.
 *
 * @param {Float32Array} energy - Pre-computed RMS energy per second
 * @param {number} centerSec - Center of search window
 * @param {number} windowSec - Search +- this many seconds
 * @returns {number} Peak timestamp in seconds
 */
export function findPeakInWindow(energy, centerSec, windowSec = 120) {
    const start = Math.max(0, Math.floor(centerSec - windowSec));
    const end = Math.min(energy.length, Math.ceil(centerSec + windowSec));

    if (start >= end) return centerSec;

    const slice = energy.slice(start, end);

    // Light median filter
    let kernelSize = Math.min(5, slice.length);
    if (kernelSize % 2 === 0) kernelSize = Math.max(1, kernelSize - 1);
    const smoothed = kernelSize >= 3 ? medianFilter(slice, kernelSize) : slice;

    let maxVal = -1;
    let maxIdx = 0;
    for (let i = 0; i < smoothed.length; i++) {
        if (smoothed[i] > maxVal) {
            maxVal = smoothed[i];
            maxIdx = i;
        }
    }

    return start + maxIdx;
}
