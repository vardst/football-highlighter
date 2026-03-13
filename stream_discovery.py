"""
stream_discovery.py — Discover sports/football streams from IPTV M3U playlists

Fetches public IPTV playlists, parses M3U format, filters for sports/football
channels, caches results, and probes stream availability.
"""

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import requests


DEFAULT_SOURCES = [
    "https://iptv-org.github.io/iptv/categories/sports.m3u",
    "https://raw.githubusercontent.com/Free-TV/IPTV/master/playlist.m3u8",
]

SPORTS_KEYWORDS = [
    "sport", "football", "soccer", "futbol", "fútbol", "futebol", "calcio",
    "espn", "bein", "sky sport", "dazn", "premier league", "la liga",
    "serie a", "bundesliga", "ligue 1", "champions league", "uefa",
    "fifa", "fox sport", "bt sport", "supersport", "eleven sport",
    "arena sport", "match tv", "movistar", "canal+",
]

CACHE_DIR = ".stream_cache"
DEFAULT_CACHE_MAX_AGE_HOURS = 6


@dataclass
class Stream:
    """A single IPTV stream entry."""
    name: str
    url: str
    group: str = ""
    tvg_id: str = ""
    tvg_name: str = ""
    tvg_logo: str = ""
    tvg_country: str = ""
    quality: str = ""
    is_acestream: bool = False
    status: str = "unknown"  # unknown, online, offline, timeout

    def __post_init__(self):
        if self.url.startswith("acestream://"):
            self.is_acestream = True
        self._infer_quality()

    def _infer_quality(self):
        """Infer stream quality from name/URL hints."""
        if self.quality:
            return
        lower = self.name.lower() + " " + self.url.lower()
        if any(q in lower for q in ["4k", "uhd", "2160"]):
            self.quality = "4K"
        elif any(q in lower for q in ["fhd", "1080", "full hd"]):
            self.quality = "FHD"
        elif any(q in lower for q in ["hd", "720"]):
            self.quality = "HD"
        elif any(q in lower for q in ["sd", "480", "360"]):
            self.quality = "SD"


def fetch_all_streams(
    sources=None,
    cache_dir=CACHE_DIR,
    max_age_hours=DEFAULT_CACHE_MAX_AGE_HOURS,
    filter_sports=True,
):
    """
    Fetch and parse streams from all IPTV sources.

    Args:
        sources: list of M3U playlist URLs (defaults to DEFAULT_SOURCES)
        cache_dir: directory for file-based cache
        max_age_hours: cache TTL in hours
        filter_sports: if True, only return sports/football streams

    Returns:
        list[Stream]: deduplicated, optionally filtered streams
    """
    if sources is None:
        sources = DEFAULT_SOURCES

    os.makedirs(cache_dir, exist_ok=True)
    all_streams = []

    for source_url in sources:
        content = _fetch_with_cache(source_url, cache_dir, max_age_hours)
        if content:
            streams = parse_m3u(content)
            all_streams.extend(streams)

    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for s in all_streams:
        if s.url not in seen_urls:
            seen_urls.add(s.url)
            unique.append(s)

    if filter_sports:
        unique = _filter_sports(unique)

    return unique


def parse_m3u(content):
    """
    Parse M3U/M3U8 playlist content into Stream objects.

    Handles the #EXTINF line format:
      #EXTINF:-1 tvg-id="..." tvg-name="..." tvg-logo="..." group-title="...", Channel Name
      http://...

    Args:
        content: raw M3U playlist text

    Returns:
        list[Stream]
    """
    streams = []
    lines = content.strip().split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("#EXTINF"):
            # Parse attributes from the EXTINF line
            attrs = _parse_extinf_attrs(line)

            # Channel name is after the last comma in the EXTINF line
            name = ""
            comma_idx = line.rfind(",")
            if comma_idx != -1:
                name = line[comma_idx + 1:].strip()

            # Next non-empty, non-comment line is the URL
            i += 1
            url = ""
            while i < len(lines):
                candidate = lines[i].strip()
                if candidate and not candidate.startswith("#"):
                    url = candidate
                    break
                i += 1

            if url:
                stream = Stream(
                    name=name or attrs.get("tvg-name", "Unknown"),
                    url=_normalize_acestream(url),
                    group=attrs.get("group-title", ""),
                    tvg_id=attrs.get("tvg-id", ""),
                    tvg_name=attrs.get("tvg-name", ""),
                    tvg_logo=attrs.get("tvg-logo", ""),
                    tvg_country=attrs.get("tvg-country", ""),
                )
                streams.append(stream)

        i += 1

    return streams


def probe_streams_batch(streams, max_workers=10, timeout=5):
    """
    Probe stream URLs concurrently to check availability.

    Updates stream.status in-place to 'online', 'offline', or 'timeout'.

    Args:
        streams: list of Stream objects
        max_workers: concurrent probe threads
        timeout: HTTP timeout per probe in seconds
    """
    def _probe_one(stream):
        try:
            resp = requests.head(
                stream.url, timeout=timeout, allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            if resp.status_code < 400:
                stream.status = "online"
            else:
                stream.status = "offline"
        except requests.Timeout:
            stream.status = "timeout"
        except requests.RequestException:
            stream.status = "offline"
        return stream

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_probe_one, s): s for s in streams}
        for future in as_completed(futures):
            future.result()


def _fetch_with_cache(url, cache_dir, max_age_hours):
    """
    Fetch URL content with file-based caching.

    Falls back to stale cache on network failure.

    Returns:
        str or None
    """
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{url_hash}.m3u")
    meta_file = os.path.join(cache_dir, f"{url_hash}.meta")

    # Check cache freshness
    if os.path.isfile(cache_file) and os.path.isfile(meta_file):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            cached_time = meta.get("fetched_at", 0)
            age_hours = (time.time() - cached_time) / 3600
            if age_hours < max_age_hours:
                with open(cache_file, encoding="utf-8", errors="replace") as f:
                    return f.read()
        except (json.JSONDecodeError, OSError):
            pass

    # Fetch fresh
    try:
        resp = requests.get(
            url, timeout=30,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        content = resp.text

        # Write cache
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(content)
        with open(meta_file, "w") as f:
            json.dump({"url": url, "fetched_at": time.time()}, f)

        return content

    except requests.RequestException as e:
        # Fall back to stale cache
        if os.path.isfile(cache_file):
            print(f"[discovery] Network error for {url}, using stale cache: {e}")
            with open(cache_file, encoding="utf-8", errors="replace") as f:
                return f.read()
        print(f"[discovery] Failed to fetch {url}: {e}")
        return None


def _parse_extinf_attrs(line):
    """
    Extract key="value" attributes from an #EXTINF line.

    Returns:
        dict of attribute name -> value
    """
    attrs = {}
    for match in re.finditer(r'([\w-]+)="([^"]*)"', line):
        attrs[match.group(1)] = match.group(2)
    return attrs


def _filter_sports(streams):
    """Filter streams to only sports/football-related channels."""
    filtered = []
    for s in streams:
        text = (s.name + " " + s.group + " " + s.tvg_name).lower()
        if any(kw in text for kw in SPORTS_KEYWORDS):
            filtered.append(s)
    return filtered


def _normalize_acestream(url):
    """Convert acestream:// URL to HTTP API URL for local Ace Stream engine."""
    if url.startswith("acestream://"):
        content_id = url.replace("acestream://", "")
        return f"http://127.0.0.1:6878/ace/getstream?id={content_id}"
    return url


