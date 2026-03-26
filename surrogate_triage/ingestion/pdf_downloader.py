"""
PDF downloader for the Surrogate Triage Pipeline.
Downloads arXiv PDFs with retry logic and cache management.
"""

import logging
import os
import time
import urllib.error
import urllib.request

from surrogate_triage.schemas import PaperMetadata

logger = logging.getLogger(__name__)

RATE_LIMIT_SECONDS = 3
DEFAULT_CACHE_DIR = "papers_cache"
DEFAULT_MAX_SIZE_BYTES = 1024 * 1024 * 1024  # 1 GB
DEFAULT_MAX_AGE_DAYS = 30
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0


class PDFDownloader:
    """Downloads and caches arXiv PDFs."""

    def __init__(self):
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce minimum delay between download requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_SECONDS:
            time.sleep(RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

    def download(self, paper: PaperMetadata, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
        """Download a paper's PDF to the cache directory.

        Args:
            paper: Paper metadata (must have pdf_url and arxiv_id).
            cache_dir: Directory to store downloaded PDFs.

        Returns:
            Local file path on success, empty string on failure.
        """
        if not paper.pdf_url or not paper.arxiv_id:
            logger.warning("Cannot download paper without pdf_url and arxiv_id")
            return ""

        os.makedirs(cache_dir, exist_ok=True)

        # Sanitize filename from arXiv ID
        safe_id = paper.arxiv_id.replace("/", "_").replace("\\", "_")
        filename = f"{safe_id}.pdf"
        filepath = os.path.join(cache_dir, filename)

        # Skip if already downloaded
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.debug("PDF already cached: %s", filepath)
            return filepath

        # Download with retries and exponential backoff
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limit()
            try:
                req = urllib.request.Request(
                    paper.pdf_url,
                    headers={"User-Agent": "autoresearch/1.0"},
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()

                if len(data) < 1000:
                    logger.warning(
                        "Downloaded PDF too small (%d bytes) for %s, attempt %d/%d",
                        len(data), paper.arxiv_id, attempt, MAX_RETRIES,
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    return ""

                with open(filepath, "wb") as f:
                    f.write(data)

                logger.info("Downloaded PDF for %s (%d bytes)", paper.arxiv_id, len(data))
                return filepath

            except urllib.error.HTTPError as e:
                logger.warning(
                    "HTTP %d downloading %s, attempt %d/%d",
                    e.code, paper.arxiv_id, attempt, MAX_RETRIES,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
            except (urllib.error.URLError, OSError) as e:
                logger.warning(
                    "Network error downloading %s: %s, attempt %d/%d",
                    paper.arxiv_id, e, attempt, MAX_RETRIES,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2

        logger.error("Failed to download PDF for %s after %d attempts", paper.arxiv_id, MAX_RETRIES)
        return ""

    def cleanup(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ):
        """Clean up the PDF cache by removing old/excess files.

        Removes files older than max_age_days first, then removes oldest files
        until total size is under max_size_bytes.

        Args:
            cache_dir: Directory containing cached PDFs.
            max_size_bytes: Maximum total cache size in bytes.
            max_age_days: Maximum age of cached files in days.
        """
        if not os.path.isdir(cache_dir):
            return

        now = time.time()
        max_age_seconds = max_age_days * 86400

        # Gather all PDF files with their stats
        files = []
        for name in os.listdir(cache_dir):
            if not name.endswith(".pdf"):
                continue
            path = os.path.join(cache_dir, name)
            try:
                stat = os.stat(path)
                files.append((path, stat.st_mtime, stat.st_size))
            except OSError:
                continue

        # Phase 1: Remove files older than max_age_days
        removed_age = 0
        remaining = []
        for path, mtime, size in files:
            if now - mtime > max_age_seconds:
                try:
                    os.remove(path)
                    removed_age += 1
                    logger.debug("Removed aged-out PDF: %s", path)
                except OSError as e:
                    logger.warning("Failed to remove %s: %s", path, e)
                    remaining.append((path, mtime, size))
            else:
                remaining.append((path, mtime, size))

        # Phase 2: Remove oldest files until under size limit
        remaining.sort(key=lambda x: x[1])  # oldest first
        total_size = sum(size for _, _, size in remaining)
        removed_size = 0
        while remaining and total_size > max_size_bytes:
            path, mtime, size = remaining.pop(0)
            try:
                os.remove(path)
                total_size -= size
                removed_size += 1
                logger.debug("Removed oversized PDF: %s", path)
            except OSError as e:
                logger.warning("Failed to remove %s: %s", path, e)

        if removed_age or removed_size:
            logger.info(
                "Cache cleanup: removed %d aged + %d oversized files, %d remaining (%.1f MB)",
                removed_age, removed_size, len(remaining),
                total_size / (1024 * 1024),
            )
