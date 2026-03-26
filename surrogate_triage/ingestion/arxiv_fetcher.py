"""
ArXiv paper fetcher for the Surrogate Triage Pipeline.
Polls arXiv API for new ML paper submissions and stores metadata.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from surrogate_triage.schemas import PaperMetadata, load_jsonl, save_jsonl

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ["cs.LG", "cs.CL", "cs.AI"]
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
RATE_LIMIT_SECONDS = 3


class ArxivFetcher:
    """Fetches new arXiv submissions and stores them as PaperMetadata."""

    def __init__(self, index_path: str = "papers_index.jsonl"):
        self.index_path = index_path
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce minimum delay between arXiv API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_SECONDS:
            time.sleep(RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

    def _build_query(self, days_back: int, start: int, max_results: int) -> str:
        """Build arXiv API query URL."""
        cat_query = " OR ".join(f"cat:{cat}" for cat in ARXIV_CATEGORIES)
        params = {
            "search_query": cat_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": str(start),
            "max_results": str(max_results),
        }
        return ARXIV_API_URL + "?" + urllib.parse.urlencode(params)

    def _fetch_page(self, url: str, timeout: int = 30) -> str:
        """Fetch a single page from arXiv API with error handling."""
        self._rate_limit()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "autoresearch/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 503:
                logger.warning("arXiv returned 503, retrying after delay")
                time.sleep(RATE_LIMIT_SECONDS * 2)
                self._rate_limit()
                req = urllib.request.Request(url, headers={"User-Agent": "autoresearch/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8")
            raise
        except (urllib.error.URLError, OSError) as e:
            logger.error("Network error fetching arXiv: %s", e)
            raise

    def _parse_entries(self, xml_text: str, cutoff_date: datetime) -> list:
        """Parse Atom XML response into PaperMetadata list."""
        papers = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error("Failed to parse arXiv XML: %s", e)
            return papers

        for entry in root.findall("atom:entry", ARXIV_NS):
            try:
                paper = self._parse_entry(entry, cutoff_date)
                if paper is not None:
                    papers.append(paper)
            except Exception as e:
                logger.warning("Failed to parse entry: %s", e)
                continue

        return papers

    def _parse_entry(self, entry, cutoff_date: datetime):
        """Parse a single Atom entry into PaperMetadata."""
        # Extract arXiv ID from the <id> tag (URL like http://arxiv.org/abs/XXXX.XXXXX)
        id_text = entry.findtext("atom:id", "", ARXIV_NS).strip()
        arxiv_id = id_text.rsplit("/", 1)[-1] if "/" in id_text else id_text
        # Strip version suffix (e.g., v1, v2)
        if arxiv_id and "v" in arxiv_id:
            base = arxiv_id.rsplit("v", 1)
            if base[1].isdigit():
                arxiv_id = base[0]

        title = entry.findtext("atom:title", "", ARXIV_NS).strip()
        title = " ".join(title.split())  # collapse whitespace

        abstract = entry.findtext("atom:summary", "", ARXIV_NS).strip()
        abstract = " ".join(abstract.split())

        published = entry.findtext("atom:published", "", ARXIV_NS).strip()

        # Check date cutoff
        if published and cutoff_date:
            try:
                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                if pub_dt < cutoff_date:
                    return None
            except ValueError:
                pass

        # Authors
        authors = []
        for author_el in entry.findall("atom:author", ARXIV_NS):
            name = author_el.findtext("atom:name", "", ARXIV_NS).strip()
            if name:
                authors.append(name)

        # Categories
        categories = []
        for cat_el in entry.findall("arxiv:primary_category", ARXIV_NS):
            term = cat_el.get("term", "")
            if term:
                categories.append(term)
        for cat_el in entry.findall("atom:category", ARXIV_NS):
            term = cat_el.get("term", "")
            if term and term not in categories:
                categories.append(term)

        # PDF URL
        pdf_url = ""
        for link_el in entry.findall("atom:link", ARXIV_NS):
            if link_el.get("title") == "pdf":
                pdf_url = link_el.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        if not arxiv_id or not title:
            return None

        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            categories=categories,
            pdf_url=pdf_url,
            published=published,
        )

    def _load_existing_ids(self) -> set:
        """Load set of arXiv IDs already in the index."""
        entries = load_jsonl(self.index_path)
        return {e.get("arxiv_id", "") for e in entries if e.get("arxiv_id")}

    def fetch_recent(self, days_back: int = 1, max_results: int = 100) -> list:
        """Fetch recent arXiv papers, with pagination.

        Args:
            days_back: How many days back to look for papers.
            max_results: Maximum total papers to return.

        Returns:
            List of PaperMetadata for new (non-duplicate) papers.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        all_papers = []
        existing_ids = self._load_existing_ids()
        page_size = min(max_results, 100)
        start = 0

        while len(all_papers) < max_results:
            url = self._build_query(days_back, start, page_size)
            logger.info("Fetching arXiv page start=%d size=%d", start, page_size)

            try:
                xml_text = self._fetch_page(url)
            except Exception as e:
                logger.error("Failed to fetch page at start=%d: %s", start, e)
                break

            papers = self._parse_entries(xml_text, cutoff)

            if not papers:
                break

            for paper in papers:
                if paper.arxiv_id not in existing_ids and len(all_papers) < max_results:
                    all_papers.append(paper)
                    existing_ids.add(paper.arxiv_id)

            start += page_size

            # If we got fewer results than requested, we've exhausted results
            if len(papers) < page_size:
                break

        logger.info("Fetched %d new papers from arXiv", len(all_papers))
        return all_papers

    def fetch_and_store(self, days_back: int = 1, max_results: int = 100) -> int:
        """Fetch recent papers and append new ones to the index.

        Args:
            days_back: How many days back to look for papers.
            max_results: Maximum papers to fetch.

        Returns:
            Number of new papers added to the index.
        """
        papers = self.fetch_recent(days_back=days_back, max_results=max_results)
        if papers:
            save_jsonl(papers, self.index_path, mode="a")
            logger.info("Stored %d new papers to %s", len(papers), self.index_path)
        return len(papers)
