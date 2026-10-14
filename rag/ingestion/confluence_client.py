"""Minimal Confluence REST client covering what ingestion needs.

* Data Center / Server: REST v1 (``/rest/api``), Bearer personal access token.
* Cloud: REST v2 (``/api/v2``, v1 content endpoints are being retired), Basic auth with
  email + API token. CONFLUENCE_URL must include ``/wiki``.

Both return the same normalised dicts, so the source code does not care which is used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import urlparse

import requests

from rag.config import ConfluenceAPIConfig, secret

_PAGE_SIZE = 50


@dataclass
class Page:
    id: str
    title: str
    version: int
    when: str
    ancestors: list[str]
    url: str


@dataclass
class Attachment:
    id: str
    title: str
    version: int
    media_type: str
    size: int
    download_url: str
    url: str


class _Http:
    def __init__(self, base_url: str, session: requests.Session, verify: bool):
        self.base = base_url.rstrip("/")
        self.origin = "{0.scheme}://{0.netloc}".format(urlparse(self.base))
        self.s = session
        self.verify = verify

    def abs(self, path: str) -> str:
        if path.startswith("http"):
            return path
        # Cloud "next"/download links may already include the /wiki context path.
        ctx = urlparse(self.base).path
        if ctx and path.startswith(ctx + "/"):
            return self.origin + path
        return self.base + (path if path.startswith("/") else "/" + path)

    def get(self, path: str, params: dict | None = None, raw: bool = False):
        for attempt in range(4):
            resp = self.s.get(self.abs(path), params=params, verify=self.verify, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(float(resp.headers.get("Retry-After", 2 ** attempt)))
                continue
            if resp.status_code in (401, 403):
                raise PermissionError(f"Confluence returned {resp.status_code} for {resp.url}: check token/permissions")
            resp.raise_for_status()
            return resp.content if raw else resp.json()
        resp.raise_for_status()


class ConfluenceDataCenter:
    def __init__(self, http: _Http):
        self.h = http

    def space_name(self, key: str) -> str:
        return self.h.get(f"/rest/api/space/{key}").get("name", key)

    def pages(self, space_key: str) -> Iterator[Page]:
        start = 0
        while True:
            data = self.h.get(
                "/rest/api/content",
                {"spaceKey": space_key, "type": "page", "status": "current",
                 "start": start, "limit": _PAGE_SIZE, "expand": "version,ancestors"},
            )
            results = data.get("results", [])
            for p in results:
                yield Page(
                    id=str(p["id"]),
                    title=p.get("title", ""),
                    version=int(p.get("version", {}).get("number", 0)),
                    when=p.get("version", {}).get("when", ""),
                    ancestors=[a.get("title", "") for a in p.get("ancestors", [])],
                    url=self.h.abs(p.get("_links", {}).get("webui", f"/pages/viewpage.action?pageId={p['id']}")),
                )
            if len(results) < _PAGE_SIZE:
                return
            start += len(results)

    def body(self, page_id: str) -> str:
        return self.h.get(f"/rest/api/content/{page_id}", {"expand": "body.storage"})["body"]["storage"]["value"]

    def attachments(self, page_id: str) -> list[Attachment]:
        out, start = [], 0
        while True:
            data = self.h.get(
                f"/rest/api/content/{page_id}/child/attachment",
                {"start": start, "limit": _PAGE_SIZE, "expand": "version"},
            )
            results = data.get("results", [])
            for a in results:
                links = a.get("_links", {})
                out.append(Attachment(
                    id=str(a["id"]),
                    title=a.get("title", ""),
                    version=int(a.get("version", {}).get("number", 0)),
                    media_type=a.get("metadata", {}).get("mediaType") or a.get("extensions", {}).get("mediaType", ""),
                    size=int(a.get("extensions", {}).get("fileSize") or 0),
                    download_url=self.h.abs(links.get("download", "")),
                    url=self.h.abs(links.get("webui") or links.get("download", "")),
                ))
            if len(results) < _PAGE_SIZE:
                return out
            start += len(results)

    def download(self, att: Attachment) -> bytes:
        return self.h.get(att.download_url, raw=True)


class ConfluenceCloud:
    def __init__(self, http: _Http):
        self.h = http
        self._space_ids: dict[str, tuple[str, str]] = {}

    def _paged(self, path: str, params: dict) -> Iterator[dict[str, Any]]:
        data = self.h.get(path, params)
        while True:
            yield from data.get("results", [])
            nxt = data.get("_links", {}).get("next")
            if not nxt:
                return
            data = self.h.get(nxt)

    def _space(self, key: str) -> tuple[str, str]:
        if key not in self._space_ids:
            res = self.h.get("/api/v2/spaces", {"keys": key}).get("results", [])
            if not res:
                raise LookupError(f"Confluence space '{key}' not found or not visible to this token")
            self._space_ids[key] = (str(res[0]["id"]), res[0].get("name", key))
        return self._space_ids[key]

    def space_name(self, key: str) -> str:
        return self._space(key)[1]

    def pages(self, space_key: str) -> Iterator[Page]:
        space_id, _ = self._space(space_key)
        raw = list(self._paged(f"/api/v2/spaces/{space_id}/pages", {"limit": 250, "status": "current"}))
        by_id = {str(p["id"]): p for p in raw}

        def ancestors(p: dict[str, Any]) -> list[str]:
            chain, parent = [], p.get("parentId")
            while parent and str(parent) in by_id and len(chain) < 50:
                chain.insert(0, by_id[str(parent)].get("title", ""))
                parent = by_id[str(parent)].get("parentId")
            return chain

        for p in raw:
            version = p.get("version", {})
            yield Page(
                id=str(p["id"]),
                title=p.get("title", ""),
                version=int(version.get("number", 0)),
                when=version.get("createdAt", ""),
                ancestors=ancestors(p),
                url=self.h.abs(p.get("_links", {}).get("webui", f"/pages/{p['id']}")),
            )

    def body(self, page_id: str) -> str:
        return self.h.get(f"/api/v2/pages/{page_id}", {"body-format": "storage"})["body"]["storage"]["value"]

    def attachments(self, page_id: str) -> list[Attachment]:
        return [
            Attachment(
                id=str(a["id"]),
                title=a.get("title", ""),
                version=int(a.get("version", {}).get("number", 0)),
                media_type=a.get("mediaType", ""),
                size=int(a.get("fileSize") or 0),
                download_url=self.h.abs(a.get("downloadLink", "")),
                url=self.h.abs(a.get("webuiLink") or a.get("downloadLink", "")),
            )
            for a in self._paged(f"/api/v2/pages/{page_id}/attachments", {"limit": 250})
        ]

    def download(self, att: Attachment) -> bytes:
        return self.h.get(att.download_url, raw=True)


def connect(cfg: ConfluenceAPIConfig) -> ConfluenceDataCenter | ConfluenceCloud:
    base = secret(cfg.url_env)
    session = requests.Session()
    session.headers["Accept"] = "application/json"
    token = secret(cfg.token_env)
    if cfg.deployment == "cloud":
        session.auth = (secret(cfg.user_env), token)
        return ConfluenceCloud(_Http(base, session, cfg.verify_ssl))
    session.headers["Authorization"] = f"Bearer {token}"
    return ConfluenceDataCenter(_Http(base, session, cfg.verify_ssl))

