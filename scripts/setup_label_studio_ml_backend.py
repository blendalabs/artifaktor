#!/usr/bin/env python3
"""Attach (or update) a Label Studio ML backend for a project.

Python: 3.11+
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_BACKEND_URL = "http://127.0.0.1:9090"
DEFAULT_PROJECT_TITLE = "AI Video Artifact Detection"
DEFAULT_BACKEND_TITLE = "Grounding DINO Artifact Prelabeler"


class LSApiError(RuntimeError):
    pass


@dataclass
class LSClient:
    base_url: str
    session: requests.Session

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        if method.upper() in {"POST", "PATCH", "PUT", "DELETE"}:
            csrf = self.session.cookies.get("csrftoken")
            if csrf:
                headers = kwargs.pop("headers", {})
                headers["X-CSRFToken"] = csrf
                kwargs["headers"] = headers
        return self.session.request(method, url, timeout=60, **kwargs)

    def json(self, method: str, path: str, expected: tuple[int, ...], **kwargs: Any) -> Any:
        resp = self.request(method, path, **kwargs)
        if resp.status_code not in expected:
            raise LSApiError(f"{method} {path} failed ({resp.status_code}): {resp.text}")
        return resp.json() if resp.text.strip() else {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create/update Label Studio ML backend and optionally run test predict")
    p.add_argument("--url", default=os.getenv("LABEL_STUDIO_URL", DEFAULT_URL), help="Label Studio URL")
    p.add_argument("--username", default=os.getenv("LABEL_STUDIO_USERNAME"), help="Label Studio username/email")
    p.add_argument("--password", default=os.getenv("LABEL_STUDIO_PASSWORD"), help="Label Studio password")
    p.add_argument("--project-id", type=int, default=None, help="Project ID")
    p.add_argument("--project-title", default=DEFAULT_PROJECT_TITLE, help="Project title if --project-id omitted")
    p.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help="ML backend URL")
    p.add_argument("--backend-title", default=DEFAULT_BACKEND_TITLE, help="ML backend title")
    p.add_argument("--timeout", type=int, default=120, help="ML backend timeout in seconds")
    p.add_argument("--test-predict", action="store_true", help="Call /predict/test?random=true after setup")
    p.add_argument("--dry-run", action="store_true", help="Show plan only")
    return p.parse_args()


def normalize_url(url: str) -> str:
    return url.rstrip("/")


def login_session(base_url: str, username: str | None, password: str | None) -> LSClient:
    if not username or not password:
        raise LSApiError("--username and --password are required for session auth")

    s = requests.Session()
    login_url = f"{base_url}/user/login"
    r = s.get(login_url, timeout=30)
    if r.status_code != 200:
        raise LSApiError(f"GET /user/login failed ({r.status_code})")

    m = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', r.text)
    if not m:
        raise LSApiError("Could not parse CSRF token from login page")

    payload = {"email": username, "password": password, "csrfmiddlewaretoken": m.group(1)}
    resp = s.post(login_url, data=payload, headers={"Referer": login_url}, allow_redirects=False, timeout=30)
    if resp.status_code not in (200, 302):
        raise LSApiError(f"Login failed ({resp.status_code}): {resp.text}")

    who = s.get(f"{base_url}/api/current-user/whoami", timeout=30)
    if who.status_code != 200:
        raise LSApiError(f"Session verification failed ({who.status_code}): {who.text}")

    return LSClient(base_url=base_url, session=s)


def list_projects(client: LSClient) -> list[dict[str, Any]]:
    payload = client.json("GET", "/api/projects?page_size=200", expected=(200,))
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    return []


def get_project_id(client: LSClient, project_id: int | None, project_title: str) -> int:
    if project_id is not None:
        return project_id
    projects = list_projects(client)
    match = next((p for p in projects if p.get("title") == project_title), None)
    if not match:
        raise LSApiError(f"Project not found by title: {project_title}")
    return int(match["id"])


def list_ml_backends(client: LSClient, project_id: int) -> list[dict[str, Any]]:
    payload = client.json("GET", f"/api/ml?project={project_id}", expected=(200,))
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    return []


def upsert_ml_backend(
    client: LSClient,
    project_id: int,
    backend_url: str,
    backend_title: str,
    timeout_s: int,
    dry_run: bool,
) -> tuple[int, str]:
    backends = list_ml_backends(client, project_id)
    existing = next((b for b in backends if b.get("url") == backend_url), None)

    payload = {
        "url": backend_url,
        "project": project_id,
        "title": backend_title,
        "timeout": timeout_s,
        "is_interactive": False,
        "auth_method": "NONE",
    }

    if existing:
        backend_id = int(existing["id"])
        if dry_run:
            return backend_id, "would-update-existing"
        client.json("PATCH", f"/api/ml/{backend_id}", expected=(200,), json=payload)
        return backend_id, "updated-existing"

    if dry_run:
        return -1, "would-create"

    created = client.json("POST", "/api/ml", expected=(201,), json=payload)
    return int(created["id"]), "created"


def run_test_predict(client: LSClient, backend_id: int, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"status": "would-run-test"}
    payload = client.json("POST", f"/api/ml/{backend_id}/predict/test?random=true", expected=(200,))
    return payload


def main() -> int:
    args = parse_args()
    base_url = normalize_url(args.url)
    backend_url = normalize_url(args.backend_url)

    try:
        client = login_session(base_url, args.username, args.password)
        project_id = get_project_id(client, args.project_id, args.project_title)

        backend_id, action = upsert_ml_backend(
            client=client,
            project_id=project_id,
            backend_url=backend_url,
            backend_title=args.backend_title,
            timeout_s=args.timeout,
            dry_run=args.dry_run,
        )

        test_payload: dict[str, Any] | None = None
        if args.test_predict:
            test_payload = run_test_predict(client, backend_id, args.dry_run)

    except (requests.RequestException, LSApiError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    summary = {
        "project_id": project_id,
        "backend_id": backend_id,
        "backend_action": action,
        "backend_url": backend_url,
        "test_predict": test_payload,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
