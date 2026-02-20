#!/usr/bin/env python3
"""Create/update a Label Studio project and attach Local Files storage.

Step-2 automation:
1) Validate labeling config XML
2) Create or update project
3) Create or reuse Local Files import storage
4) Optionally trigger storage sync

Supports auth by:
- API token (--api-token / LABEL_STUDIO_API_KEY)
- Username/password web session login (--username/--password)

Python: 3.11+
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_PROJECT_TITLE = "AI Video Artifact Detection"
DEFAULT_CONFIG_PATH = Path("label_studio/artifact_detection_config.xml")
DEFAULT_LOCAL_FILES_PATH = Path("sequences")


class LSApiError(RuntimeError):
    pass


@dataclass
class LSClient:
    base_url: str
    session: requests.Session
    auth_mode: str  # "token" | "session"

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        if self.auth_mode == "session" and method.upper() in {"POST", "PATCH", "PUT", "DELETE"}:
            csrf = self.session.cookies.get("csrftoken")
            if csrf:
                headers = kwargs.pop("headers", {})
                headers["X-CSRFToken"] = csrf
                kwargs["headers"] = headers
        return self.session.request(method=method, url=url, timeout=60, **kwargs)

    def json(self, method: str, path: str, expected: tuple[int, ...], **kwargs: Any) -> Any:
        response = self.request(method, path, **kwargs)
        if response.status_code not in expected:
            raise LSApiError(f"{method} {path} failed ({response.status_code}): {response.text}")
        if not response.text.strip():
            return {}
        return response.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up Label Studio project + local storage for artifact detection.")
    parser.add_argument("--url", default=os.getenv("LABEL_STUDIO_URL", DEFAULT_URL), help="Label Studio base URL")
    parser.add_argument(
        "--api-token",
        default=os.getenv("LABEL_STUDIO_API_KEY") or os.getenv("LABEL_STUDIO_API_TOKEN"),
        help="API token (preferred for automation)",
    )
    parser.add_argument("--username", default=os.getenv("LABEL_STUDIO_USERNAME"), help="Label Studio username/email")
    parser.add_argument("--password", default=os.getenv("LABEL_STUDIO_PASSWORD"), help="Label Studio password")
    parser.add_argument("--project-title", default=DEFAULT_PROJECT_TITLE, help="Project title")
    parser.add_argument("--project-id", type=int, default=None, help="Existing project ID (optional)")
    parser.add_argument(
        "--label-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Label config XML path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--local-files-path",
        type=Path,
        default=DEFAULT_LOCAL_FILES_PATH,
        help=f"Directory to expose as Local Files storage (default: {DEFAULT_LOCAL_FILES_PATH})",
    )
    parser.add_argument(
        "--local-storage-title",
        default="artifact-frames-local-storage",
        help="Title for Local Files storage",
    )
    parser.add_argument("--sync-storage", action="store_true", help="Trigger storage sync after setup")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, do not call write APIs")
    return parser.parse_args()


def normalize_url(url: str) -> str:
    return url.rstrip("/")


def load_and_validate_xml(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Label config not found: {path}")
    xml_text = path.read_text(encoding="utf-8")
    try:
        ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML in {path}: {exc}") from exc
    return xml_text


def parse_csrf_token_from_login_html(html: str) -> str:
    match = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', html)
    if not match:
        raise LSApiError("Could not find CSRF token on /user/login page.")
    return match.group(1)


def build_client(base_url: str, api_token: str | None, username: str | None, password: str | None) -> LSClient:
    s = requests.Session()

    if api_token:
        s.headers.update({"Authorization": f"Token {api_token}"})
        probe = s.get(f"{base_url}/api/current-user/whoami", timeout=30)
        if probe.status_code != 200:
            raise LSApiError(f"Token auth failed ({probe.status_code}): {probe.text}")
        return LSClient(base_url=base_url, session=s, auth_mode="token")

    if not username or not password:
        raise LSApiError("Provide --api-token OR --username/--password.")

    # Session login flow (works on modern LS without legacy /api/token flow)
    login_url = f"{base_url}/user/login"
    r = s.get(login_url, timeout=30)
    if r.status_code != 200:
        raise LSApiError(f"Failed to open login page ({r.status_code})")

    csrf = parse_csrf_token_from_login_html(r.text)
    form = {
        "email": username,
        "password": password,
        "csrfmiddlewaretoken": csrf,
    }
    headers = {"Referer": login_url}
    login_resp = s.post(login_url, data=form, headers=headers, allow_redirects=False, timeout=30)
    if login_resp.status_code not in (302, 200):
        raise LSApiError(f"Login failed ({login_resp.status_code}): {login_resp.text}")

    who = s.get(f"{base_url}/api/current-user/whoami", timeout=30)
    if who.status_code != 200:
        raise LSApiError(f"Session auth verification failed ({who.status_code}): {who.text}")

    return LSClient(base_url=base_url, session=s, auth_mode="session")


def list_projects(client: LSClient) -> list[dict[str, Any]]:
    payload = client.json("GET", "/api/projects?page_size=200", expected=(200,))
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    return []


def get_or_create_project(
    client: LSClient,
    project_title: str,
    project_id: int | None,
    label_config_xml: str,
    dry_run: bool,
) -> tuple[int, str]:
    if project_id is not None:
        if dry_run:
            return project_id, "would-update-by-id"
        client.json(
            "PATCH",
            f"/api/projects/{project_id}",
            expected=(200,),
            json={"title": project_title, "label_config": label_config_xml},
        )
        return project_id, "updated-by-id"

    existing = next((p for p in list_projects(client) if p.get("title") == project_title), None)
    if existing:
        pid = int(existing["id"])
        if dry_run:
            return pid, "would-update-existing"
        client.json(
            "PATCH",
            f"/api/projects/{pid}",
            expected=(200,),
            json={"title": project_title, "label_config": label_config_xml},
        )
        return pid, "updated-existing"

    if dry_run:
        return -1, "would-create"

    created = client.json(
        "POST",
        "/api/projects",
        expected=(201,),
        json={"title": project_title, "label_config": label_config_xml},
    )
    return int(created["id"]), "created"


def list_local_storages(client: LSClient, project_id: int) -> list[dict[str, Any]]:
    payload = client.json("GET", f"/api/storages/localfiles?project={project_id}", expected=(200,))
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    return []


def get_or_create_local_storage(
    client: LSClient,
    project_id: int,
    storage_title: str,
    local_path: str,
    dry_run: bool,
) -> tuple[int, str]:
    storages = list_local_storages(client, project_id)
    resolved_target = Path(local_path).resolve()
    existing = next(
        (s for s in storages if s.get("path") and Path(s["path"]).resolve() == resolved_target),
        None,
    )

    payload = {
        "title": storage_title,
        "path": str(resolved_target),
        "project": project_id,
        "use_blob_urls": True,
        "recursive_scan": True,
        "regex_filter": r"(?i).*\.(png|jpg|jpeg|webp)$",
    }

    if existing:
        sid = int(existing["id"])
        if dry_run:
            return sid, "would-update-existing"
        client.json("PATCH", f"/api/storages/localfiles/{sid}", expected=(200,), json=payload)
        return sid, "updated-existing"

    if dry_run:
        return -1, "would-create"

    created = client.json("POST", "/api/storages/localfiles", expected=(201,), json=payload)
    return int(created["id"]), "created"


def sync_storage(client: LSClient, storage_id: int, dry_run: bool) -> str:
    if dry_run:
        return "would-sync"
    client.json("POST", f"/api/storages/localfiles/{storage_id}/sync", expected=(200, 201, 202))
    return "sync-triggered"


def print_env_hint(local_files_path: Path) -> None:
    # Storage path must be a *subdirectory* of document root.
    doc_root = local_files_path.resolve().parent
    print("\nRequired env for Local Files storage (set BEFORE starting Label Studio):")
    print(f"  export LOCAL_FILES_SERVING_ENABLED=true")
    print(f"  export LOCAL_FILES_DOCUMENT_ROOT={doc_root}")
    print("\nCompatibility alias used in older docs:")
    print(f"  export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={doc_root}")


def main() -> int:
    args = parse_args()
    base_url = normalize_url(args.url)

    try:
        label_config_xml = load_and_validate_xml(args.label_config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    local_files_path = args.local_files_path.resolve()
    if not local_files_path.exists() or not local_files_path.is_dir():
        print(f"[ERROR] --local-files-path must exist and be a directory: {local_files_path}", file=sys.stderr)
        return 2

    print(f"Label config: {args.label_config.resolve()}")
    print(f"Local files path: {local_files_path}")
    print_env_hint(local_files_path)

    if args.dry_run:
        print("\n[DRY-RUN] XML validation passed.")
        print(f"[DRY-RUN] URL={base_url} Project='{args.project_title}'")
        return 0

    try:
        client = build_client(base_url, args.api_token, args.username, args.password)
        project_id, project_action = get_or_create_project(
            client=client,
            project_title=args.project_title,
            project_id=args.project_id,
            label_config_xml=label_config_xml,
            dry_run=False,
        )
        storage_id, storage_action = get_or_create_local_storage(
            client=client,
            project_id=project_id,
            storage_title=args.local_storage_title,
            local_path=str(local_files_path),
            dry_run=False,
        )
        sync_action = sync_storage(client, storage_id, dry_run=False) if args.sync_storage else "not-requested"

    except requests.RequestException as exc:
        print(f"[ERROR] Network/API request failed: {exc}", file=sys.stderr)
        return 1
    except LSApiError as exc:
        msg = str(exc)
        print(f"[ERROR] {msg}", file=sys.stderr)
        if "LOCAL_FILES_SERVING_ENABLED" in msg:
            print("\nHint: restart Label Studio after exporting the env vars shown above.", file=sys.stderr)
        return 1

    summary = {
        "project_id": project_id,
        "project_action": project_action,
        "storage_id": storage_id,
        "storage_action": storage_action,
        "sync": sync_action,
        "url": base_url,
    }
    print("\nSetup complete:")
    print(json.dumps(summary, indent=2))
    print(f"\nOpen project: {base_url}/projects/{project_id}/data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
