#!/usr/bin/env python
"""List public ``sfincs_jax`` examples by task.

The examples tree intentionally contains several topic folders. This small
browser keeps first-time navigation simple by reading the checked
``workflow_catalog.json`` and printing the recommended entry points for common
tasks such as transport matrices, autodiff, VMEC geometry, bootstrap current,
validation, and performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXAMPLES_ROOT = Path(__file__).resolve().parent
CATALOG_PATH = EXAMPLES_ROOT / "workflow_catalog.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List recommended sfincs_jax examples from examples/workflow_catalog.json.",
    )
    parser.add_argument(
        "--topic",
        default="",
        help="Filter by topic or keyword, e.g. transport, bootstrap, VMEC, autodiff, performance.",
    )
    parser.add_argument(
        "--search",
        default="",
        help="Free-text filter over workflow id, goal, topic, entrypoint, command, and keywords.",
    )
    parser.add_argument("--json", action="store_true", help="Print matching workflows as JSON.")
    parser.add_argument("--long", action="store_true", help="Include runtime budgets and dependency notes.")
    parser.add_argument("--list-topics", action="store_true", help="Print available folder topics and exit.")
    return parser


def _load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _workflow_search_blob(workflow: dict[str, Any]) -> str:
    fields = [
        workflow.get("id", ""),
        workflow.get("goal", ""),
        workflow.get("entrypoint", ""),
        workflow.get("command", ""),
        workflow.get("topic", ""),
        workflow.get("runtime_budget", ""),
        " ".join(str(item) for item in workflow.get("keywords", [])),
    ]
    return " ".join(str(field).lower() for field in fields)


def _matches(workflow: dict[str, Any], *, topic: str, search: str) -> bool:
    blob = _workflow_search_blob(workflow)
    topic_query = topic.strip().lower()
    search_query = search.strip().lower()
    if topic_query and topic_query not in blob:
        return False
    if search_query and search_query not in blob:
        return False
    return True


def _matching_workflows(catalog: dict[str, Any], *, topic: str, search: str) -> list[dict[str, Any]]:
    return [
        workflow
        for workflow in catalog.get("workflows", [])
        if _matches(workflow, topic=topic, search=search)
    ]


def _print_topics(catalog: dict[str, Any]) -> None:
    for name, metadata in sorted(catalog.get("folders", {}).items()):
        print(f"{name:22s} {metadata.get('role', '')}")


def _print_workflows(workflows: list[dict[str, Any]], *, long: bool) -> None:
    if not workflows:
        print("No workflows matched. Try --list-topics or a broader --search term.")
        return
    for workflow in workflows:
        print(f"{workflow['id']}: {workflow['goal']}")
        print(f"  entry:   examples/{workflow['entrypoint']}")
        print(f"  command: {workflow['command']}")
        if long:
            print(f"  topic:   {workflow['topic']}")
            print(f"  budget:  {workflow['runtime_budget']}")
            needs_fortran = "yes" if workflow.get("requires_fortran_v3") else "no"
            print(f"  local SFINCS Fortran v3 required for first run: {needs_fortran}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    catalog = _load_catalog()
    if args.list_topics:
        _print_topics(catalog)
        return 0
    workflows = _matching_workflows(catalog, topic=args.topic, search=args.search)
    if args.json:
        print(json.dumps({"workflows": workflows}, indent=2, sort_keys=True))
    else:
        _print_workflows(workflows, long=bool(args.long))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
