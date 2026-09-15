#!/usr/bin/env python3
"""The runner. What a customer's CI job runs; what the Action wraps.

    mo-eval-runner suite --service https://… --token …     # against the hosted service
    mo-eval-runner suite --service local                    # everything in this process

Reads the repository's own `.mo-eval/config.toml`, collects facts, lets the service choose, sends
source for the chosen few, carries out the orders that come back, reports exit codes, and writes
whatever the service says survived. It ends with a funnel — how many changes were offered, and where
the rest went — because a suite that comes back empty must say which stage emptied it.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from languages import LANGUAGES, detect_framework  # noqa: E402
from runner.client import ServiceError, client_for  # noqa: E402
from runner.collect import change_source, repo_facts  # noqa: E402
from runner.config import CONFIG_PATH, ConfigError, load_config, probe_environment  # noqa: E402
from runner.execute import Runner  # noqa: E402
from wire import TaskPackage, VerdictReport, Wire  # noqa: E402


def write_package(package: TaskPackage, into: Path) -> Path:
    """Write a package the service returned as `into/<task_id>/`. The runner writes what it is
    handed; it does not import the service to do so."""
    directory = into / package.task_id
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in package.files.items():
        (directory / name).write_text(content)
    return directory


def write_local_suite(package: TaskPackage, repo: Path, into: Path) -> Path | None:
    """Materialize `into/<task_id>/` as a one-task local-test-suite over the task's start tree.

    The start tree is rebuilt the way the validation order built it — the parent commit exported,
    the scaffold applied, one evaluator-owned commit — because mo-eval's snapshotter wants a Git
    worktree root and the order's workspace was deleted when the order finished. Returns `None`
    when the package carries no overlay (no worker image was declared).
    """
    if not package.local_suite:
        return None
    meta = json.loads(package.files["meta.json"])
    root = into / package.task_id
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    archive = subprocess.Popen(["git", "archive", "--format=tar", meta["parent_commit"]], cwd=repo, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-xf", "-", "-C", str(root)], stdin=archive.stdout, check=True)
    archive.wait()
    for path in root.rglob("*"):
        os.utime(path, None)
    (root / ".mo-eval-scaffold.patch").write_text(package.files["scaffold.patch"])
    subprocess.run(["git", "apply", "--whitespace=nowarn", ".mo-eval-scaffold.patch"], cwd=root, check=True)
    (root / ".mo-eval-scaffold.patch").unlink()
    _prepare_offline(root, meta)
    for relative, content in package.local_suite.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    git = ["git", "-c", "user.email=mo-eval@example.invalid", "-c", "user.name=mo-eval"]
    subprocess.run([*git, "init", "-q"], cwd=root, check=True)
    subprocess.run([*git, "add", "-A"], cwd=root, check=True)
    for artifact in _offline_artifacts(meta):
        if (root / artifact).exists():
            # Forced past .gitignore: the frozen snapshot takes tracked files, and an ignored
            # vendor/ would vanish between this tree and the worker that scores it.
            subprocess.run([*git, "add", "-f", artifact], cwd=root, check=True)
    subprocess.run([*git, "commit", "-q", "--no-verify", "-m", f"mo-eval start state for {package.task_id}"], cwd=root, check=True)
    return root


def _prepare_offline(root: Path, meta: dict, run=subprocess.run) -> None:
    """Make the start tree scorable with no network, the way its language does that.

    mo-eval's baseline and scorer workers have no network at all — the first live run failed on
    `dial tcp: lookup proxy.golang.org` from inside `go test`. The dependencies have to be IN the
    frozen tree. Runs before the evaluator-owned commit so they are part of the start state the agent
    receives and the scorer restores, not part of the agent's diff.

    Raises:
        subprocess.CalledProcessError: If the preparation fails; a bundle that cannot be scored
            offline must not be written as though it could.
    """
    contract = LANGUAGES.get(meta.get("generated", {}).get("language", ""))
    if contract is None or not contract.offline_prepare:
        return
    run(shlex.split(contract.offline_prepare), cwd=root, check=True, capture_output=True, text=True)


def _offline_artifacts(meta: dict) -> tuple[str, ...]:
    """What the language's offline preparation leaves behind that must be tracked."""
    contract = LANGUAGES.get(meta.get("generated", {}).get("language", ""))
    return contract.offline_artifacts if contract is not None else ()


def _repo_name(repo: Path, declared: str | None) -> str:
    if declared:
        return declared
    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=repo, capture_output=True, text=True).stdout.strip()
    tail = url.removesuffix(".git").replace(":", "/").rstrip("/")
    parts = tail.split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else tail


def _branch(repo: Path, repo_name: str) -> str:
    """The branch merged pull requests target — the repository's DEFAULT branch, not whatever is
    checked out.

    A CI job checks out the pushed branch, and a developer often has a feature branch out; listing
    merged PRs against either returns nothing, silently. `origin/HEAD` names the default for any
    clone; the forge is asked when a clone lacks it; HEAD is the last resort.
    """
    head = subprocess.run(["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"], cwd=repo,
                          capture_output=True, text=True).stdout.strip()
    if head:
        return head.split("/", 1)[-1]
    forge = subprocess.run(["gh", "repo", "view", "-R", repo_name, "--json", "defaultBranchRef", "--jq", ".defaultBranchRef.name"],
                           cwd=repo, capture_output=True, text=True).stdout.strip()
    if forge:
        return forge
    return subprocess.run(["git", "symbolic-ref", "--short", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip() or "main"


def _contract_name(repo: Path, language: str) -> str:
    """Which framework this repository actually uses, measured from its test files."""
    base = LANGUAGES[language]
    tracked = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True).stdout.splitlines()
    contents = []
    for path in [p for p in tracked if base.test_path.search(p)][:200]:
        try:
            contents.append((repo / path).read_text(errors="replace"))
        except OSError:
            continue
    return detect_framework(contents, base).name


def suite(arguments: argparse.Namespace) -> int:
    repo = arguments.repo.resolve()
    try:
        config = load_config(repo / arguments.config)
    except ConfigError as failure:
        print(f"config: {failure}", file=sys.stderr)
        return 2
    environment, missing = probe_environment(config, os.environ)
    if missing:
        print(f"config: forward_env names {missing} but the environment does not have them; "
              f"probes needing them will fail identically", file=sys.stderr)
    try:
        client = client_for(arguments.service, arguments.token or os.environ.get("MO_EVAL_TOKEN"))
    except ServiceError as failure:
        print(f"service: {failure}", file=sys.stderr)
        return 2

    out = arguments.out.resolve()
    wire = Wire(root=out / "wire")
    contract = _contract_name(repo, config.language)
    name = _repo_name(repo, config.repo)
    print(f"mo-eval-runner · {name} · {contract}")

    facts = wire.crossing("up", "repo-facts", repo_facts(
        repo, repo_name=name, language=contract, test_command=config.test_command,
        branch=arguments.branch or _branch(repo, name), limit=arguments.history,
        language_contract=LANGUAGES[config.language], setup_command=config.setup_command,
    ))
    facts = replace(facts, worker_image=config.worker_image)
    try:
        request = wire.crossing("down", "source-request", client.select(facts, arguments.candidates))
        sources = wire.crossing("up", "change-source", change_source(repo, facts, request.change_ids))
        response = wire.crossing("down", "work-orders", client.orders(facts, sources, arguments.candidates))
    except ServiceError as failure:
        print(f"service: {failure}", file=sys.stderr)
        return 1

    orders = response.orders[: arguments.validate] if arguments.validate else response.orders
    if arguments.dry_run:
        print(f"  --dry-run: {len(orders)} order(s) issued, none run")
        _funnel(facts, request, response, 0, VerdictReport([]), [], dry_run=True)
        return 0
    runner = Runner(repo=repo, workspaces=out / "workspaces", test_command=config.test_command,
                    env=environment, setup_command=config.setup_command)
    results = []
    for order in orders:
        print(f"  running {order.order_id} …", flush=True)
        results.append(runner.run(order))
    wire.crossing("up", "order-results", results)
    try:
        report = wire.crossing("down", "verdicts", client.verdicts(results))
    except ServiceError as failure:
        print(f"service: {failure}", file=sys.stderr)
        return 1

    written = []
    for verdict in report.verdicts:
        mark = "VALIDATED" if verdict.validated else "rejected "
        print(f"  {mark} {verdict.change_id[:9]}  {verdict.detail}")
        if verdict.task is not None:
            written.append(write_package(verdict.task, out / "tasks"))
            bundle = write_local_suite(verdict.task, repo, out / "local-suite")
            if bundle is not None:
                print(f"             local-suite bundle → {bundle}")

    _funnel(facts, request, response, len(orders), report, written)
    return 0 if written else 3


def _funnel(facts, request, response, ran: int, report, written: list[Path], *, dry_run: bool = False) -> None:
    """Where every offered change went. The number a user needs is not how many tasks they got,
    but why they did not get more."""
    offered = len(facts.changes)
    notes = []
    if facts.unresolved_changes:
        notes.append(f"{facts.unresolved_changes} PRs unresolvable to a parent/reference pair")
    if facts.unreadable_changes:
        notes.append(f"{facts.unreadable_changes} unreadable")
    if facts.source != "github-prs":
        notes.append(f"read from git log, squash merges only: {facts.source_note or 'forge not consulted'}")
    suffix = f"  ({'; '.join(notes)})" if notes else ""
    print(f"\n  offered    {offered:>4} merged changes via {facts.source}{suffix}")
    print(f"  selected   {len(request.change_ids):>4}   " + _reasons(request.rejections))
    print(f"  orders     {len(response.orders):>4}   " + _reasons(response.rejections))
    if dry_run:
        # Nothing ran, so nothing survived or failed to; saying so would misread a dry run as a
        # repository that yields nothing.
        print(f"  validated     —   (dry run: {len(response.orders)} order(s) would run)")
        return
    validated = sum(1 for v in report.verdicts if v.validated)
    print(f"  validated  {validated:>4} of {ran} run")
    for path in written:
        print(f"             → {path}")
    if not written:
        print("\n  no task survived; the stage that emptied the funnel is the one to look at")


def _reasons(rejections) -> str:
    if not rejections:
        return ""
    top = ", ".join(f"{r.count} {r.reason}" for r in rejections[:3])
    more = f", +{len(rejections) - 3} more" if len(rejections) > 3 else ""
    return f"rejected: {top}{more}"


def main() -> int:
    parser = argparse.ArgumentParser(prog="mo-eval-runner", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    s = commands.add_parser("suite", help="mine, validate, and write this repository's suite")
    s.add_argument("--repo", type=Path, default=Path("."))
    s.add_argument("--config", type=Path, default=CONFIG_PATH, help="relative to --repo")
    s.add_argument("--service", default="local", help="'local' or the service base URL")
    s.add_argument("--token", default=None, help="bearer token, else $MO_EVAL_TOKEN")
    s.add_argument("--branch", default=None)
    s.add_argument("--history", type=int, default=300)
    s.add_argument("--candidates", type=int, default=8)
    s.add_argument("--validate", type=int, default=0, help="cap on orders to run (0 = all)")
    s.add_argument("--out", type=Path, default=Path(".mo-eval") / "out")
    s.add_argument("--dry-run", action="store_true", help="stop after orders are issued; run nothing")
    s.set_defaults(run=suite)
    arguments = parser.parse_args()
    return arguments.run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
