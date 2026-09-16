"""Collect repository facts and source. The only forge-specific code in the system.

Everything here answers questions any git host can answer: which changes merged, what did each touch,
show me these two versions of a file, what did the human write about it. GitLab or Bitbucket is
another `_Forge` implementation, not a port — which is the reason collection lives on the runner and
the intelligence does not.

Phase 1 (`repo_facts`) sends no file content at all. Phase 2 (`change_source`) sends content for the
handful of changes the service asked for.
"""

from __future__ import annotations

import json
import re
import subprocess
from fnmatch import fnmatch
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from languages import Language
from wire import ChangeFacts, ChangeSource, FileFacts, FileSource, RepoFacts

_NUMSTAT = re.compile(r"^(\d+|-)\t(\d+|-)\t(.+)$")
_PACKAGE_NAME = re.compile(r'^\s*name\s*=\s*"([^"]+)"', re.MULTILINE)
_WORKSPACE_TABLE = re.compile(r'^\[workspace\]', re.MULTILINE)
_MEMBERS_BLOCK = re.compile(r'^members\s*=\s*\[(.*?)\]', re.MULTILINE | re.DOTALL)
_MAX_FILE_BYTES = 512 * 1024


def _git(repo: Path, *args: str) -> str:
    """Run one read-only git command in `repo` and return its stdout.

    Raises:
        subprocess.CalledProcessError: If git exits non-zero.
    """
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _build_unit(repo: Path, path: str) -> tuple[str | None, str | None]:
    """Return the nearest enclosing package name and the root of the workspace building it.

    Environment knowledge the service cannot derive from a diff: it needs the checkout to walk up to
    a manifest. Supplied as facts so the service can compose a scoped test command, and place it, in
    a repository that holds more than one workspace.

    The package is the first manifest above the file that names one. The workspace root is the
    highest manifest that declares `[workspace]` and is not excluded by it, which is where that
    package's `cargo test` has to run from.
    """
    package: str | None = None
    directory = (repo / path).parent
    while True:
        manifest = directory / "Cargo.toml"
        if manifest.is_file():
            content = manifest.read_text(errors="replace")
            if package is None:
                match = _PACKAGE_NAME.search(content)
                if match is not None:
                    package = match.group(1)
            if _WORKSPACE_TABLE.search(content) and _is_member(content, repo, directory, path):
                root = str(directory.relative_to(repo)) or "."
                return package, "." if root == "" else root
        if directory == repo or directory.parent == directory:
            return package, None
        directory = directory.parent


def _is_member(manifest: str, repo: Path, workspace: Path, path: str) -> bool:
    """Whether a workspace actually builds the package owning `path`.

    Nearness is not membership. This repository's root workspace declares `members = ["shared",
    "functions/*"]` and says nothing at all about `agent-platform/`, which is its own workspace — so
    treating the nearest enclosing `[workspace]` as the owner sends `cargo test -p mo-agent-runtime`
    to a workspace that has never heard of it, and the probe fails with "package not found" rather
    than a verdict.

    A manifest with no `members` list is a single-package workspace, which owns what is beneath it.
    """
    members = _MEMBERS_BLOCK.search(manifest)
    if members is None:
        return True
    relative = str((repo / path).parent.relative_to(workspace))
    return any(
        fnmatch(relative, pattern) or relative.startswith(pattern.rstrip("*").rstrip("/") + "/")
        for pattern in re.findall(r'"([^"]+)"', members.group(1))
    )


_PR_FIELDS = "number,title,body,mergedAt,mergeCommit,files,labels,closingIssuesReferences"
"""Deliberately without `commits`: that field drags an authors connection into the query, and
GitHub refuses the whole listing past its 500,000-node cap even at 50 pull requests. The
commit count is needed only to resolve a rebase merge, so it is fetched then, per PR."""
_PROSE_CACHE: dict[tuple[str, str], dict[str, str]] = {}
"""Prose per (repo, change) gathered while listing pull requests, so phase 2 can send it without a
second forge call per change."""


def repo_facts(
    repo: Path,
    *,
    repo_name: str,
    language: str,
    test_command: str,
    branch: str,
    limit: int,
    language_contract: Language | None = None,
    setup_command: str | None = None,
) -> RepoFacts:
    """Describe the last `limit` merged changes without disclosing any file content.

    Reads the forge's merged pull requests when it can, and falls back to first-parent git history
    when it cannot. The distinction matters: git history alone recognizes only squash merges, so on
    a repository that merges or rebases it offers almost nothing and says nothing about why.
    """
    try:
        return _facts_from_pull_requests(
            repo, repo_name=repo_name, language=language, test_command=test_command,
            branch=branch, limit=limit, language_contract=language_contract, setup_command=setup_command,
        )
    except _ForgeUnavailable as failure:
        facts = _facts_from_git_log(
            repo, repo_name=repo_name, language=language, test_command=test_command,
            branch=branch, limit=limit, language_contract=language_contract, setup_command=setup_command,
        )
        return replace(facts, source_note=str(failure)[:300])


class _ForgeUnavailable(RuntimeError):
    """The forge could not be asked — no `gh`, not authenticated, no remote, or it answered garbage."""


def _facts_from_pull_requests(repo: Path, **named: object) -> RepoFacts:
    """The merged record as the forge sees it: one entry per pull request, whatever the strategy."""
    branch = str(named["branch"])
    limit = int(named["limit"])  # type: ignore[call-overload]
    try:
        listing = subprocess.run(
            # `-R` names the repository whose pull requests are the merged record. A fork's own PR
            # list is empty, but its git history holds every upstream merge commit — so a customer
            # mining a fork declares `repo = "upstream/name"` and the export still comes from the
            # local clone.
            ["gh", "pr", "list", "-R", str(named["repo_name"]), "--state", "merged", "--base", branch,
             "--limit", str(limit), "--json", _PR_FIELDS],
            cwd=repo, capture_output=True, text=True, timeout=300, check=True,
        ).stdout
        pulls = json.loads(listing)
    except subprocess.CalledProcessError as failure:
        # The forge's own words, not Python's: "exit status 1" tells a user nothing, the stderr
        # names the auth, the rate limit, or the field the query could not serve.
        raise _ForgeUnavailable(f"gh pr list failed: {(failure.stderr or '').strip() or failure}") from failure
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as failure:
        raise _ForgeUnavailable(f"gh pr list: {failure}") from failure
    if not isinstance(pulls, list):
        raise _ForgeUnavailable("unexpected listing shape")

    changes: list[ChangeFacts] = []
    unresolved = 0
    contract = named["language_contract"]
    for pull in pulls:
        merge = (pull.get("mergeCommit") or {}).get("oid")
        paths = [entry["path"] for entry in pull.get("files") or []]
        if not merge or not paths:
            continue
        number = int(pull["number"])
        unit = _resolve_unit(repo, merge, set(paths), lambda: _commit_count(repo, number, str(named["repo_name"])))
        if unit is None:
            unresolved += 1
            continue
        parent, reference = unit
        files = []
        for entry in pull["files"]:
            package, workspace_root = (None, None)
            if contract is not None and contract.package_scoped:  # type: ignore[attr-defined]
                package, workspace_root = _build_unit(repo, entry["path"])
            files.append(FileFacts(
                path=entry["path"], insertions=int(entry.get("additions", 0)),
                deletions=int(entry.get("deletions", 0)), package=package, workspace_root=workspace_root,
            ))
        changes.append(ChangeFacts(
            change_id=reference, parent=parent, title=pull["title"], merged_at=pull["mergedAt"],
            files=files, number=int(pull["number"]), labels=[l["name"] for l in pull.get("labels") or []],
        ))
        _PROSE_CACHE[(str(repo), reference)] = _pull_prose(pull)
    return RepoFacts(
        repo=str(named["repo_name"]), forge="github", language=str(named["language"]),
        test_command=str(named["test_command"]), changes=changes,
        setup_command=named["setup_command"],  # type: ignore[arg-type]
        source="github-prs", unresolved_changes=unresolved,
    )


def _resolve_unit(
    repo: Path, merge: str, pr_paths: set[str], commit_count: Callable[[], int | None]
) -> tuple[str, str] | None:
    """Pin a pull request to the (parent, reference) commits whose diff IS the pull request.

    Three merge strategies leave three different shapes on the base branch, and the forge does not
    say which was used. So candidates are tried and VERIFIED rather than inferred:

    - squash: one commit; its first parent is the base tip.
    - merge commit: two parents; the first is the base tip and the diff against it is the PR.
    - rebase: the PR's commits replayed onto the base; `mergeCommit` is the last of them and the
      base tip is `mergeCommit~<count>`.

    A candidate is accepted only when `git diff --name-only candidate merge` touches exactly the
    files the pull request says it touched. A parent that "looks right" but yields a subset of the
    PR — a rebase read as a squash — would produce a task missing most of its own change.
    """
    try:
        parents = _git(repo, "rev-list", "--parents", "-n1", merge).split()[1:]
    except subprocess.CalledProcessError:
        # The forge names a merge commit the clone cannot see — merged into a base branch since
        # deleted, or history rewritten under it. That is one unresolvable PR to count, not a reason
        # to abandon the other three hundred: a single unreachable commit ended uber-go/zap's whole
        # mining run before this guard existed.
        return None
    if not parents:
        return None
    # Exact equality is what a correct parent yields, and it is tried first for every candidate.
    # A strict SUBSET is ambiguous: it is what a rebase looks like when misread as a squash (only the
    # last commit's files), and also what an honest squash looks like when one of the PR's files
    # netted out by merge time (observed: a PR listing `Makefile` that the merged diff never
    # touches). So a subset is accepted only where a rebase is structurally impossible — a
    # two-parent merge, or a PR the forge confirms carried a single commit. An unknown count is
    # left unresolved rather than guessed; a task built from the tail of a PR would be a task
    # missing most of its own change.
    first = _matches(repo, parents[0], merge, pr_paths)
    if first is not None:
        return first, merge
    count = commit_count() if len(parents) == 1 else 1
    if len(parents) == 1 and count is not None and count > 1:
        rebased = _matches(repo, f"{merge}~{count}", merge, pr_paths)
        if rebased is not None:
            return rebased, merge
        # A rebase's `~count` diff can only ever touch the PR's own files. If it reaches files the
        # PR never listed, those commits are unrelated history and this is a squash — one whose
        # file list simply overstates the merged diff — so its first parent may be taken as a subset.
        if not _reaches_outside(repo, f"{merge}~{count}", merge, pr_paths):
            return None
    if count is not None or len(parents) == 2:
        partial = _matches(repo, parents[0], merge, pr_paths, allow_subset=True)
        if partial is not None:
            return partial, merge
    return None


def _reaches_outside(repo: Path, candidate: str, merge: str, pr_paths: set[str]) -> bool:
    """Whether the diff from `candidate` to `merge` touches any file the PR did not list."""
    try:
        resolved = _git(repo, "rev-parse", "--verify", f"{candidate}^{{commit}}").strip()
        touched = set(_git(repo, "diff", "--name-only", resolved, merge).split("\n")) - {""}
    except subprocess.CalledProcessError:
        return False
    return bool(touched - pr_paths)


def _matches(
    repo: Path, candidate: str, merge: str, pr_paths: set[str], *, allow_subset: bool = False
) -> str | None:
    """The resolved candidate sha if its diff to `merge` is the PR's file list, else `None`.

    With `allow_subset`, a non-empty diff touching only PR-listed files also passes. A diff touching
    a file the PR never listed never does: that is the signature of the wrong parent.
    """
    try:
        resolved = _git(repo, "rev-parse", "--verify", f"{candidate}^{{commit}}").strip()
        touched = set(_git(repo, "diff", "--name-only", resolved, merge).split("\n")) - {""}
    except subprocess.CalledProcessError:
        return None
    if touched == pr_paths:
        return resolved
    if allow_subset and touched and touched < pr_paths:
        return resolved
    return None


def _commit_count(repo: Path, number: int, repo_name: str) -> int | None:
    """How many commits a pull request carried, or `None` when the forge will not say."""
    try:
        out = subprocess.run(
            ["gh", "pr", "view", str(number), "-R", repo_name, "--json", "commits", "--jq", ".commits|length"],
            cwd=repo, capture_output=True, text=True, timeout=60, check=True,
        ).stdout.strip()
        return int(out)
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def _pull_prose(pull: dict) -> dict[str, str]:
    prose = {"change_title": pull.get("title") or "", "change_body": pull.get("body") or ""}
    linked = pull.get("closingIssuesReferences") or []
    if linked:
        prose["issue_title"] = linked[0].get("title") or ""
        prose["issue_body"] = linked[0].get("body") or ""
    return prose


def _facts_from_git_log(repo: Path, **named: object) -> RepoFacts:
    """First-parent history: sees squash merges only. The fallback, and reported as such."""
    branch = str(named["branch"])
    limit = int(named["limit"])  # type: ignore[call-overload]
    contract = named["language_contract"]
    log = _git(repo, "log", "--first-parent", f"-n{limit}", "--format=%H%x00%P%x00%cI%x00%s", branch)
    changes: list[ChangeFacts] = []
    unreadable = 0
    for line in log.splitlines():
        change_id, parents, merged_at, title = line.split("\x00", 3)
        parent_list = parents.split()
        # A merge commit has no single tree the change was applied to, so this reader cannot give it
        # an honest parent_commit; the pull-request reader above can.
        if len(parent_list) != 1:
            continue
        try:
            numstat = _git(repo, "show", "--numstat", "--format=", change_id)
        except subprocess.CalledProcessError:
            unreadable += 1
            continue
        files: list[FileFacts] = []
        for entry in numstat.splitlines():
            match = _NUMSTAT.match(entry)
            if match is None:
                continue
            insertions, deletions, path = match.groups()
            if insertions == "-" or deletions == "-":
                continue  # binary
            package, workspace_root = (None, None)
            if contract is not None and contract.package_scoped:  # type: ignore[attr-defined]
                package, workspace_root = _build_unit(repo, path)
            files.append(FileFacts(path=path, insertions=int(insertions), deletions=int(deletions),
                                   package=package, workspace_root=workspace_root))
        if files:
            changes.append(ChangeFacts(change_id=change_id, parent=parent_list[0], title=title,
                                       merged_at=merged_at, files=files))
    return RepoFacts(
        repo=str(named["repo_name"]), forge="github", language=str(named["language"]),
        test_command=str(named["test_command"]), changes=changes,
        setup_command=named["setup_command"],  # type: ignore[arg-type]
        source="git-log", unreadable_changes=unreadable,
    )


def _blob(repo: Path, commit: str, path: str) -> str | None:
    """Return a file's content at `commit`, or `None` when absent or too large to send."""
    try:
        size = _git(repo, "cat-file", "-s", f"{commit}:{path}").strip()
    except subprocess.CalledProcessError:
        return None
    if int(size) > _MAX_FILE_BYTES:
        return None
    return _git(repo, "show", f"{commit}:{path}")


def _prose(repo: Path, change_id: str) -> dict[str, str]:
    """Collect what humans wrote about a change.

    The commit subject and body are always available. The linked issue is fetched through the forge
    when its CLI is authenticated, and quietly skipped when it is not — a task whose prompt is
    thinner is worse, but an unauthenticated runner should still produce one.
    """
    cached = _PROSE_CACHE.get((str(repo), change_id))
    if cached is not None:
        return cached
    subject = _git(repo, "log", "-1", "--format=%s", change_id).strip()
    body = _git(repo, "log", "-1", "--format=%b", change_id).strip()
    prose = {"change_title": subject, "change_body": body}
    number = re.search(r"\(#(\d+)\)\s*$", subject)
    if number is None:
        return prose
    try:
        linked = subprocess.run(
            [
                "gh", "pr", "view", number.group(1),
                "--json", "closingIssuesReferences",
                "--jq", ".closingIssuesReferences[0] | [.title, .body] | @tsv",
            ],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return prose
    if linked:
        title, _, issue_body = linked.partition("\t")
        prose["issue_title"] = title
        prose["issue_body"] = issue_body.replace("\\n", "\n")
    return prose


def change_source(repo: Path, facts: RepoFacts, change_ids: list[str]) -> list[ChangeSource]:
    """Send both sides of every file the named changes touched, plus their human prose.

    Args:
        repo: Local checkout.
        facts: Phase-1 facts, for the path list of each change.
        change_ids: Exactly the changes the service selected.

    Returns:
        One record per requested change, in the order requested.
    """
    by_id = {change.change_id: change for change in facts.changes}
    sources: list[ChangeSource] = []
    for change_id in change_ids:
        change = by_id[change_id]
        files = [
            FileSource(
                path=file.path,
                parent_content=_blob(repo, change.parent, file.path),
                child_content=_blob(repo, change_id, file.path),
            )
            for file in change.files
        ]
        sources.append(ChangeSource(change_id=change_id, files=files, prose=_prose(repo, change_id)))
    return sources


# --- conventions: what the repository says about how code should look ---------------------------

_CONVENTION_FILES = (
    "CONTRIBUTING.md", ".github/CONTRIBUTING.md", ".github/PULL_REQUEST_TEMPLATE.md", "PULL_REQUEST_TEMPLATE.md",
    "AGENTS.md", "CLAUDE.md", ".editorconfig",
    ".golangci.yml", ".golangci.yaml", "rustfmt.toml", ".rustfmt.toml", "clippy.toml", "ruff.toml", ".ruff.toml",
    "setup.cfg", ".flake8", ".eslintrc.json", ".eslintrc.js", "eslint.config.js", ".prettierrc", "biome.json",
    ".swiftlint.yml", "detekt.yml", ".scalafmt.conf", "analysis_options.yaml", ".rubocop.yml", "phpcs.xml",
)
_CONVENTION_FILE_CAP = 24_000
"""Bytes per file. Enough for any contributing guide; a 70 KB user manual is not a convention."""
_COMMENT_CAP = 400


def convention_files(repo: Path) -> dict[str, str]:
    """The repository's written rules, by path: contributing guide, PR template, agent instructions,
    lint and format configuration. Only files that exist and are not enormous."""
    found: dict[str, str] = {}
    for name in _CONVENTION_FILES:
        path = repo / name
        if path.is_file() and path.stat().st_size <= _CONVENTION_FILE_CAP:
            try:
                found[name] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
    return found


def review_comments(repo: Path, repo_name: str, numbers: list[int], limit: int = 400) -> list[dict]:
    """Inline review comments people left on merged pull requests — the conventions of this
    repository stated where they mattered. Bots are dropped (their comments are generic and dwarf
    the humans'), as are acknowledgements too short to carry a rule."""
    collected: list[dict] = []
    for number in numbers:
        try:
            out = subprocess.run(
                ["gh", "api", f"repos/{repo_name}/pulls/{number}/comments", "--paginate",
                 "--jq", '.[] | select(.user.type == "User") | {path, body, author: .user.login} | @json'],
                cwd=repo, capture_output=True, text=True, timeout=60, check=True,
            ).stdout
        except (subprocess.SubprocessError, OSError):
            continue
        for line in out.splitlines():
            if not line.strip():
                continue
            comment = json.loads(line)
            body = " ".join((comment.get("body") or "").split())
            if len(body) < 40 or body.lower().startswith(("done", "thanks", "thank you", "ok", "lgtm", "@")):
                continue
            collected.append({"number": number, "path": comment.get("path") or "", "author": comment["author"],
                              "body": body[:_COMMENT_CAP]})
            if len(collected) >= limit:
                return collected
    return collected
