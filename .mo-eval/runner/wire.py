"""The runner/service protocol: every type that crosses the boundary, and the audit log of crossings.

The split is the product decision. Candidate selection, diff splitting, prompt authoring and oracle
construction never leave the service, because none of them need the customer's toolchain. Everything
that *does* need the toolchain — collecting git facts, running a test suite, running an agent — is
dumb enough to hand to a runner in the customer's CI.

So a runner never receives a "task". It receives `WorkOrder`s: check out this commit, apply this
patch, run this command, report the exit code. It is not told which patch is a scaffold and which is
a gold fix, which candidate survived validation, or why any commit was chosen. The judgment stays
server-side; the runner reports exit codes.

Every value defined here is JSON, and `Wire.crossing` records each one to disk. That makes the egress
claim auditable rather than asserted: after a run, `wire/` *is* the list of what left the repository.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

# --- phase 1: facts, carrying no source content ------------------------------------------------


@dataclass(frozen=True)
class FileFacts:
    """One file touched by one change, described without any of its content."""

    path: str
    """Repository-relative path."""
    insertions: int
    """Lines added, as the forge reports them."""
    deletions: int
    """Lines removed."""
    package: str | None
    """Build unit owning the path (the nearest Cargo.toml's name), or `None` when outside one.

    Supplied by the runner because it requires the checkout; used by the service to compose a
    scoped test command. Environment knowledge, not policy.
    """
    workspace_root: str | None
    """Repository-relative directory of the workspace that package belongs to, or `None`.

    A repository is not always one workspace. This one holds several — the root workspace builds for
    wasm, and `agent-platform/` is its own native workspace — so a test command run from the
    repository root cannot see a package in another. The runner reports where each package is built
    from because only the checkout knows; the service puts probes there.
    """


@dataclass(frozen=True)
class ChangeFacts:
    """One merged change, described well enough to *select* it and no better.

    Deliberately carries no diff and no prose body: phase 1 is meant to be cheap to produce, cheap to
    send, and uninteresting if intercepted. Only the changes that survive selection have their source
    requested in phase 2.
    """

    change_id: str
    """Forge-native identity of the merged change (here, the squashed commit sha)."""
    parent: str
    """The commit the change was applied to — a task's `parent_commit`."""
    title: str
    """Subject line. Prose, but one line of it, and it is what makes selection legible in a log."""
    merged_at: str
    """ISO-8601 merge timestamp, for recency-weighted selection."""
    files: list[FileFacts]
    """Every path the change touched."""
    number: int | None = None
    """The forge's pull-request number, when the change was collected as one."""
    labels: list[str] = field(default_factory=list)
    """The pull request's labels — the cheapest available category signal."""


@dataclass(frozen=True)
class RepoFacts:
    """What a runner reports about its repository before any source is requested."""

    repo: str
    """`owner/name`, the identity a generated task records."""
    forge: str
    """Which adapter produced this (`github`, later `gitlab`, …). The only host-specific field."""
    language: str
    """Declared in the repository's own `.mo-eval/config.yml`."""
    test_command: str
    """The customer's declared test command. Orders may fill in ARGUMENTS to this and nothing else —
    an allow-list, so the service cannot ask the runner to execute a command of its choosing."""
    changes: list[ChangeFacts]
    """Merged changes offered as candidates, newest first."""
    setup_command: str | None = None
    """The customer's declared command for making a fresh checkout testable — `poetry install`,
    `npm ci`, `bundle install`.

    Declared by the repository and run by the runner. An order can ask for setup; it cannot say what
    setup is, which keeps the same allow-list as `test_command`: the service never names a command.

    A compiled language rarely needs one — `cargo test` builds from the exported source — which is
    why its absence went unnoticed until an interpreted language was tried. Without it an
    interpreted repository is tested against whatever happens to be installed in the runner's
    ambient environment, which is the environment of some OTHER commit."""
    source: str = "git-log"
    """How the merged record was read: `github-prs` (the forge's pull requests, any merge strategy)
    or `git-log` (first-parent history, which only sees squash merges). Reported because the two
    see different repositories, and a suite mined from the wrong one is thin for a reason the user
    cannot otherwise discover."""
    worker_image: str | None = None
    """The repository's declared worker image, pinned by digest, for containerized agent runs."""
    source_note: str = ""
    """Why the record was read the way it was, when that was not the first choice — the forge's own
    error text, trimmed. A fallback that does not say why it fell back is a failure that reads as a
    thin repository."""
    unresolved_changes: int = 0
    """Pull requests whose merged unit could not be pinned to a (parent, reference) pair whose diff
    matches the PR's own file list. Counted, not dropped silently."""
    unreadable_changes: int = 0
    """How many merged changes the runner could not describe, and therefore never offered.

    Reported rather than swallowed. A repository with a damaged object, a rewritten history or a
    partial clone will refuse to show some commits, and silently skipping them would shrink the
    candidate pool with no trace — the same shape as every other under-collection failure here. A
    non-zero value means the offered list is not the whole history.
    """


# --- phase 2: source for selected changes only ---------------------------------------------------


@dataclass(frozen=True)
class RejectionSummary:
    """How many changes one reason turned away. Coarse on purpose: enough for a user to learn why a
    repository yields little, not enough to reconstruct the selection model."""

    reason: str
    count: int


@dataclass(frozen=True)
class SourceRequest:
    """The service naming which changes it wants source for. The selection model stays server-side;
    only its output crosses — plus a tally of why the rest did not qualify, because a repository that
    yields nothing must be able to tell "nothing qualified" from "the service did not understand me"."""

    change_ids: list[str]
    contract: str = ""
    """The language/framework contract the service resolved for this repository, e.g. `go+ginkgo`."""
    rejections: list[RejectionSummary] = field(default_factory=list)
    """Selection rejections, most common first."""


@dataclass(frozen=True)
class FileSource:
    """Both sides of one changed file. Sending whole files rather than a diff keeps the split exact;
    a production runner can send hunks with context when egress volume matters."""

    path: str
    parent_content: str | None
    """Content at the parent commit, or `None` when the change added the file."""
    child_content: str | None
    """Content after the change, or `None` when the change deleted it."""


@dataclass(frozen=True)
class ChangeSource:
    """The source and prose for one selected change. This is the only payload carrying repository
    content, and it covers the selected changes alone."""

    change_id: str
    files: list[FileSource]
    prose: dict[str, str]
    """Free-form human context (change title, linked issue title/body) used to author a prompt."""


# --- phase 3: work orders, and what comes back ---------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One primitive a runner knows how to perform.

    `export` materializes a commit's tree into a clean directory; `apply` applies a patch to it;
    `probe` runs the declared test command with service-supplied arguments and reports how it exited.
    A probe carries no expectation — the runner is never told which way it is supposed to go.
    """

    op: str
    """`export`, `setup`, `apply`, or `probe`."""
    step_id: str
    """Identifies this step's result in the report."""
    commit: str | None = None
    """`export`: the tree to materialize."""
    patch: str | None = None
    """`apply`: a unified diff."""
    cwd: str | None = None
    """`probe`: repository-relative directory to run in, for a repository of several workspaces.

    A path inside the exported tree, which the runner enforces; it is not a way to reach a command
    elsewhere on the machine.
    """
    args: list[str] | None = None
    """`probe`: arguments APPENDED to the repository's own declared `test_command`.

    Not a script, and not a command. The service can say which tests to run; it cannot say what to
    run them with. That makes the bound an allow-list the runner enforces structurally rather than a
    deny-list of things a script must not contain — the same posture this repository already takes
    with request-header forwarding.
    """


@dataclass(frozen=True)
class WorkOrder:
    """A unit of work for the runner, opaque as to purpose."""

    order_id: str
    """Correlates the result. The service keeps the mapping from order to candidate; the runner
    cannot reconstruct it."""
    steps: list[Step]


@dataclass(frozen=True)
class StepResult:
    """What one step did, with no interpretation attached."""

    step_id: str
    exit_code: int
    duration_seconds: float
    output_tail: str
    """Bounded trailing output. Diagnostic for the operator; the verdict is the exit code."""


@dataclass(frozen=True)
class OrderResult:
    order_id: str
    steps: list[StepResult]
    error: str | None = None
    """Set when the runner could not carry the order out at all, as distinct from a step that ran
    and exited non-zero. Conflating the two would let an infrastructure failure read as a verdict."""


@dataclass(frozen=True)
class OrdersResponse:
    """The orders to run, and a tally of the selected changes that could not become one — so a user
    sees "7 selected, 5 split away for X" rather than a silently shorter list."""

    orders: list[WorkOrder]
    rejections: list[RejectionSummary] = field(default_factory=list)


# --- phase 4: verdicts, and the task packages that earned one -----------------------------------


@dataclass(frozen=True)
class TaskPackage:
    """A validated task as files, in the corpus layout mo-eval already reads."""

    task_id: str
    files: dict[str, str]
    """`meta.json`, `prompt.txt`, `scaffold.patch`, `acceptance.sh` — by name, as text."""
    local_suite: dict[str, str] = field(default_factory=dict)
    """Files to lay over the task's start tree to make it a one-task `local-test-suite` bundle
    (`.mo-eval/suite.yaml`, `.mo-eval/prompt.md`, `.mo-eval/acceptance/acceptance.sh`). Empty when
    the repository declared no worker image."""


@dataclass(frozen=True)
class OrderVerdict:
    """What one order's exit codes meant, decided server-side."""

    change_id: str
    validated: bool
    detail: str
    task: TaskPackage | None = None
    """Present exactly when `validated`."""


@dataclass(frozen=True)
class VerdictReport:
    verdicts: list[OrderVerdict]


# --- phase 5: handing validated bundles to the service's storage, and asking for a run ----------


@dataclass(frozen=True)
class UploadRequest:
    """The runner naming the bundles it holds for one suite. The service answers with one presigned
    PUT per bundle, so the bytes go to storage directly and never through the service itself."""

    repo: str
    suite_id: str
    """Identifies this mining run of this repository; the runner chooses it (a timestamp + sha)."""
    task_ids: list[str]


@dataclass(frozen=True)
class UploadTargets:
    urls: dict[str, str]
    """Presigned PUT URL per task id, short-lived."""
    keys: dict[str, str]
    """Where each bundle will live, so a run can name it."""


@dataclass(frozen=True)
class ReviewComment:
    """One inline review comment a person left on a merged change — the repository's conventions,
    stated in its own words at the moment they mattered."""

    number: int
    path: str
    author: str
    body: str


@dataclass(frozen=True)
class ConventionSources:
    """What the conventions judge learns from: the repository's written rules and its review
    history. Collected by the runner, distilled on the service's side."""

    files: dict[str, str] = field(default_factory=dict)
    """Contributing guide, PR template, lint and format configuration, agent instructions — by path."""
    comments: list[ReviewComment] = field(default_factory=list)


@dataclass(frozen=True)
class RunRequest:
    """Ask the service to evaluate a suite: which arms, how many repeats. Recorded as a job; a lane
    picks it up. The runner never talks to the lane."""

    repo: str
    suite_id: str
    task_ids: list[str]
    arms: list[str]
    """Model routes, one arm each (`anthropic/claude-opus-5`, `momento/zai-org/GLM-5.3`)."""
    repeats: int = 1
    titles: dict[str, str] = field(default_factory=dict)
    """Task id -> human title, carried so results can be read without the packages."""
    conventions: ConventionSources | None = None
    """Absent: correctness only, no conventions score."""


@dataclass(frozen=True)
class RunTicket:
    run_id: str
    job_key: str
    results_prefix: str
    """Where `cells.json` and the reports will appear when the lane is done."""


# --- decoding ------------------------------------------------------------------------------------


def from_json(kind: type, data: Any) -> Any:
    """Rebuild a boundary type from the JSON `to_json` produced.

    Only what the wire uses: dataclasses, lists and dicts of them, optionals, and scalars. Unknown
    keys are refused rather than ignored — a runner and a service that disagree about a field should
    fail at the boundary, where the message names the field, not three stages later.

    Raises:
        ValueError: If `data` does not fit `kind`.
    """
    origin = get_origin(kind)
    if origin is Union or origin is UnionType:
        members = [member for member in get_args(kind) if member is not NoneType]
        if data is None:
            return None
        if len(members) != 1:
            raise ValueError(f"cannot decode into {kind}")
        return from_json(members[0], data)
    if origin is list:
        (item,) = get_args(kind)
        if not isinstance(data, list):
            raise ValueError(f"expected a list for {kind}, got {type(data).__name__}")
        return [from_json(item, entry) for entry in data]
    if origin is dict:
        _, value = get_args(kind)
        if not isinstance(data, dict):
            raise ValueError(f"expected an object for {kind}, got {type(data).__name__}")
        return {str(key): from_json(value, entry) for key, entry in data.items()}
    if is_dataclass(kind) and isinstance(kind, type):
        if not isinstance(data, dict):
            raise ValueError(f"expected an object for {kind.__name__}, got {type(data).__name__}")
        hints = get_type_hints(kind)
        known = {f.name for f in fields(kind)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{kind.__name__} does not have field(s) {sorted(unknown)}")
        values = {}
        for f in fields(kind):
            if f.name in data:
                values[f.name] = from_json(hints[f.name], data[f.name])
            elif f.default is MISSING and f.default_factory is MISSING:
                raise ValueError(f"{kind.__name__} is missing required field {f.name!r}")
        return kind(**values)
    if kind is Any or isinstance(data, kind):
        return data
    if kind is float and isinstance(data, int):
        return float(data)
    raise ValueError(f"expected {getattr(kind, '__name__', kind)}, got {type(data).__name__}")


# --- the audit log -------------------------------------------------------------------------------

_DIRECTIONS = {"up": "runner->service", "down": "service->runner"}


@dataclass
class Wire:
    """Records every payload that crosses the boundary, in order, as JSON on disk.

    The point is not debugging. A customer asking "what leaves our repository?" gets a directory of
    the literal answer instead of a paragraph of reassurance.
    """

    root: Path
    sequence: int = field(default=0)

    def crossing(self, direction: str, name: str, payload: Any) -> Any:
        """Record one payload and return it unchanged.

        Args:
            direction: `up` for runner->service, `down` for service->runner.
            name: Short label for the payload, used in the filename.
            payload: Any dataclass, or list of them, defined in this module.

        Returns:
            `payload`, so a call site can wrap the value it was already passing.

        Raises:
            ValueError: If `direction` is not a known direction.
        """
        if direction not in _DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}")
        self.sequence += 1
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{self.sequence:02d}-{direction}-{name}.json"
        path.write_text(json.dumps(to_json(payload), indent=2, sort_keys=True) + "\n")
        return payload

    def manifest(self) -> str:
        """Render the recorded crossings as a table, newest last."""
        rows = []
        for path in sorted(self.root.glob("*.json")):
            _, direction, name = path.stem.split("-", 2)
            rows.append(f"  {_DIRECTIONS[direction]:>18}  {name:<22} {path.stat().st_size:>9,} bytes")
        return "\n".join(rows)


def to_json(value: Any) -> Any:
    """Convert dataclasses (and containers of them) to plain JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list):
        return [to_json(item) for item in value]
    if isinstance(value, dict):
        return {key: to_json(item) for key, item in value.items()}
    return value
