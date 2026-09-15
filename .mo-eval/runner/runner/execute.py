"""Execute work orders. The runner's whole intelligence, and there is deliberately very little of it.

Three primitives — export a tree, apply a patch, probe with the repository's own test command — and
a report of how each exited. Nothing here knows what a task is, which patch is a scaffold, or which
way a probe is supposed to go. That is not an accident of the spike: it is what lets the service keep
its method while the work runs on a machine the customer controls.

The bound on a probe is structural. `Step.args` are appended to the `test_command` the repository
itself declares, so the service chooses which tests run and never what runs them.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from wire import OrderResult, Step, StepResult, WorkOrder

_OUTPUT_TAIL_BYTES = 4000
_EXPORT_TIMEOUT_SECONDS = 600
_PROBE_TIMEOUT_SECONDS = 3600
_SETUP_TIMEOUT_SECONDS = 1800


class OrderError(RuntimeError):
    """The runner could not carry an order out — distinct from a step that ran and exited non-zero."""


def _host_target() -> str:
    """Resolve this machine's Rust host triple.

    Environment knowledge, so the runner answers it. A repository whose test command needs the triple
    writes `{host_target}` in its declared command and the runner fills it in; the service never
    supplies it, because the service does not know what machine this is.
    """
    probe = subprocess.run(["rustc", "-vV"], check=True, capture_output=True, text=True)
    for line in probe.stdout.splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ").strip()
    raise OrderError("rustc reported no host triple")


def _expand(command: str) -> list[str]:
    """Split the declared test command, expanding the placeholders a runner is allowed to fill."""
    return [part.replace("{host_target}", _host_target()) for part in shlex.split(command)]


class Runner:
    """Executes orders against one checkout, materializing each order in its own directory."""

    def __init__(
        self,
        repo: Path,
        workspaces: Path,
        test_command: str,
        env: dict[str, str],
        setup_command: str | None = None,
    ) -> None:
        """
        Args:
            repo: The local checkout orders export trees from.
            workspaces: Directory each order's workspace is created beneath.
            test_command: The repository's declared command — the allow-list probes are bound to.
            setup_command: The repository's declared command for making a checkout testable, if any.
            env: Extra environment for probes (a shared `CARGO_TARGET_DIR` keeps builds incremental
                across candidates, which is the difference between minutes and hours).
        """
        self.repo = repo
        self.workspaces = workspaces
        self.test_command = test_command
        self.setup_command = setup_command
        self.env = env

    def run(self, order: WorkOrder) -> OrderResult:
        """Carry out one order, reporting each step's exit code and nothing more.

        Returns:
            The step results, or an `OrderResult` carrying `error` when the order could not be
            carried out at all. A failure to run is never reported as a step verdict.
        """
        workspace = self.workspaces / order.order_id
        results: list[StepResult] = []
        try:
            for step in order.steps:
                results.append(self._step(step, workspace))
        except OrderError as failure:
            return OrderResult(order_id=order.order_id, steps=results, error=str(failure))
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
        return OrderResult(order_id=order.order_id, steps=results)

    def _step(self, step: Step, workspace: Path) -> StepResult:
        started = time.monotonic()
        if step.op == "export":
            self._export(step, workspace)
            return self._result(step, 0, started, "")
        if step.op == "setup":
            return self._setup(step, workspace, started)
        if step.op == "apply":
            return self._apply(step, workspace, started)
        if step.op == "probe":
            return self._probe(step, workspace, started)
        raise OrderError(f"unknown step op {step.op!r}")

    def _export(self, step: Step, workspace: Path) -> None:
        """Materialize a commit's tree into a clean directory.

        A tree export, not a clone: the workspace gets no history and no remote, so nothing in it can
        be used to look up how the change was actually made. mo-eval's own t-suite materializer takes
        the same approach, for the same reason.
        """
        if step.commit is None:
            raise OrderError("export step carries no commit")
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        archive = subprocess.Popen(
            ["git", "archive", "--format=tar", step.commit],
            cwd=self.repo,
            stdout=subprocess.PIPE,
        )
        extract = subprocess.run(
            ["tar", "-xf", "-", "-C", str(workspace)],
            stdin=archive.stdout,
            capture_output=True,
            text=True,
            timeout=_EXPORT_TIMEOUT_SECONDS,
        )
        if archive.stdout is not None:
            archive.stdout.close()
        if archive.wait() != 0 or extract.returncode != 0:
            raise OrderError(f"could not export {step.commit}: {extract.stderr[-400:]}")
        _freshen(workspace)
        _own_repository(workspace)

    def _setup(self, step: Step, workspace: Path, started: float) -> StepResult:
        """Make the exported checkout testable, using the command the REPOSITORY declared.

        The step carries no command and cannot: the service asks for setup and the runner decides
        what that means. A repository that declares none is assumed to build from its source, which
        is true of every compiled language here and of none of the interpreted ones.
        """
        if self.setup_command is None:
            return self._result(step, 0, started, "no setup command declared")
        completed = subprocess.run(
            shlex.split(self.setup_command),
            cwd=workspace,
            capture_output=True,
            text=True,
            env={**os.environ, **self.env},
            timeout=_SETUP_TIMEOUT_SECONDS,
        )
        tail = (completed.stdout + completed.stderr)[-_OUTPUT_TAIL_BYTES:]
        return self._result(step, completed.returncode, started, tail)

    def _apply(self, step: Step, workspace: Path, started: float) -> StepResult:
        if step.patch is None:
            raise OrderError("apply step carries no patch")
        patch_file = workspace / ".mo-eval-order.patch"
        patch_file.write_text(step.patch)
        completed = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_file)],
            cwd=workspace,
            capture_output=True,
            text=True,
        )
        patch_file.unlink(missing_ok=True)
        tail = (completed.stdout + completed.stderr)[-_OUTPUT_TAIL_BYTES:]
        return self._result(step, completed.returncode, started, tail)

    def _probe(self, step: Step, workspace: Path, started: float) -> StepResult:
        """Run the repository's declared test command with the service's arguments appended."""
        if step.args is None:
            raise OrderError("probe step carries no args")
        directory = self._resolve(workspace, step.cwd)
        command = [*_expand(self.test_command), *step.args]
        try:
            completed = subprocess.run(
                command,
                cwd=directory,
                capture_output=True,
                text=True,
                env={**os.environ, **self.env},
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return self._result(step, 124, started, "probe timed out")
        tail = (completed.stdout + completed.stderr)[-_OUTPUT_TAIL_BYTES:]
        return self._result(step, completed.returncode, started, tail)

    def _resolve(self, workspace: Path, relative: str | None) -> Path:
        """Resolve a probe's working directory inside the workspace.

        A directory the service names is still a path the runner validates. Anything resolving
        outside the exported tree is refused rather than clamped, so a traversal is an error the
        operator sees instead of a silent relocation.

        Raises:
            OrderError: If the path escapes the workspace or does not exist.
        """
        if relative is None or relative == ".":
            return workspace
        resolved = (workspace / relative).resolve()
        if not resolved.is_relative_to(workspace.resolve()):
            raise OrderError(f"probe directory escapes the workspace: {relative!r}")
        if not resolved.is_dir():
            raise OrderError(f"probe directory does not exist: {relative!r}")
        return resolved

    def _result(self, step: Step, exit_code: int, started: float, tail: str) -> StepResult:
        return StepResult(
            step_id=step.step_id,
            exit_code=exit_code,
            duration_seconds=round(time.monotonic() - started, 2),
            output_tail=tail,
        )


def _own_repository(workspace: Path) -> None:
    """Make an exported tree its own Git repository, so `git apply` resolves paths against IT.

    Run inside a repository, `git apply` takes patch paths relative to that repository's root and
    silently ignores any that fall outside the current directory — exit 0, nothing applied. A
    workspace under `.mo-eval/out/` inside a customer's checkout is exactly that case: on GitHub
    Actions every scaffold "applied" with exit 0 and every graded test was then "not found" at the
    start state, which the judge read as "already passes". An empty repository in the workspace
    stops discovery at the workspace boundary.
    """
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True, capture_output=True)


def _freshen(workspace: Path) -> None:
    """Stamp every exported file with the current time.

    `git archive` gives each file its COMMIT's timestamp, so an exported tree from last week looks
    older than any build artifact produced since. Cargo decides a path dependency is fresh by
    comparing mtimes, so it then skips recompiling and links this tree's crate against a `.rmeta`
    built from a different commit's source. The result is a compile error that belongs to neither
    tree — observed here as `missing fields ... in initializer of RecordedFact`, from a candidate
    whose reference state builds perfectly on its own.

    That failure is silent in the worst way: it does not look like a bug, it looks like a candidate
    that failed validation, so a real task is discarded and nothing anywhere reports a problem.

    The cost is a full rebuild per exported tree, which is the honest price of a correct verdict.
    Sharing one target directory across trees is the thing that was wrong, not the rebuild.
    """
    for path in workspace.rglob("*"):
        try:
            os.utime(path, None)
        except OSError:
            continue
