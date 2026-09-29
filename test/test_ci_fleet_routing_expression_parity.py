"""Pin every inline fleet route and resolver consumer to its event policy.

Actor admission here is an availability decision: an unlisted actor or missing
repository variable gets a hosted runner, not a queued job the webhook rejects.
AWS webhook filtering remains the security boundary. Fast Gate computes this
inline because its gates must not depend on another job, and the only condition
they carry is the merge-queue push skip pinned below.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"

# The one `if:` a Fast Gate job carries: skip a push to main only while the
# repository variable MERGE_QUEUE_ENABLED is 'true', when the merge group
# already ran the gate on that exact tree.
_FAST_GATE_PUSH_SKIP_CLAUSE = "github.event_name != 'push' || vars.MERGE_QUEUE_ENABLED != 'true'"

_ACTOR_PREDICATE = "contains(fromJSON(vars.CODEBUILD_ACTOR_IDS || '[]'), github.actor_id)"
_CANONICAL_ROUTING_EXPR = (
    "${{ github.repository == 'kirodotdev/KiroCrew' && "
    + _ACTOR_PREDICATE
    + " && (github.event_name == 'push' || (github.event_name == 'pull_request' && "
    "(github.event.action == 'opened' || github.event.action == 'synchronize') && "
    "github.event.pull_request.head.repo.full_name == github.repository)) && "
    "format('codebuild-kirocrew-gha-linux-{0}-{1}', github.run_id, github.run_attempt) "
    "|| 'ubuntu-latest' }}"
)
# The workflows the merge queue runs. A merge_group run's actor is the person
# who queued the pull request, so the same actor predicate admits it; the event
# is named so a merge group's jobs reach the fleet instead of falling to hosted.
_CANONICAL_MERGE_GROUP_ROUTING_EXPR = (
    "${{ github.repository == 'kirodotdev/KiroCrew' && "
    + _ACTOR_PREDICATE
    + " && (github.event_name == 'push' || github.event_name == 'merge_group' || "
    "(github.event_name == 'pull_request' && "
    "(github.event.action == 'opened' || github.event.action == 'synchronize') && "
    "github.event.pull_request.head.repo.full_name == github.repository)) && "
    "format('codebuild-kirocrew-gha-linux-{0}-{1}', github.run_id, github.run_attempt) "
    "|| 'ubuntu-latest' }}"
)
_CANONICAL_PUSH_EXPR = (
    "${{ github.repository == 'kirodotdev/KiroCrew' && "
    + _ACTOR_PREDICATE
    + " && github.event_name == 'push' && "
    "format('codebuild-kirocrew-gha-linux-{0}-{1}', github.run_id, github.run_attempt) "
    "|| 'ubuntu-latest' }}"
)
_CANONICAL_DISPATCH_EXPR = (
    "${{ github.repository == 'kirodotdev/KiroCrew' && "
    + _ACTOR_PREDICATE
    + " && (github.event_name == 'push' || github.event_name == 'workflow_dispatch') && "
    "format('codebuild-kirocrew-gha-linux-{0}-{1}', github.run_id, github.run_attempt) "
    "|| 'ubuntu-latest' }}"
)
# The ratchet audit: push, manual dispatch, and the merge group, which is the
# integrated tree one step before it lands.
_CANONICAL_DISPATCH_MERGE_GROUP_EXPR = (
    "${{ github.repository == 'kirodotdev/KiroCrew' && "
    + _ACTOR_PREDICATE
    + " && (github.event_name == 'push' || github.event_name == 'workflow_dispatch' || "
    "github.event_name == 'merge_group') && "
    "format('codebuild-kirocrew-gha-linux-{0}-{1}', github.run_id, github.run_attempt) "
    "|| 'ubuntu-latest' }}"
)

# These lanes remain hosted: untrusted-content model execution, schedule/issue
# triggers, or an isolation contract tied to hosted paths. The scope-review
# generate job's action-code Write fence is one such path-sensitive contract.
_PERMANENT_EXCEPTIONS = {
    ("security-scope-review.yml", "generate"),
    ("security-scope-review.yml", "validate"),
    ("security-scope-review.yml", "publish"),
    ("issue-summary.yml", "summarize"),
    ("issue-triage.yml", "triage"),
    ("ai-review-human-override.yml", "record"),
    ("disposition-deferral-check.yml", "validate-deferral"),
    ("nightly.yml", "version"),
    ("connections-l0.yml", "probe"),
    ("memory-benchmark.yml", "accept"),
    ("fix-loop-analysis.yml", "metrics"),
    ("fix-loop-analysis.yml", "analyze"),
    ("deferred-findings-audit.yml", "audit"),
    ("add-contributor.yml", "add"),
    ("ship-report.yml", "report"),
    ("first-principles-review.yml", "first-principles-review"),
    ("ux-review.yml", "ux-review"),
    ("design-review.yml", "design-review"),
    ("code-review.yml", "sast"),
    ("ci-runner-watchdog.yml", "watchdog"),
}

# Fixed expectations, never inferred from the workflow contents: a route
# silently removed or a new unreviewed fleet job must fail the inventory check.
_EXPECTED_ROUTED_JOBS = {
    ("fast-gate.yml", "vendor-manifest"),
    ("fast-gate.yml", "brand-lint"),
    ("fast-gate.yml", "comment-history-lint"),
    ("fast-gate.yml", "focus-cue-lint"),
    ("fast-gate.yml", "feature-map-lint"),
    ("fast-gate.yml", "changelog-history"),
    ("fast-gate.yml", "decision-ledger-history"),
    ("fast-gate.yml", "builtin-skill-scope"),
    ("fast-gate.yml", "loop-bound-locks"),
    ("fast-gate.yml", "testpaths-coverage"),
    ("fast-gate.yml", "cwd-relative-repo-reads"),
    ("fast-gate.yml", "harness-parity"),
    ("fast-gate.yml", "memory-store-seam"),
    ("fast-gate.yml", "docs-lint"),
    ("build.yml", "build-wheel"),
    ("build.yml", "desktop-matrix"),
    ("main-ratchet-audit.yml", "ratchet-gates"),
    ("main-ratchet-audit.yml", "frontend-ceiling"),
    ("main-ratchet-audit.yml", "bundle-ceiling"),
    ("main-ratchet-audit.yml", "report"),
    ("release.yml", "version"),
    ("release.yml", "resolve-promotion"),
    ("release.yml", "stable-gate"),
    ("release.yml", "github-release"),
    ("release.yml", "record-promotion"),
    ("pages.yml", "build"),
    ("pages.yml", "deploy"),
    ("cross-platform.yml", "cross-platform"),
    ("dependency-review.yml", "license-gate"),
    ("pr-scope.yml", "pr-scope"),
    ("screenshot-evidence.yml", "screenshot-evidence"),
    ("macos-on-demand.yml", "decide"),
    ("ci.yml", "changes"),
    ("ci.yml", "await-fast-gate"),
    ("code-review.yml", "autosde-rules"),
    ("code-review.yml", "inclusive-language"),
    ("code-review.yml", "pr-hygiene"),
    ("pr-merge-conflict-label.yml", "label"),
    ("build-wheel.yml", "build-wheel"),
    ("dependency-vulnerability.yml", "audit-production-dependencies"),
}
_PUSH_ONLY_WORKFLOWS = {
    "release.yml",
    "pr-merge-conflict-label.yml",
    "build-wheel.yml",
    "dependency-vulnerability.yml",
}
_DISPATCH_WORKFLOWS = {"pages.yml"}
_DISPATCH_MERGE_GROUP_WORKFLOWS = {"main-ratchet-audit.yml"}
# The fleet-routed workflows that declare a `merge_group` trigger.
_MERGE_GROUP_WORKFLOWS = {"ci.yml", "fast-gate.yml", "build.yml"}
# merge_group-triggered workflows with no fleet route: the queue's required
# check (a hosted poll), the OIDC content scan (a reusable-workflow call) and
# the hosted Python client suite.
_MERGE_GROUP_HOSTED_WORKFLOWS = {
    "merge-queue-readiness.yml",
    "internal-content-scan-gate.yml",
    "client-py.yml",
}

# Resolver consumers pin their complete expressions, including hosted fallbacks.
# Backend shards, backend lint, frontend tests and bundle size use large Linux
# compute; Windows and boot-matrix consumers pin their respective OS mappings.
# A literal fleet label must never bypass the shared actor/event/fork admission.
_CANONICAL_CONSUMER_EXPR = "${{ needs.changes.outputs.linux_runner || 'ubuntu-latest' }}"
_CANONICAL_CONSUMER_EXPR_LARGE = (
    "${{ needs.changes.outputs.linux_runner_large || 'ubuntu-latest' }}"
)
_CANONICAL_WINDOWS_CONSUMER_EXPR = "${{ needs.changes.outputs.windows_runner || 'windows-latest' }}"
_CANONICAL_BOOT_MATRIX_EXPR = (
    "${{ matrix.os == 'ubuntu-latest' && "
    "(needs.changes.outputs.linux_runner_large || 'ubuntu-latest') || "
    "matrix.os == 'windows-latest' && "
    "(needs.changes.outputs.windows_runner || 'windows-latest') || matrix.os }}"
)
_EXPECTED_RESOLVER_CONSUMER_JOBS = {
    ("ci.yml", "backend-lint"): _CANONICAL_CONSUMER_EXPR_LARGE,
    ("ci.yml", "backend-test"): _CANONICAL_CONSUMER_EXPR_LARGE,
    ("ci.yml", "backend-test-windows"): _CANONICAL_WINDOWS_CONSUMER_EXPR,
    ("ci.yml", "backend-test-windows-fail-closed"): _CANONICAL_WINDOWS_CONSUMER_EXPR,
    ("ci.yml", "backend-test-crew-container"): _CANONICAL_CONSUMER_EXPR,
    ("ci.yml", "coverage-combine"): _CANONICAL_CONSUMER_EXPR,
    ("ci.yml", "coverage-gate"): _CANONICAL_CONSUMER_EXPR,
    ("ci.yml", "frontend-lint"): _CANONICAL_CONSUMER_EXPR,
    ("ci.yml", "lockfile-engines-floor"): _CANONICAL_CONSUMER_EXPR,
    ("ci.yml", "cfn-lint"): _CANONICAL_CONSUMER_EXPR,
    ("ci.yml", "electron-test"): _CANONICAL_CONSUMER_EXPR,
    ("ci.yml", "electron-test-windows"): _CANONICAL_WINDOWS_CONSUMER_EXPR,
    ("ci.yml", "frontend-test"): _CANONICAL_CONSUMER_EXPR_LARGE,
    ("ci.yml", "frontend-coverage-merge"): _CANONICAL_CONSUMER_EXPR,
    ("ci.yml", "bundle-size"): _CANONICAL_CONSUMER_EXPR_LARGE,
    ("ci.yml", "e2e"): _CANONICAL_CONSUMER_EXPR_LARGE,
    ("ci.yml", "integration"): _CANONICAL_CONSUMER_EXPR_LARGE,
    ("ci.yml", "e2e-boot-matrix"): _CANONICAL_BOOT_MATRIX_EXPR,
    ("ci.yml", "real-adapter-contract"): _CANONICAL_CONSUMER_EXPR,
}


def _all_workflow_files() -> list[Path]:
    return sorted(_WORKFLOWS_DIR.glob("*.yml"))


def _expected_inline_expression(workflow_name: str) -> str:
    if workflow_name in _PUSH_ONLY_WORKFLOWS:
        return _CANONICAL_PUSH_EXPR
    if workflow_name in _DISPATCH_WORKFLOWS:
        return _CANONICAL_DISPATCH_EXPR
    if workflow_name in _DISPATCH_MERGE_GROUP_WORKFLOWS:
        return _CANONICAL_DISPATCH_MERGE_GROUP_EXPR
    if workflow_name in _MERGE_GROUP_WORKFLOWS:
        return _CANONICAL_MERGE_GROUP_ROUTING_EXPR
    return _CANONICAL_ROUTING_EXPR


def test_merge_group_trigger_parity() -> None:
    """A workflow names merge_group in its routing iff it is triggered by it.

    The routing expression admits `merge_group` only where the event can occur;
    a workflow triggered by merge_group but routed with the plain expression
    would send every merge group's jobs to hosted runners, and one routed for
    merge_group without the trigger claims an event it never receives.
    """
    triggered: set[str] = set()
    for path in _all_workflow_files():
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        triggers = workflow.get(True, workflow.get("on"))
        if isinstance(triggers, dict) and "merge_group" in triggers:
            triggered.add(path.name)
    assert triggered == (
        _MERGE_GROUP_WORKFLOWS | _DISPATCH_MERGE_GROUP_WORKFLOWS | _MERGE_GROUP_HOSTED_WORKFLOWS
    )


def test_merge_queue_readiness_polls_every_merge_group_workflow() -> None:
    """The queue's required check waits on exactly the workflows the queue runs.

    A workflow that gains the `merge_group` trigger without joining the poll
    would run on the group and still be ignored by its verdict; one polled
    without the trigger would never appear and fail every group at the appear
    window. The env is the one place the list lives in code, so it is pinned
    to the trigger set here rather than restated.
    """
    workflow = yaml.safe_load(
        (_WORKFLOWS_DIR / "merge-queue-readiness.yml").read_text(encoding="utf-8")
    )
    (job,) = workflow["jobs"].values()
    assert job["name"] == "PR Readiness", "the job name is the ruleset's required check"
    (step,) = job["steps"]
    polled = set(step["env"]["WORKFLOWS"].split())
    triggered = {
        path.name
        for path in _all_workflow_files()
        if "merge_group"
        in (
            (lambda w: w.get(True, w.get("on")) or {})(
                yaml.safe_load(path.read_text(encoding="utf-8"))
            )
        )
    }
    assert polled == triggered - {"merge-queue-readiness.yml"}


def test_every_copy_of_the_routing_expression_matches_the_canonical_one() -> None:
    drifted: list[str] = []
    found_any = False
    for path in _all_workflow_files():
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_id, spec in workflow["jobs"].items():
            value = spec.get("runs-on")
            # Parse the value, not a line regex: folded YAML and label arrays
            # must not hide a fleet route from this check. Anchor on the fleet
            # prefix, never on the predicate whose correctness we are checking.
            if "codebuild-" not in str(value):
                continue
            found_any = True
            if value != _expected_inline_expression(path.name):
                drifted.append(f"{path.name}:{job_id}: {value}")
    assert found_any, "no fleet routes found; the inventory must not pass vacuously"
    assert not drifted, "fleet routing expression drift:\n" + "\n".join(drifted)


def test_every_job_using_the_routing_expression_is_accounted_for() -> None:
    """Both removed routes and unreviewed additions fail, as do missing exceptions."""
    routed: set[tuple[str, str]] = set()
    exception_runs_on: dict[tuple[str, str], object] = {}
    for path in _all_workflow_files():
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_id, spec in workflow["jobs"].items():
            key = (path.name, job_id)
            runs_on = spec.get("runs-on")
            if "codebuild-" in str(runs_on):
                routed.add(key)
            if key in _PERMANENT_EXCEPTIONS:
                exception_runs_on[key] = runs_on

    assert routed == _EXPECTED_ROUTED_JOBS, (
        f"missing routes: {_EXPECTED_ROUTED_JOBS - routed}; "
        f"unexpected routes: {routed - _EXPECTED_ROUTED_JOBS}"
    )
    assert (
        set(exception_runs_on) == _PERMANENT_EXCEPTIONS
    ), f"missing hosted exceptions: {_PERMANENT_EXCEPTIONS - set(exception_runs_on)}"
    wrong_runner = {
        key: value for key, value in exception_runs_on.items() if value != "ubuntu-latest"
    }
    assert not wrong_runner, f"hosted exception changed runner: {wrong_runner}"


def test_every_ci_yml_resolver_consumer_reads_the_resolver_not_a_literal() -> None:
    """Pin the complete Linux/Windows consumer set, tiers and empty-output fallback."""
    workflow = yaml.safe_load((_WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8"))
    observed_consumers = {
        ("ci.yml", job_id): spec["runs-on"]
        for job_id, spec in workflow["jobs"].items()
        if any(
            f"needs.changes.outputs.{os_name}_runner" in str(spec.get("runs-on"))
            for os_name in ("linux", "windows")
        )
    }
    assert observed_consumers == _EXPECTED_RESOLVER_CONSUMER_JOBS


def test_all_fast_gates_skip_only_the_queued_push() -> None:
    """Every gate keeps the fleet route and carries exactly one condition.

    The one `if:` a gate may carry is the push/variable clause: with the merge
    queue on, a push to main already had every gate run on its merge group, so
    the push run skips whole and holds no fleet job an orphan could sit on. Pinned
    by equality so an extra term cannot dodge a gate on a PR or a merge group.
    """
    workflow = yaml.safe_load((_WORKFLOWS_DIR / "fast-gate.yml").read_text(encoding="utf-8"))
    assert workflow["jobs"]
    for job_id, spec in workflow["jobs"].items():
        assert "needs" not in spec, job_id
        assert spec.get("if") == _FAST_GATE_PUSH_SKIP_CLAUSE, job_id
        assert spec["runs-on"] == _CANONICAL_MERGE_GROUP_ROUTING_EXPR, job_id


def test_ci_resolvers_share_actor_policy_and_emit_large_labels() -> None:
    workflow = yaml.safe_load((_WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8"))
    changes = workflow["jobs"]["changes"]
    expected_eligibility = _CANONICAL_MERGE_GROUP_ROUTING_EXPR.split(" && format(", 1)[0] + " }}"
    steps = {step.get("id"): step for step in changes["steps"]}
    for step_id, output, os_name in (
        ("runner", "linux_runner_large", "linux"),
        ("windows-runner", "windows_runner", "windows"),
    ):
        step = steps[step_id]
        assert step["env"]["ELIGIBLE"] == expected_eligibility
        assert changes["outputs"][output] == f"${{{{ steps.{step_id}.outputs.{output} }}}}"
        assert step["env"]["LABEL"] == (
            f"codebuild-kirocrew-gha-{os_name}-${{{{ github.run_id }}}}-"
            "${{ github.run_attempt }}"
        )
        script = step["run"]
        assert 'if [ "$ELIGIBLE" = "true" ]; then' in script
        assert f'echo "{output}=$LABEL instance-size:large"' in script
        hosted = "ubuntu-latest" if os_name == "linux" else "windows-latest"
        assert f'echo "{output}={hosted}"' in script


def test_merge_queue_readiness_budget_covers_ci_critical_path() -> None:
    """The poll's TOTAL_BUDGET must outlast the slowest chain of ci.yml job caps.

    The budget is a ceiling on how long a green merge group may take before the
    queue's required check gives up on it. ci.yml's longest `needs` chain of
    `timeout-minutes` is that ceiling's floor: raise a shard cap without raising
    the budget and a slow-but-green group is dequeued. The job's own
    `timeout-minutes` must in turn exceed the budget, so the step's error -- not
    the job cap -- is what names the lane still pending.
    """
    import re

    readiness = yaml.safe_load(
        (_WORKFLOWS_DIR / "merge-queue-readiness.yml").read_text(encoding="utf-8")
    )
    (job,) = readiness["jobs"].values()
    (step,) = job["steps"]
    match = re.search(r"^\s*TOTAL_BUDGET=(\d+)\s*$", step["run"], re.MULTILINE)
    assert match, "TOTAL_BUDGET must be a literal integer assignment in the poll"
    budget_minutes = int(match.group(1)) / 60

    ci = yaml.safe_load((_WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8"))
    jobs = ci["jobs"]

    def longest_path(job_id: str) -> int:
        spec = jobs[job_id]
        needs = spec.get("needs", [])
        needs = [needs] if isinstance(needs, str) else list(needs)
        upstream = max((longest_path(n) for n in needs), default=0)
        return upstream + int(spec["timeout-minutes"])

    critical_path = max(longest_path(job_id) for job_id in jobs)
    assert budget_minutes >= critical_path, (
        f"TOTAL_BUDGET is {budget_minutes:.0f} min but ci.yml's longest needs-chain "
        f"of timeout-minutes is {critical_path} min; raise the budget with the cap"
    )
    assert int(job["timeout-minutes"]) > budget_minutes
