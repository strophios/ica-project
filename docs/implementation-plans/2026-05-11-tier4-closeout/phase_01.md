# Tier 4 Phase 1: Hygiene Fixes Implementation Plan

**Goal:** Close M1 + M3 + M4 from the Tier 2/3 deferred lists via three small, mechanical changes plus tests, landing as a single commit.

**Architecture:** Three independent in-scope items — (a) `test_script.py` second-half deletion (absorbs M1: the dead block at lines 188-209 referencing the retired `classifier_from_dapt_checkpoint`, including the blocking `raise` at line 200); (b) M3 head-name `/`-validation in `HeadConfig.__post_init__`; (c) M4 dual head-name validation across `ClassificationHead.__init__` (reject `name=None`) and `build_endpoint_model` (assert unique head names). The dual M4 validation applies the boundary-inventory pattern: same invariant checked at construction-site and call-site.

**Tech Stack:** Python 3.12, frozen dataclasses, pytest. No new dependencies.

**Scope:** 1 of 3 phases. Design reference: `docs/notes/tier4-design.md` "Piece 1: Hygiene fixes".

**Codebase verified:** 2026-05-11.

---

<!-- START_TASK_1 -->
### Task 1: test_script.py second-half cleanup

**Files:**
- Modify: `src/test_script.py` (delete lines 188-209; verify/update docstring at lines 1-3)

**Context.** Investigator confirmed: lines 1-185 contain working endpoint-layer wiring code; lines 188-209 are dead code referencing `classifier_from_dapt_checkpoint` (deleted in T2P4c). Line 200 has `raise RuntimeError(...)`. Existing module docstring at lines 1-3.

**Step 1: Confirm current state**

Use the Read tool:
- `src/test_script.py` with `offset=186, limit=25` to inspect the dead block (lines 186-210)
- `src/test_script.py` with `offset=1, limit=10` to inspect the existing module docstring

Expected: Lines 188-193 instantiate a second preprocessor; 195-199 stub-comment about the retired API; 200-203 contain the raise; 204-209 attempt to use unreachable `cca_classifier`. Lines 1-3 are the existing module docstring.

**Step 2: Delete lines 188-209**

Use the Edit tool to remove the entire block from line 188 through line 209. The file should terminate at the existing last working line (around 185 or 186 depending on blank lines).

**Step 3: Read existing docstring and update if stale**

The Read tool output from Step 1 already shows the docstring. If it describes content that lived in the deleted second half (standard-mode training, `classifier_from_dapt_checkpoint`, etc.), replace it with:

```python
"""Sandbox script exercising the endpoint-layer training pattern
with the Tier 2 abstractions (load_dapt_backbone +
ClassificationHead + build_endpoint_model).

Does NOT exercise standard-mode training — that path is covered
by tests/test_heads.py and tests/test_assembly.py.
"""
```

If the existing docstring is already accurate, leave it unchanged.

**Step 4: Verify Python syntax is valid**

Run:
```bash
python -c "import ast; ast.parse(open('src/test_script.py').read())"
```

Expected: No output (parse succeeds).

**Step 5: Verify pytest still collects all tests**

Run:
```bash
pytest --collect-only 2>&1 | tail -3
```

Expected: 192 tests collected (no changes to test files yet).

No commit yet — bundling with later tasks.
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: M3 — HeadConfig `/`-validation

**Files:**
- Modify: `src/cca_config.py` (extend `HeadConfig.__post_init__` at lines 156-179)
- Modify: `tests/test_cca_config.py` (add test method to `TestHeadConfigValidation` class)

**Context.** `HeadConfig.__post_init__` validates `name` as a non-empty string (lines 157-161) but doesn't reject `/`. `_default_group_fn` in `src/model_setup/assembly.py:54-65` splits `variable.path` on `/`. A head named `"cca/v2"` would silently group as `"cca"`, breaking discriminative-LR grouping in `LayerLRModel`.

**Step 1: Write the failing test**

Locate the existing `TestHeadConfigValidation` class in `tests/test_cca_config.py`. Add:

```python
def test_rejects_name_containing_slash(self):
    """Head names with '/' collide with Keras's variable-path
    separator used by _default_group_fn (assembly.py:54-65)."""
    with pytest.raises(ValueError, match="/"):
        HeadConfig(
            name="cca/v2",
            source_column="cca_label",
            hidden_dim=768,
            loss=FLPULossConfig(prior=0.02),
        )
```

(If `FLPULossConfig` isn't already imported in the test module, use the same import pattern as existing tests in the file.)

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_cca_config.py::TestHeadConfigValidation::test_rejects_name_containing_slash -v
```

Expected: FAIL — `HeadConfig` constructs without raising.

**Step 3: Add the `/` check to `HeadConfig.__post_init__`**

In `src/cca_config.py`, find the existing non-empty-string check for `name` (around lines 157-161). Immediately after it, add:

```python
if "/" in self.name:
    raise ValueError(
        f"HeadConfig.name must not contain '/'; got {self.name!r}. "
        f"'/' is the Keras variable-path separator that "
        f"_default_group_fn in src/model_setup/assembly.py splits "
        f"on to group variables by head; a name containing '/' "
        f"would silently mis-group, breaking discriminative LR."
    )
```

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/test_cca_config.py::TestHeadConfigValidation::test_rejects_name_containing_slash -v
```

Expected: PASS.

**Step 5: Regression check on all HeadConfig validation tests**

Run:
```bash
pytest tests/test_cca_config.py::TestHeadConfigValidation -v
```

Expected: all existing tests still pass plus the new one.

No commit yet.
<!-- END_TASK_2 -->

<!-- START_SUBCOMPONENT_A (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: M4a — ClassificationHead rejects `name=None`

**Files:**
- Modify: `src/model_setup/heads.py` (`ClassificationHead.__init__` at lines 109-116, including docstring at lines 96-99)
- Modify: `tests/test_heads.py` (add test methods; place in existing class for construction tests)

**Context.** Current signature: `def __init__(self, hidden_dim, dropout=0.1, loss_fn=None, metrics=None, name=None)`. Keras's `Layer` base allows `name=None` with auto-fallback (e.g., `"classification_head_1"`); two heads built without explicit names would silently collide. Removing the default forces explicit naming.

**Step 1: Discover existing test class**

Run:
```bash
grep -n "^class Test" tests/test_heads.py
```

Identify the class that holds construction tests (likely `TestClassificationHeadConstruction` or similar based on the investigator's report; tests like `test_standard_mode_constructs_with_only_hidden_dim` live there). Use this class for the new tests.

**Step 2: Write the failing tests**

Add to the identified construction-tests class:

```python
def test_rejects_name_none(self):
    """name must be explicit; name=None would fall back to Keras
    auto-generated names that collide silently across heads."""
    with pytest.raises(ValueError, match="name"):
        ClassificationHead(hidden_dim=768, name=None)

def test_rejects_missing_name_arg(self):
    """name has no default — must be passed."""
    with pytest.raises(TypeError):
        ClassificationHead(hidden_dim=768)
```

**Step 3: Run tests to verify they fail**

Run:
```bash
pytest tests/test_heads.py -v -k "rejects_name or rejects_missing_name"
```

Expected: both new tests FAIL — `name=None` doesn't raise; `ClassificationHead(hidden_dim=768)` succeeds because the current default is `name=None`.

**Step 4: Discover existing call sites that need updating**

Before changing the signature, identify call sites that construct `ClassificationHead` without an explicit `name=`. Run:
```bash
grep -rn "ClassificationHead(" src/ tests/ scripts/ --include="*.py"
```

Inspect each match. Production paths (`src/cca_config.py` `DEFAULT_CCA_CONFIG`, training/eval scripts) should already pass explicit names. Test files (especially `tests/test_heads.py`) likely have constructions that rely on the `name=None` default and will need explicit `name=` added.

**Step 5: Update `ClassificationHead.__init__`**

In `src/model_setup/heads.py`:

(a) Make `name` keyword-only and required. The new signature uses `*` to force keyword-only:

```python
def __init__(self, hidden_dim, dropout=0.1, loss_fn=None, metrics=None, *, name):
```

(A positional parameter without a default cannot follow defaulted positionals — that would be a `SyntaxError`. The `*` syntax converts `name` to keyword-only, making it both required and forcing explicit `name=` at every call site.)

(b) Add an explicit `None`-rejection check at the start of `__init__` (defends against deliberate `name=None`):

```python
if name is None:
    raise ValueError(
        "ClassificationHead requires an explicit name; name=None "
        "would fall back to Keras auto-generated names "
        "(e.g., 'classification_head_1') which collide silently "
        "across heads in a multi-head model."
    )
```

(The `*` makes `name=None` a deliberate choice the caller has to make rather than an accidental default; the runtime check handles the deliberate case.)

(c) Update the docstring at lines 96-99 to note `name` is now a required keyword-only parameter.

**Step 6: Update at-risk call sites identified in Step 4**

For each call site found in Step 4 that doesn't pass `name=`, add an explicit name argument. Test files will typically use a name reflecting the test scope (e.g., `name="test_head"`, `name="primary"`). For any test that specifically tested `name=None` behavior (now changed by Task 3 above), update it to assert the new ValueError behavior or remove it as redundant with the new tests.

**Step 7: Run tests to verify they pass**

Run:
```bash
pytest tests/test_heads.py -v -k "rejects_name or rejects_missing_name"
```

Expected: both new tests PASS.

**Step 8: Regression-check the full test_heads.py suite**

Run:
```bash
pytest tests/test_heads.py -v
```

Expected: all tests pass after the Step 6 updates. If any tests still fail, locate the remaining missing-name call sites and fix.

No commit yet.
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: M4b — build_endpoint_model asserts unique head names

**Files:**
- Modify: `src/model_setup/assembly.py` (`build_endpoint_model` at lines 68-166; add assertion near function entry)
- Modify: `tests/test_assembly.py` (add test method to `TestBuildEndpointModel`)

**Context.** `heads` is currently a `dict`, so duplicate keys can't structurally occur. The assert is a forward-compat guard against future API change (e.g., `heads` becoming a list of `(name, head)` pairs). Testing it requires a mapping-like object that reports duplicate keys.

**Step 1: Discover the fixture pattern and test class**

Run:
```bash
grep -n "^class Test\|@pytest.fixture\|def fresh_" tests/test_assembly.py
```

Identify (a) the test class for `build_endpoint_model` tests (per the investigator's report, `TestBuildEndpointModel` starting around line 120), and (b) the fixture name and signature for the backbone fixture (per investigator, `fresh_backbone` at line 101). If the actual fixture name differs from `fresh_backbone`, substitute the correct name throughout the Step 2 test code.

**Step 2: Write the failing test**

Add to the test class identified in Step 1 (likely `TestBuildEndpointModel`) in `tests/test_assembly.py`:

```python
def test_build_endpoint_model_rejects_duplicate_head_names(
    self, fresh_backbone
):
    """Forward-compat boundary-inventory check: same unique-names
    invariant enforced at call-site (here) and construction-site
    (ClassificationHead.__init__ requires explicit name). Dict
    structurally prevents duplicates today, but a future API
    change (heads as list of pairs) could allow them — the
    assert is the guard. Test triggers the assert via a
    mapping-like fake that reports duplicate keys."""

    class _DuplicateKeyHeads:
        """Test helper: mimics dict.keys() returning duplicates."""
        def keys(self):
            return ["x", "x"]
        def items(self):
            return []  # Not reached; assert fires first
        def __len__(self):
            return 2

    fake_heads = _DuplicateKeyHeads()
    with pytest.raises(ValueError, match="duplicate"):
        build_endpoint_model(backbone=fresh_backbone, heads=fake_heads)
```

(If `build_endpoint_model` isn't already imported at module level in `tests/test_assembly.py`, add the import.)

**Step 3: Run test to verify it fails**

Run:
```bash
pytest tests/test_assembly.py -v -k "rejects_duplicate_head_names"
```

Expected: FAIL — `build_endpoint_model` either doesn't raise or raises something other than `ValueError("duplicate...")`.

**Step 4: Add the unique-names assertion to `build_endpoint_model`**

In `src/model_setup/assembly.py`, locate the entry of `build_endpoint_model` (after parameter handling but before any model-construction logic). Add:

```python
names = list(heads.keys())
if len(set(names)) != len(names):
    duplicates = sorted({n for n in names if names.count(n) > 1})
    raise ValueError(
        f"build_endpoint_model requires unique head names; got "
        f"duplicates: {duplicates}"
    )
```

**Step 5: Run test to verify it passes**

Run:
```bash
pytest tests/test_assembly.py -v -k "rejects_duplicate_head_names"
```

Expected: PASS.

**Step 6: Regression-check the assembly test suite**

Run:
```bash
pytest tests/test_assembly.py -v
```

Expected: all tests pass.

No commit yet.
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_TASK_5 -->
### Task 5: Full suite + commit

**Step 1: Run the complete test suite**

Run:
```bash
pytest
```

Expected: All tests pass. Baseline was 192 (per CLAUDE.md). New tests added in this phase: Task 2 adds 1, Task 3 adds 2, Task 4 adds 1 = +4 new tests, for a target of 196. Some existing tests in `tests/test_heads.py` may have been *modified* (Task 3 Step 6) to pass explicit `name=` — those don't change the count.

**If the final count differs from 196:** That means either some Task 3 modifications converted tests rather than just adjusting their inputs, or Task 4's test triggered an existing fixture issue. Either is acceptable as long as all tests pass; record the actual count in the commit message.

**If any tests fail:** Diagnose and fix. Most likely failure mode is missed `name=` updates from Task 3 Step 6 in test files that weren't checked. Re-run `grep -rn "ClassificationHead(" tests/ --include="*.py" | grep -v "name=" | grep -v "name ="` to locate.

**Step 2: Review the diff**

Run:
```bash
git status
git diff --stat
```

Expected: changes only to `src/test_script.py`, `src/cca_config.py`, `src/model_setup/heads.py`, `src/model_setup/assembly.py`, and the three test files. No unintended changes elsewhere.

**Step 3: Stage and commit**

Run:
```bash
git add src/test_script.py src/cca_config.py src/model_setup/heads.py src/model_setup/assembly.py tests/test_cca_config.py tests/test_heads.py tests/test_assembly.py
git commit -m "$(cat <<'EOF'
Tier 4 Piece 1: hygiene fixes (test_script.py + M3 + M4)

Three small in-scope items from the Tier 2/3 review backlog,
bundled as one commit per the compact 3-piece structure in
docs/notes/tier4-design.md.

- test_script.py second-half deletion (lines 188-209, including
  the blocking raise at line 200 that was the M1 item). Absorbs
  M1. Keeps lines 1-185 as a sandbox for the endpoint-layer
  pattern; standard-mode training is covered in tests.
- M3: HeadConfig.__post_init__ rejects head names containing
  '/'. '/' is Keras's variable-path separator, used by
  _default_group_fn (assembly.py) to group variables by head
  for discriminative LR; collision would silently mis-group.
- M4: dual head-name validation. ClassificationHead.__init__
  no longer accepts name=None (now a keyword-only required
  parameter with explicit ValueError on None); build_endpoint_model
  asserts unique head names across the supplied collection.
  Catches the invariant at both construction-site and
  call-site (boundary-inventory pattern).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Step 4: Verify commit landed**

Run:
```bash
git log --oneline -1
git status
```

Expected: New HEAD commit; working tree clean for in-scope files.

**Step 5: Request code review (per project Tier 2/3 convention)**

Per the Tier 2/3 pattern, dispatch the code-reviewer subagent to validate this piece before moving to Phase 2. Review prompt should reference `docs/notes/tier4-design.md` "Piece 1" as the spec.
<!-- END_TASK_5 -->
