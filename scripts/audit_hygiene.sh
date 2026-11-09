#!/usr/bin/env bash
# Asserts repository hygiene:
#   - no absolute host paths in tracked or about-to-be-tracked files
#   - every tool cache is routed under ./.cache/
#   - no stray cache directories at the project root
#   - .gitignore covers .cache/, .venv/, .env
#   - every container image pulled by Dockerfile / compose.yaml is pinned by digest
#
# Usage: sh scripts/audit_hygiene.sh [--workspace]
#   --workspace  also require a pristine working tree: no modified, staged or
#                untracked (non-ignored) files. Run it after tooling such as
#                scripts/mutate.sh to prove nothing outside ./.cache/ was touched.

# Re-exec under real bash when launched via `sh` (POSIX mode lacks process substitution).
if [ -z "${BASH_VERSION:-}" ] || shopt -qo posix 2>/dev/null; then exec bash "$0" "$@"; fi
set -euo pipefail

workspace=0
for arg in "$@"; do
  case "$arg" in
    --workspace) workspace=1 ;;
    *) printf 'audit_hygiene.sh: unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"

fail=0
err() { printf 'FAIL: %s\n' "$*" >&2; fail=1; }
ok() { printf 'ok:   %s\n' "$*"; }

# --- 1. Absolute host paths ---------------------------------------------------
# Character classes keep these patterns from matching this script itself.
readonly ABS_PATH_RE='/[U]sers/|/[h]ome/[A-Za-z0-9_.-]+/|/[V]olumes/|/[p]rivate/(var|tmp)/|/[v]ar/folders/|[A-Za-z]:\\[U]sers\\|(^|[^A-Za-z0-9_.$])[~]/'

files=()
while IFS= read -r -d '' f; do
  [[ -f "$f" ]] && files+=("$f")
done < <(git ls-files -z --cached --others --exclude-standard)

if ((${#files[@]})); then
  if hits="$(grep -EIn "$ABS_PATH_RE" -- "${files[@]}")"; then
    err "absolute host paths found:"$'\n'"$hits"
  else
    ok "no absolute host paths in ${#files[@]} files"
  fi
fi

# --- 2. Cache routing in pyproject.toml ---------------------------------------
check_key() {
  local section="$1" key="$2" expected="$3"
  local value
  value="$(awk -v sec="[$section]" -v key="$key" '
    $0 == sec { in_sec = 1; next }
    /^\[/     { in_sec = 0 }
    in_sec && $1 == key { sub(/^[^=]*=[[:space:]]*/, ""); gsub(/"/, ""); print; exit }
  ' pyproject.toml)"
  if [[ "$value" == "$expected" ]]; then
    ok "[$section] $key = $value"
  else
    err "[$section] $key is '${value:-<unset>}', expected '$expected'"
  fi
}
check_key "tool.uv"                cache-dir  ".cache/uv"
check_key "tool.ruff"              cache-dir  ".cache/ruff"
check_key "tool.mypy"              cache_dir  ".cache/mypy"
check_key "tool.pytest.ini_options" cache_dir ".cache/pytest"
check_key "tool.coverage.run"      data_file  ".cache/coverage/.coverage"
check_key "tool.coverage.html"     directory  ".cache/coverage/html"
check_key "tool.coverage.xml"      output     ".cache/coverage/coverage.xml"

grep -q 'set_hypothesis_home_dir(HYPOTHESIS_DIR)' tests/conftest.py \
  && ok "hypothesis home routed via tests/conftest.py" \
  || err "tests/conftest.py does not route hypothesis storage"

# --- 3. Stray caches at the project root --------------------------------------
for stray in .pytest_cache .mypy_cache .ruff_cache .hypothesis .coverage htmlcov coverage.xml mutants; do
  [[ -e "$stray" ]] && err "stray cache at project root: $stray"
done
ok "no stray root-level caches"

# --- 4. .gitignore coverage ---------------------------------------------------
for path in .cache/probe .venv/probe .env; do
  git check-ignore -q --no-index "$path" && ok "ignored: $path" || err "not ignored: $path"
done

# --- 5. Container images pinned by immutable digest ---------------------------
# Dockerfile: the syntax frontend, image ARGs and literal FROMs (FROM ${ARG} is
# covered by its ARG). compose.yaml: every pulled image; the locally built
# embedding-feature-store/* image has no registry digest.
readonly DIGEST_RE='@sha256:[0-9a-f]{64}'
if [[ -f Dockerfile ]]; then
  if unpinned="$(grep -nE '^# syntax=|^ARG [A-Z_]+_IMAGE=|^FROM ' Dockerfile \
      | grep -vE '^[0-9]+:FROM \$\{[A-Z_]+\}' | grep -vE "$DIGEST_RE")"; then
    err "Dockerfile image references without a digest:"$'\n'"$unpinned"
  else
    ok "Dockerfile images pinned by digest"
  fi
fi
if [[ -f compose.yaml ]]; then
  if unpinned="$(grep -nE '^[[:space:]]+image:' compose.yaml \
      | grep -v 'image: embedding-feature-store/' | grep -vE "$DIGEST_RE")"; then
    err "compose.yaml images without a digest:"$'\n'"$unpinned"
  else
    ok "compose.yaml images pinned by digest"
  fi
fi

# --- 6. Pristine working tree (--workspace) -----------------------------------
if ((workspace)); then
  if dirty="$(git status --porcelain --untracked-files=all)" && [[ -z "$dirty" ]]; then
    ok "working tree is pristine"
  else
    err "working tree has changes outside ignored paths:"$'\n'"$dirty"
  fi
fi

if ((fail)); then
  printf '\naudit_hygiene: FAILED\n' >&2
  exit 1
fi
printf '\naudit_hygiene: PASSED\n'
