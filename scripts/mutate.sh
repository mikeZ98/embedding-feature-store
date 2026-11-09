#!/usr/bin/env bash
# Mutation testing of the core modules, confined to ./.cache/mutmut/.
#
# mutmut 3.x always builds its mutated copy in ./mutants/ relative to the
# directory it is started from, and the mutated test runs create their own
# caches next to it. So it is never started from the checkout: this script
# copies the working tree's pyproject.toml, uv.lock, src/ and tests/ into a
# throwaway workspace, runs mutmut there with the project's locked
# environment, and gates on the mutation score. Tracked files are read, never
# written.
#
# Which modules are mutated and which pytest flags the mutant runs get
# (--no-cov -q: no coverage data file contention) live in [tool.mutmut].
#
# Usage: sh scripts/mutate.sh [--min-score PCT] [--max-children N]
#   --min-score PCT    fail unless the score is strictly greater than PCT (default 90)
#   --max-children N   parallel mutant workers (default: CPU count)
#
# Score = (killed + timeout) / (total - skipped), the same formula as `mutmut badge`.
# Inspect survivors afterwards with:
#   cd .cache/mutmut/workspace && uv run --project ../../.. mutmut browse

# Re-exec under real bash when launched via `sh` (POSIX mode lacks arrays/[[ ]]).
if [ -z "${BASH_VERSION:-}" ] || shopt -qo posix 2>/dev/null; then exec bash "$0" "$@"; fi
set -euo pipefail

min_score=90
max_children=""
while (($#)); do
  case "$1" in
    --min-score) min_score="${2:?--min-score needs a value}"; shift 2 ;;
    --max-children) max_children="${2:?--max-children needs a value}"; shift 2 ;;
    -h | --help) sed -n '2,22p' "$0"; exit 0 ;;
    *) printf 'mutate.sh: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

root="$(git rev-parse --show-toplevel)"
cd "$root"

readonly workspace=".cache/mutmut/workspace"
export UV_CACHE_DIR="$root/.cache/uv"
# Deterministic Hypothesis examples, so no mutant is killed or spared by chance.
export HYPOTHESIS_PROFILE=mutation

if [[ -e mutants ]]; then
  printf 'mutate.sh: ./mutants/ exists in the checkout; remove it (mutmut was run outside this script)\n' >&2
  exit 1
fi
git check-ignore -q --no-index "$workspace/probe" || {
  printf 'mutate.sh: %s is not git-ignored\n' "$workspace" >&2
  exit 1
}

uv sync --frozen --quiet

# Fresh workspace on every run: a stale mutants/ tree would report stale results.
rm -rf "$workspace"
mkdir -p "$workspace"
git ls-files -z --cached --others --exclude-standard -- pyproject.toml uv.lock src tests \
  | tar -cf - --null -T - \
  | tar -xf - -C "$workspace"

run_args=(run)
[[ -n "$max_children" ]] && run_args+=(--max-children "$max_children")

(
  cd "$workspace"
  uv run --project "$root" --frozen --no-sync mutmut "${run_args[@]}"
  uv run --project "$root" --frozen --no-sync mutmut export-cicd-stats >/dev/null
)

stats="$workspace/mutants/mutmut-cicd-stats.json"
if ! uv run --frozen --no-sync python - "$stats" "$min_score" <<'PY'
import json
import sys

stats = json.load(open(sys.argv[1]))
threshold = float(sys.argv[2])
tested = stats["total"] - stats["skipped"]
detected = stats["killed"] + stats["timeout"]
score = 100.0 * detected / tested if tested else 0.0
print(json.dumps(stats, sort_keys=True))
print(f"mutation score: {score:.2f}% ({detected}/{tested} detected; threshold > {threshold:g}%)")
sys.exit(0 if tested and score > threshold else 1)
PY
then
  printf '\nsurviving mutants:\n' >&2
  (cd "$workspace" && uv run --project "$root" --frozen --no-sync mutmut results) >&2 || true
  printf '\nmutate: FAILED\n' >&2
  exit 1
fi

[[ -e mutants ]] && { printf 'mutate.sh: mutmut leaked ./mutants/ into the checkout\n' >&2; exit 1; }
printf '\nmutate: PASSED\n'
