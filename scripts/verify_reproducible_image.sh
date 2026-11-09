#!/usr/bin/env bash
# Proves the Docker image is bit-for-bit reproducible.
#
# Builds the image twice from scratch (--no-cache) with the same
# SOURCE_DATE_EPOCH, exports each build as an OCI tarball with layer
# timestamps rewritten to that epoch, and requires the two to be identical:
# same tarball bytes, same manifest digest, same layer digests. Attestations
# are disabled because provenance records build times and IDs.
#
# Usage: sh scripts/verify_reproducible_image.sh [--platform OS/ARCH]
#   SOURCE_DATE_EPOCH  defaults to the commit time of HEAD
#
# Artefacts go to ./.cache/docker/. Needs a running Docker daemon with
# BuildKit/buildx and an image store that can export OCI (containerd store or
# a docker-container builder).

# Re-exec under real bash when launched via `sh` (POSIX mode lacks arrays/[[ ]]).
if [ -z "${BASH_VERSION:-}" ] || shopt -qo posix 2>/dev/null; then exec bash "$0" "$@"; fi
set -euo pipefail

platform_args=()
while (($#)); do
  case "$1" in
    --platform) platform_args=(--platform "${2:?--platform needs a value}"); shift 2 ;;
    -h | --help) sed -n '2,16p' "$0"; exit 0 ;;
    *) printf 'verify_reproducible_image.sh: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"

epoch="${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct)}"
readonly out=".cache/docker"
mkdir -p "$out"

build() {
  local dest="$1"
  rm -f "$dest"
  docker buildx build \
    --no-cache \
    --provenance=false \
    --sbom=false \
    "${platform_args[@]}" \
    --build-arg SOURCE_DATE_EPOCH="$epoch" \
    --output "type=oci,dest=$dest,rewrite-timestamp=true" \
    --progress=plain \
    . >"$dest.log" 2>&1 || { tail -n 40 "$dest.log" >&2; return 1; }
}

manifest_digest() { tar -xOf "$1" index.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["manifests"][0]["digest"])'; }
blob_list() { tar -tvf "$1" | awk '{print $NF}' | sort; }

printf 'SOURCE_DATE_EPOCH=%s\n' "$epoch"
for n in 1 2; do
  printf 'cold build %s ...\n' "$n"
  build "$out/image-$n.tar"
done

sum1="$(shasum -a 256 "$out/image-1.tar" | cut -d' ' -f1)"
sum2="$(shasum -a 256 "$out/image-2.tar" | cut -d' ' -f1)"
man1="$(manifest_digest "$out/image-1.tar")"
man2="$(manifest_digest "$out/image-2.tar")"

printf 'build 1: tar sha256:%s manifest %s\n' "$sum1" "$man1"
printf 'build 2: tar sha256:%s manifest %s\n' "$sum2" "$man2"

if [[ "$sum1" == "$sum2" && "$man1" == "$man2" ]] && diff <(blob_list "$out/image-1.tar") <(blob_list "$out/image-2.tar") >/dev/null; then
  printf '\nreproducible: PASSED\n'
else
  printf '\nblob differences:\n' >&2
  diff <(blob_list "$out/image-1.tar") <(blob_list "$out/image-2.tar") >&2 || true
  printf '\nreproducible: FAILED\n' >&2
  exit 1
fi
