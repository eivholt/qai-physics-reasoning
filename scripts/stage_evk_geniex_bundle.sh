#!/usr/bin/env bash
set -euo pipefail

evk_target="${EVK_TARGET:-ubuntu@192.168.1.158}"
destination_dir="${DESTINATION_DIR:?Set DESTINATION_DIR to a new EVK-local staging path}"
source_dir="${SOURCE_DIR:-}"
local_archive="${LOCAL_ARCHIVE:-}"

if [[ "${destination_dir}" != /home/ubuntu/qai-conveyor/* ]]; then
  printf 'DESTINATION_DIR must be below /home/ubuntu/qai-conveyor: %s\n' \
    "${destination_dir}" >&2
  exit 2
fi
if [[ -n "${source_dir}" && -n "${local_archive}" ]]; then
  printf 'Set only one of SOURCE_DIR or LOCAL_ARCHIVE.\n' >&2
  exit 2
fi
if [[ -z "${source_dir}" && -z "${local_archive}" ]]; then
  printf 'Set SOURCE_DIR to an EVK-local bundle or LOCAL_ARCHIVE to a local .tar.gz.\n' >&2
  exit 2
fi

if [[ -n "${local_archive}" ]]; then
  test -f "${local_archive}"
  archive_sha256="$(sha256sum "${local_archive}" | cut -d' ' -f1)"
  remote_archive="/home/ubuntu/qai-conveyor/cache/${archive_sha256}.tar.gz"
  remote_upload="${remote_archive}.uploading"

  ssh -o BatchMode=yes -o ConnectTimeout=5 "${evk_target}" \
    "mkdir -p /home/ubuntu/qai-conveyor/cache"
  remote_sha256="$(
    ssh -o BatchMode=yes -o ConnectTimeout=5 "${evk_target}" \
      "if test -f '${remote_archive}'; then sha256sum '${remote_archive}' | cut -d' ' -f1; fi"
  )"
  if [[ "${remote_sha256}" != "${archive_sha256}" ]]; then
    scp -q -o BatchMode=yes -o ConnectTimeout=5 \
      "${local_archive}" "${evk_target}:${remote_upload}"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "${evk_target}" \
      "test \"\$(sha256sum '${remote_upload}' | cut -d' ' -f1)\" = '${archive_sha256}'; mv '${remote_upload}' '${remote_archive}'"
  fi

  ssh -o BatchMode=yes -o ConnectTimeout=5 "${evk_target}" \
    bash -s -- "${remote_archive}" "${destination_dir}" "${archive_sha256}" <<'REMOTE'
set -euo pipefail
remote_archive="$1"
destination_dir="$2"
archive_sha256="$3"
importing="${destination_dir}.importing.$$"

if [ -e "${destination_dir}" ]; then
  printf 'Destination already exists: %s\n' "${destination_dir}" >&2
  exit 1
fi
trap 'rm -rf -- "${importing}"' EXIT
mkdir -p "${importing}"
tar -xzf "${remote_archive}" --strip-components=1 -C "${importing}"
for context in vision_encoder.bin part1_of_4.bin part2_of_4.bin part3_of_4.bin part4_of_4.bin; do
  test -f "${importing}/${context}"
  test ! -L "${importing}/${context}"
done
test -r "${importing}/geniex_compat.json"
printf '%s\n' "${archive_sha256}" >"${importing}/.qai-conveyor-archive-sha256"
mv "${importing}" "${destination_dir}"
trap - EXIT
du -sh "${destination_dir}"
REMOTE
  printf 'staged_archive_sha256=%s\n' "${archive_sha256}"
  exit 0
fi

ssh -o BatchMode=yes -o ConnectTimeout=5 "${evk_target}" \
  bash -s -- "${source_dir}" "${destination_dir}" <<'REMOTE'
set -euo pipefail
source_dir="$1"
destination_dir="$2"

test -d "${source_dir}"
if [ -e "${destination_dir}" ]; then
  printf 'Destination already exists: %s\n' "${destination_dir}" >&2
  exit 1
fi
mkdir -p "${destination_dir}"
cp -aL "${source_dir}/." "${destination_dir}/"
for context in vision_encoder.bin part1_of_4.bin part2_of_4.bin part3_of_4.bin part4_of_4.bin; do
  test -f "${destination_dir}/${context}"
  test ! -L "${destination_dir}/${context}"
done
du -sh "${destination_dir}"
REMOTE
