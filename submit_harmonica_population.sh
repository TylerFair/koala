#!/bin/bash

set -euo pipefail

WORKDIR="/project/ekempton/tfairnington/JWST"
WORKER="${WORKDIR}/run_harmonica_population.sbatch"
SOURCE_MANIFEST="${WORKDIR}/harmonica_population_manifest.txt"
MAX_POPULATION_JOBS=5

cd "${WORKDIR}"
mkdir -p logs/harmonica_population queue_runs

if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
    echo "Missing manifest: ${SOURCE_MANIFEST}" >&2
    exit 1
fi

total=$(wc -l < "${SOURCE_MANIFEST}")
if (( total != 83 )); then
    echo "Expected 83 configs, found ${total} in ${SOURCE_MANIFEST}." >&2
    exit 1
fi

while IFS= read -r config; do
    if [[ ! -f "${WORKDIR}/${config}" ]]; then
        echo "Missing config listed in manifest: ${config}" >&2
        exit 1
    fi
done < "${SOURCE_MANIFEST}"

existing_jobs=$(squeue -h -u "${USER}" -o '%i' | wc -l)
available_slots=$((MAX_POPULATION_JOBS - existing_jobs))
if (( available_slots < 1 )); then
    echo "There are already ${existing_jobs} submitted jobs; no slot remains under the five-job cap." >&2
    exit 1
fi
if (( available_slots > total )); then
    available_slots="${total}"
fi

run_tag="$(date +%Y%m%dT%H%M%S)_$$"
state_dir="${WORKDIR}/queue_runs/harmonica_${run_tag}"
mkdir -p "${state_dir}"
cp "${SOURCE_MANIFEST}" "${state_dir}/manifest.txt"
printf '0\n' > "${state_dir}/next_index"
printf 'timestamp\tjob_id\tslot\tresult\texit_code\tconfig\n' > "${state_dir}/status.tsv"

array_max=$((available_slots - 1))
job_id=$(
    sbatch --parsable \
        --array="0-${array_max}" \
        --export=ALL,QUEUE_STATE_DIR="${state_dir}",MANIFEST_PATH="${state_dir}/manifest.txt" \
        "${WORKER}"
)
printf '%s\n' "${job_id}" > "${state_dir}/array_job_id"
printf '%s\n' "${state_dir}" > "${WORKDIR}/queue_runs/latest_harmonica_run"

echo "Submitted harmonica population worker pool."
echo "  array job:  ${job_id}"
echo "  workers:    ${available_slots}"
echo "  configs:    ${total}"
echo "  state:      ${state_dir}"
echo "  status:     ${state_dir}/status.tsv"
echo "Each worker requeues itself after a config, keeping at most five jobs submitted."

