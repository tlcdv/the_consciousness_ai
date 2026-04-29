#!/usr/bin/env bash
# Ablation campaign launcher (Phase 4).
# Runs A_current + 7 ablations sequentially, 200 episodes each on dark_room.
# Writes a one-line status to runs/ablation/_campaign_status.txt before each
# run so an external monitor (e.g. the assistant via ScheduleWakeup) can poll
# progress without scanning every log file.
#
# Total wall-clock estimate: ~14h on CPU (Run D crashes fast, ~5 min).
# Logs go to runs/ablation/<name>/ and runs/ablation/_logs/<name>.log.

set -u  # do NOT use `set -e`: Run D is expected to crash, the campaign
        # must continue past it.

CAMPAIGN_DIR="runs/ablation"
LOGS_DIR="${CAMPAIGN_DIR}/_logs"
STATUS_FILE="${CAMPAIGN_DIR}/_campaign_status.txt"
mkdir -p "${LOGS_DIR}"

# Order: A first (reference), D second (will crash fast, validates ablation
# wiring), then the remaining 5 in plan order.
RUNS=(
  "A_current::"
  "D_no_consfix:--ablate-consolidation-fix:"
  "C_no_replay:--ablate-memory-replay:"
  "G_no_rnd_zero:--ablate-rnd-zero-on-reward:"
  "E_no_div:--ablate-gate-diversity:"
  "F_no_fb:--ablate-gate-feedback:"
  "H_no_pad:--ablate-pad-loop:"
  "I_no_bptt:--ablate-bptt:"
)

CAMPAIGN_START=$(date +%s)
echo "campaign_started=$(date -Iseconds)" > "${STATUS_FILE}"

for entry in "${RUNS[@]}"; do
  name="${entry%%:*}"
  rest="${entry#*:}"
  flag="${rest%%:*}"

  log_dir="${CAMPAIGN_DIR}/${name}"
  log_file="${LOGS_DIR}/${name}.log"
  mkdir -p "${log_dir}"

  start_ts=$(date +%s)
  {
    echo "current_run=${name}"
    echo "current_flag=${flag}"
    echo "current_started=$(date -Iseconds)"
    echo "campaign_elapsed_min=$(( (start_ts - CAMPAIGN_START) / 60 ))"
  } > "${STATUS_FILE}"

  python -m scripts.training.train_rlhf \
    --episodes 200 --max-steps 200 \
    --env dark_room \
    --log-dir "${log_dir}" \
    --log-ei-every 50 \
    ${flag:+${flag}} \
    > "${log_file}" 2>&1
  rc=$?

  end_ts=$(date +%s)
  duration_min=$(( (end_ts - start_ts) / 60 ))
  {
    echo "${name} exit=${rc} duration_min=${duration_min} ts=$(date -Iseconds)"
  } >> "${CAMPAIGN_DIR}/_campaign_log.txt"
done

{
  echo "campaign_finished=$(date -Iseconds)"
  echo "total_minutes=$(( ($(date +%s) - CAMPAIGN_START) / 60 ))"
} >> "${STATUS_FILE}"

echo "Campaign complete." >> "${LOGS_DIR}/_done.marker"
