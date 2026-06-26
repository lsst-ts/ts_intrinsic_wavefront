#!/usr/bin/env bash
# Launch the AOS Snakemake pipeline, locally or as a Slurm batch job.
#
# Two modes (select with --mode; default local):
#
#   local  — run snakemake on THIS node, detached via nohup so a dropped
#            SSH/network connection won't kill it.  Tuned for the RSP terminal
#            allocation (~4 cores, 16 GiB): -j4 + mem_mb throttling.  Use this
#            from the USDF JupyterLab RSP (which cannot submit to Slurm).
#
#   batch  — submit ONE sbatch job to the s3df cluster that runs snakemake with
#            many parallel slots on an allocated compute node (a roma node has
#            far more than 4 cores).  Submit this from an s3df interactive node
#            (e.g. via the `slacrd` ssh alias) where sbatch + the LSST stack are
#            available; the job inherits the submitting shell's environment
#            (--export=ALL), so `snakemake`/`python` resolve to the same stack.
#            Long/parallel runs belong here, not on the shared login node.
#
# Usage:
#   ./run_snake.sh                        # local, standard run
#   ./run_snake.sh -n                     # local dry-run (extra args pass through)
#   ./run_snake.sh --until combine_donuts # local, partial
#   ./run_snake.sh --mode batch           # batch on roma (32 cpus, 96G, 8h)
#   ./run_snake.sh --mode batch -n        # batch dry-run (validates submission)
#
# Batch tunables (env vars; defaults in parens):
#   SB_PARTITION (roma)  SB_CPUS (32)  SB_MEM (96G)  SB_TIME (08:00:00)
#   SB_RESMEM (90000)    # snakemake --resources mem_mb budget on the node
#   SB_ACCOUNT (rubin:developers@roma)  SB_QOS (normal)   # non-preemptable
#     USDF account names encode the partition; pair with SB_PARTITION.
#     `normal` QOS won't be preempted; `rubin:default@*` only offers
#     `preemptable`.  roma/milano also expose `expedite` for short urgent jobs.
#
# Remember to `git pull` (or sync.sh) first to pick up code changes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

# ---- parse a leading --mode; everything else passes through to snakemake ----
mode=local
passthru=()
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)   mode="$2"; shift 2;;
        --mode=*) mode="${1#*=}"; shift;;
        *)        passthru+=("$1"); shift;;
    esac
done

ts=$(date +%Y%m%d_%H%M%S)

case "$mode" in
    local)
        log="logs/run_${ts}.log"
        args=(-j 4 --resources mem_mb=14000 --keep-going "${passthru[@]}")
        nohup snakemake "${args[@]}" > "$log" 2>&1 &
        pid=$!
        echo "snakemake (local) launched: pid $pid"
        echo "  args:   ${args[*]}"
        echo "  log:    $log"
        echo "  follow: tail -f $log"
        echo "  check:  pgrep -af snakemake"
        ;;
    batch)
        command -v sbatch >/dev/null || {
            echo "error: sbatch not found — submit batch mode from an s3df" \
                 "interactive node (e.g. slacrd), not the RSP pod." >&2
            exit 2; }
        part=${SB_PARTITION:-roma}
        cpus=${SB_CPUS:-32}
        mem=${SB_MEM:-96G}
        tlim=${SB_TIME:-08:00:00}
        resmem=${SB_RESMEM:-90000}
        acct=${SB_ACCOUNT:-rubin:developers@roma}
        qos=${SB_QOS:-normal}
        jlog="logs/batch_${ts}.out"
        sb=(sbatch --partition="$part" --account="$acct" --qos="$qos"
            --cpus-per-task="$cpus" --mem="$mem" --time="$tlim"
            --job-name=aos_snake --output="$jlog")
        # -j == cpus so rules schedule cpus-wide; mem_mb budget caps concurrent
        # memory so the summed per-rule mem_mb fits the node allocation.
        smk="snakemake -j ${cpus} --resources mem_mb=${resmem} --keep-going ${passthru[*]}"
        "${sb[@]}" --wrap "cd '$PWD' && ${smk}"
        echo "submitted batch job -> '$part' acct=$acct qos=$qos (${cpus} cpus, ${mem}, ${tlim})"
        echo "  snakemake: ${smk}"
        echo "  job log:   $jlog"
        echo "  watch:     squeue --me   |   tail -f $jlog"
        ;;
    *)
        echo "error: unknown --mode '$mode' (use 'local' or 'batch')" >&2
        exit 2;;
esac
