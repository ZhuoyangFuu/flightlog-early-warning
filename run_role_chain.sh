#!/usr/bin/env bash
# Follow-up chain after run_role_probes.sh (2026-09-18, Zhuoyang: "automatically update the
# result, no permission needed"): wait for the probes -> rule sweep of the new dumps ->
# paper_tables -> paper_figs_v2 -> fill_role_rows (RESULTS.md 5.6) -> docx -> RESUME note.
#   bash run_role_chain.sh start | status | tail | stop
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zhuoyang/venvs/flightlog/bin/python
LOG="$ROOT/logs/role_chain.log"; PIDF="$ROOT/state/role_chain.pid"; DONE="$ROOT/state/role_chain.done"
running() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
log() { echo "=== $(date '+%F %T') $*"; }
waitpid() { while kill -0 "$1" 2>/dev/null; do sleep 60; done; }
work() {
  cd "$ROOT"; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  log "chain armed; waiting for run_role_probes.sh (pid $(cat state/role_probes.pid 2>/dev/null))"
  [[ -f state/role_probes.pid ]] && waitpid "$(cat state/role_probes.pid)"
  n=$(ls meta/horizon_pt_*role.json 2>/dev/null | wc -l)
  log "probes finished: $n / 54 role json; failures: $(grep -c '!! FAILED\|Killed' logs/role_probes.log)"
  log "step 1: rule sweep of the new score dumps"
  bash run_seq_all.sh start; sleep 5; waitpid "$(cat state/seq_all.pid)"
  log "sweep done: $(ls meta/sequential_parts/*.csv | wc -l) parts / $(ls meta/scores/*.parquet | wc -l) dumps"
  log "step 2: paper_tables.py"; $PY paper_tables.py > logs/paper_tables_role.log 2>&1 || log "!! paper_tables failed"
  grep -E "role" logs/paper_tables_role.log | head -40
  log "step 3: paper_figs_v2.py"; $PY paper_figs_v2.py 2>&1 | grep -v Warning || log "!! figs failed"
  log "step 4: fill_role_rows.py"; $PY fill_role_rows.py || log "!! fill_role_rows failed"
  log "step 5: docx"; bash build_results_docx.sh || log "!! docx failed"
  $PY - <<PYEOF
p="$ROOT/RESUME.md"; s=open(p).read()
note=("> **✅ AUTO $(date '+%F %T') — role probes landed ($n/54 json), rules swept, tables/figs/docx rebuilt, "
      "RESULTS §5.6 role rows auto-filled by fill_role_rows.py (log: logs/role_chain.log). Read §5.6 before quoting.**\n>\n")
h="# FlightLog — Resume Here\n\n"; s=s.replace(h, h+note, 1); open(p,"w").write(s)
PYEOF
  touch "$DONE"; log "chain complete"
}
case "${1:-}" in
  start) running && { echo "already running (pid $(cat "$PIDF"))"; exit 1; }
    rm -f "$DONE"
    setsid nohup bash -c "$(declare -f work log waitpid); ROOT='$ROOT'; PY='$PY'; DONE='$DONE'; work" </dev/null >>"$LOG" 2>&1 &
    echo $! >"$PIDF"; echo "started pid $(cat "$PIDF"); log: $LOG" ;;
  status) running && echo "[role_chain] RUNNING (pid $(cat "$PIDF"))" || { [[ -f "$DONE" ]] && echo "[role_chain] DONE" || echo "[role_chain] not running"; }
    tail -n 4 "$LOG" 2>/dev/null; echo "--- probes:"; bash "$ROOT/run_role_probes.sh" status | tail -n 3 ;;
  tail) tail -f "$LOG" ;;
  stop) running && { pkill -P "$(cat "$PIDF")"; kill "$(cat "$PIDF")"; echo stopped; } || echo "not running" ;;
  *) echo "usage: bash run_role_chain.sh start|status|tail|stop" ;;
esac
