#!/bin/bash
# CROWN_PLAN.md Step 8, the 3D run on the Step 7 genome (R9).  R4's queue (the phase hold-out)
# with the genome swapped: `export/stage3_crown_standin_best.step` (64e5068 on the (1.5, 1.0)
# rim), all eight stencil phases, SVK, h 2.0 / hc 0.25, the default box, `--r-out` 51.0,
# R3/R4's per-phase windows.  Phases 0 and 3.75 run FIRST: they are D7's registered check.
# Serial, each solve in a 32G MemoryMax scope, a refused solve retried with the window widened
# 4 then 8 mm each side, a failed gmsh mesh retried up to 3 times (§203).  Then patch3d.py
# (the force control, every solve) and post3d_cyl.py (band hoop / s_zz) over every field.
#
#   OUT=<dir> studies/probe_3d/q_step8.sh      (run from the repo root, nothing else on the box)
#
# V3D is the python3.12 venv from requirements-3d.txt and GLLIB the unpacked libGLU/libOpenGL
# (see the 3D toolchain notes, PLAN.md §202); the defaults are where they lived on 2026-09-26.
W=$(pwd)
[ -e $W/export/stage3_crown_standin_best.step ] || { echo "run from the repo root"; exit 1; }
S=${OUT:?set OUT to a scratch directory}
T=/tmp/claude-1000/-home-eric-bodhi-github-wheel/b55e1094-2d08-4060-b992-ba2d70a81f1c/scratchpad
PY=${V3D:-$T/v3d}/bin/python
export LD_LIBRARY_PATH=${GLLIB:-$T/debs/root/usr/lib/x86_64-linux-gnu}
[ -x $PY ] || { echo "no 3D venv at $PY"; exit 1; }
mkdir -p $S && cd $S
tag=s7
cp $W/export/stage3_crown_standin_best.step $tag.step
cp $W/export/stage3_crown_standin_best_step_manifest.json $tag.manifest.json
cp $W/studies/probe_3d/{fe3d,mesh3d,rot,patch3d,post3d_cyl}.py .
# phase, x0, x1, seed0, seed1 -- R3/R4's windows for the crown on top, (1.5, 1.0)
cat > windows.txt <<'EOF'
0 -1.00 4.00 0.40 2.40
3.75 -6.04 8.24 -0.33 2.48
7.5 -7.21 6.03 -1.66 0.78
11.25 -7.34 5.10 -2.07 0.08
15 -7.10 5.02 -1.91 0.02
18.75 -6.53 5.35 -1.45 0.35
22.5 -5.87 5.89 -0.84 0.87
26.25 -5.26 6.42 -0.26 1.36
EOF
log() { echo "$(date +%FT%T) $*" >> q.log; }
win() { awk -v p=$1 -v w=$2 '$1==p{print "--x0",$2-w,"--x1",$3+w,"--seed0",$4,"--seed1",$5}' windows.txt; }
sha1sum $tag.step fe3d.py mesh3d.py rot.py >> q.log
log "QUEUE STEP8 START 64e5068"
for ph in 0 3.75 7.5 11.25 15 18.75 22.5 26.25; do
  m=m_${tag}_$ph.npz
  if [ ! -e $m ]; then
    if [ $ph = 0 ]; then src=$tag.step; else $PY rot.py $tag.step $ph ${tag}_$ph.step; src=${tag}_$ph.step; fi
    for try in 1 2 3; do
      /usr/bin/time -f "  wall %e s  peak %M KB" $PY mesh3d.py $src $m 2.0 0.25 > mlog_${tag}_$ph.txt 2>&1 && break
      log "mesh $tag $ph try $try FAILED"; rm -f $m
    done
    log "meshed $tag $ph $(tail -1 mlog_${tag}_$ph.txt)"
  fi
  [ -e $m ] || { log "$tag $ph NO MESH"; continue; }
  for wid in 0 4 8; do
    r=r_${tag}_$ph.npz
    [ -e $r ] && break
    systemd-run --user --scope -q -p MemoryMax=32G /usr/bin/time -f "  wall %e s  peak %M KB" \
      $PY fe3d.py $m $r --faces free --kinematics svk --r-out 51.0 $(win $ph $wid) > log_${tag}_${ph}_w$wid.txt 2>&1
    rc=$?
    log "solve $tag $ph widen=$wid rc=$rc $(grep -E '^  axle_drop_mm|^  axle_drop_linear' log_${tag}_${ph}_w$wid.txt | awk '{print $2}' | tr '\n' ' ') $(tail -1 log_${tag}_${ph}_w$wid.txt)"
    [ $rc = 0 ] && break
  done
done
pairs=$(for ph in 0 3.75 7.5 11.25 15 18.75 22.5 26.25; do [ -e r_${tag}_$ph.npz ] && echo m_${tag}_$ph.npz:r_${tag}_$ph.npz; done)
$PY patch3d.py $pairs > collate_patch.txt 2>&1; log "patch3d rc=$?"
$PY post3d_cyl.py $pairs > collate_cyl.txt 2>&1; log "post3d_cyl rc=$?"
log "QUEUE STEP8 DONE"
