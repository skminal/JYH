#!/bin/bash
# one-line status of each run directory given as argument
W=${RUNS:-/tmp/claude-0/-home-user-JYH/c62af2be-1b9c-5e55-83fe-02b1a29689e9/scratchpad}
for v in "$@"; do d=$W/$v
  t0=$(python3 -c "import json;print(json.load(open('$d/case.json'))['derived']['t0']*1e3)")
  echo "$v: step $(tail -1 $d/log_time | awk '{print $2, "dt", $6*'$t0'*1e3 "us wall", $10 "s"}') | t=$(tail -1 $d/mon_Th | awk '{printf "%.3f ms theta=%s", $3*'$t0', $NF}') Vgas=$(tail -1 $d/mon_Vgas | awk '{print $NF}') Pgas=$(tail -1 $d/mon_Pgas | awk '{print $NF}') Sd=$(tail -1 $d/mon_Sd | awk '{print $NF}') Sth=$(tail -1 $d/mon_Sth | awk '{print $NF}') U=$(tail -1 $d/mon_Vel | awk '{print $NF}')"
  grep -v "^$" $d/err.log | grep -v poisson_electric | tail -1 | cut -c1-130
done
