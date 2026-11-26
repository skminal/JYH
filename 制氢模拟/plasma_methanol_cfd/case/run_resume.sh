#!/bin/bash
# Start or resume a Gerris run in RUNDIR.  usage: run_resume.sh RUNDIR INITIAL.gfs
# Resumes from the newest complete save-*.gfs if there is one. Monitor files are
# rewritten by Gerris on every start, so their previous content is kept in hist_*.
GFS=${GFS:-/tmp/claude-0/-home-user-JYH/c62af2be-1b9c-5e55-83fe-02b1a29689e9/scratchpad/gfs}
export PATH=$GFS/bin:$PATH LD_LIBRARY_PATH=$GFS/lib PKG_CONFIG_PATH=$GFS/lib/pkgconfig
cd "$1" || exit 1
for f in mon_* log_time; do [ -s "$f" ] && cat "$f" >> "hist_$f"; done
start="$2"
for f in $(ls save-*.gfs 2>/dev/null | sort -r); do
    # a save cut short by a kill has no closing data; accept only files that
    # are at least 90% the size of the save before it
    prev=$(ls save-*.gfs | sort | grep -B1 "^$f$" | head -1)
    if [ "$prev" = "$f" ] || [ $(stat -c %s "$f") -ge $(( $(stat -c %s "$prev") * 9 / 10 )) ]; then
        start="$f"; break
    fi
done
echo "$(date +%T) starting from $start" >> resume.log
exec gerris2D "$start" > out.log 2>> err.log
