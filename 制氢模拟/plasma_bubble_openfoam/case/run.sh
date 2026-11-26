#!/bin/bash
# usage: run.sh CASEDIR  -- mesh, initialise (only if new) and run plasmaInterFoam
S=/tmp/claude-0/-home-user-JYH/c62af2be-1b9c-5e55-83fe-02b1a29689e9/scratchpad
source $S/ofenv.sh
cd "$1" || exit 1
if [ ! -f constant/polyMesh/points ]; then blockMesh > log.blockMesh 2>&1 || exit 2; fi
if [ ! -f 0/alpha.ethanol ]; then cp 0/alpha.ethanol.orig 0/alpha.ethanol; setFields > log.setFields 2>&1 || exit 3; fi
exec plasmaInterFoam >> log.run 2>&1
