# environment for compiling against the Ubuntu OpenFOAM v1912 binaries
export FOAM_INST_DIR=/tmp/claude-0/-home-user-JYH/c62af2be-1b9c-5e55-83fe-02b1a29689e9/scratchpad/ofsrc
. /tmp/claude-0/-home-user-JYH/c62af2be-1b9c-5e55-83fe-02b1a29689e9/scratchpad/ofsrc/openfoam-OpenFOAM-v1912.200626/etc/bashrc WM_COMPILER_TYPE=system WM_MPLIB=SYSTEMOPENMPI > /dev/null 2>&1
export FOAM_LIBBIN=/usr/lib
export FOAM_USER_APPBIN=/tmp/claude-0/-home-user-JYH/c62af2be-1b9c-5e55-83fe-02b1a29689e9/scratchpad/ofuser/bin
export FOAM_USER_LIBBIN=/tmp/claude-0/-home-user-JYH/c62af2be-1b9c-5e55-83fe-02b1a29689e9/scratchpad/ofuser/lib
export PATH=$FOAM_USER_APPBIN:$WM_PROJECT_DIR/wmake:/usr/bin:$PATH
export LD_LIBRARY_PATH=$FOAM_USER_LIBBIN:/usr/lib/openmpi/lib:/usr/lib:$LD_LIBRARY_PATH
