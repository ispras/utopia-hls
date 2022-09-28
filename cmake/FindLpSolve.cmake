##############################################################################
## From the Ball Project http://www.ball-project.org/                       ##
## https://bitbucket.org/ball/ball/src/b4bdcec12ee5/cmake/FindLPSolve.cmake ##
##                                                                          ##
## Some modifications have been done here to make it work better on Windows ##
##############################################################################

## Detect lpsolve
INCLUDE(CheckCXXSourceCompiles)

SET(LPSOLVE_INCLUDE_DIR "" CACHE STRING "Full path to the lpsolve headers")
MARK_AS_ADVANCED(LPSOLVE_INCLUDE_DIR)


SET(LPSOLVE_LIBRARIES "" CACHE STRING "Full path to the lpsolve55 library (including the library)")
MARK_AS_ADVANCED(LPSOLVE_LIBRARIES)
		
SET(LPSOLVE_INCLUDE_PATH /opt/homebrew/Cellar/lp_solve/5.5.2.11/include)
SET(LPSOLVE_LIBRARIES /opt/homebrew/Cellar/lp_solve/5.5.2.11/lib)
SET(LPSOLVE_LINKS /opt/homebrew/Cellar/lp_solve/5.5.2.11/lib)
SET(LPSOLVE_INCLUDE_TRIAL_PATH
    /sw/include
    /usr/include
    /usr/local/include
    /opt/include
    /opt/local/include
    /opt/homebrew/Cellar/lp_solve/5.5.2.11/include
    C:/Program\ Files
    )
MESSAGE("+")
FIND_PATH(LPSOLVE_INCLUDE_PATH lpsolve/lp_lib.h ${LPSOLVE_INCLUDE_PATH} ${LPSOLVE_INCLUDE_TRIAL_PATH})
MESSAGE("before if")


INCLUDE(FindPackageHandleStandardArgs)

MESSAGE("after include args")

FIND_PACKAGE_HANDLE_STANDARD_ARGS(LpSolve DEFAULT_MSG
    LPSOLVE_LIBRARIES
    LPSOLVE_INCLUDE_PATH
    LPSOLVE_LINKS
    )
MESSAGE("the end")
