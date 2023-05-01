# Set the minimum cmake version to that which supports cuda natively.
# 3.10 required for cuda -std=c++14, however 3.12 fixes some device linker errors
cmake_minimum_required(VERSION VERSION 3.12 FATAL_ERROR)

# Name the project and set languages
project(fgpu2_neuroblastoma CUDA CXX)

# Option to enable/disable building the visualisation
option(VISUALISATION "Enable visualisation support" OFF)

# Optionaly set the version of flamegpu which should be used.
# Use a branch name such as `master`, or a tagged version such as `v2.0.0-alpha` 
set(FLAMEGPU_VERSION "797d09a00f591e88e67f34647f662f180c97c4c5" CACHE STRING "Git branch or tag to use")

# Our core dependency is FLAMEGPU2 lib, first lets find it
include(${CMAKE_CURRENT_LIST_DIR}/cmake/flamegpu2.cmake)

# Include common rules.
include(${FLAMEGPU_ROOT}/cmake/common.cmake)

# Define output location of binary files
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}/)

# Prepare list of source files
# Can't do this automatically, as CMake wouldn't know when to regen (as CMakeLists.txt would be unchanged)
SET(COMMON_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/chemo_len.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/header.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/environment.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/agents/neuroblastoma.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/agents/schwann.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/agents/grid.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/submodels/force_resolution.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hostfns.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model.cu
)
SET(MAIN_SRC
    ${COMMON_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu
)
# Add the executable and set required flags for the target
add_flamegpu_executable("${PROJECT_NAME}" "${MAIN_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" TRUE)

# Mark the root of our source directory as an include directory, so we can include files relative to root
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Also set as startup project
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"  PROPERTY VS_STARTUP_PROJECT "${PROJECT_NAME}")

# Set the default (visual studio) debugger configure_file
set_target_properties("${PROJECT_NAME}" PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                                                   VS_DEBUGGER_COMMAND_ARGUMENTS "-s 100")

# Create a secondary target for the orchestrator interface
project(orchestrator_FGPUNB CUDA CXX)
SET(ORCHESTRATOR_SRC
    ${COMMON_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/orchestrator_src/structures.h
    ${CMAKE_CURRENT_SOURCE_DIR}/orchestrator_src/main.h
    ${CMAKE_CURRENT_SOURCE_DIR}/orchestrator_src/main.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/orchestrator_src/json.h
    ${CMAKE_CURRENT_SOURCE_DIR}/orchestrator_src/json_in.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/orchestrator_src/json_in2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/orchestrator_src/json_out.cpp
)
add_flamegpu_executable(${PROJECT_NAME} "${ORCHESTRATOR_SRC}" "${FLAMEGPU_ROOT}" "${PROJECT_BINARY_DIR}" TRUE)  
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/orchestrator_src")

# Reset the project so the host solution has the same name
project(fgpu2_neuroblastoma CUDA CXX)