cmake_minimum_required(VERSION 3.17)
project(cosmos_geniex_raw_video LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GENIEX_SOURCE "" CACHE PATH "Path to the pinned Qualcomm GenieX checkout")
if(NOT GENIEX_SOURCE)
    message(FATAL_ERROR "Set -DGENIEX_SOURCE=/path/to/GenieX")
endif()

set(GENIEX_QAIRT_SOURCE "${GENIEX_SOURCE}/third-party/geniex-qairt")
if(NOT EXISTS "${GENIEX_QAIRT_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR
        "GenieX QAIRT submodule is missing: ${GENIEX_QAIRT_SOURCE}")
endif()
if(NOT EXISTS "${GENIEX_QAIRT_SOURCE}/third-party/geniex-proc/CMakeLists.txt")
    message(FATAL_ERROR
        "GenieX geniex-proc submodule is not initialized. Run the upstream "
        "recursive submodule checkout before using the overlay build.")
endif()

set(GENIEX_BUILD_VLM ON CACHE BOOL "" FORCE)
set(GENIEX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GENIEX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${GENIEX_QAIRT_SOURCE}" geniex-qairt)

add_executable(cosmos_geniex_raw_video
    raw_video_deepstack_runner.cpp
)
target_link_libraries(cosmos_geniex_raw_video PRIVATE
    geniex_vlm
    geniex-proc
)
set_target_properties(cosmos_geniex_raw_video PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Put the shared libraries beside the runner. QNN/HTP runtime libraries are
# already copied to bin/htp-files by geniex_core's upstream post-build rule.
foreach(runtime_target IN ITEMS
        geniex_core
        geniex_vlm
        geniex-proc
        geniex-proc-vision)
    if(TARGET "${runtime_target}")
        add_custom_command(TARGET cosmos_geniex_raw_video POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:${runtime_target}>"
                "$<TARGET_FILE_DIR:cosmos_geniex_raw_video>"
            VERBATIM
        )
    endif()
endforeach()
