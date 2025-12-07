cmake_minimum_required (VERSION 3.20)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project("GigaLearnCPP" LANGUAGES CXX CUDA)

include_directories("${PROJECT_SOURCE_DIR}/src/")

# Set CUDA flags for unsupported compiler
if(NOT DEFINED CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
endif()

# Robust libtorch detection:
set(LIBTORCH_ROOT "")
if(DEFINED ENV{LIBTORCH})
    set(LIBTORCH_ROOT $ENV{LIBTORCH})
endif()

if(NOT LIBTORCH_ROOT)
    if(EXISTS "${PROJECT_SOURCE_DIR}/libtorch")
        set(LIBTORCH_ROOT "${PROJECT_SOURCE_DIR}/libtorch")
    elseif(EXISTS "${PROJECT_SOURCE_DIR}/../libtorch")
        set(LIBTORCH_ROOT "${PROJECT_SOURCE_DIR}/../libtorch")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/libtorch")
        set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/libtorch")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/../libtorch")
        set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/../libtorch")
    endif()
endif()

if(LIBTORCH_ROOT)
    message(STATUS "LibTorch root detected: ${LIBTORCH_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}/share/cmake/Torch")
else()
    message(STATUS "LibTorch root not auto-detected. If CMake fails, set LIBTORCH env var or pass -DCMAKE_PREFIX_PATH=<path-to-libtorch>")
endif()

# Add libtorch
find_package(Torch REQUIRED)
if(Torch_FOUND)
    message(STATUS "Found LibTorch: ${TORCH_INSTALL_PREFIX}")
endif()

# Determine a prefix to use for forced includes
if(DEFINED TORCH_INSTALL_PREFIX)
    set(_torch_prefix ${TORCH_INSTALL_PREFIX})
elseif(LIBTORCH_ROOT)
    set(_torch_prefix ${LIBTORCH_ROOT})
elseif(DEFINED CMAKE_PREFIX_PATH)
    # take first entry
    list(GET CMAKE_PREFIX_PATH 0 _torch_prefix)
else()
    set(_torch_prefix "")
endif()

if(_torch_prefix)
    # Force include directories for compiler/IDE to find ATen and c10 headers
    message(STATUS "Forcing libtorch include dirs from: ${_torch_prefix}")
    include_directories("${_torch_prefix}/include")
    include_directories("${_torch_prefix}/include/ATen")
    include_directories("${_torch_prefix}/include/ATen/core")
    include_directories("${_torch_prefix}/include/c10")
    include_directories("${_torch_prefix}/include/torch/csrc/api/include")
endif()

# TORCH_CXX_FLAGS may include useful defines
if(DEFINED TORCH_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit found at: ${CUDAToolkit_ROOT}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp" "libsrc/*.cpp" "libsrc/.h" "libsrc/.hpp")

add_library(GigaLearnCPP STATIC ${FILES_SRC})
target_compile_definitions(GigaLearnCPP PUBLIC -DWITHIN_GGL)
target_include_directories(GigaLearnCPP PUBLIC "src/public")
target_include_directories(GigaLearnCPP PRIVATE "src/private")
if(CUDAToolkit_INCLUDE_DIRS)
    target_include_directories(GigaLearnCPP PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Ensure compiler can find libtorch headers on the target
if(_torch_prefix)
    target_include_directories(GigaLearnCPP PUBLIC
        ${_torch_prefix}/include
        ${_torch_prefix}/include/ATen
        ${_torch_prefix}/include/ATen/core
        ${_torch_prefix}/include/c10
        ${_torch_prefix}/include/torch
        ${_torch_prefix}/include/torch/csrc/api/include
    )
endif()

# Prefer modern target if available; force linking to Torch::Torch when possible
if(TARGET Torch::Torch)
    message(STATUS "Linking with Torch::Torch target")
    target_link_libraries(GigaLearnCPP PUBLIC Torch::Torch)
else()
    message(STATUS "Linking with TORCH_LIBRARIES variable")
    target_link_libraries(GigaLearnCPP PUBLIC ${TORCH_LIBRARIES})
endif()

# Enable compile define when CUDA libs present
if (TORCH_CUDA_LIBRARIES)
    message(STATUS "Enabling CUDA support...")
    target_compile_definitions(GigaLearnCPP PRIVATE -DRG_CUDA_SUPPORT)
endif()

# Set C++ version to 20
set_target_properties(GigaLearnCPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnCPP PROPERTIES CXX_STANDARD 20)

# Include RLGymCPP
add_subdirectory(RLGymCPP)
target_link_libraries(GigaLearnCPP PUBLIC RLGymCPP)

# Include JSON
target_include_directories(GigaLearnCPP PUBLIC "${PROJECT_SOURCE_DIR}/libsrc/json")

# Include python
find_package(Python COMPONENTS Interpreter Development)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(GigaLearnCPP PUBLIC ${PYTHON_LIBRARIES})
message(STATUS "Found Python:")
message(STATUS "PYTHON_LIBRARIES: ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
add_definitions(-DPY_EXEC_PATH="${Python_EXECUTABLE}") # Give C++ access to the executable path

# Include pybind11
add_subdirectory(pybind11)
target_link_libraries(GigaLearnCPP PUBLIC pybind11::embed)

foreach(_py_file IN ITEMS metric_receiver.py render_receiver.py)
    configure_file("./python_scripts/${_py_file}" "../python_scripts/${_py_file}" COPY)
endforeach()

# After detecting LIBTORCH_ROOT / finding Torch, print diagnostics and confirm key headers
if(DEFINED TORCH_INSTALL_PREFIX)
    set(_torch_prefix ${TORCH_INSTALL_PREFIX})
elseif(DEFINED LIBTORCH_ROOT)
    set(_torch_prefix ${LIBTORCH_ROOT})
else()
    set(_torch_prefix "")
endif()

if(_torch_prefix)
    message(STATUS "Using libtorch root: ${_torch_prefix}")
    set(_include_candidates
        "${_torch_prefix}/include"
        "${_torch_prefix}/include/ATen"
        "${_torch_prefix}/include/c10"
        "${_torch_prefix}/include/torch/csrc/api/include"
        "${_torch_prefix}/include/torch")

    foreach(_inc IN LISTS _include_candidates)
        if(EXISTS "${_inc}")
            message(STATUS "Found include dir: ${_inc}")
        else()
            message(WARNING "Missing include dir: ${_inc}")
        endif()
    endforeach()

    # Check for critical headers
    if(EXISTS "${_torch_prefix}/include/ATen/Context.h")
        message(STATUS "Found key header: ${_torch_prefix}/include/ATen/Context.h")
    else()
        message(WARNING "Key header ATen/Context.h not found under ${_torch_prefix}/include. "
            "If this file exists in a different location, set LIBTORCH env or -DCMAKE_PREFIX_PATH to the libtorch root.")
    endif()
else()
    message(WARNING "No libtorch root detected yet. Configure with -DCMAKE_PREFIX_PATH=<path-to-libtorch> or set LIBTORCH env var.")
endif()
