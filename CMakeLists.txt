cmake_minimum_required (VERSION 3.20)

project("GigaLearnBot" LANGUAGES CXX CUDA)

# Normalize CUDA_PATH-like env vars early to avoid backslash escape issues in FindCUDA
if(DEFINED ENV{CUDA_PATH})
    file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _cuda_path_norm)
    set(ENV{CUDA_PATH} "${_cuda_path_norm}")
endif()
if(DEFINED ENV{CUDA_BIN_PATH})
    file(TO_CMAKE_PATH "$ENV{CUDA_BIN_PATH}" _cuda_bin_path_norm)
    set(ENV{CUDA_BIN_PATH} "${_cuda_bin_path_norm}")
endif()

# Enforce C++ standard globally
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# If user didn't pass a CMAKE_PREFIX_PATH, try to default it to the known local libtorch path
if (NOT DEFINED CMAKE_PREFIX_PATH)
    if (EXISTS "C:/Giga/GigaLearnCPP-Leak/libtorch")
        message(STATUS "No CMAKE_PREFIX_PATH provided, defaulting to C:/Giga/GigaLearnCPP-Leak/libtorch")
        set(CMAKE_PREFIX_PATH "C:/Giga/GigaLearnCPP-Leak/libtorch" CACHE PATH "Default libtorch path" FORCE)
    endif()
endif()

# If the user hasn't defined C/C++ compilers, default to the MSVC cl.exe you provided so compiler-id detection succeeds.
if (NOT DEFINED CMAKE_C_COMPILER)
    if (EXISTS "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")
        message(STATUS "Defaulting CMAKE_C_COMPILER to MSVC cl.exe")
        set(CMAKE_C_COMPILER "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" CACHE FILEPATH "Default MSVC C compiler" FORCE)
    endif()
endif()
if (NOT DEFINED CMAKE_CXX_COMPILER)
    if (EXISTS "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")
        message(STATUS "Defaulting CMAKE_CXX_COMPILER to MSVC cl.exe")
        set(CMAKE_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" CACHE FILEPATH "Default MSVC C++ compiler" FORCE)
    endif()
endif()

# Ensure we use the expected nvcc if present (use forward slashes to avoid CMake escape issues)
set(_cuda_default_root "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(EXISTS "${_cuda_default_root}/bin/nvcc.exe")
        message(STATUS "Defaulting CMAKE_CUDA_COMPILER to ${_cuda_default_root}/bin/nvcc.exe")
        set(CMAKE_CUDA_COMPILER "${_cuda_default_root}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
endif()
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR AND EXISTS "${_cuda_default_root}")
    set(CUDA_TOOLKIT_ROOT_DIR "${_cuda_default_root}" CACHE PATH "CUDA root" FORCE)
endif()
set(_cuda_include_dir "")
if(DEFINED CUDA_TOOLKIT_ROOT_DIR AND EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/include")
    set(_cuda_include_dir "${CUDA_TOOLKIT_ROOT_DIR}/include")
endif()

# If a C compiler is already detected, use it as the CUDA host compiler to help compiler-id detection
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    if(DEFINED CMAKE_C_COMPILER)
        message(STATUS "Setting CMAKE_CUDA_HOST_COMPILER to detected C compiler: ${CMAKE_C_COMPILER}")
        set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_C_COMPILER}" CACHE FILEPATH "CUDA host compiler" FORCE)
    elseif(DEFINED ENV{VCINSTALLDIR})
        # Best-effort fallback when running from Visual Studio; do not hardcode MSVC layout
        message(STATUS "VCINSTALLDIR detected in environment; please run configuration from a Developer Command Prompt so CMake can detect the MSVC compiler.")
    endif()
endif()

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    # Avoid configure-time GPU detection failures by defaulting to a modern arch; override with -DCMAKE_CUDA_ARCHITECTURES=...
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler")
else()
    string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler")
endif()

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
add_executable(GigaLearnBot ${FILES_SRC})

# Ensure RLBot headers use the bundled flatbuffers first to avoid ABI mismatches with system/vcpkg flatbuffers.
target_include_directories(GigaLearnBot BEFORE PRIVATE
    "${CMAKE_SOURCE_DIR}/RLBotCPP/lib/inc"
    "${CMAKE_SOURCE_DIR}/RLBotCPP/inc"
    ${_cuda_include_dir}
)
if(MSVC)
    target_compile_options(GigaLearnBot PRIVATE /GS-)
endif()

# Delay-load heavy CUDA/Torch DLLs so startup can surface clear errors instead of failing before main.
if(MSVC)
endif()

# Copy all runtime DLL dependencies (MSVC CRT, CUDA, Torch, etc.) next to the executable for reliable double-click runs.
if(MSVC)
    set(_copy_dest $<TARGET_FILE_DIR:GigaLearnBot>)
    add_custom_command(TARGET GigaLearnBot POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:GigaLearnBot>
                ${_copy_dest}
        COMMAND_EXPAND_LISTS
    )

    # Explicitly copy libtorch DLLs in case they are not picked up by TARGET_RUNTIME_DLLS
    set(_torch_dll_dir "${CMAKE_SOURCE_DIR}/libtorch/lib")
    if(EXISTS "${_torch_dll_dir}")
        file(GLOB _torch_dlls "${_torch_dll_dir}/*.dll")
        if(_torch_dlls)
            add_custom_command(TARGET GigaLearnBot POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${_torch_dlls}
                        ${_copy_dest}
            )
        endif()
    endif()

    # Copy key CUDA runtime DLLs alongside the exe for out-of-IDE runs
    if(DEFINED CUDA_TOOLKIT_ROOT_DIR AND EXISTS "${CUDA_TOOLKIT_ROOT_DIR}/bin")
        file(GLOB _cuda_dlls
            "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64*.dll"
            "${CUDA_TOOLKIT_ROOT_DIR}/bin/cublas64*.dll"
            "${CUDA_TOOLKIT_ROOT_DIR}/bin/cublasLt64*.dll"
            "${CUDA_TOOLKIT_ROOT_DIR}/bin/cufft64*.dll"
            "${CUDA_TOOLKIT_ROOT_DIR}/bin/curand64*.dll"
            "${CUDA_TOOLKIT_ROOT_DIR}/bin/cusolver64*.dll"
            "${CUDA_TOOLKIT_ROOT_DIR}/bin/cusparse64*.dll"
            "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvrtc64*.dll"
        )
        if(_cuda_dlls)
            add_custom_command(TARGET GigaLearnBot POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${_cuda_dlls}
                        ${_copy_dest}
            )
        endif()
    endif()

    # Copy the Python DLL we are linking against so embedding works from the build output folder
    find_file(_py_dll
        NAMES python313.dll python312.dll python311.dll python310.dll
        PATHS
            ${Python_RUNTIME_LIBRARY_DIRS}
            "C:/Users/hight/AppData/Local/Programs/Python/Python313"
            "C:/Users/hight/AppData/Local/Programs/Python/Python312"
            "C:/Users/hight/AppData/Local/Programs/Python/Python311"
            "C:/Users/hight/AppData/Local/Programs/Python/Python310"
    )
    if(_py_dll)
        add_custom_command(TARGET GigaLearnBot POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_py_dll}"
                    ${_copy_dest}
        )
    endif()
endif()

# Set target C++ standard explicitly
set_target_properties(GigaLearnBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnBot PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

# Make sure GigaLearnCPP is going to build in the same directory as us
# Otherwise, we won't be able to import it at runtime
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# If a local libtorch exists at the repository root, add it to CMAKE_PREFIX_PATH (helpful for IDEs)
if (EXISTS "${CMAKE_SOURCE_DIR}/libtorch/")
    message(STATUS "Top-level: Using local libtorch folder: ${CMAKE_SOURCE_DIR}/libtorch")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")
    include_directories("${CMAKE_SOURCE_DIR}/libtorch/include")
endif()

# Ensure compilers see includes from any CMAKE_PREFIX_PATH entries (helps find ATen headers)
if(DEFINED CMAKE_PREFIX_PATH)
    foreach(_p IN LISTS CMAKE_PREFIX_PATH)
        if(EXISTS "${_p}/include")
            message(STATUS "Adding include dir from CMAKE_PREFIX_PATH: ${_p}/include")
            include_directories("${_p}/include")
        endif()
    endforeach()
endif()

# Include RLGymSim_PPO
add_subdirectory(GigaLearnCPP)
target_link_libraries(GigaLearnBot PRIVATE GigaLearnCPP)

# Include RLBot
add_subdirectory(RLBotCPP)
target_link_libraries(GigaLearnBot PRIVATE RLBotCPP)

# Helpful note if compiler-id detection fails (CMakeDetermineCompilerId)
if(CMAKE_GENERATOR MATCHES "Visual Studio" AND NOT DEFINED CMAKE_CUDA_COMPILER)
    message(WARNING "Visual Studio generator selected but CMAKE_CUDA_COMPILER not set. If you intend to build with CUDA, pass -DCMAKE_CUDA_COMPILER=<path/to/nvcc> or set CMAKE_PREFIX_PATH to libtorch and run from a Developer Prompt.")
endif()
