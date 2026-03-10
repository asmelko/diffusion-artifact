cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
  diffuse
  VERSION 0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_ASAN OFF)
option(RISCV_CROSSCOMPILE OFF)
option(NO_PAPI OFF)
option(REDUCED_KERNELS_SET OFF)

if(USE_ASAN)
  add_compile_options(-fsanitize=address -g)
  add_link_options(-fsanitize=address)
endif()

if(RISCV_CROSSCOMPILE)
  set(NO_PAPI ON CACHE BOOL "Disable PAPI when cross-compiling for RISC-V" FORCE)
endif()

if(MSVC)
  set(DIFFUSE_CPP_COMPILE_OPTIONS /W4 /bigobj)
else()
  if(RISCV_CROSSCOMPILE)
    set(DIFFUSE_CPP_COMPILE_OPTIONS -Wall -Wextra -pedantic)

    set(CMAKE_CXX_FLAGS
        "-O3 -ffast-math -mepi -mllvm -combiner-store-merging=0 -mllvm -vectorizer-use-vp-strided-load-store -mcpu=avispado -mllvm -disable-loop-idiom-memcpy -mllvm -disable-loop-idiom-memset -Xclang -target-feature -Xclang +does-not-implement-vszext -Xclang -target-feature -Xclang +does-not-implement-tu -mllvm -riscv-uleb128-reloc=0 -fno-slp-vectorize"
    )
    # set(CMAKE_CXX_FLAGS "-Rpass-analysis=loop-vectorize -Rpass-missed=loop-vectorize -Rpass=loop-vectorize ")
  else()
    set(DIFFUSE_CPP_COMPILE_OPTIONS -Wall -Wextra -pedantic
                                    -Wno-interference-size -march=native)
  endif()
endif()

include(FetchContent)

# Fetch noarr
FetchContent_Declare(
  noarr_structures
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL file://${CMAKE_CURRENT_SOURCE_DIR}/submodules/noarr-structures)
FetchContent_MakeAvailable(noarr_structures)

# Fetch argparse
FetchContent_Declare(
  argparse
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL file://${CMAKE_CURRENT_SOURCE_DIR}/submodules/argparse)
FetchContent_MakeAvailable(argparse)

# Fetch json
FetchContent_Declare(
  json
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL file://${CMAKE_CURRENT_SOURCE_DIR}/submodules/json)
FetchContent_MakeAvailable(json)

# Fetch highway
FetchContent_Declare(
  highway
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL file://${CMAKE_CURRENT_SOURCE_DIR}/submodules/highway)

option(HWY_ENABLE_TESTS OFF)
option(HWY_ENABLE_CONTRIB OFF)
option(HWY_ENABLE_INSTALL OFF)
option(HWY_ENABLE_EXAMPLES OFF)

FetchContent_MakeAvailable(highway)

file(GLOB_RECURSE src_files "src/*")

if(REDUCED_KERNELS_SET)
  list(REMOVE_ITEM src_files 
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cubed_mix_thomas_solver_t.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cubed_thomas_solver_t.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/least_compute_thomas_solver_s_t.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/least_memory_thomas_solver_d_f.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/least_memory_thomas_solver_d_t.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/least_memory_thomas_solver_t.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/space-dependent-diffusion/least_memory_thomas_solver_t.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/space-dependent-diffusion/partial_blocking.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/space-dependent-diffusion/full_blocking.cpp"
  )
endif()

add_executable(diffuse ${src_files})

if(REDUCED_KERNELS_SET)
  target_compile_definitions(diffuse PRIVATE REDUCED_KERNELS_SET)
endif()

target_compile_options(
  diffuse PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${DIFFUSE_CPP_COMPILE_OPTIONS}>)

target_include_directories(
  diffuse PRIVATE ${noarr_structures_SOURCE_DIR}/include
                  ${argparse_SOURCE_DIR}/include ${highway_SOURCE_DIR})

target_link_libraries(diffuse nlohmann_json::nlohmann_json)

target_link_libraries(diffuse hwy)

if(RISCV_CROSSCOMPILE)
  find_package(OpenMP REQUIRED)
  link_libraries(${CMAKE_CURRENT_SOURCE_DIR}/.rave/build/parallel/ompt)
else()
  find_package(OpenMP REQUIRED)
  target_link_libraries(diffuse OpenMP::OpenMP_CXX)
endif()

find_package(LAPACK)
if(NOT LAPACK_FOUND)
  message(STATUS "LAPACK not found. LAPACK-based kernels will not run.")
else()
  target_link_libraries(diffuse LAPACK::LAPACK)
  target_compile_definitions(diffuse PRIVATE HAS_LAPACK)
endif()

# Find and link PAPI
find_package(PkgConfig REQUIRED)
pkg_check_modules(PAPI IMPORTED_TARGET papi)

if (NO_PAPI)
  target_compile_definitions(diffuse PRIVATE NO_PAPI)
endif()

if(PAPI_FOUND)

  target_link_libraries(diffuse PkgConfig::PAPI)

elseif(NOT NO_PAPI)

  message(STATUS "'papi' will be built from source:")

  # Fetch papi
  FetchContent_Declare(
    papi
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL file://${CMAKE_CURRENT_SOURCE_DIR}/submodules/papi)
  FetchContent_MakeAvailable(papi)

  execute_process(COMMAND ./configure --prefix=${papi_BINARY_DIR}
                  WORKING_DIRECTORY ${papi_SOURCE_DIR}/src)

  add_custom_target(
    papi_build
    BYPRODUCTS ${papi_BINARY_DIR}/lib/libpapi.a
    COMMAND make -j
    COMMAND make install
    WORKING_DIRECTORY ${papi_SOURCE_DIR}/src)

  add_dependencies(diffuse papi_build)

  target_include_directories(diffuse PRIVATE ${papi_BINARY_DIR}/include)
  target_link_libraries(diffuse ${papi_BINARY_DIR}/lib/libpapi.a)

  set(CMAKE_PREFIX_PATH
      ${papi_BINARY_DIR}
      CACHE PATH "Path to installed packages" FORCE)

endif()
