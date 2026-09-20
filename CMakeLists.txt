cmake_minimum_required (VERSION 3.20)

project (szp
  VERSION 1.0.0
  DESCRIPTION "SZp Error Bounded Lossy Compressor"
  LANGUAGES C CXX
)

# Optional CUDA support
option(ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)
if(ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  message(STATUS "CUDA support enabled (arch: ${CMAKE_CUDA_ARCHITECTURES})")
endif()

# Default build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type was set. Setting build type to ${default_build_type}.")
  set(CMAKE_BUILD_TYPE ${default_build_type} CACHE
    STRING "Choose the type to build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "build shared libraries over static libraries" ON)

include(CTest)
include(GNUInstallDirs)

find_package(OpenMP COMPONENTS C CXX)
find_package(PkgConfig)
pkg_search_module(ZSTD IMPORTED_TARGET libzstd)

# ---- OpenMP SZp library (only if OpenMP CXX available) ----
if(OpenMP_CXX_FOUND)
  add_library(SZp
      SZp/src/szp.cc
      SZp/src/szp_CompressionToolkit.cc
      SZp/src/szp_TypeManager.cc
      SZp/src/szp_double.cc
      SZp/src/szp_float.cc
      SZp/src/szp_helper.cc
      SZp/src/szp_rw.cc
      SZp/src/szpd_double.cc
      SZp/src/szpd_float.cc
      SZp/src/szp_topology_3d.cc
  )
  target_compile_features(SZp PUBLIC c_std_99 cxx_std_14)
  target_link_libraries(SZp PUBLIC OpenMP::OpenMP_CXX)
  if(ZSTD_FOUND)
    target_link_libraries(SZp PUBLIC PkgConfig::ZSTD)
  endif()
  target_include_directories(SZp
      PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/SZp/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SZp>
  )
  install(TARGETS SZp EXPORT SZpTargets)
else()
  message(STATUS "OpenMP CXX not found — SZp (OpenMP) library will not be built.")
endif()

# ---- CUDA library ----
if(ENABLE_CUDA)
  add_library(SZp_cuda
      SZp/src/szp_cuda_compress.cu
      SZp/src/szp_cuda_decompress.cu
      SZp/src/szp_cuda_topology.cu
      SZp/src/szp_cuda_topology_3d.cu
  )
  target_compile_features(SZp_cuda PUBLIC cxx_std_17)
  target_link_libraries(SZp_cuda PUBLIC CUDA::cudart)
  target_include_directories(SZp_cuda
      PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/SZp/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SZp>
  )
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(SZp_cuda PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
  endif()
  install(TARGETS SZp_cuda EXPORT SZpTargets)

  # Test programs need both SZp (OpenMP) and SZp_cuda
  if(OpenMP_CXX_FOUND)
    add_executable(test_cuda_compress example/test_cuda_compress.cu)
    target_link_libraries(test_cuda_compress SZp SZp_cuda OpenMP::OpenMP_CXX)
    target_include_directories(test_cuda_compress PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/SZp/include)

    add_executable(test_cuda_decompress example/test_cuda_decompress.cu)
    target_link_libraries(test_cuda_decompress SZp SZp_cuda OpenMP::OpenMP_CXX)
    target_include_directories(test_cuda_decompress PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/SZp/include)

    add_executable(test_cuda_topology example/test_cuda_topology.cu)
    target_link_libraries(test_cuda_topology SZp SZp_cuda OpenMP::OpenMP_CXX)
    target_include_directories(test_cuda_topology PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/SZp/include)

    add_executable(test_cuda_topology_3d example/test_cuda_topology_3d.cu)
    target_link_libraries(test_cuda_topology_3d SZp SZp_cuda OpenMP::OpenMP_CXX)
    target_include_directories(test_cuda_topology_3d PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/SZp/include)

    add_executable(compare_with_lopc example/compare_with_lopc.cu)
    target_link_libraries(compare_with_lopc SZp SZp_cuda OpenMP::OpenMP_CXX)
    target_include_directories(compare_with_lopc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/SZp/include)
  else()
    message(STATUS "OpenMP CXX not found — test programs skipped. SZp_cuda library will still be built.")
  endif()
endif()

# ---- Install / export (only if at least one target was created) ----
get_property(_export_targets GLOBAL PROPERTY PACKAGES_FOUND)
# Check if any installable targets exist
if(OpenMP_CXX_FOUND OR ENABLE_CUDA)
  install(EXPORT SZpTargets NAMESPACE SZp:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SZp/)
  export(EXPORT SZpTargets FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake/SZpTargets.cmake NAMESPACE SZp::)
  include(CMakePackageConfigHelpers)
  configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/SZpConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/SZpConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SZp
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SZpConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY AnyNewerVersion
  )
  install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/SZpConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/SZpConfigVersion.cmake"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SZp
  )
endif()

install(DIRECTORY SZp/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/SZp)
