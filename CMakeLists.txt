cmake_minimum_required (VERSION 3.20)

#define the project
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
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Allow __host__ __device__ lambdas for Thrust
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
  message(STATUS "CUDA support enabled (arch: ${CMAKE_CUDA_ARCHITECTURES})")
endif()

#correct was to set a default build type
# https://blog.kitware.com/cmake-and-the-default-build-type/
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
pkg_search_module(ZSTD  IMPORTED_TARGET libzstd)

# Build the OpenMP SZp library only if OpenMP CXX was found
if(OpenMP_CXX_FOUND)
add_library(
    SZp
        SZp/src/szp.cc
        SZp/src/szp_CompressionToolkit.cc
        SZp/src/szp_TypeManager.cc
        SZp/src/szp_double.cc
        SZp/src/szp_float.cc
        SZp/src/szp_helper.cc
        SZp/src/szp_rw.cc
        SZp/src/szpd_double.cc
        SZp/src/szpd_float.cc
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
endif() # OpenMP_CXX_FOUND

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
install(DIRECTORY SZp/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/SZp)

# ---- CUDA library and example targets ----
if(ENABLE_CUDA)
  add_library(SZp_cuda
      SZp/src/szp_cuda_compress.cu
      SZp/src/szp_cuda_decompress.cu
      SZp/src/szp_cuda_topology.cu
  )
  target_compile_features(SZp_cuda PUBLIC cxx_std_14)
  target_link_libraries(SZp_cuda PUBLIC CUDA::cudart)
  target_include_directories(SZp_cuda
      PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/SZp/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/SZp>
  )
  # Set CUDA architectures (default to common ones if not specified)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(SZp_cuda PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
  endif()

  install(TARGETS SZp_cuda EXPORT SZpTargets)

  install(TARGETS SZp_cuda EXPORT SZpTargets)

  # Example/test programs (need both SZp and SZp_cuda for cross-validation)
  if(OpenMP_CXX_FOUND)
    add_executable(test_cuda_compress example/test_cuda_compress.cu)
    target_link_libraries(test_cuda_compress SZp SZp_cuda OpenMP::OpenMP_CXX)
    target_include_directories(test_cuda_compress PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/SZp/include)

    add_executable(test_cuda_decompress example/test_cuda_decompress.cu)
    target_link_libraries(test_cuda_decompress SZp SZp_cuda OpenMP::OpenMP_CXX)
    target_include_directories(test_cuda_decompress PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/SZp/include)
  else()
    message(STATUS "OpenMP CXX not found — test programs (which cross-validate CUDA vs OpenMP) are skipped. SZp_cuda library will still be built.")
  endif()
endif()
