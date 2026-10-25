# tests/CMakeLists.txt — CTest integration for topoSZp-cuda
#
# Build and register automated tests. Requires both SZp (OpenMP) and
# SZp_cuda (CUDA) targets — guarded by the parent CMakeLists.txt.

# ---- Compress / decompress round-trip ----
add_executable(test_compress_decompress test_compress_decompress.cu)
target_link_libraries(test_compress_decompress SZp SZp_cuda OpenMP::OpenMP_CXX)
add_test(NAME compress_decompress COMMAND test_compress_decompress)

# ---- 2D topology preservation ----
add_executable(test_topology_2d test_topology_2d.cu)
target_link_libraries(test_topology_2d SZp SZp_cuda OpenMP::OpenMP_CXX)
add_test(NAME topology_2d COMMAND test_topology_2d)

# ---- 3D topology preservation ----
add_executable(test_topology_3d test_topology_3d.cu)
target_link_libraries(test_topology_3d SZp SZp_cuda OpenMP::OpenMP_CXX)
add_test(NAME topology_3d COMMAND test_topology_3d)

# ---- OpenMP ↔ CUDA cross-path round-trip ----
add_executable(test_roundtrip test_roundtrip.cu)
target_link_libraries(test_roundtrip SZp SZp_cuda OpenMP::OpenMP_CXX)
add_test(NAME roundtrip COMMAND test_roundtrip)
