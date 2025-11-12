AUTOMAKE_OPTIONS=foreign
AM_LDFLAGS = @OPENMP_CXXFLAGS@
AM_FCFLAGS = -I../SZp/include -I../SZp
AM_CXXFLAGS = -I../SZp/include -I../SZp
if OPENMP
AM_CXXFLAGS+=-fopenmp
endif

bin_PROGRAMS=testfloat_compress_fastmode1 testfloat_decompress_fastmode1 test_multidim

testfloat_compress_fastmode1_SOURCES=testfloat_compress_fastmode1_topology.cc
testfloat_compress_fastmode1_CXXFLAGS=-I../SZp/include -I../SZp
testfloat_compress_fastmode1_LDADD=../SZp/.libs/libszp.a -lm -lgomp
testfloat_decompress_fastmode1_SOURCES=testfloat_decompress_fastmode1_topology.cc
testfloat_decompress_fastmode1_CXXFLAGS=-I../SZp/include -I../SZp
testfloat_decompress_fastmode1_LDADD=../SZp/.libs/libszp.a -lm -lgomp
test_multidim_SOURCES=test_multidim.cc
test_multidim_LDADD=../SZp/.libs/libszp.a -lm -lgomp