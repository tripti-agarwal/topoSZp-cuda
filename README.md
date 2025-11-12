# SZp (Also known as fZ-light)

* Developers: Tripti Agarwal(kernels, entries, and examples), Sheng Di (utility)
* Email: tripti.coer@gmail.com   tripti.agarwal@utah.edu
  

This is the official repository of SZp, an extreme-fast error-bounded lossy compressor. It is a CPU compressor (supporting OpenMP). 
The design and optimizations of SZp are published under the name fZ-light in SC '24.

## Installation
Configure and build the SZp:
```bash
# Decompress the downloaded files
# Change to the SZp directory and set up the installation directory:
cd SZp
mkdir install

# Run the configuration script:
./configure --prefix=$(pwd)/install --enable-openmp

# Compile the SZp using multiple threads:
make -j

# Install the compiled SZp:
make install

```

## Run SZp
```bash
export OMP_NUM_THREADS=$NUMTHREADS
```


