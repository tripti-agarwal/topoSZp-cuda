# SZp (Also known as fZ-light)

* Developers: Tripti Agarwal(kernels, entries, and examples), Sheng Di (utility)
* Email: tripti.coer@gmail.com   tripti.agarwal@utah.edu
  



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


