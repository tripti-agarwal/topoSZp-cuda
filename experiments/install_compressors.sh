#!/bin/bash
#
# Install competing compressors for comparison experiments
# Builds: SZ3, ZFP, Tthresh, TopoSZ (if not already installed)
#
# Usage: bash install_compressors.sh [INSTALL_DIR]
#   Default: /lcrc/project/ECP-EZ/tripti/compressors

set -e

INSTALL_DIR="${1:-/lcrc/project/ECP-EZ/tripti/compressors}"
SRC_DIR="$INSTALL_DIR/src"
BIN_DIR="$INSTALL_DIR/bin"
LIB_DIR="$INSTALL_DIR/lib"
mkdir -p "$SRC_DIR" "$BIN_DIR" "$LIB_DIR"

echo "Installing compressors to: $INSTALL_DIR"
echo ""

###############################################################################
# SZ3 (https://github.com/szcompressor/SZ3)
###############################################################################
if [ ! -f "$BIN_DIR/sz3" ]; then
    echo "=== Building SZ3 ==="
    cd "$SRC_DIR"
    if [ ! -d SZ3 ]; then
        git clone --depth 1 https://github.com/szcompressor/SZ3.git
    fi
    cd SZ3
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    make install
    echo "  SZ3 installed: $BIN_DIR/sz3"
else
    echo "=== SZ3: already installed ==="
fi
echo ""

###############################################################################
# ZFP (https://github.com/LLNL/zfp)
###############################################################################
if [ ! -f "$BIN_DIR/zfp" ]; then
    echo "=== Building ZFP ==="
    cd "$SRC_DIR"
    if [ ! -d zfp ]; then
        git clone --depth 1 https://github.com/LLNL/zfp.git
    fi
    cd zfp
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_UTILITIES=ON
    make -j$(nproc)
    make install
    echo "  ZFP installed: $BIN_DIR/zfp"
else
    echo "=== ZFP: already installed ==="
fi
echo ""

###############################################################################
# Tthresh (https://github.com/rballester/tthresh)
###############################################################################
if [ ! -f "$BIN_DIR/tthresh" ]; then
    echo "=== Building Tthresh ==="
    cd "$SRC_DIR"
    if [ ! -d tthresh ]; then
        git clone --depth 1 https://github.com/rballester/tthresh.git
    fi
    cd tthresh
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cp tthresh "$BIN_DIR/" 2>/dev/null || true
    echo "  Tthresh installed: $BIN_DIR/tthresh"
else
    echo "=== Tthresh: already installed ==="
fi
echo ""

###############################################################################
# SZ 1.4 / SZ 2.x (https://github.com/szcompressor/SZ)
###############################################################################
if [ ! -f "$BIN_DIR/sz" ]; then
    echo "=== Building SZ (1.4/2.x) ==="
    cd "$SRC_DIR"
    if [ ! -d SZ ]; then
        git clone --depth 1 https://github.com/szcompressor/SZ.git
    fi
    cd SZ
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    make install
    echo "  SZ installed: $BIN_DIR/sz"
else
    echo "=== SZ: already installed ==="
fi
echo ""

###############################################################################
# Summary
###############################################################################
echo "========== INSTALLED COMPRESSORS =========="
for cmd in sz3 sz zfp tthresh; do
    if [ -f "$BIN_DIR/$cmd" ]; then
        echo "  ✓ $cmd: $BIN_DIR/$cmd"
    else
        # Check if the binary exists with a different name
        found=$(find "$INSTALL_DIR" -name "$cmd" -type f -executable 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            echo "  ✓ $cmd: $found"
        else
            echo "  ✗ $cmd: NOT FOUND"
        fi
    fi
done
echo ""
echo "Add to PATH: export PATH=$BIN_DIR:\$PATH"
echo "Add to LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LIB_DIR:\$LD_LIBRARY_PATH"
