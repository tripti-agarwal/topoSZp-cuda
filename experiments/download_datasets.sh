#!/bin/bash
#
# Download SDRBench datasets for TopoSZp-CUDA experiments
#
# Usage: bash download_datasets.sh [DEST_DIR]
#   Default DEST_DIR: /lcrc/project/ECP-EZ/tripti/SDRBench-data

set -e

DEST=${1:-/lcrc/project/ECP-EZ/tripti/SDRBench-data}
mkdir -p "$DEST"
cd "$DEST"

echo "Downloading SDRBench datasets to: $DEST"
echo ""

###############################################################################
# Hurricane ISABEL — 100x500x500, float32, 13 fields
###############################################################################
if [ ! -d "$DEST/Hurricane-ISABEL" ]; then
    echo "=== Downloading Hurricane ISABEL (100x500x500) ==="
    wget -q --show-progress -O Hurricane-ISABEL.tar.gz \
        "https://g-d0cd3f.fd635.8443.data.globus.org/raw-data/Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz"
    mkdir -p Hurricane-ISABEL
    tar xzf Hurricane-ISABEL.tar.gz -C Hurricane-ISABEL --strip-components=1
    rm -f Hurricane-ISABEL.tar.gz
    echo "  Done: $(ls Hurricane-ISABEL/*.f32 2>/dev/null | wc -l) files"
else
    echo "=== Hurricane ISABEL: already exists, skipping ==="
fi
echo ""

###############################################################################
# NYX — 512x512x512, float32, 6 fields
###############################################################################
if [ ! -d "$DEST/NYX" ]; then
    echo "=== Downloading NYX (512x512x512) ==="
    wget -q --show-progress -O NYX.tar.gz \
        "https://g-d0cd3f.fd635.8443.data.globus.org/raw-data/EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz"
    mkdir -p NYX
    tar xzf NYX.tar.gz -C NYX --strip-components=1
    rm -f NYX.tar.gz
    echo "  Done: $(ls NYX/*.f32 2>/dev/null | wc -l) files"
else
    echo "=== NYX: already exists, skipping ==="
fi
echo ""

###############################################################################
# SCALE-LETKF — 98x1200x1200, float32, 13 fields
###############################################################################
if [ ! -d "$DEST/SCALE-LETKF" ]; then
    echo "=== Downloading SCALE-LETKF (98x1200x1200) ==="
    wget -q --show-progress -O SCALE.tar.gz \
        "https://g-d0cd3f.fd635.8443.data.globus.org/raw-data/SCALE_LETKF/SDRBENCH-SCALE-98x1200x1200.tar.gz"
    mkdir -p SCALE-LETKF
    tar xzf SCALE.tar.gz -C SCALE-LETKF --strip-components=1
    rm -f SCALE.tar.gz
    echo "  Done: $(ls SCALE-LETKF/*.f32 2>/dev/null | wc -l) files"
else
    echo "=== SCALE-LETKF: already exists, skipping ==="
fi
echo ""

###############################################################################
# Miranda — 256x384x384, float64, 7 fields
###############################################################################
if [ ! -d "$DEST/Miranda" ]; then
    echo "=== Downloading Miranda (256x384x384) ==="
    wget -q --show-progress -O Miranda.tar.gz \
        "https://g-d0cd3f.fd635.8443.data.globus.org/raw-data/Miranda/SDRBENCH-Miranda-256x384x384.tar.gz"
    mkdir -p Miranda
    tar xzf Miranda.tar.gz -C Miranda --strip-components=1
    rm -f Miranda.tar.gz
    echo "  Done: $(ls Miranda/*.f64 Miranda/*.f32 2>/dev/null | wc -l) files"
else
    echo "=== Miranda: already exists, skipping ==="
fi
echo ""

###############################################################################
# Summary
###############################################################################
echo "========== DOWNLOAD COMPLETE =========="
echo ""
for d in Hurricane-ISABEL NYX SCALE-LETKF Miranda; do
    if [ -d "$DEST/$d" ]; then
        count=$(ls "$DEST/$d"/*.f32 "$DEST/$d"/*.f64 "$DEST/$d"/*.bin "$DEST/$d"/*.dat 2>/dev/null | wc -l)
        size=$(du -sh "$DEST/$d" | cut -f1)
        echo "  $d: $count files, $size"
    fi
done
echo ""
echo "Total: $(du -sh "$DEST" | cut -f1)"
