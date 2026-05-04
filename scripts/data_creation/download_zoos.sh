#!/bin/bash
# Fetch the GHRP model zoos used by the paper's reproduction pipeline.
#
# Required:
#   MODELXRAY_GHRP_DIR        : destination for the SCZ STL10 zoo (D1)
#   MODELXRAY_RESNET_MZ_ROOT  : destination for the ResNet18-TinyImageNet zoo (D5)
#
# Both are large (a few GB each unzipped) and only required for end-to-end reruns.

set -euo pipefail

if [ -z "${MODELXRAY_GHRP_DIR:-}" ]; then
    echo "MODELXRAY_GHRP_DIR not set; refusing to download." >&2
    echo "  export MODELXRAY_GHRP_DIR=/path/to/ghrp/zoos" >&2
    exit 1
fi
if [ -z "${MODELXRAY_RESNET_MZ_ROOT:-}" ]; then
    echo "MODELXRAY_RESNET_MZ_ROOT not set; refusing to download." >&2
    echo "  export MODELXRAY_RESNET_MZ_ROOT=/path/to/resnet/checkpoints" >&2
    exit 1
fi

mkdir -p "$MODELXRAY_GHRP_DIR" "$MODELXRAY_RESNET_MZ_ROOT"

# D1: SCZ STL10 small-CNN zoo (Schurholt et al. 2022, GHRP)
echo "[1/2] Downloading SCZ STL10 zoo into $MODELXRAY_GHRP_DIR ..."
cd "$MODELXRAY_GHRP_DIR"
for f in stl_small_hyp_fix stl_small_hyp_rand stl_small_seed; do
    if [ ! -f "${f}.zip" ]; then
        wget -q --show-progress \
            "https://zenodo.org/records/6631784/files/${f}.zip?download=1" \
            -O "${f}.zip"
    fi
    unzip -n -q "${f}.zip"
done

# D5: ResNet18-TinyImageNet subset
echo "[2/2] Downloading ResNet18-TinyImageNet zoo into $MODELXRAY_RESNET_MZ_ROOT ..."
cd "$MODELXRAY_RESNET_MZ_ROOT"
if [ ! -f "tiny-imagenet_resnet18_subset.zip" ]; then
    wget -q --show-progress \
        "https://zenodo.org/records/7023278/files/tiny-imagenet_resnet18_subset.zip?download=1" \
        -O tiny-imagenet_resnet18_subset.zip
fi
unzip -n -q tiny-imagenet_resnet18_subset.zip

echo "Done."
