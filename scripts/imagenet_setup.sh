# Copyright (c) 2023. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# This script assumes a cluster node with $SLURM_TMPDIR as the data folder and the ImageNet dataset stored at $IMAGENET.

# Before running this script, run the following two commands (depending on the cluster configuration):
# export IMAGENET="/network/datasets/imagenet/"  # your path to the ImageNet dataset
# chmod +x scripts/imagenet_setup.sh             # make this script executable

# then run the script and get a coffee (it will take a few minutes to unpack the data)
# ./scripts/imagenet_setup.sh

root="$(dirname "$(readlink -f -- "$0" || realpath -- "$0")")"  # get the current directory

cd "$SLURM_TMPDIR";
mkdir imagenet
cd imagenet
mkdir train

echo "Setting up ImageNet data (takes a few minutes depending on the environment)..."

# unpack train data (can be commented for eval)
tar -xf "$IMAGENET/ILSVRC2012_img_train.tar" -C train/
cd train
for i in *.tar; do dir=${i%.tar}; mkdir -p $dir; tar xf $i -C $dir; done

cp "$IMAGENET/ILSVRC2012_devkit_t12.tar.gz" "$SLURM_TMPDIR/imagenet/";  # copy meta-data

# unpack validation data (2 options)

# option 1 (if $IMAGENET already contains validation data in the image folder format)
# cp -r "$IMAGENET/../imagenet.var/imagenet_torchvision/val" "$SLURM_TMPDIR/imagenet/"  # copy all validation images

# option 2 (based on https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
cd ..
mkdir val
tar -xf "$IMAGENET/ILSVRC2012_img_val.tar" -C val/
cd val
chmod +x "$root/valprep.sh"
"$root/valprep.sh"

echo "ImageNet preparation done!"