# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Assuming a cluster node with $SLURM_TMPDIR as the data folder and the ImageNet dataset downloaded and stored at
# $SOME_DIR_TO_IMAGENET, a typical ImageNet data preparation is the following:

cd $SLURM_TMPDIR;
mkdir imagenet
cd imagenet
mkdir train

echo "setting up imagenet data (can take a few minutes)..."
tar -xf $SOME_DIR_TO_IMAGENET/ILSVRC2012_img_train.tar -C train/
cd train
for i in *.tar; do dir=${i%.tar}; mkdir -p $dir; tar xf $i -C $dir; done

cp -r $SOME_DIR_TO_IMAGENET/val "$SLURM_TMPDIR/imagenet/"  # copy all validation images

cp $SOME_DIR_TO_IMAGENET/ILSVRC2012_devkit_t12.tar.gz "$SLURM_TMPDIR/imagenet/";