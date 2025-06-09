import os

import numpy as np

# we want the image fed to the model to be 244x244.

NETWORK_INPUT_SIZE = 224

# we would like there to be some flexibility in the size and position of the tagret
# cell within the image fed into the network.  to do this we set a target cell size
# which is approximately the size we would like the cell to be when it is fed into
# the network.

# NOTE: with this settingrm tt the cell will take up < 30% of the image (possibly quite
# a bit less depending on the cell shape). this may seem small but it can help to
# force the network to look for the cell in the image rather than just learning
# to look in a specific location.

# especially if we add noise outside of the cell ! TODO

TARGET_CELL_SIZE = 128

# for random rotations we need to allow some extra space around the target cell

ROTATED_CELL_SIZE = np.ceil(np.sqrt(2) * NETWORK_INPUT_SIZE)

# the we will allow scaling of the cell in the X and Y directions so we will need some
# extra space to allow for this.

MAX_CROP = 32
MAX_PAD = 32

SCALED_CELL_SIZE = ROTATED_CELL_SIZE + MAX_CROP

# finally we will allow the cell to be shifted in the X and Y directions so that it is
# not always in the center of the image. that needs some more space

MAX_LEFT_SHIFT = 32
MIN_RIGHT_SHIFT = 32

SHIFTED_CELL_SIZE = SCALED_CELL_SIZE + 2 * MAX_LEFT_SHIFT

# this gives us the required patch size when we initially grab the cell patch from the
# image - but assumes the transforms will be applied in this order (the final
# transform will crop the NETWORK_INPUT_SIZE patch from the center of the patch)

PATCH_SIZE = SHIFTED_CELL_SIZE

# set the learning rate for the model and other training parameters

LEARNING_RATE = 0.0001
BATCH_SIZE = 16
EPOCHS = 20
WEIGHT_DECAY = 0.00001
AMS_GRAD = True
MOMENTUM_DECAY = 0.0004
SGD_MOMENTUM = 0
DROPOUT = 0.0

# set the number of workers for the data loader
# this is equal to the number of CPUs on the machine
# os.cpu_count is the total number of CPUs on the system
# we want the number of cpus available to us

NUM_WORKERS = len(os.sched_getaffinity(0))  #  // 2

# some filenames and paths etc

TRAINING_LOG = "../output/training_log.csv"


KERNEL_SIZE = 31
