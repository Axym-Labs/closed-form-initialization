SEED = 7

DATASETS = ["cifar10", "cifar100"]
SUITES = ["random-affine", "random-crop", "same-class"]

W = 512
DEPTH = 3
N_TRAIN = 6000
N_TEST = 1000

ANALYTIC_AUG_REPEATS = 2

BATCH_SIZE = 256
EPOCHS = 10
BACKPROP_EPOCHS = 10
GREEDY_BT_EPOCHS = 10
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

HEAD_REG = 100.0
LAMBDA_REG = 1.0
ACTIVATION = "relu"
DUAL_MAPPING = False
OUTPUT_SOURCE = "pre-hidden"
CENTER_AFTER_HIDDEN = False
BT_USE_PROJECTOR = True

ANALYTIC_MODELS = [
    "closed-form-barlow",
    "paper-cca-shared",
    "whitened-shared-pca",
]

LINEAR_MODELS = [
    "linear-regression",
    "pca",
    "random",
]

LEARNED_MODELS = [
    "supervised-backprop",
    "barlow-twins-backprop",
    "barlow-twins-greedy-post",
]
