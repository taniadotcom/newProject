import torch

EXP_ID = "exp_01_2d_sine"
PROBLEM = "sine"
METHOD = "sobol"
DIM = 2
SIGMA = 0.2
N_POINTS = 8192
N_BOUNDARY = 512
EPOCHS = 5000
LR = 1e-3
DTYPE = torch.float64
RESULTS_DIR = None  # ορίζεται στο κάθε experiment
NOTES = ""
