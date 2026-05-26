import torch
import torch.nn as nn
from datetime import datetime

import Filters.EKF_test as EKF_test

from Simulations.Extended_sysmdl import SystemModelWithStateUncertainty
import Simulations.config as config
from Simulations.utils import Decimate_and_perturbate_Data,Short_Traj_Split
from Simulations.Lorenz_Atractor.parameters import m1x_0, m2x_0, m, n,delta_t_gen,delta_t,\
f, h, h_nobatch, fInacc, Q_structure, R_structure, f_gen

from Pipelines.Pipeline_EKF import Pipeline_EKF
from KNet.KalmanNet_nn import KalmanNetNN

from Plot import Plot_extended as Plot

print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
### dataset parameters
args.N_E = 1000
args.N_CV = 10
args.N_T = 10
args.T = 3000
args.T_test = 3000
### training parameters
args.use_cuda = True # use GPU or not
args.n_steps = 2000
args.n_batch = 8
args.lr = 1e-4
args.wd = 1e-4

if args.use_cuda:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print("Using GPU")
   else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

offset = 0 # offset for the data
chop = False # whether to chop the dataset sequences into smaller ones
path_results = 'KNet/'
DatafolderName = 'Simulations/Lorenz_Atractor/data/'
DatafileName = 'data_gen5.pt'
r = torch.tensor([1])
lambda_q = torch.tensor([0.3873])#([0.00000000001])

Q = (lambda_q[0]**2) * Q_structure
R = (r[0]**2) * R_structure 
# True Model
sys_model_true = SystemModelWithStateUncertainty(f_gen, Q, h, R, args.T, args.T_test,m,n)
sys_model_true.InitSequence(m1x_0.unsqueeze(0), m2x_0.unsqueeze(0))

sys_model_true.GenerateSequence(Q, R, 6000000)

torch.save([sys_model_true.x.unsqueeze(0), sys_model_true.sigma_prior.unsqueeze(0), sys_model_true.sigma_post.unsqueeze(0)], DatafolderName + DatafileName)

# Model with partial Info
#sys_model = SystemModel(fInacc, Q, h, R, args.T, args.T_test,m,n)
#sys_model.InitSequence(m1x_0, m2x_0)