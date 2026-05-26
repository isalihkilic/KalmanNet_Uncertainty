import torch.nn as nn
import torch
import time
from Filters.EKF import ExtendedKalmanFilter


def EKFTest(args, SysModel, test_input, test_target, test_target_cov, allStates=True,\
     randomInit = False,test_init=None, test_lengthMask=None):
    # Number of test samples
    N_T = test_target.size()[0]
    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')
    cov_loss_fn = nn.MSELoss(reduction='mean')  
    # MSE [Linear]
    MSE_EKF_linear_arr = torch.zeros(N_T)
    MSE_EKF_cov_linear_arr = torch.zeros(N_T)
    # Allocate empty tensor for output
    EKF_out = torch.zeros([N_T, SysModel.m, test_input.size()[2]]) # N_T x m x T
    KG_array = torch.zeros([N_T, SysModel.m, SysModel.n, test_input.size()[2]]) # N_T x m x n x T
    
    if not allStates:
        loc = torch.tensor([True,False,False]) # for position only
        if SysModel.m == 2: 
            loc = torch.tensor([True,False]) # for position only

    start = time.time()
    EKF = ExtendedKalmanFilter(SysModel, args)
    # Init and Forward Computation   
    if(randomInit):
        EKF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(N_T,-1,-1))        
    else:
        EKF.Init_batched_sequence(SysModel.m1x_0.view(1,SysModel.m,1).expand(N_T,-1,-1), SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(N_T,-1,-1))           
    EKF.GenerateBatch(test_input)
     
    end = time.time()
    t = end - start

    KG_array = EKF.KG_array
    EKF_out = EKF.x

    # MSE loss
    for j in range(N_T):# cannot use batch due to different length and std computation   
        if(allStates):
            if args.randomLength:
                MSE_EKF_linear_arr[j] = loss_fn(EKF.x[j,:,test_lengthMask[j]], test_target[j,:,test_lengthMask[j]]).item()
                MSE_EKF_cov_linear_arr[j] = cov_loss_fn(EKF.sigma[j,:,:,test_lengthMask[j]], test_target_cov[j,:,:,test_lengthMask[j]]).item()
            else:      
                MSE_EKF_linear_arr[j] = loss_fn(EKF.x[j,:,:], test_target[j,:,:]).item()
                MSE_EKF_cov_linear_arr[j] = cov_loss_fn(EKF.sigma[j,:,:], test_target_cov[j,:,:]).item()
        else: # mask on state
            if args.randomLength:
                MSE_EKF_linear_arr[j] = loss_fn(EKF.x[j,loc,test_lengthMask[j]], test_target[j,loc,test_lengthMask[j]]).item()
                MSE_EKF_cov_linear_arr[j] = cov_loss_fn(EKF.sigma[j,:,:,test_lengthMask[j]], test_target_cov[j,:,:,test_lengthMask[j]]).item()
            else:           
                MSE_EKF_linear_arr[j] = loss_fn(EKF.x[j,loc,:], test_target[j,loc,:]).item()
                MSE_EKF_cov_linear_arr[j] = cov_loss_fn(EKF.sigma[j,:,:], test_target_cov[j,:,:]).item()


    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)
    MSE_EKF_cov_linear_avg = torch.mean(MSE_EKF_cov_linear_arr)
    MSE_EKF_cov_dB_avg = 10 * torch.log10(MSE_EKF_cov_linear_avg)

    MSE_symmetricity = cov_loss_fn(EKF.sigma, EKF.sigma.permute(0, 2, 1, 3)).item()

    # Standard deviation
    MSE_EKF_linear_std = torch.std(MSE_EKF_linear_arr, unbiased=True)

    # Confidence interval
    EKF_std_dB = 10 * torch.log10(MSE_EKF_linear_std + MSE_EKF_linear_avg) - MSE_EKF_dB_avg
    
    print("Extended Kalman Filter - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("Extended Kalman Filter - STD:", EKF_std_dB, "[dB]")
    print("Extended Kalman Filter - MSE Cov Test:", MSE_EKF_cov_dB_avg, "[dB]")
    print("Extended Kalman Filter - Covariance Symmetricity (0 is optimal):", MSE_symmetricity)
    # Print Run Time
    print("Inference Time:", t)

    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out]


