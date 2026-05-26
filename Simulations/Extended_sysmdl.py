"""# **Class: System Model for Non-linear Cases**

1 Store system model parameters: 
    state transition function f, 
    observation function h, 
    process noise Q, 
    observation noise R, 
    train&CV dataset sequence length T,
    test dataset sequence length T_test,
    state dimension m,
    observation dimension n, etc.

2 Generate datasets for non-linear cases
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class SystemModel:

    def __init__(self, f, Q, h, R, T, T_test, m, n, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.f = f
        self.m = m
        self.Q = Q
        #########################
        ### Observation Model ###
        #########################
        self.h = h
        self.n = n
        self.R = R
        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S

        self.Sigma_post = None

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

        self.m1x_0_batch = m1x_0_batch
        self.x_prev = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        # Pre allocate an array for current state
        self.x = torch.zeros(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.zeros(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate Sequence Iteratively
        for t in range(0, T):

            ########################
            #### State Evolution ###
            ########################   
            if torch.equal(Q_gen,torch.zeros(self.m,self.m)):# No noise
                 xt = self.f(self.x_prev)   
            elif self.m == 1: # 1 dim noise
                xt = self.f(self.x_prev)
                eq = torch.normal(mean=0, std=Q_gen)
                # Additive Process Noise
                xt = torch.add(xt,eq)
            else:            
                xt = self.f(self.x_prev)
                mean = torch.zeros([self.m])              
                distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
                eq = distrib.rsample()
                eq = torch.reshape(eq[:], xt.size())
                # Additive Process Noise
                xt = torch.add(xt,eq)

            ################
            ### Emission ###
            ################
            yt = self.h(xt)
            # Observation Noise         
            if self.n == 1: # 1 dim noise
                er = torch.normal(mean=0, std=R_gen)
                # Additive Observation Noise
                yt = torch.add(yt,er)
            else:  
                mean = torch.zeros([self.n])            
                distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
                er = distrib.rsample()
                er = torch.reshape(er[:], yt.size())       
                # Additive Observation Noise
                yt = torch.add(yt,er)
            
            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt)

            # Save Current Observation to Trajectory Array
            self.y[:, t] = torch.squeeze(yt)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, args, size, T, randomInit=False):
        if(randomInit):
            # Allocate Empty Array for Random Initial Conditions
            self.m1x_0_rand = torch.zeros(size, self.m, 1)
            if args.distribution == 'uniform':
                ### if Uniform Distribution for random init
                for i in range(size):           
                    initConditions = torch.rand_like(self.m1x_0) * args.variance
                    self.m1x_0_rand[i,:,0:1] = initConditions.view(self.m,1)     
            
            elif args.distribution == 'normal':
                ### if Normal Distribution for random init
                for i in range(size):
                    distrib = MultivariateNormal(loc=torch.squeeze(self.m1x_0), covariance_matrix=self.m2x_0)
                    initConditions = distrib.rsample().view(self.m,1)
                    self.m1x_0_rand[i,:,0:1] = initConditions
            else:
                raise ValueError('args.distribution not supported!')
            
            self.Init_batched_sequence(self.m1x_0_rand, self.m2x_0)### for sequence generation
        else: # fixed init
            initConditions = self.m1x_0.view(1,self.m,1).expand(size,-1,-1)
            self.Init_batched_sequence(initConditions, self.m2x_0)### for sequence generation
    
        if(args.randomLength):
            # Allocate Array for Input and Target (use zero padding)
            self.Input = torch.zeros(size, self.n, args.T_max)
            self.Target = torch.zeros(size, self.m, args.T_max)
            self.lengthMask = torch.zeros((size,args.T_max), dtype=torch.bool)# init with all false
            # Init Sequence Lengths
            T_tensor = torch.round((args.T_max-args.T_min)*torch.rand(size)).int()+args.T_min # Uniform distribution [100,1000]
            for i in range(0, size):
                # Generate Sequence
                self.GenerateSequence(self.Q, self.R, T_tensor[i].item())
                # Training sequence input
                self.Input[i, :, 0:T_tensor[i].item()] = self.y             
                # Training sequence output
                self.Target[i, :, 0:T_tensor[i].item()] = self.x
                # Mask for sequence length
                self.lengthMask[i, 0:T_tensor[i].item()] = True

        else:
            # Allocate Empty Array for Input
            self.Input = torch.empty(size, self.n, T)
            # Allocate Empty Array for Target
            self.Target = torch.empty(size, self.m, T)

            # Set x0 to be x previous
            self.x_prev = self.m1x_0_batch
            xt = self.x_prev

            # Generate in a batched manner
            for t in range(0, T):
                ########################
                #### State Evolution ###
                ########################   
                if torch.equal(self.Q,torch.zeros(self.m,self.m)):# No noise
                    xt = self.f(self.x_prev)
                elif self.m == 1: # 1 dim noise
                    xt = self.f(self.x_prev)
                    eq = torch.normal(mean=torch.zeros(size), std=self.Q).view(size,1,1)
                    # Additive Process Noise
                    xt = torch.add(xt,eq)
                else:            
                    xt = self.f(self.x_prev)
                    mean = torch.zeros([size, self.m])              
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=self.Q)
                    eq = distrib.rsample().view(size,self.m,1)
                    # Additive Process Noise
                    xt = torch.add(xt,eq)

                ################
                ### Emission ###
                ################
                # Observation Noise
                if torch.equal(self.R,torch.zeros(self.n,self.n)):# No noise
                    yt = self.h(xt)
                elif self.n == 1: # 1 dim noise
                    yt = self.h(xt)
                    er = torch.normal(mean=torch.zeros(size), std=self.R).view(size,1,1)
                    # Additive Observation Noise
                    yt = torch.add(yt,er)
                else:  
                    yt =  self.h(xt)
                    mean = torch.zeros([size,self.n])            
                    distrib = MultivariateNormal(loc=mean, covariance_matrix=self.R)
                    er = distrib.rsample().view(size,self.n,1)          
                    # Additive Observation Noise
                    yt = torch.add(yt,er)

                ########################
                ### Squeeze to Array ###
                ########################

                # Save Current State to Trajectory Array
                self.Target[:, :, t] = torch.squeeze(xt,2)

                # Save Current Observation to Trajectory Array
                self.Input[:, :, t] = torch.squeeze(yt,2)

                ################################
                ### Save Current to Previous ###
                ################################
                self.x_prev = xt

class SystemModelWithStateUncertainty:

    def __init__(self, f, Q, h, R, T, T_test, m, n, particle_count=1000, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.f = f
        self.m = m
        self.Q = Q
        #########################
        ### Observation Model ###
        #########################
        self.h = h
        self.n = n
        self.R = R
        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S



        self.particle_count = particle_count

    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

        self.m1x_0_batch = m1x_0_batch
        self.x_prev = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R

    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen, T):
        device = self.m1x_0.device

        # =========================
        # STORAGE
        # =========================
        self.x = torch.zeros(self.m, T, device=device)
        self.y = torch.zeros(self.n, T, device=device)
        self.sigma_prior = torch.zeros(self.m, self.m, T, device=device)
        self.sigma_post  = torch.zeros(self.m, self.m, T, device=device)

        # =========================
        # TRUE STATE INIT
        # =========================
        x_true = self.m1x_0.view(-1)   # [m]

        # =========================
        # PARTICLE INIT
        # =========================
        Q0 = Q_gen if Q_gen.ndim == 2 else Q_gen[0]
        # Add a tiny epsilon to the diagonal to prevent Cholesky crashes
        L0 = torch.linalg.cholesky(Q0 + 1e-6 * torch.eye(self.m, device=device))

        # Default to 500 particles if not defined elsewhere
        Np_half = self.particle_count // 2 if hasattr(self, 'particle_count') else 500
        eps = torch.randn(Np_half, self.m, device=device)
        eps = torch.cat([eps, -eps], dim=0)  # Antithetic sampling
        Np = eps.shape[0]

        particles = x_true.unsqueeze(0) + eps @ L0.T   # [Np, m]
        weights = torch.ones(Np, device=device) / Np

        for t in range(T):
            print(t)

            # =========================
            # 1. TRUE STATE EVOLUTION
            # =========================
            # Format to [batch_size=1, m, 1] for f(), then back to [m]
            x_true_input = x_true.view(1, self.m, 1)
            x_true = self.f(x_true_input).view(-1)   

            # =========================
            # 2. PARTICLE PROPAGATION
            # =========================
            # Format to [batch_size=Np, m, 1] for f(), then back to [Np, m]
            particles_input = particles.view(Np, self.m, 1)
            particles_f = self.f(particles_input).view(Np, self.m)

            if self.m == 1:
                noise = torch.normal(mean=0.0, std=Q_gen, size=(Np, 1), device=device)
            else:
                noise = MultivariateNormal(
                    torch.zeros(self.m, device=device), Q_gen
                ).rsample((Np,))

            particles = particles_f + noise   # [Np, m]

            # =========================
            # 3. PRIOR COVARIANCE
            # =========================
            mu_prior = particles.mean(dim=0)          # [m]
            dx = particles - mu_prior.unsqueeze(0)    # [Np, m]
            
            # Covariance math: [m, Np] @ [Np, m] -> [m, m]
            Sigma_prior = (dx.T @ dx) / (Np - 1)      
            
            self.sigma_prior[:, :, t] = Sigma_prior

            # =========================
            # 4. OBSERVATION
            # =========================
            # Format to [batch_size=1, m, 1] for h(), then back to [n]
            x_true_input = x_true.view(1, self.m, 1)
            y_true = self.h(x_true_input).view(-1)   

            y_t = y_true          # [n]
            self.y[:, t] = y_t

            # =========================
            # 5. LIKELIHOOD
            # =========================
            # Format to [batch_size=Np, m, 1] for h(), then back to [Np, n]
            particles_input = particles.view(Np, self.m, 1)
            y_particles = self.h(particles_input).view(Np, self.n)   

            diff = y_particles - y_t.unsqueeze(0)   # [Np, n]

            if self.n == 1:
                log_w = -0.5 * (diff.squeeze(-1)**2) / (R_gen**2 + 1e-8)
            else:
                invR = torch.inverse(R_gen)
                # diff is [Np, n], invR is [n, n]
                log_w = -0.5 * torch.sum((diff @ invR) * diff, dim=1)  # [Np]

            # Numerical stability: subtract max before softmax to prevent NaNs
            log_w = log_w - torch.max(log_w)
            weights = torch.softmax(log_w, dim=0)

            # =========================
            # 6. POSTERIOR
            # =========================
            mu_post = torch.sum(weights.unsqueeze(1) * particles, dim=0)  # [m]

            dx_post = particles - mu_post.unsqueeze(0)  # [Np, m]
            
            # Weighted Covariance math: [m, Np] @ [Np, m] -> [m, m]
            Sigma_post = (weights.unsqueeze(1) * dx_post).T @ dx_post  

            self.sigma_post[:, :, t] = Sigma_post

            # =========================
            # 7. RESAMPLING
            # =========================
            idx = torch.multinomial(weights, Np, replacement=True)
            particles = particles[idx]

            # Reset weights uniformly
            weights = torch.ones(Np, device=device) / Np

            # =========================
            # 8. SAVE STATE
            # =========================
            self.x[:, t] = x_true
    # def GenerateSequence(self, Q_gen, R_gen, T):
    #     # Pre allocate an array for current state
    #     self.x = torch.zeros(size=[self.m, T])
    #     # Pre allocate an array for current observation
    #     self.y = torch.zeros(size=[self.n, T])
        
    #     # Pre allocate an array for current observation
    #     self.sigma = torch.zeros(size=[self.m, self.m, T])

    #     # Set x0 to be x previous
    #     self.x_prev = self.m1x_0
    #     xt = self.x_prev
        
    #     # Generate Sequence Iteratively
    #     for t in range(0, T):

    #         ########################
    #         #### State Evolution ###
    #         ########################   
    #         if torch.equal(Q_gen,torch.zeros(self.m,self.m)):# No noise
    #             xt = self.f(self.x_prev)   
    #         elif self.m == 1: # 1 dim noise
    #             xt = self.f(self.x_prev)
    #             xt_particles = []
    #             for i in range(self.particle_count):
    #                 eq = torch.normal(mean=0, std=Q_gen)
    #                 # Additive Process Noise
    #                 xt_particles.append(torch.add(xt,eq))
    #             xt_particles = torch.stack(xt_particles)
    #             sigma_prior = torch.cov(xt_particles)
    #         else:            
    #             xt = self.f(self.x_prev)
    #             mean = torch.zeros([self.m])              
    #             distrib = MultivariateNormal(loc=mean, covariance_matrix=Q_gen)
    #             xt_particles = []
    #             for i in range(self.particle_count):
    #                 eq = distrib.rsample()
    #                 eq = torch.reshape(eq[:], xt.size())
    #                 # Additive Process Noise
    #                 xt_particles.append(torch.add(xt,eq))
    #             xt_particles = torch.stack(xt_particles)
    #             sigma_prior = torch.cov(xt_particles)
    #             xt = xt_particles[0]

    #         ################
    #         ### Emission ###
    #         ################
    #         yt = self.h(xt)
    #         # Observation Noise         
    #         if self.n == 1: # 1 dim noise
    #             er = torch.normal(mean=0, std=R_gen)
    #             # Additive Observation Noise
    #             yt = torch.add(yt,er)
    #         else:  
    #             mean = torch.zeros([self.n])            
    #             distrib = MultivariateNormal(loc=mean, covariance_matrix=R_gen)
    #             er = distrib.rsample()
    #             er = torch.reshape(er[:], yt.size())       
    #             # Additive Observation Noise
    #             yt = torch.add(yt,er)
            
    #         ########################
    #         ### Squeeze to Array ###
    #         ########################

    #         # Save Current State to Trajectory Array
    #         self.x[:, t] = torch.squeeze(xt,1)

    #         # Save Current Observation to Trajectory Array
    #         self.y[:, t] = torch.squeeze(yt,1)

    #         ################################
    #         ### Save Current to Previous ###
    #         ################################
    #         self.x_prev = xt

    


    ######################
    ### Generate Batch ###
    ######################
    # def GenerateBatch(self, args, size, T, randomInit=False):
    #     if(randomInit):
    #         # Allocate Empty Array for Random Initial Conditions
    #         self.m1x_0_rand = torch.zeros(size, self.m, 1)
    #         if args.distribution == 'uniform':
    #             ### if Uniform Distribution for random init
    #             for i in range(size):           
    #                 initConditions = torch.rand_like(self.m1x_0) * args.variance
    #                 self.m1x_0_rand[i,:,0:1] = initConditions.view(self.m,1)     
            
    #         elif args.distribution == 'normal':
    #             ### if Normal Distribution for random init
    #             for i in range(size):
    #                 distrib = MultivariateNormal(loc=torch.squeeze(self.m1x_0), covariance_matrix=self.m2x_0)
    #                 initConditions = distrib.rsample().view(self.m,1)
    #                 self.m1x_0_rand[i,:,0:1] = initConditions
    #         else:
    #             raise ValueError('args.distribution not supported!')
            
    #         self.Init_batched_sequence(self.m1x_0_rand, self.m2x_0)### for sequence generation
    #     else: # fixed init
    #         initConditions = self.m1x_0.view(1,self.m,1).expand(size,-1,-1)
    #         self.Init_batched_sequence(initConditions, self.m2x_0)### for sequence generation
    
    #     if(args.randomLength):
    #         # Allocate Array for Input and Target (use zero padding)
    #         self.Input = torch.zeros(size, self.n, args.T_max)
    #         self.Target = torch.zeros(size, self.m, args.T_max)
    #         self.lengthMask = torch.zeros((size,args.T_max), dtype=torch.bool)# init with all false
    #         # Init Sequence Lengths
    #         T_tensor = torch.round((args.T_max-args.T_min)*torch.rand(size)).int()+args.T_min # Uniform distribution [100,1000]
    #         for i in range(0, size):
    #             # Generate Sequence
    #             self.GenerateSequence(self.Q, self.R, T_tensor[i].item())
    #             # Training sequence input
    #             self.Input[i, :, 0:T_tensor[i].item()] = self.y             
    #             # Training sequence output
    #             self.Target[i, :, 0:T_tensor[i].item()] = self.x
    #             # Mask for sequence length
    #             self.lengthMask[i, 0:T_tensor[i].item()] = True

    #     else:
    #         # Allocate Empty Array for Input
    #         self.Input = torch.empty(size, self.n, T)
    #         # Allocate Empty Array for Target
    #         self.Target = torch.empty(size, self.m, T)

    #         # Set x0 to be x previous
    #         self.x_prev = self.m1x_0_batch
    #         xt = self.x_prev

    #         # Generate in a batched manner
    #         for t in range(0, T):
    #             ########################
    #             #### State Evolution ###
    #             ########################   
    #             if torch.equal(self.Q,torch.zeros(self.m,self.m)):# No noise
    #                 xt = self.f(self.x_prev)
    #             elif self.m == 1: # 1 dim noise
    #                 xt = self.f(self.x_prev)
    #                 eq = torch.normal(mean=torch.zeros(size), std=self.Q).view(size,1,1)
    #                 # Additive Process Noise
    #                 xt = torch.add(xt,eq)
    #             else:            
    #                 xt = self.f(self.x_prev)
    #                 mean = torch.zeros([size, self.m])              
    #                 distrib = MultivariateNormal(loc=mean, covariance_matrix=self.Q)
    #                 eq = distrib.rsample().view(size,self.m,1)
    #                 # Additive Process Noise
    #                 xt = torch.add(xt,eq)

    #             ################
    #             ### Emission ###
    #             ################
    #             # Observation Noise
    #             if torch.equal(self.R,torch.zeros(self.n,self.n)):# No noise
    #                 yt = self.h(xt)
    #             elif self.n == 1: # 1 dim noise
    #                 yt = self.h(xt)
    #                 er = torch.normal(mean=torch.zeros(size), std=self.R).view(size,1,1)
    #                 # Additive Observation Noise
    #                 yt = torch.add(yt,er)
    #             else:  
    #                 yt =  self.h(xt)
    #                 mean = torch.zeros([size,self.n])            
    #                 distrib = MultivariateNormal(loc=mean, covariance_matrix=self.R)
    #                 er = distrib.rsample().view(size,self.n,1)          
    #                 # Additive Observation Noise
    #                 yt = torch.add(yt,er)

    #             ########################
    #             ### Squeeze to Array ###
    #             ########################

    #             # Save Current State to Trajectory Array
    #             self.Target[:, :, t] = torch.squeeze(xt,2)

    #             # Save Current Observation to Trajectory Array
    #             self.Input[:, :, t] = torch.squeeze(yt,2)

    #             ################################
    #             ### Save Current to Previous ###
    #             ################################
    #             self.x_prev = xt

    def GenerateBatch(self, args, size, T, randomInit=False):
        device = self.m1x_0.device

        # =========================
        # INIT CONDITIONS
        # =========================
        if randomInit:
            self.m1x_0_rand = torch.zeros(size, self.m, 1, device=device)

            if args.distribution == 'uniform':
                for i in range(size):
                    initConditions = torch.rand_like(self.m1x_0) * args.variance
                    self.m1x_0_rand[i,:,0:1] = initConditions.view(self.m,1)

            elif args.distribution == 'normal':
                for i in range(size):
                    distrib = MultivariateNormal(
                        torch.squeeze(self.m1x_0),
                        self.m2x_0
                    )
                    initConditions = distrib.rsample().view(self.m,1)
                    self.m1x_0_rand[i,:,0:1] = initConditions
            else:
                raise ValueError('args.distribution not supported!')

        else:
            initConditions = self.m1x_0.view(1,self.m,1).expand(size,-1,-1)

        # =========================
        # RANDOM LENGTH
        # =========================
        if args.randomLength:

            self.Input = torch.zeros(size, self.n, args.T_max, device=device)
            self.Target = torch.zeros(size, self.m, args.T_max, device=device)

            self.Sigma_prior = torch.zeros(size, self.m, self.m, args.T_max, device=device)
            self.Sigma_post  = torch.zeros(size, self.m, self.m, args.T_max, device=device)

            self.lengthMask = torch.zeros((size,args.T_max), dtype=torch.bool, device=device)

            T_tensor = torch.round(
                (args.T_max-args.T_min)*torch.rand(size)
            ).int() + args.T_min

            for i in range(size):

                Ti = T_tensor[i].item()

                # set initial
                if randomInit:
                    self.m1x_0 = self.m1x_0_rand[i]
                else:
                    self.m1x_0 = initConditions[i]

                # generate sequence
                self.GenerateSequence(self.Q, self.R, Ti)

                # store
                self.Input[i, :, 0:Ti] = self.y
                self.Target[i, :, 0:Ti] = self.x

                self.Sigma_prior[i, :, :, 0:Ti] = self.sigma_prior
                self.Sigma_post[i, :, :, 0:Ti]  = self.sigma_post

                self.lengthMask[i, 0:Ti] = True

        # =========================
        # FIXED LENGTH
        # =========================
        else:

            self.Input = torch.empty(size, self.n, T, device=device)
            self.Target = torch.empty(size, self.m, T, device=device)

            self.Sigma_prior = torch.empty(size, self.m, self.m, T, device=device)
            self.Sigma_post  = torch.empty(size, self.m, self.m, T, device=device)

            for i in range(size):

                # set initial
                if randomInit:
                    self.m1x_0 = self.m1x_0_rand[i]
                else:
                    self.m1x_0 = initConditions[i]

                # generate MC sequence
                self.GenerateSequence(self.Q, self.R, T)

                # store
                self.Input[i] = self.y
                self.Target[i] = self.x

                self.Sigma_prior[i] = self.sigma_prior
                self.Sigma_post[i]  = self.sigma_post
