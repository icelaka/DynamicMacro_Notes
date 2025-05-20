%% File Info.

%{

    model.m
    -------
    This code sets up the model.

%}

%% Model class.

classdef model
    methods(Static)
        %% Set up structure array for model parameters and set the simulation parameters.
        
        function par = setup()
            %% Structure array for model parameters.
            
            par = struct();
            
            %% Preferences.
            
            par.beta = 0.94; % Discount factor: Lower values of this mean that consumers are impatient and consume more today.
            par.sigma = 2.00; % CRRA: Higher values of this mean that consumers are risk averse and do not want to consume too much today.
            par.lambda = 1.5; %Financial constraint parameter
            par.phi = 0.5; % Financial constraint parameter for investment (i_t ≤ φπ_t)
            
            assert(par.beta > 0 && par.beta < 1.00,'Discount factor should be between 0 and 1.\n')
            assert(par.sigma > 0,'CRRA should be at least 0.\n')
            
            %% Technology.
            
            par.alpha = 0.33; % Capital's share of income.
            par.delta = 0.03; % Depreciation rate of physical capital.
            
            assert(par.alpha > 0 && par.alpha < 1.00,'Capital share of income should be between 0 and 1.\n')
            assert(par.delta >= 0 && par.delta <= 1.00,'The depreciation rate should be from 0 to 1.\n')
            
            par.sigma_eps = 0.07; % Std. dev of productivity shocks.
            par.rho = 0.85; % Persistence of AR(1) process.
            par.mu = 0.0; % Intercept of AR(1) process.
            
            assert(par.sigma_eps > 0,'The standard deviation of the shock must be positive.\n')
            assert(abs(par.rho) < 1,'The persistence must be less than 1 in absolute value so that the series is stationary.\n')
            
            %% Prices.
            
            par.r = 0.06; % Real interest rate.
            par.omega = 0.2; % Weight for updating interest rate in general equilibrium.
            
            %% Simulation parameters.
            
            par.seed = 2025; % Seed for simulation.
            par.T = 10000; % Number of time periods.
            par.N = 3000; % Number of people.
            
            
            %% Proprietary?
            par.gamma = 0.2;
            par.delta_q = 0.1;
            par.theta = 0.5;
            par.eta = 0.5;
            par.sigma_q = 0.1;
            
            par.kappa_0 = 0.1; % Linear cost
            par.kappa_1 = 0.2; % Quadratic cost
            par.kappa_2 = 0.05; % Maintenance cost
            
            par.subsidy = 0.0; % subsidy rate (0 to 1)
            
        end
        
        %% Generate state grids.
        
        function par = gen_grids(par)
            %% Asset grid.
            
            par.alen = 500; % Grid size for k.
            par.amax = 300; % Upper bound for k.
            par.amin = 0; % Minimum k.
            
            %% Idea qual grid.
            par.qlen = 100; % Grid size for q.
            par.qmax = 1; % Upper bound for q.
            
            assert(par.alen > 5,'Grid size for a should be positive and greater than 5.\n')
            assert(par.amax > par.amin,'Minimum a should be less than maximum value.\n')
            
            par.agrid = linspace(par.amin,par.amax,par.alen)'; % Equally spaced, linear grid for a and a'.
            par.qgrid = linspace(0,par.qmax,par.qlen)'; % Equally spaced, linear grid for q and q'.
            
            %% Discretized productivity process.
            
            par.zlen = 7; % Grid size for z.
            par.m = 3; % Scaling parameter for Tauchen.
            
            assert(par.zlen > 3,'Grid size for z should be positive and greater than 3.\n')
            assert(par.m > 0,'Scaling parameter for Tauchen should be positive.\n')
            
            [zgrid,pmat] = model.tauchen(par.mu,par.rho,par.sigma_eps,par.zlen,par.m); % Tauchen's Method to discretize the AR(1) process for log productivity.
            par.zgrid = exp(zgrid); % The AR(1) is in z so exponentiate it to get exp(z).
            par.pmat = pmat; % Transition matrix.
            
        end
        
        %% Tauchen's Method
        
        function [y,pi] = tauchen(mu,rho,sigma,N,m)
            %% Construct equally spaced grid.
            
            ar_mean = mu/(1-rho); % The mean of a stationary AR(1) process is mu/(1-rho).
            ar_sd = sigma/((1-rho^2)^(1/2)); % The std. dev of a stationary AR(1) process is sigma/sqrt(1-rho^2)
            
            y1 = ar_mean-(m*ar_sd); % Smallest grid point is the mean of the AR(1) process minus m*std.dev of AR(1) process.
            yn = ar_mean+(m*ar_sd); % Largest grid point is the mean of the AR(1) process plus m*std.dev of AR(1) process.
            
            y = linspace(y1,yn,N); % Equally spaced grid.
            d = y(2)-y(1); % Step size.
            
            %% Compute transition probability matrix from state j (row) to k (column).
            
            ymatk = repmat(y,N,1); % States next period.
            ymatj = mu+rho*ymatk'; % States this period.
            
            pi = normcdf(ymatk,ymatj-(d/2),sigma) - normcdf(ymatk,ymatj+(d/2),sigma); % Transition probabilities to state 2, ..., N-1.
            pi(:,1) = normcdf(y(1),mu+rho*y-(d/2),sigma); % Transition probabilities to state 1.
            pi(:,N) = 1 - normcdf(y(N),mu+rho*y+(d/2),sigma); % Transition probabilities to state N.
            
        end
        
        %% Utility function.
        
        function u = utility(c,par)
            %% CRRA utility.
            
            if par.sigma == 1
                u = log(c); % Log utility.
            else
                u = (c.^(1-par.sigma))./(1-par.sigma); % CRRA utility.
            end
            
        end
        
    end
end