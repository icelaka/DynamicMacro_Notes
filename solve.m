%% File Info.

%{

    solve.m
    -------
    This code solves the model.

%}

%% Solve class.

classdef solve
    methods(Static)
        %% Solve the household's problem using VFI.
        function sol = hh_problem_baseline(par,sol)
            %% Model parameters, grids and functions.
            
            beta = par.beta; % Discount factor.
            
            alen = par.alen; % Grid size for a.
            agrid = par.agrid; % Grid for a (state and choice).
            
            zlen = par.zlen; % Grid size for z.
            zgrid = par.zgrid; % Grid for z.
            pmat = par.pmat; % Transition matrix for z.
            
            r = par.r; % Real interest rate, taken as given by households.
            w = par.w; % Real wage rate, taken as given by households.
            
            %phi = -w*par.zgrid(1)/r; % The natural borrowing constraint---the present value of the worst possible realization of income. Set amin in the model file to be a negative number.
            phi = 0; % The zero borrowing constraint---no borrowing allowed. Set amin in the model file to be zero.
            
            %% Value Function Iteration.
            
            v1 = nan(alen,zlen); % Container for V.
            a1 = nan(alen,zlen); % Container for a'.
            c1 = nan(alen,zlen); % Container for c'.
            
            crit = 1e-6;
            maxiter = 10000;
            diff = 1;
            iter = 0;
            
            fprintf('------------Beginning Value Function Iteration.------------\n\n')
            
            c0 = (1+r)*agrid+w.*zgrid; % Guess of consumption is to consume everything; this is a matrix because agrid is a column vector and zgrid is a row vector.
            v0 = model.utility(c0,par)./(1-beta);
            
            while diff > crit && iter < maxiter % Iterate on the Bellman Equation until convergence.
                
                for p = 1:alen % Loop over the a-states.
                    if agrid(p) >= phi % Only solve the model when the borrowing constraint isnt violated.
                        for j = 1:zlen % Loop over the y-states.
                            
                            % Consumption.
                            c = (1+r)*agrid(p)+w.*zgrid(j)-agrid; % Possible values for consumption, c(t) = (1+r)a(t-1) + y(t) - a(t+1).
                            
                            % Solve the maximization problem.
                            ev = v0*pmat(j,:)'; %  The next-period value function is the expected value function over each possible next-period A, conditional on the current state j.
                            vall = model.utility(c,par) + beta*ev; % Compute the value function for each choice of k', given k.
                            vall(c<=0) = -inf; % Set the value function to negative infinity when c < 0.
                            [vmax,ind] = max(vall); % Maximize: vmax is the maximized value function; ind is where it is in the grid.
                            
                            % Store values.
                            v1(p,j) = vmax; % Maximized v.
                            c1(p,j) = c(ind); % Optimal c'.
                            a1(p,j) = agrid(ind); % Optimal a'.
                            
                        end
                    end
                end
                
                diff = norm(v1-v0); % Check for convergence.
                v0 = v1; % Update guess of v.
                
                iter = iter + 1; % Update counter.
                
                % Print counter.
                if mod(iter,25) == 0
                    fprintf('Iteration: %d.\n',iter)
                end
                
            end
            
            fprintf('\nConverged in %d iterations.\n\n',iter)
            
            fprintf('------------End of Value Function Iteration.------------\n')
            
            %% Macro variables, value, and policy functions.
            
            sol.a = a1; % Savings policy function.
            sol.c = c1; % Consumption policy function.
            sol.v = v1; % Value function.
            
        end
        
        
        function sol = hh_problem_profit(par, sol)
            % Setup
            beta = par.beta;
            agrid = par.agrid; alen = par.alen;
            zgrid = par.zgrid; zlen = par.zlen;
            pmat = par.pmat;
            
            r = par.r; w = par.w;
            phi = 0;
            
            % Get profits from firm solution (dividends)
            profit = sol.profit; % Dividends = firm's profit
            
            % VFI
            v0 = zeros(alen, zlen);
            v1 = zeros(alen, zlen);
            a1 = zeros(alen, zlen);
            c1 = zeros(alen, zlen);
            
            crit = 1e-6;
            maxiter = 10000;
            diff = 1;
            iter = 0;
            
            fprintf('\n------------Beginning Value Function Iteration.------------\n\n')
            
            while diff > crit && iter < maxiter
                iter = iter + 1;
                
                for i = 1:alen
                    a = agrid(i);
                    
                    if a >= phi
                        for j = 1:zlen
                            z = zgrid(j);
                            
                            % Budget constraint with dividends: c + a' = (1+r)a + wl*e^z + d
                            % where d = profit
                            cash_on_hand = (1+r)*a + w*z + profit(i,j);
                            
                            vtemp = zeros(alen, 1);
                            ctemp = zeros(alen, 1);
                            
                            for k = 1:alen
                                ap = agrid(k);
                                c = cash_on_hand - ap;
                                
                                if c > 0
                                    ev = 0;
                                    for l = 1:zlen
                                        ev = ev + v0(k,l)*pmat(j,l);
                                    end
                                    
                                    u = model.utility(c, par);
                                    vtemp(k) = u + beta*ev;
                                    ctemp(k) = c;
                                else
                                    vtemp(k) = -1e10;
                                    ctemp(k) = 0;
                                end
                            end
                            
                            [v1(i,j), idx] = max(vtemp);
                            a1(i,j) = agrid(idx);
                            c1(i,j) = ctemp(idx);
                        end
                    else
                        v1(i,:) = -1e10;
                        a1(i,:) = agrid(1);
                        c1(i,:) = 0;
                    end
                end
                
                diff = norm(v1(:)-v0(:));
                v0 = v1;
                
                if mod(iter, 10) == 0
                    fprintf('Iteration: %d, Diff: %f\n', iter, diff)
                end
            end
            
            fprintf('\nConverged in %d iterations.\n\n', iter)
            fprintf('------------End of Value Function Iteration.------------\n')
            
            % Save
            sol.v = v1;
            sol.a = a1;
            sol.c = c1;
        end
        
        function sol = hh_problem_idea(par,sol)
            %% Setup
            beta = par.beta;
            agrid = par.agrid; alen = par.alen;
            zgrid = par.zgrid; zlen = par.zlen;
            qgrid = par.qgrid; qlen = par.qlen;  % Add idea quality grid
            pmat = par.pmat;
            
            r = par.r; w = par.w;
            delta_q = par.delta_q;    % Idea depreciation rate
            theta = par.theta;        % Idea investment productivity
            eta = par.eta;            % Idea investment elasticity
            sigma_q = par.sigma_q;    % Idea shock std dev
            kappa_0 = par.kappa_0;    % Linear investment cost
            kappa_1 = par.kappa_1;    % Quadratic investment cost
            kappa_2 = par.kappa_2;    % Maintenance cost
            profit = sol.profit;      % Firm profits
            
            %% VFI for Abandon case (baseline income)
            % This is the value if household abandons the idea
            [A, Z] = ndgrid(agrid, zgrid);
            v0 = model.utility((1 + r) * A + w * Z, par) / (1 - beta);  % Initial value guess
            v1 = nan(alen, zlen);
            a1 = nan(alen, zlen);
            c1 = nan(alen, zlen);
            
            crit = 1e-6; maxiter = 10000; diff = 1; iter = 0;
            
            fprintf('\n------------Beginning Value Function Iteration.------------\n\n')
            
            while diff > crit && iter < maxiter
                iter = iter + 1;
                for p = 1:alen
                    for j = 1:zlen
                        % Standard budget constraint without ideas
                        c = (1 + r) * agrid(p) + w * zgrid(j) - agrid;
                        
                        % Expected value
                        ev = 0;
                        for k = 1:zlen
                            ev = ev + v0(p,k) * pmat(j,k);
                        end
                        
                        % Value
                        val = model.utility(c, par) + beta * ev;
                        val(c <= 0) = -inf;
                        [v1(p,j), ind] = max(val);
                        c1(p,j) = c(ind);
                        a1(p,j) = agrid(ind);
                    end
                end
                
                diff = norm(v1(:) - v0(:));
                v0 = v1;
            end
            
            fprintf('\nConverged in %d iterations.\n\n', iter)
            
            % Store value of abandoning idea
            v_abandon = v1;
            
            %% Full model with proprietary ideas
            v = nan(alen, zlen, qlen);           % Value function with idea quality
            a_pol = nan(alen, zlen, qlen);       % Savings policy
            c_pol = nan(alen, zlen, qlen);       % Consumption policy
            i_pol = nan(alen, zlen, qlen);       % Idea investment policy
            keep_pol = nan(alen, zlen, qlen);    % Keep/abandon policy
            
            % Initialize value function
            for a_idx = 1:alen
                for z_idx = 1:zlen
                    for q_idx = 1:qlen
                        if q_idx == 1  % q = 0 means no idea
                            v(a_idx, z_idx, q_idx) = v_abandon(a_idx, z_idx);
                        else
                            % Initial guess: value with idea
                            c_guess = (1 + r) * agrid(a_idx) + profit(a_idx, z_idx, q_idx) - kappa_2 * qgrid(q_idx);
                            v(a_idx, z_idx, q_idx) = model.utility(c_guess, par) / (1 - beta);
                        end
                    end
                end
            end
            
            % Solve with VFI
            diff = 1; iter = 0;
            while diff > crit && iter < maxiter
                iter = iter + 1;
                v_next = v;
                
                for a_idx = 1:alen
                    a = agrid(a_idx);
                    
                    for z_idx = 1:zlen
                        z = zgrid(z_idx);
                        
                        for q_idx = 1:qlen
                            q = qgrid(q_idx);
                            
                            % If no idea quality (q=0), use abandoned value
                            if q_idx == 1
                                v_next(a_idx, z_idx, q_idx) = v_abandon(a_idx, z_idx);
                                a_pol(a_idx, z_idx, q_idx) = a1(a_idx, z_idx);
                                c_pol(a_idx, z_idx, q_idx) = c1(a_idx, z_idx);
                                i_pol(a_idx, z_idx, q_idx) = 0;
                                keep_pol(a_idx, z_idx, q_idx) = 0;
                                continue;
                            end
                            
                            % Option 1: Keep idea and invest
                            v_keep_max = -inf;
                            a_keep_opt = a;
                            c_keep_opt = 0;
                            i_keep_opt = 0;
                            
                            % Search over savings and investment decisions
                            for a_next_idx = 1:alen
                                a_next = agrid(a_next_idx);
                                
                                % Try different investment levels
                                for i_val = linspace(0, 0.5, 20)  % Discretized investment options
                                    % Calculate costs of maintaining and investing in idea
                                    idea_cost = kappa_0 * i_val + kappa_1 * i_val^2 + kappa_2 * q;
                                    
                                    % Budget constraint with stochastic profit
                                    c = (1 + r) * a + profit(a_idx, z_idx, q_idx) * z - a_next - idea_cost;
                                    
                                    % Skip infeasible choices
                                    if c <= 0
                                        continue;
                                    end
                                    
                                    % Calculate expected value
                                    ev = 0;
                                    
                                    % Discretize idea quality shocks
                                    shock_val = [-2*sigma_q, -sigma_q, 0, sigma_q, 2*sigma_q];
                                    shock_prob = [0.1, 0.2, 0.4, 0.2, 0.1];
                                    
                                    % For each possible shock
                                    for shock_idx = 1:length(shock_val)
                                        % Evolution of idea quality
                                        q_next = (1 - delta_q) * q + theta * i_val^eta + shock_val(shock_idx);
                                        q_next = max(0, min(q_next, par.qmax));
                                        
                                        % Find nearest grid point for next q
                                        [~, q_next_idx] = min(abs(qgrid - q_next));
                                        
                                        % Calculate expected value over productivity transitions
                                        for z_next_idx = 1:zlen
                                            ev = ev + pmat(z_idx, z_next_idx) * shock_prob(shock_idx) * v(a_next_idx, z_next_idx, q_next_idx);
                                        end
                                    end
                                    
                                    % Calculate value of keeping idea
                                    v_keep = model.utility(c, par) + beta * ev;
                                    
                                    % Update if better than current best
                                    if v_keep > v_keep_max
                                        v_keep_max = v_keep;
                                        a_keep_opt = a_next;
                                        c_keep_opt = c;
                                        i_keep_opt = i_val;
                                    end
                                end
                            end
                            
                            % Option 2: Abandon idea (use v_abandon)
                            
                            % Choose the better option: keep or abandon
                            if v_keep_max > v_abandon(a_idx, z_idx)
                                % Keep idea
                                v_next(a_idx, z_idx, q_idx) = v_keep_max;
                                a_pol(a_idx, z_idx, q_idx) = a_keep_opt;
                                c_pol(a_idx, z_idx, q_idx) = c_keep_opt;
                                i_pol(a_idx, z_idx, q_idx) = i_keep_opt;
                                keep_pol(a_idx, z_idx, q_idx) = 1;
                            else
                                % Abandon idea
                                v_next(a_idx, z_idx, q_idx) = v_abandon(a_idx, z_idx);
                                a_pol(a_idx, z_idx, q_idx) = a1(a_idx, z_idx);
                                c_pol(a_idx, z_idx, q_idx) = c1(a_idx, z_idx);
                                i_pol(a_idx, z_idx, q_idx) = 0;
                                keep_pol(a_idx, z_idx, q_idx) = 0;
                            end
                        end
                    end
                end
                
                % Check convergence
                diff = norm(v_next(:) - v(:));
                v = v_next;
                
                if mod(iter, 10) == 0
                    fprintf('Iteration %d, diff = %.6f\n', iter, diff);
                end
            end
            
            fprintf('------------End of Value Function Iteration.------------\n')
            
            % Save
            sol.v = v;
            sol.a = a_pol;
            sol.c = c_pol;
            sol.i = i_pol;
            sol.keep = keep_pol;
        end
        
        function [par,sol] = firm_problem_baseline(par)
            %% Model parameters, grids and functions.
            
            delta = par.delta; % Depreciation rate.
            alpha = par.alpha; % Capital's share of income.
            
            r = par.r; % Real interest rate, which the firm takes as given.
            k = ((r+delta)/alpha)^(1/(alpha-1)); % Capital stock solved from the first-order condition.
            
            %% Capital and wages.
            
            sol = struct();
            
            sol.k = k; % Capital stock.
            par.w = (1-alpha)*k^alpha; % Real wage rate.
            
        end
        
        %% Solve the firm's static problem.
        function [par,sol] = firm_problem_profit(par)
            % Model Parameter
            delta = par.delta; % Depreciation rate.
            alpha = par.alpha; % Capital's share of income.
            lambda = par.lambda; % Financial constraint parameter
            r = par.r;
            
            % Original unconstrained capital (keep this for aggregate calculations)
            k_unconstrained = ((r+delta)/alpha)^(1/(alpha-1));
            
            % Initialize solution structure
            sol = struct();
            sol.k = k_unconstrained; % Keep this for economy-wide equilibrium
            
            % Initialize arrays for capital, profit and investment
            sol.k_policy = zeros(par.alen, par.zlen);
            sol.profit = zeros(par.alen, par.zlen);
            sol.investment = zeros(par.alen, par.zlen);
            
            % Calculate constrained capital and profits for each asset and productivity level
            for a_idx = 1:par.alen
                a = par.agrid(a_idx);
                
                % Financial constraint: k ≤ λa
                k_constrained = lambda * a;
                
                for z_idx = 1:par.zlen
                    z = par.zgrid(z_idx);
                    
                    % Unconstrained optimal capital for this productivity
                    k_optimal = ((r+delta)/(alpha*z))^(1/(alpha-1));
                    
                    % Actual capital is minimum of unconstrained and constrained
                    k = min(k_optimal, k_constrained);
                    
                    % Calculate profit at this state
                    profit = z * k^alpha - (r+delta) * k;
                    
                    % Financial constraint on investment: i_t ≤ φπ_t
                    max_investment = par.phi * profit;
                    
                    % Store results
                    sol.k_policy(a_idx, z_idx) = k;
                    sol.profit(a_idx, z_idx) = profit;
                    sol.investment(a_idx, z_idx) = max_investment;
                end
            end
            
            % Calculate wage rate (for the baseline model income)
            par.w = (1-alpha)*k_unconstrained^alpha;
        end
        
        function [par,sol] = firm_problem_idea(par)
            
            % Model Parameter
            delta = par.delta; % Depreciation rate.
            alpha = par.alpha; % Capital's share of income.
            lambda = par.lambda;
            gamma = par.gamma; % idea quality elasticity
            r = par.r;
            %Original unconstrained capital (keep this for aggregate calculations)
            k_unconstrained = ((r+delta)/alpha)^(1/(alpha-1));
            
            % Initialize solution structure
            sol = struct();
            sol.k = k_unconstrained; % Keep this for economy-wide equilibrium
            
            %Initialize arrays for capital and profit at each asset level and productivity
            sol.k_policy = zeros(par.alen, par.zlen, par.qlen);
            sol.profit = zeros(par.alen, par.zlen, par.qlen);
            
            %Calculate constrained capital and profits for each asset and productivity level
            for a_idx = 1:par.alen
                a = par.agrid(a_idx);
                
                %Financial constraint: k ≤ λa
                k_constrained = lambda * a;
                
                for z_idx = 1:par.zlen
                    z = par.zgrid(z_idx);
                    
                    for q_idx = 1:par.qlen
                        q = par.qgrid(q_idx);
                        
                        if q > 0
                            k_optimal = ((r+delta)/(alpha*z*q^gamma))^(1/(alpha-1));
                            %Unconstrained optimal capital for this productivity
                        else
                            k_optimal = ((r+delta)/(alpha*z))^(1/(alpha-1));
                        end
                        
                        %Actual capital is minimum of unconstrained and constrained
                        k = min(k_optimal, k_constrained);
                        
                        %Calculate profit at this state
                        if q > 0
                            profit = z * q^gamma * k^alpha - (r+delta) * k;
                        else
                            profit = z * k^alpha - (r+delta) * k;
                        end
                        % profit = z * k^alpha - (r+delta) * k;
                        
                        %Store results
                        sol.k_policy(a_idx, z_idx, q_idx) = k;
                        sol.profit(a_idx, z_idx, q_idx) = profit;
                    end
                end
            end
            %Calculate wage rate (for the baseline model income)
            par.w = (1-alpha)*k_unconstrained^alpha;
        end
    end
    
end


%     %    function [par,sol] = firm_problem(par)
%     %        Model parameters, grids and functions.
%     %
%     %        delta = par.delta; % Depreciation rate.
%     %        alpha = par.alpha; % Capital's share of income.
%     %
%     %        r = par.r; % Real interest rate, which the firm takes as given.
%     %        k = ((r+delta)/alpha)^(1/(alpha-1)); % Capital stock solved from the first-order condition.
%     %
%     %        % Capital and wages.
%     %
%     %        sol = struct();
%     %        sol.k = k; % Capital stock.
%     %        par.w = (1-alpha)*k^alpha; % Real wage rate.
%     %    end
%     % end
% end





