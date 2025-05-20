%% File Info.

%{

    simulate.m
    ----------
    This code simulates the model.

%}

%% Simulate class.

classdef simulate
    methods(Static)
        function sim = economy_baseline(par,sol)
            %% Set up.
            
            agrid = par.agrid; % Assets today (state variable).
            zgrid = par.zgrid; % Productivity.
            pmat = par.pmat; % Transition matrix.
            
            apol = sol.a; % Policy function for capital.
            cpol = sol.c; % Poliscy function for consumption.
            
            T = par.T+10; % Time periods.
            N = par.N; % People.
            
            zsim = nan(T,N); % Container for simulated income.
            asim = nan(T,N); % Container for simulated savings.
            csim = nan(T,N); % Container for simulated consumption.
            usim = nan(T,N); % Container for simulated utility.
            
            %% Begin simulation.
            
            rng(par.seed);
            
            pmat0 = pmat^100; % Stationary distirbution.
            cmat = cumsum(pmat,2); % CDF matrix.
            
            z0_ind = randsample(par.zlen,N,true,pmat0(1,:))'; % Index for initial income.
            a0_ind = randsample(par.alen,N,true)'; % Index for initial wealth.
            
            for i = 1:N % Person loop.
                zsim(1,i) = zgrid(z0_ind(i)); % Productivity in period 0.
                csim(1,i) = cpol(a0_ind(i),z0_ind(i)); % Consumption in period 0 given a0.
                asim(1,i) = apol(a0_ind(i),z0_ind(i)); % Savings for period 1 given a0.
                usim(1,i) = model.utility(csim(1,i),par); % Utility in period 0 given a0.
            end
            
            %% Simulate endogenous variables.
            
            for j = 2:T % Time loop.
                for i = 1:N % Person loop.
                    at_ind = find(asim(j-1,i)==agrid); % Savings choice in the previous period is the state today. Find where the latter is on the grid.
                    zsim(j,i) = zgrid(z0_ind(i)); % Productivity in period 0.
                    csim(j,i) = cpol(at_ind,z0_ind(i)); % Consumption in period t.
                    asim(j,i) = apol(at_ind,z0_ind(i)); % Savings for period t+1.
                    usim(j,i) = model.utility(csim(j,i),par); % Utility in period t.
                    z1_ind = find(rand<=cmat(z0_ind(i),:)); % Draw income shock for next period.
                    z0_ind(i) = z1_ind(1);
                end
                
                if j >= 11 && norm(mean(asim(j-9:j,:),'all')-mean(asim(j-10:j-1,:),'all')) < 0.0001
                    break
                end
                
            end
            
            sim = struct();
            
            sim.zsim = zsim(j-9:j,:); % Simulated productivity.
            sim.asim = asim(j-9:j,:); % Simulated savings.
            sim.csim = csim(j-9:j,:); % Simulated consumption.
            sim.usim = usim(j-9:j,:); % Simulated utility.
            sim.asup = mean(asim(j-9:j,:),'all'); % Simulated savings.
            
        end
        
        function sim = economy_profit(par, sol)
            % Set up
            agrid = par.agrid;
            zgrid = par.zgrid;
            pmat = par.pmat;
            
            apol = sol.a;
            cpol = sol.c;
            
            T = par.T+10;
            N = par.N;
            
            zsim = nan(T,N);
            asim = nan(T,N);
            csim = nan(T,N);
            usim = nan(T,N);
            
            % Add tracking for dividends and disposable income
            divs = sol.profit;  % Dividends from firm profits
            divsim = nan(T,N);  % Simulated dividends
            
            % Markov process for disposable income shock (ξ)
            xi_rho = 0.8;  % Persistence parameter for xi
            xi_sigma = 0.1; % Standard deviation of shock
            xisim = nan(T,N);  % Simulated xi shocks
            ydsim = nan(T,N);  % Simulated disposable income
            
            % Begin simulation
            rng(par.seed);
            
            pmat0 = pmat^100;
            cmat = cumsum(pmat,2);
            
            z0_ind = randsample(par.zlen,N,true,pmat0(1,:))';
            a0_ind = randsample(par.alen,N,true)';
            
            for i = 1:N
                zsim(1,i) = zgrid(z0_ind(i));
                csim(1,i) = cpol(a0_ind(i),z0_ind(i));
                asim(1,i) = apol(a0_ind(i),z0_ind(i));
                usim(1,i) = model.utility(csim(1,i),par);
                
                % Initial dividend based on firm's profit for the state
                divsim(1,i) = divs(a0_ind(i),z0_ind(i));
                
                % Initial xi shock
                xisim(1,i) = 1.0;  % Start at 1 (no shock)
                
                % Initial disposable income
                ydsim(1,i) = divsim(1,i) * xisim(1,i);
            end
            
            % Simulate endogenous variables
            for j = 2:T
                for i = 1:N
                    [~, at_ind] = min(abs(asim(j-1,i) - agrid));
                    
                    % Update productivity
                    z1_ind = find(rand<=cmat(z0_ind(i),:));
                    z0_ind(i) = z1_ind(1);
                    zsim(j,i) = zgrid(z0_ind(i));
                    
                    % Update dividends
                    divsim(j,i) = divs(at_ind, z0_ind(i));
                    
                    % Update xi shock (Markov process)
                    xisim(j,i) = xi_rho * xisim(j-1,i) + normrnd(0, xi_sigma);
                    
                    % Calculate disposable income
                    ydsim(j,i) = divsim(j,i) * xisim(j,i);
                    
                    % Update consumption and savings
                    csim(j,i) = cpol(at_ind, z0_ind(i));
                    asim(j,i) = apol(at_ind, z0_ind(i));
                    usim(j,i) = model.utility(csim(j,i), par);
                end
                
                if j >= 11 && norm(mean(asim(j-9:j,:),'all')-mean(asim(j-10:j-1,:),'all')) < 0.0001
                    break
                end
            end
            
            % Save results
            sim = struct();
            sim.zsim = zsim(j-9:j,:);
            sim.asim = asim(j-9:j,:);
            sim.csim = csim(j-9:j,:);
            sim.usim = usim(j-9:j,:);
            sim.divsim = divsim(j-9:j,:);
            sim.xisim = xisim(j-9:j,:);
            sim.ydsim = ydsim(j-9:j,:);
            
            % Calculate aggregate statistics
            sim.amean = mean(asim(j-9:j,:), 'all');
            sim.cmean = mean(csim(j-9:j,:), 'all');
            sim.zmean = mean(zsim(j-9:j,:), 'all');
            sim.divmean = mean(divsim(j-9:j,:), 'all');
            sim.ydmean = mean(ydsim(j-9:j,:), 'all');
            
            % Asset supply
            sim.asup = sim.amean;
        end
        
        %% Simulate the model.
        function sim = economy_idea(par, sol)
            % Set up
            agrid = par.agrid;
            zgrid = par.zgrid;
            pmat = par.pmat;
            
            apol = sol.a;
            cpol = sol.c;
            
            T = par.T+10;
            N = par.N;
            
            zsim = nan(T,N);
            asim = nan(T,N);
            csim = nan(T,N);
            usim = nan(T,N);
            
            % Add tracking for dividends and disposable income
            divs = sol.profit;  % Dividends from firm profits
            divsim = nan(T,N);  % Simulated dividends
            
            % Markov process for disposable income shock (ξ)
            xi_rho = 0.8;  % Persistence parameter for xi
            xi_sigma = 0.1; % Standard deviation of shock
            xisim = nan(T,N);  % Simulated xi shocks
            ydsim = nan(T,N);  % Simulated disposable income
            
            % Begin simulation
            rng(par.seed);
            
            pmat0 = pmat^100;
            cmat = cumsum(pmat,2);
            
            z0_ind = randsample(par.zlen,N,true,pmat0(1,:))';
            a0_ind = randsample(par.alen,N,true)';
            
            for i = 1:N
                zsim(1,i) = zgrid(z0_ind(i));
                csim(1,i) = cpol(a0_ind(i),z0_ind(i));
                asim(1,i) = apol(a0_ind(i),z0_ind(i));
                usim(1,i) = model.utility(csim(1,i),par);
                
                % Initial dividend based on firm's profit for the state
                divsim(1,i) = divs(a0_ind(i),z0_ind(i));
                
                % Initial xi shock
                xisim(1,i) = 1.0;  % Start at 1 (no shock)
                
                % Initial disposable income
                ydsim(1,i) = divsim(1,i) * xisim(1,i);
            end
            
            % Simulate endogenous variables
            for j = 2:T
                for i = 1:N
                    [~, at_ind] = min(abs(asim(j-1,i) - agrid));
                    
                    % Update productivity
                    z1_ind = find(rand<=cmat(z0_ind(i),:));
                    z0_ind(i) = z1_ind(1);
                    zsim(j,i) = zgrid(z0_ind(i));
                    
                    % Update dividends
                    divsim(j,i) = divs(at_ind, z0_ind(i));
                    
                    % Update xi shock (Markov process)
                    xisim(j,i) = xi_rho * xisim(j-1,i) + normrnd(0, xi_sigma);
                    
                    % Calculate disposable income
                    ydsim(j,i) = divsim(j,i) * xisim(j,i);
                    
                    % Update consumption and savings
                    csim(j,i) = cpol(at_ind, z0_ind(i));
                    asim(j,i) = apol(at_ind, z0_ind(i));
                    usim(j,i) = model.utility(csim(j,i), par);
                end
                if j >= 11 && norm(mean(asim(j-9:j,:),'all')-mean(asim(j-10:j-1,:),'all')) < 0.0001
                    break
                end
            end
            
            
            % Save results
            sim = struct();
            sim.zsim = zsim(j-9:j,:);
            sim.asim = asim(j-9:j,:);
            sim.csim = csim(j-9:j,:);
            sim.usim = usim(j-9:j,:);
            sim.divsim = divsim(j-9:j,:);
            sim.xisim = xisim(j-9:j,:);
            sim.ydsim = ydsim(j-9:j,:);
            
            % Calculate aggregate statistics
            sim.amean = mean(asim(j-9:j,:), 'all');
            sim.cmean = mean(csim(j-9:j,:), 'all');
            sim.zmean = mean(zsim(j-9:j,:), 'all');
            sim.divmean = mean(divsim(j-9:j,:), 'all');
            sim.ydmean = mean(ydsim(j-9:j,:), 'all');
            
            % Asset supply
            sim.asup = sim.amean;
        end
    end
end
%         function sim = economy(par,sol)
%             %% Set up.
%
%             agrid = par.agrid; % Assets today (state variable).
%             zgrid = par.zgrid; % Productivity.
%             pmat = par.pmat; % Transition matrix.
%
%             apol = sol.a; % Policy function for capital.
%             cpol = sol.c; % Policy function for consumption.
%
%             T = par.T+10; % Time periods.
%             N = par.N; % People.
%
%             zsim = nan(T,N); % Container for simulated income.
%             asim = nan(T,N); % Container for simulated savings.
%             csim = nan(T,N); % Container for simulated consumption.
%             usim = nan(T,N); % Container for simulated utility.
%             % Add idea quality tracking
%             qgrid = par.qgrid;
%             ipol = sol.i;         % R&D investment policy
%             keepol = sol.keep;    % Keep/abandon policy
%
%             % Initialize containers for ideas
%             qsim = nan(T, N);     % Idea quality
%             isim = nan(T, N);     % R&D investment
%             keepsim = nan(T, N);  % Keep/abandon decisions
%             %% Begin simulation.
%
%             rng(par.seed);
%
%             q0_ind = ones(N, 1);  % Start at first grid point (q=0)
%             qsim(1,:) = qgrid(q0_ind);
%
%             pmat0 = pmat^100; % Stationary distirbution.
%             cmat = cumsum(pmat,2); % CDF matrix.
%
%             z0_ind = randsample(par.zlen,N,true,pmat0(1,:))'; % Index for initial income.
%             a0_ind = randsample(par.alen,N,true)'; % Index for initial wealth.
%
%             for i = 1:N % Person loop.
%                 zsim(1,i) = zgrid(z0_ind(i)); % Productivity in period 0.
%                 csim(1,i) = cpol(a0_ind(i),z0_ind(i)); % Consumption in period 0 given a0.
%                 asim(1,i) = apol(a0_ind(i),z0_ind(i)); % Savings for period 1 given a0.
%                 usim(1,i) = model.utility(csim(1,i),par); % Utility in period 0 given a0.
%             end
%
%             %% Simulate endogenous variables.
%
%             for j = 2:T % Time loop.
%                 for i = 1:N % Person loop.
%                     [~, at_ind] = min(abs(asim(j-1,i) - agrid));
%                     q_ind = q0_ind(i);
%
%                     % at_ind = find(asim(j-1,i)==agrid); % Savings choice in the previous period is the state today. Find where the latter is on the grid.
%                     % zsim(j,i) = zgrid(z0_ind(i)); % Productivity in period 0.
%                     % csim(j,i) = cpol(at_ind,z0_ind(i)); % Consumption in period t.
%                     % asim(j,i) = apol(at_ind,z0_ind(i)); % Savings for period t+1.
%                     % usim(j,i) = model.utility(csim(j,i),par); % Utility in period t.
%                     % z1_ind = find(rand<=cmat(z0_ind(i),:)); % Draw income shock for next period.
%                     % z0_ind(i) = z1_ind(1);
%                     if keepol(at_ind, z0_ind(i), q_ind) == 1
%                         % Keep the idea
%                         keepsim(j,i) = 1;
%
%                         % Apply R&D investment
%                         isim(j,i) = ipol(at_ind, z0_ind(i), q_ind);
%
%                         % Idea evolves
%                         delta_q = par.delta_q;
%                         theta = par.theta;
%                         eta = par.eta;
%
%                         % Draw shock
%                         eps_q = normrnd(0, par.sigma_q);
%
%                         % Evolve idea quality
%                         q_next = (1-delta_q)*qgrid(q_ind) + theta*isim(j,i)^eta + eps_q;
%                         q_next = max(0, min(q_next, par.qmax));
%
%                         % Find nearest grid point
%                         [~, q0_ind(i)] = min(abs(qgrid - q_next));
%                         qsim(j,i) = q_next;
%                     else
%                         % Abandon the idea
%                         keepsim(j,i) = 0;
%                         isim(j,i) = 0;
%                         q0_ind(i) = 1;  % q = 0
%                         qsim(j,i) = 0;
%                     end
%
%                     % Update other variables using the right indices
%                     csim(j,i) = cpol(at_ind, z0_ind(i), q0_ind(i));
%                     asim(j,i) = apol(at_ind, z0_ind(i), q0_ind(i));
%                     usim(j,i) = model.utility(csim(j,i), par);
%
%                     % Draw next productivity
%                     z1_ind = find(rand<=cmat(z0_ind(i),:));
%                     z0_ind(i) = z1_ind(1);
%                 end
%
%                 if j >= 11 && norm(mean(asim(j-9:j,:),'all')-mean(asim(j-10:j-1,:),'all')) < 0.0001
%                     break
%                 end
%
%             end
%
%             sim = struct();
%
%             sim.qsim = qsim(j-9:j,:);
%             sim.isim = isim(j-9:j,:);
%             sim.keepsim = keepsim(j-9:j,:);
%             sim.qavg = mean(qsim(j-9:j,:), 'all');
%             sim.iavg = mean(isim(j-9:j,:), 'all');
%             sim.keeprate = mean(keepsim(j-9:j,:), 'all');
%         end
%
%     end
% end