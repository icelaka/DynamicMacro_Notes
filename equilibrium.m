%% File Info.

%{

    equilibrium.m
    -------------
    This code compute the equilibrium of the model.

%}

%% Solve class.

classdef equilibrium
    methods(Static)
        
        function F = obj_fun_baseline(r0,par)
            %% Find the value of r so that markets clear.
            
            par.r = r0; % Guess of r.
            
            [par,sol] = solve.firm_problem_baseline(par); % Firms.
            sol = solve.hh_problem_baseline(par,sol); % Households.
            sim = simulate.economy_baseline(par,sol);
            
            F = norm(sim.asup-sol.k);
            
        end

        function F = obj_fun_profit(r0,par)
            %% Find the value of r so that markets clear.
            
            par.r = r0; % Guess of r.
            
            [par,sol] = solve.firm_problem_profit(par); % Firms.
            sol = solve.hh_problem_profit(par,sol); % Households.
            sim = simulate.economy_profit(par,sol);
            
            F = norm(sim.asup-sol.k);
            
        end

        function F = obj_fun_idea(r0,par)
            %% Find the value of r so that markets clear.
            
            par.r = r0; % Guess of r.
            
            [par,sol] = solve.firm_problem_idea(par); % Firms.
            sol = solve.hh_problem_idea(par,sol); % Households.
            sim = simulate.economy_idea(par,sol);
            
            F = norm(sim.asup-sol.k);
            
        end
        
    end
end