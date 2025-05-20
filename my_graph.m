%% File Info.

%{

    my_graph.m
    ----------
    This code plots the value and policy functions.

%}

%% Graph class.

classdef my_graph
    methods(Static)
        %% Plot value and policy functions.
        
        function [] = plot_dist_pe(sim)
            %% Plot density function (partial equilibrium).
            
            figure(1)
            
            histogram(sim.asim,20)
            xlabel({'$a$'},'Interpreter','latex')
            ylabel({'Frequency'},'Interpreter','latex')
            title('Distribution of Wealth')
            
        end
        
        function [] = plot_dist_ge(sim_ge)
            %% Plot density function (general equilibrium).
            
            figure(2)
            
            histogram(sim_ge.asim,20)
            xlabel({'$a$'},'Interpreter','latex')
            ylabel({'Frequency'},'Interpreter','latex')
            title('Distribution of Wealth')
            
        end
        
        function [] = cfun(par,sol)
            
            %% Plot consumption policy function.
            
            figure(3)
            plot(par.agrid,sol.c)
            xlabel({'$a_{t}$'},'Interpreter','latex')
            ylabel({'$c_{t}$'},'Interpreter','latex')
            title('Consumption Policy Function')
            
        end
        
        function [] = idea_policy(par, sol)
            % Plot R&D investment policy
            figure(6);
            clf;
            
            % Select middle productivity and asset level
            z_idx = ceil(par.zlen/2);
            a_idx = ceil(par.alen/2);
            
            % Plot R&D investment as function of idea quality
            plot(par.qgrid, squeeze(sol.i(a_idx, z_idx, :)), 'b-', 'LineWidth', 2);
            
            xlabel('Idea Quality (q)');
            ylabel('R&D Investment (i)');
            title('R&D Investment Policy');
            grid on;
            
            % Add threshold line showing when ideas are abandoned
            hold on;
            for q_idx = 1:par.qlen
                if sol.keep(a_idx, z_idx, q_idx) == 0
                    plot(par.qgrid(q_idx), 0, 'rx', 'MarkerSize', 10);
                end
            end
            hold off;
            legend('R&D Investment', 'Abandoned Ideas');
        end
        
        function [] = compare_ideas_costs(par, sol_with_costs, sol_without_costs)
            % Compare results with and without costs of investing in ideas
            figure(7);
            
            % Select middle productivity state
            z_idx = ceil(par.zlen/2);
            
            % Create 2x2 subplot
            subplot(2, 2, 1);
            
            % Compare investment policies for different assets
            a_indices = [ceil(par.alen*0.25), ceil(par.alen*0.5), ceil(par.alen*0.75)];
            colors = {'b', 'g', 'r'};
            
            hold on;
            for i = 1:length(a_indices)
                a_idx = a_indices(i);
                plot(par.qgrid, squeeze(sol_with_costs.i(a_idx, z_idx, :)), [colors{i}, '-'], 'LineWidth', 2);
                plot(par.qgrid, squeeze(sol_without_costs.i(a_idx, z_idx, :)), [colors{i}, '--'], 'LineWidth', 1.5);
            end
            hold off;
            
            xlabel('Idea Quality (q)');
            ylabel('R&D Investment (i)');
            title('R&D Investment Comparison');
            legend('With Costs (Low A)', 'Without Costs (Low A)', ...
                'With Costs (Med A)', 'Without Costs (Med A)', ...
                'With Costs (High A)', 'Without Costs (High A)');
            
            % Compare keep/abandon decisions
            subplot(2, 2, 2);
            
            % Plot threshold curves
            hold on;
            a_vec = linspace(par.amin, par.amax, 100);
            q_threshold_with = zeros(length(a_vec), 1);
            q_threshold_without = zeros(length(a_vec), 1);
            
            for i = 1:length(a_vec)
                a = a_vec(i);
                [~, a_idx] = min(abs(par.agrid - a));
                
                % Find lowest q that is kept for each case
                for q_idx = 1:par.qlen
                    if sol_with_costs.keep(a_idx, z_idx, q_idx) == 1
                        q_threshold_with(i) = par.qgrid(q_idx);
                        break;
                    end
                end
                
                for q_idx = 1:par.qlen
                    if sol_without_costs.keep(a_idx, z_idx, q_idx) == 1
                        q_threshold_without(i) = par.qgrid(q_idx);
                        break;
                    end
                end
            end
            
            plot(a_vec, q_threshold_with, 'b-', 'LineWidth', 2);
            plot(a_vec, q_threshold_without, 'r--', 'LineWidth', 2);
            hold off;
            
            xlabel('Assets (a)');
            ylabel('Minimum Idea Quality Kept');
            title('Abandonment Thresholds');
            legend('With Costs', 'Without Costs');
            
            % Compare consumption
            subplot(2, 2, 3);
            
            % Pick middle asset level
            a_idx = ceil(par.alen/2);
            
            hold on;
            plot(par.qgrid, squeeze(sol_with_costs.c(a_idx, z_idx, :)), 'b-', 'LineWidth', 2);
            plot(par.qgrid, squeeze(sol_without_costs.c(a_idx, z_idx, :)), 'r--', 'LineWidth', 2);
            hold off;
            
            xlabel('Idea Quality (q)');
            ylabel('Consumption (c)');
            title('Consumption Policy');
            legend('With Costs', 'Without Costs');
            
            % Compare savings
            subplot(2, 2, 4);
            
            hold on;
            plot(par.qgrid, squeeze(sol_with_costs.a(a_idx, z_idx, :)), 'b-', 'LineWidth', 2);
            plot(par.qgrid, squeeze(sol_without_costs.a(a_idx, z_idx, :)), 'r--', 'LineWidth', 2);
            hold off;
            
            xlabel('Idea Quality (q)');
            ylabel('Next Period Assets (a'')');
            title('Savings Policy');
            legend('With Costs', 'Without Costs');
            
            suptitle('Comparison: With vs. Without R&D Costs');
        end
    end
end