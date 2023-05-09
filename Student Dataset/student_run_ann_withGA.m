function [population, fitness_score, progress] = student_run_ann_withGA()

% set RNG seed number to get reproducible results; 
% change seed number to get different results
% seed 9420, 7243, 321
seed = 321;
rng('default');
rng(seed);

% load dataset
data = dlmread('student_academic_success.csv');

X = data(:, 1:end-1); % assuming all columns are attributes
Y = data(:, end); % except for the last column for labels

[~, num_X_cols] = size(X);
num_Y_value = length(unique(Y));

% set maximum values for the layers'parameter
% seed 9420 -> max 4 layers, 50 units per layer
% seed 7243 -> max 4 layers, 30 units per layer
% seed 321 -> max 4 layers, 30 units per layer
max_hidden_layer = 4;
max_unit_per_layer = 30;

% set parameters for GA
population_size = 200; % 200 chromosomes for population
generations_max = 1000; % run for 1000 generations
selrate = 0.2; % SelectionRate
mutrate = 0.8; % MutationRate

% parameter for simulated annealing
initial_temperature = 0.5;
cooling_rate = 0.8;
num_iterations = 100;

progress = [];

convergence_maxcount = 10; % stop the GA if the average fitness score stopped increasing for 10 generations
convergence_count = 0;
convergence_avg = 0;

% initialize population
% determine the maximum possible chromosome length
max_chromosome_length = (num_X_cols*max_unit_per_layer+max_unit_per_layer)+((max_unit_per_layer*max_unit_per_layer)+max_unit_per_layer)*(max_hidden_layer-1)+(max_unit_per_layer*num_Y_value+num_Y_value)+1+max_hidden_layer;

% pre-allocate the population matrix with zero values
% contain information of chromosome, including number of hidden layers,
% number of units per layer, weights and biases
population = zeros(population_size, max_chromosome_length); 

for i = 1:population_size
    hidden_layers = randi([1, max_hidden_layer]);
    hidden_layers_units = randi([1, max_unit_per_layer], 1, hidden_layers);
    net = create_network([hidden_layers, hidden_layers_units], num_X_cols, num_Y_value);
    chromosome = network_to_chromosome(net, max_chromosome_length, max_hidden_layer, max_unit_per_layer);
    population(i, 1:length(chromosome)) = chromosome;
end

fitness_score = zeros(population_size, 1);

% store the information for first and last generation
first_generation_population = population(:, 1:max_hidden_layer+1);
last_generation_population = [];

generations_current = 1;
while generations_current < generations_max

    % test all chromosomes that haven't been tested
    for i = 1:population_size

        if fitness_score(i,1) == 0
            % fitness testing a chromosome
            fitness_score(i,1) = fitness_function(population(i, :), X, Y, num_X_cols, num_Y_value, max_hidden_layer, max_unit_per_layer);
            
            if generations_current ~= 1
                % apply simulated annealing to the current chromosome
                optimized_chromosome = simulated_annealing(population(i, :), X, Y, num_X_cols, num_Y_value, max_hidden_layer, max_unit_per_layer, initial_temperature, cooling_rate, num_iterations);
                % update the population with the optimized chromosome
                population(i, :) = optimized_chromosome;
                % re-evaluate the fitness of the optimized chromosome
                fitness_score(i,1) = fitness_function(optimized_chromosome, X, Y, num_X_cols, num_Y_value, max_hidden_layer, max_unit_per_layer);
            end
        end
    end

    % find out statistics of the population
    fit_avg = mean(fitness_score);
    fit_max = max(fitness_score);
    progress = [progress; fit_avg, fit_max];

    % print current progress
    disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));

    % Convergence criteria
    % convergence? 
    if fit_avg > convergence_avg
        convergence_avg = fit_avg;
        convergence_count = 0;
    else
        convergence_count = convergence_count + 1;
    end

    % perform triggered hypermutation to promote diversity and escape local
    % optima at every 30th generation
    if (mod(generations_current, 30) == 0)
        % define the update factors
        update_factor_mutrate = 0.15; % 15% increase
        update_factor_selrate = 0.15; % 15% decrease
        
        % update mutation and selection rates
        mutrate = mutrate * (1 + update_factor_mutrate);
        selrate = selrate * (1 - update_factor_selrate);
        
        % limit mutation and selection rates within a range
        mutrate = min(max(mutrate,0),1);
        selrate = min(max(selrate,0),1);
    end

    % stop the GA if reach 100% accuracy or reach convergence?
    % instead of stopping immediately, slowly adjust selrate and mutrate
    if (fit_max >= 1)
        generations_max = 0;
        last_generation_population = population(:, 1:max_hidden_layer+1);
        disp("Reached convergence.")
    
    elseif (convergence_count > convergence_maxcount)
        % what to do if fitness haven't improved?
        % stop the GA?
        % generations_max = 0;
        % disp("Reached convergence.")
        % disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
    
        % or adjust selection rate and mutation rate for fine-grained search
        if (selrate < 0.98)
            [selrate, mutrate] = update_rates(fit_avg, fit_max, selrate, mutrate);
        else
            generations_max = 0;
            last_generation_population = population(:, 1:max_hidden_layer+1);
            disp("Reached convergence.")
        end
    end

    generations_current = generations_current + 1;

    % do genetic operators
    [population, fitness_score] = genetic_operators(population, fitness_score, selrate, mutrate, max_hidden_layer, max_unit_per_layer, fit_avg, fit_max);
end

% plot a graph of average fitness score, best fitness score vs number of
% generations
[num_prog_rows, ~] = size(progress);
generation = 1:num_prog_rows;
avg_fit = progress(:, 1);
best_fit = progress(:, 2);
plot(generation, avg_fit, '-', generation, best_fit, '-');
xlabel('Number of Generations');
ylabel('Fitness Scores');
title('Fitness Scores VS Number of Generations');
legend('Average Fitness', 'Maximum Fitness');

% display information of first and last generation
disp("First Generation");
disp(first_generation_population);

disp("");
disp("Last Generation");
disp(last_generation_population);

end

% decode the information for chromosome from network structure
function chromosome = network_to_chromosome(net, max_chromosome_length, max_hidden_layer, max_unit_per_layer)

hidden_layers = net.hidden_layers;
hidden_layer_units = [];
weights_and_biases = [];

input_units = size(net.W1, 2);
output_units = size(net.(['W', num2str(hidden_layers + 1)]), 1);

for i = 1:max_hidden_layer
    if i <= hidden_layers
        hidden_layer_units = [hidden_layer_units, size(net.(['W', num2str(i)]), 1)];
    else 
        hidden_layer_units = [hidden_layer_units, 0];
    end
end

for i = 1:max_hidden_layer
    layer_name = 'W' + string(i);
    bias_name = 'b' + string(i);

    if i <= hidden_layers
        weights = net.(layer_name)(:);
        biases = net.(bias_name)(:);
    else
        weights = [];
        biases = [];
    end

    % calculate remaining weights and biases for current layer
    if i == 1
        max_weights = max_unit_per_layer * input_units;
    else
        max_weights = max_unit_per_layer * max_unit_per_layer;
    end

    max_biases = max_unit_per_layer;
    
    remaining_weights = max_weights - length(weights);
    remaining_biases = max_biases - length(biases);
    
    % pad current layer's weights and biases with zeros
    weights_and_biases = [weights_and_biases, weights', zeros(1, remaining_weights), biases', zeros(1, remaining_biases)];
end

% add the weights and biases of the output layer with padding
output_layer = hidden_layers + 1;
output_weights = net.(['W', num2str(output_layer)])(:);
output_biases = net.(['b', num2str(output_layer)])(:);

max_output_weights = max_unit_per_layer * output_units;
max_output_biases = output_units;

remaining_output_weights = max_output_weights - length(output_weights);
remaining_output_biases = max_output_biases - length(output_biases);

weights_and_biases = [weights_and_biases, output_weights', zeros(1, remaining_output_weights), output_biases', zeros(1, remaining_output_biases)];

% combine 
chromosome_partial = [hidden_layers, hidden_layer_units, weights_and_biases];
chromosome = [chromosome_partial, zeros(1, max_chromosome_length - length(chromosome_partial))];

end


% decode the information for network structure from chromosome
function net = chromosome_to_network(chromosome, num_X_cols, num_Y_value, max_hidden_layers, max_unit_per_layer)

    hidden_layers = chromosome(1);
    hidden_layer_units = chromosome(2:1+max_hidden_layers);

    net = struct();
    net.hidden_layers = hidden_layers;

    current_index = 1 + max_hidden_layers + 1;

    % handle hidden layers
    for i = 1:hidden_layers
        layer_name = 'W' + string(i);
        bias_name = 'b' + string(i);

        if i == 1
            layer_size = [hidden_layer_units(i), num_X_cols];
            max_weights = max_unit_per_layer * num_X_cols;
        else
            layer_size = [hidden_layer_units(i), hidden_layer_units(i-1)];
            max_weights = max_unit_per_layer * max_unit_per_layer;
        end

        num_weights = prod(layer_size);
        num_biases = layer_size(1);
        max_biases = max_unit_per_layer;
        
        net.(layer_name) = reshape(chromosome(current_index:current_index+num_weights - 1), layer_size);
        net.(bias_name) = chromosome(current_index + max_weights : current_index + max_weights + num_biases - 1)';

        current_index = current_index + max_weights + max_biases;
    end

    % handle output layer separately
    current_index = 1 + max_hidden_layers + 1 + (max_hidden_layers-1)*(max_unit_per_layer * max_unit_per_layer + max_unit_per_layer) + (num_X_cols*max_unit_per_layer+max_unit_per_layer);
    
    i = i + 1;
    layer_name = 'W' + string(i);
    bias_name = 'b' + string(hidden_layers + 1);
    layer_size = [num_Y_value, hidden_layer_units(hidden_layers)];

    num_weights = prod(layer_size);
    max_weights = max_unit_per_layer * num_Y_value;

    net.(layer_name) = reshape(chromosome(current_index:current_index+num_weights - 1), layer_size);
    net.(bias_name) = chromosome(current_index + max_weights : current_index + max_weights + num_Y_value - 1)';
end


% test the accuracy of the chromosome
function score = fitness_function(chromosome, X, Y, num_X_cols, num_Y_value, max_hidden_layers, max_unit_per_layer)

    net = chromosome_to_network(chromosome, num_X_cols, num_Y_value, max_hidden_layers, max_unit_per_layer);

    % now test the new network
    Y_pred = test(net, X);
    score = mean(Y == Y_pred');
end


% perform operations on the genes of chromosome
function [population, fitness_score] = genetic_operators(population, fitness_score, selrate, mutrate, max_hidden_layer, max_unit_per_layer, fit_avg, fit_max)

% how many chromosomes to reject?
popsize = size(population, 1);
num_reject = round((1-selrate) * popsize);

    for i = 1:num_reject
        % find lowest fitness score and remove the chromosome
        [~, lowest] = min(fitness_score);
        population(lowest, :) = [];
        fitness_score(lowest) = [];
    end
    
    % for each rejection, create a new chromosome
    num_parents = size(population, 1);
    
    for i = 1:num_reject
    
        % several methods can be used to select parent chromosomes
    
        % 1. rank-based selection
        % [~, sorted_indices] = sort(fitness_score, 'descend');
        % parent1 = population(sorted_indices(parent_idx(1)), :);
        % parent2 = population(sorted_indices(parent_idx(2)), :);
    
        % 2. stochastic universal sampling
        % select parents based on a probability distribution proportional to
        % their fitness value
        selected_indices = stochastic_universal_sampling(fitness_score, 2);
        if selected_indices(1) == 0 || selected_indices(2) == 0
            parent1 = population(randi([1, 10]), :);
            parent2 = population(randi([1, 10]), :);
        else
            parent1 = population(selected_indices(1), :);
            parent2 = population(selected_indices(2), :);
        end
    
        % 3. random permutation method
        % order = randperm(num_parents);
        % parent1 = population(order(1), :);
        % parent2 = population(order(2), :);
    
        % mix-and-match
        offspring = crossover(parent1, parent2, max_hidden_layer);
    
        % mutation
        offspring = mutation(offspring, max_hidden_layer, max_unit_per_layer, mutrate);
        
        % add new offspring to population
        population = [population; offspring];
        fitness_score = [fitness_score;0];
    end

end


function offspring = crossover(parent1, parent2, max_hidden_layer)

% crossover on number of hidden layer
if rand < 0.5
    offspring_layer = parent1(1);
else
    offspring_layer = parent2(1);
end

% check for longest and shortest parent
longest_parent = [];
shortest_parent = [];
min_hidden_layer = min(parent1(1), parent2(1));
if min_hidden_layer == parent1(1)
    longest_parent = [longest_parent, parent2];
    shortest_parent = [shortest_parent, parent1];
else
    longest_parent = [longest_parent, parent1];
    shortest_parent = [shortest_parent, parent2];
end

% crossover on number of units per hidden layer
offspring_hidden_unit = [];
for i = 1:offspring_layer
    if i > min_hidden_layer
        offspring_hidden_unit = [offspring_hidden_unit, longest_parent(i+1)];
    else
        prob = rand;
        if prob < 0.33
            offspring_hidden_unit = [offspring_hidden_unit, longest_parent(i+1)];
        elseif prob < 0.66
            offspring_hidden_unit = [offspring_hidden_unit, ceil((longest_parent(i+1)+shortest_parent(i+1))/2)];
        else
            offspring_hidden_unit = [offspring_hidden_unit, shortest_parent(i+1)];
        end
    end
end

offspring_hidden_unit = [offspring_hidden_unit, zeros(1, (max_hidden_layer - length(offspring_hidden_unit)))];

offspring_weights_biases = [(parent1(max_hidden_layer+2:end) + parent2(max_hidden_layer+2:end))/2];


% add new value for newly formed layer with zero values
% find the indices of zero values
zero_indices = offspring_weights_biases == 0;

% generate random numbers between -1 and 1 for the zero values
random_values = 2 * rand(size(offspring_weights_biases)) - 1;

% replace the zero values with random values between -1 and 1
offspring_weights_biases(zero_indices) = random_values(zero_indices);

offspring = [offspring_layer, offspring_hidden_unit, offspring_weights_biases];

end


function offspring = mutation(offspring, max_hidden_layer, max_unit_per_layer, mutrate)

% Creep mutation parameters
% introduce small changes to the values of genes
% creep_rate = 0.2;
% creep_min = -0.5 - mutrate;
% creep_max = 0.5 + mutrate;

% mutate on number of hidden layer
if rand < mutrate
    offspring_layer = randi([1, max_hidden_layer]);
else
    offspring_layer = offspring(1);
end

% mutate on number of units per hidden layer
offspring_hidden_unit = [];
for i = 1:offspring_layer
    if rand < mutrate
        if offspring(i+1) == 0 || offspring(i+1) == max_unit_per_layer
            offspring_hidden_unit(i) = ceil(randi([2, max_unit_per_layer])/2);
        else
            offspring_hidden_unit(i) = randi([offspring(i + 1), offspring(i + 1) + 1]);
        end
    else
        if offspring(i+1) == 0
            offspring_hidden_unit(i) = randi([1, max_unit_per_layer]);
        else    
            offspring_hidden_unit(i) = offspring(i + 1);
        end
    end
end
offspring_hidden_unit = [offspring_hidden_unit, zeros(1, (max_hidden_layer - length(offspring_hidden_unit)))];


offspring_weights_biases = [];
for j = max_hidden_layer+2:length(offspring)
    if rand < mutrate

        % 1. Add a small random value to the gene (creep mutation)
        % sigma = 0.1;
        % creep_value = creep_min + (creep_max - creep_min) * rand;
        % offspring_weights_biases = [offspring_weights_biases, (offspring(j) + creep_rate * creep_value + sigma * randn)];

        % 2. add Gaussian noise with mean 0 and standard deviation 'sigma'
        sigma = 0.1;
        offspring_weights_biases = [offspring_weights_biases, (offspring(j) + sigma * randn)];

    else
        offspring_weights_biases= [offspring_weights_biases, offspring(j)];
    end
end

% add new value for newly formed layer with zero values
% find the indices of zero values
zero_indices = offspring_weights_biases == 0;

% generate random numbers between -1 and 1 for the zero values
sigma = 0.1;
random_values = 2 * rand(size(offspring_weights_biases)) - 1 + sigma * randn;

% replace the zero values with random values between -1 and 1
offspring_weights_biases(zero_indices) = random_values(zero_indices);

offspring = [offspring_layer, offspring_hidden_unit, offspring_weights_biases];

end


% adapative rate
function [selrate, mutrate] = update_rates(fit_avg, fit_max, selrate, mutrate)

ratio = fit_avg / fit_max;
selrate = selrate + 0.1 * ratio;
mutrate = mutrate - 0.1 * ratio;

end


function [selected_indices] = stochastic_universal_sampling(fitness_score, num_individuals_to_select)

fitness_sum = sum(fitness_score);

% calculates the distance between adjacent selection pointers based on the number of individuals to select and the total fitness score of the population
pointer_distance = fitness_sum / num_individuals_to_select;
start_pointer = rand * pointer_distance;

% generates a vector of selection pointers that are evenly spaced across the fitness range of the population
% first pointer starts at start_pointer, and subsequent pointers are spaced by pointer_distance
% last pointer is at the end of the fitness range, which is fitness_sum minus the sum of all previous pointer distances
pointers = start_pointer:pointer_distance:(fitness_sum - (fitness_sum - start_pointer - pointer_distance * (num_individuals_to_select - 1)));

selected_indices = zeros(1, num_individuals_to_select);

% calculates the cumulative sum of the fitness scores, which will be used to determine which individuals are selected based on the positions of the selection pointers
cum_fitness = cumsum(fitness_score);
pointer_idx = 1;

for i = 1:length(fitness_score)
    % checks whether the current selection pointer (determined by pointer_idx) falls within the fitness range of that individual
    while pointer_idx <= length(pointers) && pointers(pointer_idx) <= cum_fitness(i)
        selected_indices(pointer_idx) = i;
        pointer_idx = pointer_idx + 1;
    end
end

end


% metaheuristic optimization algorithm inspired by the process of annealing in metallurgy, which aims to find the global optimum of a cost or objective function by allowing uphill moves with a decreasing probability as the algorithm progresses
function new_chromosome = simulated_annealing(chromosome, X, Y, num_X_cols, num_Y_value, max_hidden_layer, max_unit_per_layer, initial_temperature, cooling_rate, num_iterations)

% initialize the current solution as the input chromosome
current_solution = chromosome;

% compute the fitness of the current solution
current_fitness = fitness_function(current_solution, X, Y, num_X_cols, num_Y_value, max_hidden_layer, max_unit_per_layer);

% set the initial temperature
temperature = initial_temperature;

% loop for a fixed number of iterations
for i = 1:num_iterations

    % generate a new solution by perturbing the current solution
    new_solution = perturb(current_solution, max_hidden_layer, max_unit_per_layer);

    % compute the fitness of the new solution
    new_fitness = fitness_function(new_solution, X, Y, num_X_cols, num_Y_value, max_hidden_layer, max_unit_per_layer);

    % calculate the change in fitness
    delta_fitness = new_fitness - current_fitness;

    % if the new solution is better or accepted by the annealing criterion, update the current solution
    if delta_fitness > 0 || exp(delta_fitness / temperature) > rand
        current_solution = new_solution;
        current_fitness = new_fitness;
    end

    % update the temperature
    temperature = temperature * cooling_rate;
end

new_chromosome = current_solution;

end


% induce changes on the chromosomes during annealing
function new_solution = perturb(current_solution, max_hidden_layer, max_unit_per_layer)

% initialize the new solution as the current solution
new_solution = current_solution;

% modify the weights and biases with small random changes
for j = max_hidden_layer + 2:length(new_solution)
    sigma = 0.1;
    new_solution(j) = new_solution(j) + sigma * randn;
end

end