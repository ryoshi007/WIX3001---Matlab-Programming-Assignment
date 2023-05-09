function Y_pred = test(net, X)
% tests network with input X
% assuming dimensions of X is [num_data, num_features]

% how many layers?
layers = (length(fieldnames(net))-2) / 2;

% forward pass
for i = 1:layers
    layer_name = 'W' + string(i);
    bias_name = 'b' + string(i);
    if i == 1
        A = net.(layer_name) * X';
    else
        A = net.(layer_name) * A;
    end
    
    A = A + net.(bias_name);

    if i == layers
        % sigmoid for the last layer
        A = sigmoid(A);
    else
        % ReLU for all other layers
        A = ReLU(A);
    end   
end

% dimensions of A should be [num_outputs, num_data]
% get predictions

[~, Y_pred] = max(A, [], 1);

end


function A = ReLU(Y)

A = max(Y, 0);

end


function Y = sigmoid(A)

Y = 1 ./ (1+exp(-A));

end

