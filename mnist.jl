using MLDatasets
using ImageShow
using Flux
using Infiltrator

train_x_raw, train_y_raw = MNIST(split=:train)[:]
test_x_raw, test_y_raw = MNIST(split=:test)[:]

train_x = Flux.flatten(train_x_raw)
test_x = Flux.flatten(test_x_raw)

train_y = Flux.onehotbatch(train_y_raw, 0:9)
test_y = Flux.onehotbatch(test_y_raw, 0:9)

#=
4LS: A 4-layer model using 16 nodes in the 
inner layers and the sigmoid activation function
=#

model4LS = Chain(
    Dense(28 * 28, 16, sigmoid),        # 784 x 16 + 16 = 12560 parameters
    Dense(16, 16, sigmoid),             #  16 x 16 + 16 =   272 parameters
    Dense(16, 16, sigmoid),             #  16 x 16 + 16 =   272 parameters
    Dense(16, 10, sigmoid)              #  16 x 10 + 10 =   170 parameters
)                                     #             --> 13274 parameters total

#=
3LS: A 3-layer model using 60 nodes in the inner
layers and the sigmoid activation function
=#

model3LS = Chain(
    Dense(28 * 28, 60, sigmoid),        # 784 x 60 + 60 = 47100 parameters
    Dense(60, 60, sigmoid),             #  60 x 60 + 60 =  3660 parameters
    Dense(60, 10, sigmoid)              #  60 x 10 + 10 =   610 parameters
)                                     #             --> 51370 parameters total

#=
2LR: A 2-layer model using 32 nodes in the inner
layer and the relu activation function

This model uses the relu-activation function only 
on the first layer. On the second layer no such 
function is specified. This means that the Flux-default 
is used (identity). As a consequence, this layer may 
produce values in the range [-∞, ∞], which is not what we want. 
Therefore the softmax-function is applied on the results standardizing
them to a range from 0 to 1 and ensuring that they sum up to 1.
=#

model2LR = Chain(
    Dense(28 * 28, 32, relu),           # 784 x 32 + 32 = 25120 parameters
    Dense(32, 10),                      #  32 x 10 + 10 =   330 parameters
    softmax                             #             --> 25450 parameters total
)

#= 
Get the model parameters in order to use them
as input in Flux.train
=#

params4LS = Flux.params(model4LS)
params3LS = Flux.params(model3LS)
params2LR = Flux.params(model2LR)

#=
Define the loss functions
=#

loss4LSmse(x, y) = Flux.Losses.mse(model4LS(x), y)
loss4LSce(x, y) = Flux.Losses.crossentropy(model4LS(x), y)
loss3LSmse(x, y) = Flux.Losses.mse(model3LS(x), y)
loss3LSce(x, y) = Flux.Losses.crossentropy(model3LS(x), y)
loss2LRce(x, y) = Flux.Losses.crossentropy(model2LR(x), y)

function train_batch(X, y, loss, opt, params, epochs)
    data = [(X, y)]
    for epoch in 1:epochs
        Flux.train!(loss, params, data, opt)
        println("Mean square error:", loss4LSmse(x_train, y_train))
        @exfiltrate
    end
end

train_batch(train_x, train_y, loss4LSmse,
    Descent(0.1), params4LS, 100)

printstyled(safehouse.params)
