# Loose notes

Convolutional neural network for the Mnih et al. (2015) architecture

84x84x4 input
first hidden layer:
32 8x8 filters with stride 4, ReLU
second hidden layer:
64 4x4 filters with stride 2, ReLU
third hidden layer:
64 3x3 filters with stride 1, ReLU
fully connected layer:
512 units, ReLU
output layer:
1 for each action

since in the NEC algorithm, we dont use the output of the network to predict the Q-value,
we can use the output of the network to predict the latent space representation of the state
in our case the input is the state for the envionment and the output is the latent space representation of the state
