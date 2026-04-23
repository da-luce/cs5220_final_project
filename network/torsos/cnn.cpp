#include "architectures/torsos/cnn.h"

namespace architectures {
namespace torsos {

// ==========================================
// ResidualBlock Implementation
// ==========================================
ResidualBlockImpl::ResidualBlockImpl(int channels) {
    // Spatial dimensions are preserved using Kernel=3, Padding=1, Stride=1
    auto conv_options = torch::nn::Conv2dOptions(channels, channels, 3)
                            .padding(1)
                            .stride(1)
                            .bias(false); // BN handles the bias

    conv1 = register_module("conv1", torch::nn::Conv2d(conv_options));
    bn1   = register_module("bn1",   torch::nn::BatchNorm2d(channels));
    
    conv2 = register_module("conv2", torch::nn::Conv2d(conv_options));
    bn2   = register_module("bn2",   torch::nn::BatchNorm2d(channels));
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x;
    
    // Pass 1
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    
    // Pass 2
    x = conv2->forward(x);
    x = bn2->forward(x);
    
    // Skip connection before final activation
    x += residual;
    return torch::relu(x);
}

// ==========================================
// CNNTorso Implementation
// ==========================================
CNNTorsoImpl::CNNTorsoImpl(int in_channels, int num_filters, int num_blocks) 
    : hidden_channels(num_filters) {
    
    // 1. Initial Convolution (Projects your flat encoding into the hidden dimension)
    auto initial_conv_options = torch::nn::Conv2dOptions(in_channels, num_filters, 3)
                                    .padding(1)
                                    .stride(1)
                                    .bias(false);
                                    
    layers->push_back(torch::nn::Conv2d(initial_conv_options));
    layers->push_back(torch::nn::BatchNorm2d(num_filters));
    layers->push_back(torch::nn::ReLU());

    // 2. Residual Tower
    for (int i = 0; i < num_blocks; ++i) {
        layers->push_back(ResidualBlock(num_filters));
    }

    // 3. Register the entire sequential container
    register_module("layers", layers);
}

torch::Tensor CNNTorsoImpl::forward(torch::Tensor x) {
    // Input x: [Batch, Channels, Height, Width]
    // Output  : [Batch, hidden_channels, Height, Width]
    return layers->forward(x);
}

int CNNTorsoImpl::get_output_channels() const {
    return hidden_channels;
}

} // namespace torsos
} // namespace architectures