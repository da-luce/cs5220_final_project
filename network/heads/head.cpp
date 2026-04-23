#include "architectures/heads/heads.h"

namespace architectures {
namespace heads {

// ==========================================
// Policy Head Implementation
// ==========================================
PolicyHeadImpl::PolicyHeadImpl(int in_channels, int D) {
    // A 1x1 convolution acts like an independent Dense layer for every single square on the board.
    // It maps your hidden features (e.g., 128) down to your D move actions.
    conv = register_module("conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, D, 1) // Kernel size 1, Stride 1, Padding 0
    ));
}

torch::Tensor PolicyHeadImpl::forward(torch::Tensor x) {
    // Input : [Batch, in_channels, H, W]
    // Output: [Batch, D, H, W]
    return conv->forward(x); 
}

// ==========================================
// Value Head Implementation
// ==========================================
ValueHeadImpl::ValueHeadImpl(int in_channels, int board_size) {
    // 1. Reduce the channel depth to 1 using a 1x1 conv to save parameters before flattening
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 1, 1)));
    bn = register_module("bn", torch::nn::BatchNorm2d(1));
    
    // 2. Dense layers to process the flattened board down to a single value
    fc1 = register_module("fc1", torch::nn::Linear(board_size, 256));
    fc2 = register_module("fc2", torch::nn::Linear(256, 1));
}

torch::Tensor ValueHeadImpl::forward(torch::Tensor x) {
    // 1. Pass through 1x1 conv and activation
    x = conv->forward(x);
    x = bn->forward(x);
    x = torch::relu(x);
    
    // 2. Flatten the spatial dimensions [Batch, 1, H, W] -> [Batch, H * W]
    x = x.view({x.size(0), -1});
    
    // 3. Dense layers
    x = fc1->forward(x);
    x = torch::relu(x);
    
    x = fc2->forward(x);
    
    // 4. Tanh squashes the output to a range of [-1.0, 1.0] representing Win/Loss
    return torch::tanh(x);
}

} // namespace heads
} // namespace architectures