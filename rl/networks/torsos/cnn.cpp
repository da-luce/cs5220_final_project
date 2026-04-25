#include "networks/torsos/cnn.h"

namespace networks {
namespace torsos {

ResidualBlockImpl::ResidualBlockImpl(int channels) {
    auto options = torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(true);
    conv1 = register_module("conv1", torch::nn::Conv2d(options));
    conv2 = register_module("conv2", torch::nn::Conv2d(options));
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x;
    x = torch::relu(conv1->forward(x));
    x = conv2->forward(x);
    return torch::relu(x + residual);
}

CNNTorsoImpl::CNNTorsoImpl(int in_channels, int num_filters, int num_blocks) 
    : hidden_channels(num_filters) {
    
    // Initial Conv
    layers->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, num_filters, 3).padding(1)));
    layers->push_back(torch::nn::ReLU());

    // If num_blocks is 0, we just do one more conv (like the Python model)
    if (num_blocks == 0) {
        layers->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, num_filters * 2, 3).padding(1)));
        layers->push_back(torch::nn::ReLU());
        hidden_channels = num_filters * 2;
    } else {
        for (int i = 0; i < num_blocks; ++i) {
            layers->push_back(ResidualBlock(num_filters));
        }
    }

    register_module("layers", layers);
}

torch::Tensor CNNTorsoImpl::forward(torch::Tensor x) {
    return layers->forward(x);
}

int CNNTorsoImpl::get_output_channels() const {
    return hidden_channels;
}

} // namespace torsos
} // namespace networks
