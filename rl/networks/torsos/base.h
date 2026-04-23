namespace networks {
namespace torsos {

class TorsoBase : public torch::nn::Module {
public:
    virtual ~TorsoBase() = default;
    
    // Every torso must take a tensor and return a processed feature tensor
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    // Useful for heads to know the depth of the feature map they are receiving
    virtual int get_output_channels() const = 0;
};

} // namespace torsos
} // namespace networks