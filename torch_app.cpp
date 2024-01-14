#include <torch/torch.h>
#include <iostream>

class Net : public torch::nn::Module 
{
    public:
        Net(): conv1(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 5)))),
                conv2(register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 10, 5)))),
                fc1(register_module("fc1", torch::nn::Linear(320,50))),
                fc2(register_module("fc2", torch::nn::Linear(50,10)))
        {}
    
    private:
        torch::nn::Conv2d conv1;
        torch::nn::Conv2d conv2;
        torch::nn::Linear fc1;
        torch::nn::Linear fc2;


};

int main()
{
    torch::Tensor tensor = torch::rand({2,3});
    std::cout << tensor << std::endl;

    Net myNetwork = Net();
    
    std::cout << myNetwork.name() <<std::endl;
}