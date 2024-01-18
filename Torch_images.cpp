#include <torch/torch.h>
#include <iostream>

class Net : public torch::nn::Module 
{
    public:
        Net()
        {
        }
        torch::Tensor forward(torch::Tensor x)
        {
            return x;
        }
    private:

};

int main()
{   
    std::cout << "Images..." << std::endl;
    return 0;
}