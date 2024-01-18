#include <torch/torch.h>
#include <iostream>

class Net : public torch::nn::Module 
{
    public:
        Net(): 
                conv1(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 5)))),
                conv2(register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 5)))),
                fc1(register_module("fc1", torch::nn::Linear(320,50))),
                fc2(register_module("fc2", torch::nn::Linear(50,10))),
                fc3(register_module("fc3", torch::nn::Linear(5,2)))

        {
            register_module("conv2_drop", conv2_drop);
        }
        torch::Tensor forward(torch::Tensor x)
        {
            x = torch::relu(torch::max_pool2d(conv1->forward(x),2));
            x = torch::relu(torch::max_pool2d(conv2_drop->forward(conv2->forward(x)),2));
            x = x.view({-1, 320});
            x = torch::relu(fc1->forward(x));
            x = torch::dropout(x, 0.5, is_training());
            x = fc2->forward(x);
            //x = fc3->forward(x);
            x = torch::log_softmax(x,1);
            return x;
        }
    private:
        torch::nn::Conv2d conv1;
        torch::nn::Conv2d conv2;
        torch::nn::Dropout2d conv2_drop;
        torch::nn::Linear fc1;
        torch::nn::Linear fc2;
        torch::nn::Linear fc3;
};

int main()
{   
    std::cout << "Images..." << std::endl;
    return 0;
}