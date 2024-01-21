#include <torch/torch.h>
#include <iostream>

// some global var
std::string datasetPath = "./caltech-101/101_ObjectCategories/";
std::string infoPath = "./caltech-101/";



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