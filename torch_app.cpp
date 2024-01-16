#include <torch/torch.h>
#include <iostream>

const int64_t interval = 10;

class Net : public torch::nn::Module 
{
    public:
        Net(): 
                conv1(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 5)))),
                conv2(register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 5)))),
                fc1(register_module("fc1", torch::nn::Linear(320,50))),
                fc2(register_module("fc2", torch::nn::Linear(50,10)))

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
            x = torch::log_softmax(x,1);
            return x;
        }
    private:
        torch::nn::Conv2d conv1;
        torch::nn::Conv2d conv2;
        torch::nn::Dropout2d conv2_drop;
        torch::nn::Linear fc1;
        torch::nn::Linear fc2;
};

int main()
{
    // The batch size for training.
    const int64_t batchSizeTrain = 64;
    const int64_t batchSizeTest = 1000;
    const int64_t epochs = 10;
    const int64_t interval = 10;

    // Set to CPU , GPU tbd
    torch::Device device = torch::kCPU;

    torch::Tensor tensor = torch::rand({2,3});
    std::cout << tensor << std::endl;

    Net myNetwork = Net();

    // Train Data
    auto train_dataset = torch::data::datasets::MNIST("./")
        .map(torch::data::transforms::Stack<>());

    auto train_data_loader = torch::data::make_data_loader(
        std::move(train_dataset), 64);
    
    // Test data
     auto test_dataset = torch::data::datasets::MNIST("./", torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Stack<>());
    auto test_data_loader = torch::data::make_data_loader(
        std::move(test_dataset), 64);

    torch::optim::SGD optimizer(myNetwork.parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
    torch::nn::NLLLoss criterion;
    torch::Tensor loss_print;
    torch::Tensor test_loss_print;

    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        for (auto& batch : *train_data_loader)
        {
            optimizer.zero_grad();

            torch::Tensor prediction = myNetwork.forward(batch.data);
            torch::Tensor loss = criterion->forward(prediction, batch.target);

            loss.backward();
            optimizer.step();
            loss_print = loss;   
        }
        for (auto& batch : *test_data_loader)
        {
            torch::Tensor prediction = myNetwork.forward(batch.data);
            torch::Tensor loss = criterion->forward(prediction, batch.target);
            test_loss_print = loss;   
        }
        std::cout << loss_print << std::endl;
        std::cout << test_loss_print << std::endl;
    }
    
}