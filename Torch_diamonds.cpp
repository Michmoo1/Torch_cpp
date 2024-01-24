#include <torch/torch.h>
#include <fstream>
#include "build/matplotlib-cpp/matplotlibcpp.h"

// some global var
namespace plt = matplotlibcpp;

class Net : public torch::nn::Module 
{
    public:
        Net():
            fc1(register_module("fc1", torch::nn::Linear(3,64))),
            fc2(register_module("fc2", torch::nn::Linear(64,32))),
            fc3(register_module("fc3", torch::nn::Linear(32,1)))
        {
        }
        torch::Tensor forward(torch::Tensor x)
        {

            x = torch::tanh(fc1->forward(x));
            x = torch::tanh(fc2->forward(x));
            x = fc3->forward(x);

            return x;
        }
    private:
        torch::nn::Linear fc1;
        torch::nn::Linear fc2;
        torch::nn::Linear fc3;
};

class DiamondsDataset : public torch::data::datasets::Dataset<DiamondsDataset>
{
    using Example = torch::data::Example<>;
    std::vector<std::vector <std::string>> content_diamonds_dataset = read_dataset_csv();

    public:
        DiamondsDataset(){}
        

        Example get(size_t index)
        {
            float carat = std::stoi(content_diamonds_dataset[index][0]);
            float depth = std::stoi(content_diamonds_dataset[index][4]);
            float table = std::stoi(content_diamonds_dataset[index][5]);
            float target = std::stoi(content_diamonds_dataset[index][6]);
            
            return {torch::tensor({carat, depth, table}), torch::tensor({target})};
        }

        torch::optional<size_t> size() const
        {
            return content_diamonds_dataset.size();
        }

        std::vector<std::vector<std::string>> read_dataset_csv()
        {
            std::vector<std::vector<std::string>> content;
            std::vector<std::string> row;
            std::string line, word;

            std::string fileName = "diamonds.csv";
            std::fstream file (fileName, std::ios::in);

            if (file.is_open())
            {
                while(getline(file, line))
                {
                    row.clear();
                    std::stringstream str(line);
                    
                    while(getline(str, word, ','))
                    {
                        row.push_back(word);
                        
                    }
                    content.push_back(row);
                
                }
                std::cout << "File successfully opened !" << std::endl;
            }
            else
            {
                std::cout << "Error could not open file!" << std::endl;
            }
            return content;
        }
};

int main()
{   
    std::cout << "Diamonds..." << std::endl;
    Net myNetwork = Net();
    DiamondsDataset dataset;
    auto example = dataset.get(0);
    std::cout << example.data << std::endl;
    std::cout << example.target << std::endl;
    
    torch::optim::SGD optimizer(myNetwork.parameters(), torch::optim::SGDOptions(0.01));
    torch::nn::MSELoss criterion;

    //Train loop
    for (int i = 0; i< 500; ++i)
    {
        example = dataset.get(i);
        optimizer.zero_grad();

        torch::Tensor prediction  = myNetwork.forward(example.data);
        torch::Tensor loss = criterion(prediction, example.target);

        loss.backward();
        optimizer.step();
        
        if (i % 50 == 0)
        {
            std::cout << "Epoch: " << i << " Loss: " << loss.item<float>() << std::endl;
        }
        
    }
    
    //Test loop
    for (int i = 600; i< 650; ++i)
    {
        example = dataset.get(i);
        
        torch::Tensor prediction  = myNetwork.forward(example.data);
        std::cout << "Prediction: " << prediction << std::endl;
    }
    plt::plot({1,3,2,4});
    plt::show();
    return 0;
}