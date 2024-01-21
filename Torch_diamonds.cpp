#include <torch/torch.h>
#include <fstream>

// some global var

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
            float target = std::stoi(content_diamonds_dataset[index][6]);
            
            return {torch::tensor({carat, depth}), torch::tensor({target})};
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
    DiamondsDataset dataset;
    auto example = dataset.get(0);
    std::cout << example.data << std::endl;
    std::cout << example.target << std::endl;
    return 0;
}