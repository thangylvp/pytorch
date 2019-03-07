#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

int main() {
	torch::Tensor tensor = torch::rand({2, 3});
	std::cout << tensor << std::endl;

	std::string model_path = "../../data/model/model.pt";

	std::ifstream is (model_path, std::ifstream::binary);
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(is);

	if (module == nullptr) {
        std::cerr << "model load error " << std::endl;
    }
    std::cout << "Model load ok.\n";
  	assert(module != nullptr);
  	std::cout << "ok\n";
}
