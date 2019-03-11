#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

#define CUDA 

int main() {
	/* test tensor
	torch::Tensor tensor = torch::rand({2, 3});
	std::cout << tensor << std::endl;
	*/


	std::string model_path = "../../data/model/model.pt";

	std::ifstream is (model_path, std::ifstream::binary);
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(is);

	if (module == nullptr) {
        std::cerr << "model load error " << std::endl;
    }

#ifdef CUDA
	module->to(at::kCUDA);
#endif

    std::cout << "Model load ok.\n";
  	assert(module != nullptr);
  	std::cout << "ok\n";

	std::vector<torch::jit::IValue> inputs;
#ifdef CUDA
	inputs.push_back(torch::ones({1, 3, 224, 224}).to(at::kCUDA));
#else
	inputs.push_back(torch::ones({1, 3, 224, 224}));
#endif
	// Execute the model and turn its output into a tensor.
	at::Tensor output = module->forward(inputs).toTensor();

	std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
