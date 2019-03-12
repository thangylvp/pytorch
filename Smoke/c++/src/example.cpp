#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#define CUDA 

#define kIMAGE_SIZE 224
#define kCHANNELS 3
#define kTOP_K 1

bool LoadImage(std::string file_name, cv::Mat &image) {
	image = cv::imread(file_name);  // CV_8UC3
	
	if (image.empty() || !image.data) {
		return false;
	}
	cv::imshow("src", image);
  	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	std::cout << "== image size: " << image.size() << " ==" << std::endl;

	// scale image to fit
	cv::Size scale(kIMAGE_SIZE, kIMAGE_SIZE);
	cv::resize(image, image, scale);
	std::cout << "== simply resize: " << image.size() << " ==" << std::endl;

	// convert [unsigned int] to [float]
	image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

	return true;
	
}

int main() {
	


	std::string model_path = "../../data/model/model.pt";

	std::ifstream is (model_path, std::ifstream::binary);
    std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(is);

	if (module == nullptr) {
        std::cerr << "model load error " << std::endl;
    }

#ifdef CUDA
	if (torch::cuda::is_available()) {
		std::cerr << "CUDA ok" << std::endl;
		module->to(at::kCUDA);
	}
#endif

	std::string labels[2];
	labels[1] = "Smoke";
	labels[0] = "No Smoke";
    std::cout << "Model load ok.\n";
  	assert(module != nullptr);
  	std::cout << "ok\n";

	std::string filePath = "../../data/Smoke/val/pos/";
	cv::Mat srcImg;

	std::vector<cv::String> fn;
    cv::glob(filePath,fn,true);

	double totalTime = 0;
	for (size_t k=0; k<fn.size(); ++k) {
		std::cerr << fn[k] << std::endl;
		if (LoadImage(fn[k], srcImg)) {
			std::cerr << "Done load Image " << std::endl;
			auto input_tensor = torch::from_blob(srcImg.data, {1, kIMAGE_SIZE, kIMAGE_SIZE, kCHANNELS});
			input_tensor = input_tensor.permute({0, 3, 1, 2});
			input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
			input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
			input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);
		#ifdef CUDA
			if (torch::cuda::is_available()) {
				input_tensor = input_tensor.to(at::kCUDA);
			}

		#endif

			double e1 = cv::getTickCount();
			at::Tensor output = module->forward({input_tensor}).toTensor();
			double e2 = cv::getTickCount();

			totalTime += (e2 - e1) / cv::getTickFrequency();
			std::cerr << "FPS : " << (1.0) / ((e2 - e1) / cv::getTickFrequency()) << std::endl;;
			std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
			
			auto results = output.sort(-1, true);
			auto softmaxs = std::get<0>(results)[0].softmax(0);
			auto indexs = std::get<1>(results)[0];

			for (int i = 0; i < kTOP_K; ++i) {
				auto idx = indexs[i].item<int>();
				std::cout << "    ============= Top-" << i + 1
						<< " =============" << std::endl;
				std::cout << "    Label:  " << labels[idx] << std::endl;
				std::cout << "    With Probability:  "
						<< softmaxs[i].item<float>() * 100.0f << "%" << std::endl;
			}
			/*
			auto idx = indexs[0].item<int>();
			std::cout << "    ============= Top-" << 0 + 1<< " =============" << std::endl;
			std::cout << "    Label:  " << labels[idx] << std::endl;
			std::cout << "    With Probability:  " << softmaxs[0].item<float>() * 100.0f << "%" << std::endl;
			*/
			cv::waitKey();
		}
	}
	/*
	std::vector<torch::jit::IValue> inputs;
#ifdef CUDA
	inputs.push_back(torch::ones({1, 3, 224, 224}).to(at::kCUDA));
#else
	inputs.push_back(torch::ones({1, 3, 224, 224}));
#endif
	// Execute the model and turn its output into a tensor.
	at::Tensor output = module->forward(inputs).toTensor();
*/
//	std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
/*	
	auto results = output.sort(-1, true);
	auto softmaxs = std::get<0>(results)[0].softmax(0);
    auto indexs = std::get<1>(results)[0];

	auto idx = indexs[0].item<int>();
	std::cout << "    ============= Top-" << 0 + 1<< " =============" << std::endl;
    std::cout << "    Label:  " << labels[idx] << std::endl;
    std::cout << "    With Probability:  " << softmaxs[0].item<float>() * 100.0f << "%" << std::endl;
*/
}
