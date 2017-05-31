#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::activation;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

inline void parse_kaggle_dogcat_test
	(
	const std::string &filename,
	std::vector<vec_t> *test_images,
	std::vector<label_t> *test_labels,
	float_t scale_min,
	float_t scale_max,
	int x_padding,
	int y_padding
	)
{
	const int DC_IMAGE_DEPTH = 3;
	const int DC_IMAGE_WIDTH = 64;
	const int DC_IMAGE_HEIGHT = 64;
	const int DC_IMAGE_AREA = DC_IMAGE_WIDTH * DC_IMAGE_HEIGHT;
	const int DC_IMAGE_SIZE = DC_IMAGE_AREA * DC_IMAGE_DEPTH;
	const int DC_NUM_CLASS_SAMPLE = 1000; // manual label 1000 samples in test folder

	if (x_padding < 0 || y_padding < 0)
		{
		throw nn_error("padding size must not be negative");
		}
	if (scale_min >= scale_max)
		{
		throw nn_error("scale_max must be greater than scale_min");
		}

	std::ifstream labelfile(filename + "/test/test_label1000.txt");
	std::string line;

	std::vector<unsigned char> buf(DC_IMAGE_SIZE);
	for (int i = 1; i <= DC_NUM_CLASS_SAMPLE; ++i)
		{
		Mat imtest = imread(filename + "/test/" + to_string(i) + ".jpg", IMREAD_COLOR);
		vec_t img;
		int w = DC_IMAGE_WIDTH + 2 * x_padding;
		int h = DC_IMAGE_HEIGHT + 2 * y_padding;
		img.resize(w * h * DC_IMAGE_DEPTH, scale_min);
		for (int c = 0; c < DC_IMAGE_DEPTH; c++)
			{
			for (int y = 0; y < DC_IMAGE_HEIGHT; y++)
				{
	        		for (int x = 0; x < DC_IMAGE_WIDTH; x++)
					{
					//cat
					img[c * w * h + (y + y_padding) * w + x + x_padding] =
					scale_min +
					(scale_max - scale_min) *
					imtest.at<Vec3b>(y, x)[c] / 255;
					}
				}
			}

		test_images->push_back(img);
		std::getline(labelfile, line);

		test_labels->push_back(std::stoi(line));
		}
	labelfile.close();
}

inline void parse_kaggle_dogcat_train
	(
	const std::string &filename,
	std::vector<vec_t> *train_images,
	std::vector<label_t> *train_labels,
	float_t scale_min,
	float_t scale_max,
	int x_padding,
	int y_padding
	)
{
	const int DC_IMAGE_DEPTH = 3;
	const int DC_IMAGE_WIDTH = 64;
	const int DC_IMAGE_HEIGHT = 64;
	const int DC_IMAGE_AREA = DC_IMAGE_WIDTH * DC_IMAGE_HEIGHT;
	const int DC_IMAGE_SIZE = DC_IMAGE_AREA * DC_IMAGE_DEPTH;
	const int DC_NUM_CLASS_SAMPLE = 12500;

	if (x_padding < 0 || y_padding < 0)
		{
		throw nn_error("padding size must not be negative");
		}
	if (scale_min >= scale_max)
		{
		throw nn_error("scale_max must be greater than scale_min");
		}

	std::vector<unsigned char> buf(DC_IMAGE_SIZE);
	for (int i = 0; i < DC_NUM_CLASS_SAMPLE; ++i)
		{
		Mat imcat = imread(filename + "/train/cat." + to_string(i) + ".jpg", IMREAD_COLOR);
		Mat imdog = imread(filename + "/train/dog." + to_string(i) + ".jpg", IMREAD_COLOR);
		vec_t img0;
		vec_t img1;
		int w = DC_IMAGE_WIDTH + 2 * x_padding;
		int h = DC_IMAGE_HEIGHT + 2 * y_padding;
		img0.resize(w * h * DC_IMAGE_DEPTH, scale_min);
		img1.resize(w * h * DC_IMAGE_DEPTH, scale_min);
		for (int c = 0; c < DC_IMAGE_DEPTH; c++)
			{
			for (int y = 0; y < DC_IMAGE_HEIGHT; y++)
				{
				for (int x = 0; x < DC_IMAGE_WIDTH; x++)
					{
					//cat
					img0[c * w * h + (y + y_padding) * w + x + x_padding] =
					scale_min +
					(scale_max - scale_min) *
					imcat.at<Vec3b>(y, x)[c] / 255;
					//dog
					img1[c * w * h + (y + y_padding) * w + x + x_padding] =
					scale_min +
					(scale_max - scale_min) *
					imdog.at<Vec3b>(y, x)[c] / 255;
					}
				}
			}
		train_images->push_back(img0);
		train_labels->push_back(0);
		train_images->push_back(img1);
		train_labels->push_back(1);
		}
}

template <typename N>
void construct_net
	(
	N &nn,
	core::backend_t backend_type
	)
{
    using conv    = convolutional_layer;
    using pool    = max_pooling_layer;
    using fc      = fully_connected_layer;
    using relu    = relu_layer;
    using softmax = softmax_layer;
    using dp      = dropout_layer;

    const serial_size_t n_fmaps  = 30;  // number of feature maps for upper layer
    const serial_size_t n_fmaps2 = 50;  // number of feature maps for lower layer

    nn << conv(64, 64, 3, 3, n_fmaps, padding::same, true, 1, 1, backend_type) // C1
       << relu() // activation
       << pool(64, 64, n_fmaps, 2, backend_type) // P2
       << conv(32, 32, 3, n_fmaps, n_fmaps2, padding::same, true, 1, 1, backend_type) // C3
       << relu() // activation
       << conv(32, 32, 3, n_fmaps2, n_fmaps2, padding::same, true, 1, 1, backend_type) // C4
       << relu() // activation
       << pool(32, 32, n_fmaps2, 2, backend_type) // P5
       << fc(16 * 16 * n_fmaps2, 256, true, backend_type) // FC6
       << relu() // activation
       << dp(256, 0.5)
       << fc(256, 2, true, backend_type) //FC7
       << softmax(2); // FC8
}

void train_cifar10_dog_cat
	(
	std::string data_dir_path,
	double learning_rate,
	const int n_train_epochs,
	const int n_minibatch,
	core::backend_t backend_type,
	std::ostream &log
	)
{
	// specify loss-function and learning strategy
	network<sequential> nn;
	adam optimizer;
	construct_net(nn, backend_type);

	for (int i = 0; i < nn.depth(); i++)
		{
		std::cout << "#layer:" << i << "\n";
		std::cout << "layer type:" << nn[i]->layer_type() << "\n";
		std::cout << "input:" << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
		std::cout << "output:" << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
		}

	std::cout << "load models..." << std::endl;

	// load dataset
	std::vector<label_t> train_labels;
	std::vector<vec_t> train_images;
	std::vector<label_t> test_labels;
	std::vector<vec_t> test_images;
	parse_kaggle_dogcat_train(data_dir_path, &train_images, &train_labels, -1.0, 1.0, 0, 0);
	parse_kaggle_dogcat_test(data_dir_path, &test_images, &test_labels, -1.0, 1.0, 0, 0);

	//std::cout << "train label size: " << train_labels.size() << std::endl;
	//std::cout << "train image size: " << train_images.size() << std::endl;
	//std::cout << "test label size: " << test_labels.size() << std::endl;
	//std::cout << "test image size: " << test_images.size() << std::endl;

	std::cout << "start learning" << std::endl;

	progress_display disp(train_images.size());
	timer t;
	optimizer.alpha *= static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate);

	int epoch = 1;
	// create callback
	auto on_enumerate_epoch = [&]()
		{
		std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
			  << t.elapsed() << "s elapsed." << std::endl;
		++epoch;
		tiny_dnn::result res = nn.test(test_images, test_labels);
		log << res.num_success << "/" << res.num_total << std::endl;
		disp.restart(train_images.size());
		t.restart();
		};

	auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

	// training
	nn.train<cross_entropy>(optimizer, train_images, train_labels, n_minibatch,
				n_train_epochs, on_enumerate_minibatch, on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	// test and show results
	nn.test(test_images, test_labels).print_detail(std::cout);
	// save networks
	std::ofstream ofs("kaggle-dog-cat-weights");
	ofs << nn;
}

static core::backend_t parse_backend_name
	(
	const std::string &name
	)
{
	const std::array<const std::string, 5> names =
		{
		"internal", "nnpack", "libdnn", "avx", "opencl",
		};
	for (size_t i = 0; i < names.size(); ++i)
		{
		if (name.compare(names[i]) == 0)
			{
			return static_cast<core::backend_t>(i);
			}
		}
	return core::default_engine();
}

static void usage
	(
	const char *argv0
	)
{
	std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
		<< " --learning_rate 0.001"
		<< " --epochs 50"
		<< " --minibatch_size 50"
		<< " --backend_type avx" << std::endl;
}

int main
	(
	int argc,
	char **argv
	)
{
	double learning_rate         = 0.001;
	int epochs                   = 50;
	std::string data_path        = "";
	int minibatch_size           = 50;
	core::backend_t backend_type = core::default_engine();

	if (argc == 2)
		{
		std::string argname(argv[1]);
		if (argname == "--help" || argname == "-h")
			{
			usage(argv[0]);
			return 0;
			}
		}
	for (int count = 1; count + 1 < argc; count += 2)
		{
		std::string argname(argv[count]);
		if (argname == "--learning_rate")
			{
			learning_rate = atof(argv[count + 1]);
			}
		else if (argname == "--epochs")
			{
			epochs = atoi(argv[count + 1]);
			}
		else if (argname == "--minibatch_size")
			{
			minibatch_size = atoi(argv[count + 1]);
			}
		else if (argname == "--backend_type")
			{
			backend_type = parse_backend_name(argv[count + 1]);
			}
		else if (argname == "--data_path")
			{
			data_path = std::string(argv[count + 1]);
			}
		else
			{
			std::cerr << "Invalid parameter specified - \"" << argname << "\"" << std::endl;
			usage(argv[0]);
			return -1;
			}
		}
	if (data_path == "")
		{
		std::cerr << "Data path not specified." << std::endl;
		usage(argv[0]);
		return -1;
		}
	if (learning_rate <= 0)
		{
		std::cerr << "Invalid learning rate. The learning rate must be greater than 0." << std::endl;
		return -1;
		}
	if (epochs <= 0)
		{
		std::cerr << "Invalid number of epochs. The number of epochs must be greater than 0." << std::endl;
		return -1;
		}
	if (minibatch_size <= 0 || minibatch_size > 25000)
		{
		std::cerr << "Invalid minibatch size. The minibatch size must be greater than 0"
				" and less than dataset size (25000)." << std::endl;
		return -1;
		}
	std::cout << "Running with the following parameters:" << std::endl
		      << "Data path: " << data_path << std::endl
              << "Learning rate: " << learning_rate << std::endl
              << "Minibatch size: " << minibatch_size << std::endl
              << "Number of epochs: " << epochs << std::endl
              << "Backend type: " << backend_type << std::endl
              << std::endl;
	try
		{
		train_cifar10_dog_cat(data_path, learning_rate, epochs, minibatch_size, backend_type, std::cout);
		}
	catch (tiny_dnn::nn_error &err)
		{
		std::cerr << "Exception: " << err.what() << std::endl;
		}

	return 0;
}
