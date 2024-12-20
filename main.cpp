#include<iostream>
#include<conio.h>
#include<unordered_map>
#include<random>
#include<cmath>
#include<omp.h>
#include<windows.h>
#include<string>
#include<fstream>
#include<filesystem>
#include<thread>
using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> ins(0, 50);
uniform_int_distribution<> spec(1, 20);

#define hhh1 1536
#define hhh2 768
#define hhh3 384
filesystem::path executable = filesystem::absolute(".");
filesystem::path current_file;

double grad_hidden1[hhh1][4] = { 0 };
double grad_hidden2[hhh2][hhh1] = { 0 };
double grad_hidden3[hhh3][hhh2] = { 0 };
double grad_output_layer[hhh3] = { 0 };
double h1_out[hhh1] = { 0.0 };
double h2_out[hhh2] = { 0.0 };
double h3_out[hhh3] = { 0.0 };
double delta_hidden1[hhh1] = {};
double delta_hidden2[hhh2] = {};
double delta_hidden3[hhh3] = {};

double relu(double x) {
	if (x > 0) return x;
	return 0;
}
double relu_d(double x) {
	if (x > 0) return 1;
	return 0;
}
void data_norm(double** data) {
	for (int i = 0; i < 20000; i++) {
		for (int j = 0; j < 5; j++) {
			data[i][j] = log10(data[i][j]);
		}
	}
}

double** data_gen(double** data) {
#pragma omp parallel for
	for (int i = 0; i < 20000; i++) {
		int x = spec(gen);
		switch (x) {
		case 1: {
			data[i][0] = ins(gen);
			for (int j = 1; j < 5; j++) {
				data[i][j] = data[i][0];
			}
			break;
		}
		case 2: {
			data[i][0] = ins(gen);
			for (int j = 1; j < 5; j++) {
				data[i][j] = data[i][j - 1] + 1;
			}
			break;
		}
		case 3: {
			data[i][0] = ins(gen);
			for (int j = 1; j < 5; j++) {
				data[i][j] = sqrt(data[i][j - 1]);
			}
			break;
		}
		case 4: {
			data[i][0] = ins(gen);
			double fd = ins(gen);
			for (int j = 1; j < 5; j++) {
				data[i][j] = data[i][j - 1] + fd;
			}
			break;
		}
		case 5: {
			data[i][0] = ins(gen);
			double fd = ins(gen);
			for (int j = 1; j < 5; j++) {
				data[i][j] = data[i][j - 1] * fd;
			}
			break;
		}
		case 6: {
			data[i][0] = ins(gen);
			double fd = ins(gen);
			for (int j = 1; j < 5; j++) {
				data[i][j] = data[i][j - 1] / (fd + 0.01);
			}
			break;
		}
		case 7: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			for (int j = 2; j < 5; j++) {
				data[i][j] = data[i][j - 1] * data[i][j - 2];
			}
			break;
		}
		case 8: {
			data[i][0] = ins(gen);
			for (int j = 1; j < 5; j++) {
				data[i][j] = pow(data[i][j - 1], j + 1);
			}
			break;
		}
		case 9: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			for (int j = 2; j < 5; j++) {
				data[i][j] = data[i][j - 1] + data[i][j - 2];
			}
			break;
		}
		case 10: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			for (int j = 2; j < 5; j++) {
				data[i][j] = data[i][j - 2];
			}
			break;
		}
		case 11: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			for (int j = 2; j < 5; j++) {
				data[i][j] = data[i][j - 2] + 1;
			}
			break;
		}
		case 12: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			double fd = ins(gen);
			for (int j = 2; j < 5; j++) {
				data[i][j] = data[i][j - 2] + fd;
			}
			break;
		}
		case 13: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			for (int j = 2; j < 5; j++) {
				data[i][j] = sqrt(data[i][j - 2]);
			}
			break;
		}
		case 14: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			for (int j = 2; j < 5; j++) {
				data[i][j] = data[i][j - 2] * 2;
			}
			break;
		}
		case 15: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			double fd = ins(gen);
			for (int j = 2; j < 5; j++) {
				data[i][j] = data[i][j - 2] / (fd + 0.01);
			}
			break;
		}
		case 16: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			double fd = ins(gen);
			data[i][2] = data[i][0] * fd;
			data[i][3] = data[i][1] * fd;
			data[i][4] = data[i][2] * fd;
			break;
		}
		case 17: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			double fd = ins(gen);
			data[i][2] = data[i][0] / fd;
			data[i][3] = data[i][1] / fd;
			data[i][4] = data[i][2] / fd;
			break;
		}
		case 18: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			for (int j = 2; j < 5; j++) {
				data[i][j] = data[i][j - 1] * data[i][j - 2];
			}
			break;
		}
		case 19: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			for (int j = 2; j < 5; j++) {
				data[i][j] = data[i][j - 1] * data[i][0];
			}
			break;
		}
		case 20: {
			data[i][0] = ins(gen);
			data[i][1] = ins(gen);
			for (int j = 2; j < 5; j++) {
				data[i][j] = data[i][j - 1] + data[i][0];
			}
			break;
		}
		}
	}
	data_norm(data);
	return data;
}

class NeuralMLP {
private:
	double** hidden1;
	double** hidden2;
	double** hidden3;
	double* output_weights;
	int input_size;
	int h1_size;
	int h2_size;
	int h3_size;
	int output_size;
	double minloss;
public:
	NeuralMLP(bool mode, int in_size, int h1, int h2, int h3, int out_size) : input_size(in_size), h1_size(h1), h2_size(h2), h3_size(h3), output_size(out_size) {

		hidden1 = new double* [h1_size];
		for (int i = 0; i < h1_size; ++i) hidden1[i] = new double[input_size];
		hidden2 = new double* [h2_size];
		for (int i = 0; i < h2_size; ++i) hidden2[i] = new double[h1_size];
		hidden3 = new double* [h3_size];
		for (int i = 0; i < h3_size; ++i) hidden3[i] = new double[h2_size];
		output_weights = new double[h3_size];
		minloss = 999999;

		if (mode) {
			auto init_weights = [](double** weights, int rows, int cols, int fan_in) {
				double stddev = sqrt(2.0 / fan_in);
				normal_distribution<double> dist(0.0, stddev);
				for (int i = 0; i < rows; ++i) {
					for (int j = 0; j < cols; ++j) {
						weights[i][j] = dist(gen);
					}
				}
				};
			init_weights(hidden1, h1_size, input_size, input_size);
			init_weights(hidden2, h2_size, h1_size, h1_size);
			init_weights(hidden3, h3_size, h2_size, h2_size);
			double stddev = sqrt(2.0 / h3_size);
			normal_distribution<double> dist(0.0, stddev);
			for (int j = 0; j < h3_size; ++j) {
				output_weights[j] = dist(gen);
			}
		}
		else {
			load_model();
			clog << "\033[31M MODEL LOADED.\n";
		}
	}

	~NeuralMLP() {
		for (int i = 0; i < h1_size; ++i) delete[] hidden1[i];
		delete[] hidden1;
		for (int i = 0; i < h2_size; ++i) delete[] hidden2[i];
		delete[] hidden2;
		for (int i = 0; i < h3_size; ++i) delete[] hidden3[i];
		delete[] hidden3;
		delete[] output_weights;
	}

	void save_model() {
		std::ofstream ofile("weights.bin", ios::out | ios::binary);
		if (!ofile.is_open()) {
			std::cerr << "Can't open file " << std::endl;
			return;
		}
		for (int i = 0; i < 1536; ++i) {
			ofile.write(reinterpret_cast<char*>(hidden1[i]), 4 * sizeof(double));
		}
		for (int i = 0; i < 768; ++i) {
			ofile.write(reinterpret_cast<char*>(hidden2[i]), h1_size * sizeof(double));
		}
		for (int i = 0; i < 384; ++i) {
			ofile.write(reinterpret_cast<char*>(hidden3[i]), h2_size * sizeof(double));
		}
		ofile.write(reinterpret_cast<char*>(output_weights), h3_size * sizeof(double));
		ofile.close();
		std::ofstream onfile("minloss.bin", ios::out | ios::binary);
		onfile.write(reinterpret_cast<char*>(&minloss), sizeof(double));
		onfile.close();
	}

	void load_model() {
		std::ifstream ifile("weights.bin", ios::in | ios::binary);
		if (!ifile.is_open()) {
			throw::out_of_range("No save found.");
			return;
		}
		for (int i = 0; i < 1536; ++i) {
			ifile.read(reinterpret_cast<char*>(hidden1[i]), 4 * sizeof(double));
		}
		for (int i = 0; i < 768; ++i) {
			ifile.read(reinterpret_cast<char*>(hidden2[i]), h1_size * sizeof(double));
		}
		for (int i = 0; i < 384; ++i) {
			ifile.read(reinterpret_cast<char*>(hidden3[i]), h2_size * sizeof(double));
		}
		ifile.read(reinterpret_cast<char*>(output_weights), h3_size * sizeof(double));
		ifile.close();
		ifstream infile("minloss.bin", ios::in | ios::binary);
		if (!ifile.is_open()) {
			cout << "No minimal error file found. Creating new...";
			infile.close();
			ofstream nfile("minloss.bin", ios::out | ios::binary);
			double x = 999999;
			nfile.write(reinterpret_cast<char*>(&x), sizeof(double));
			minloss = x;
			nfile.close();
		}
		else {
			infile.read(reinterpret_cast<char*>(&minloss), sizeof(double));
			infile.close();
		}
	}

	double forward(double* input) {
#pragma omp parallel for
		for (int i = 0; i < h1_size; i++) {
			h1_out[i] = 0;
			if (i < h2_size) {
				h2_out[i] = 0;
				if (i < h3_size);
				h3_out[i] = 0;
			}
		}
#pragma omp parallel for
		for (int i = 0; i < h1_size; ++i) {
			for (int j = 0; j < input_size; ++j) {
				h1_out[i] += hidden1[i][j] * input[j];
			}
			h1_out[i] = relu(h1_out[i]);
		}
#pragma omp parallel for
		for (int i = 0; i < h2_size; ++i) {
			for (int j = 0; j < h1_size; ++j) {
				h2_out[i] += hidden2[i][j] * h1_out[j];
			}
			h2_out[i] = relu(h2_out[i]);
		}
		for (int i = 0; i < h3_size; ++i) {
			for (int j = 0; j < h2_size; ++j) {
				h3_out[i] += hidden3[i][j] * h2_out[j];
			}
			h3_out[i] = relu(h3_out[i]);
		}
		double output = 0;
		for (int j = 0; j < h3_size; ++j) {
			output += output_weights[j] * h3_out[j];
		}
		return output;
	}
	void train(double** data, double lr, int epochs, int batch_size) {
		int wait = 0;
		int data_size = batch_size;
		epochs += 1;
		double curtotalerror = 999999;
		for (int epoch = 1; epoch < epochs; ++epoch) {
			clog << "\033[2m Epoch " << epoch << " started.\n";
			double total_error = 0;

			for (int batch_start = 0; batch_start < data_size; batch_start += batch_size) {
				clog << "\033[2m Batch " << batch_start << " started.\n";
				int batch_end = min(batch_start + batch_size, data_size);
				int current_batch_size = batch_end - batch_start;
#pragma omp parallel for
				for (int i = 0; i < h1_size; i++) {
					for (int j = 0; j < h1_size; j++) {
						if (j < 4) grad_hidden1[i][j] = 0;
						if (i < h2_size) {
							grad_hidden2[i][j] = 0;
							if (i < h3_size && j < h2_size) grad_hidden3[i][j] = 0;
						}
					}
				}

				for (int i = 0; i < hhh3; i++) {
					grad_output_layer[i] = 0;
				}
				clog << "\033[2m Grads zeroed\n";
				for (int i = batch_start; i < batch_end; ++i) {
					char chose = 'g';
					if (!(i % 150)) cout << "\033[1m\n" << (float)i / batch_end * 100 << "%\n";
					while (_kbhit()) chose = _getch();;
					if (chose == 'p' || chose == 'P') {
						cout << "\033[0m\nChose operation: ";
						string operation;
						cin >> operation;
						if (operation == "lr") {
							cin >> lr;
						}
						else if (operation == "stop") {
							epoch = epochs;
							break;
						}
						else if (operation == "skip") {
							break;
						}
						else if (operation == "epoch_set") {
							cin >> epoch;
						}
						else if (operation == "epochs_set") {
							cin >> epochs;
						}
						else if (operation == "forward") {
							cout << "\nEnter nums: \n";
							double validation[4];
							for (int i = 0; i < 4; i++) {
								cin >> validation[i];
								validation[i] = log(validation[i]);
							}
							cout << "\nResult: " << pow(10, forward(validation)) << "\n";
						}

					}
					else if (chose == 's' || chose == 's') {
						save_model();
					}

					if (data[i][4] > 700) continue;
					double input[4] = {};
					for (int k = 0; k < 4; k++) {
						input[k] = data[i][k];
					}
#pragma omp parallel for
					for (int k = 0; k < h1_size; k++)
					{
						h1_out[k] = 0;
						if (k < h2_size) {
							h2_out[k] = 0;
							if (k < h3_size);
							h3_out[k] = 0;
						}
					}
#pragma omp parallel for
					for (int i = 0; i < h1_size; ++i) {
						for (int j = 0; j < input_size; ++j) {
							h1_out[i] += hidden1[i][j] * input[j];
						}
						h1_out[i] = relu(h1_out[i]);
					}
#pragma omp parallel for
					for (int i = 0; i < h2_size; ++i) {
						for (int j = 0; j < h1_size; ++j) {
							h2_out[i] += hidden2[i][j] * h1_out[j];
						}
						h2_out[i] = relu(h2_out[i]);
					}
#pragma omp parallel for
					for (int i = 0; i < h3_size; ++i) {
						for (int j = 0; j < h2_size; ++j) {
							h3_out[i] += hidden3[i][j] * h2_out[j];
						}
						h3_out[i] = relu(h3_out[i]);
					}
					double output = 0;
					for (int j = 0; j < h3_size; ++j) {
						output += output_weights[j] * h3_out[j];
					}

					if (i == batch_start) clog << "\033[92m Forward success\n";

					double error = data[i][4] - output;
					error /= abs(data[i][4]);
					total_error += abs(error);
					double delta_output = 2 * error;
					if (!(i % 150)) {
						clog << "\n\n\033[38;5;226m Error " << error << " counted\n";
						clog << "Res: " << data[i][0] << " " << data[i][1] << " " << data[i][2] << " " << data[i][3];
						clog << ". Ans " << output << '\n';
					}
#pragma omp parallel for
					for (int i = 0; i < h1_size; i++) {
						delta_hidden1[i] = 0;
						if (i < h2_size) {
							delta_hidden2[i] = 0;
							if (i < h3_size);
							delta_hidden3[i] = 0;
						}
					}

					for (int j = 0; j < h3_size; ++j) {
						delta_hidden3[j] = delta_output * output_weights[j] * relu_d(h3_out[j]);
					}
#pragma omp parallel for
					for (int j = 0; j < h2_size; j++) {
						delta_hidden2[j] = 0;
						for (int k = 0; k < h3_size; k++) {
							delta_hidden2[j] += delta_hidden3[k] * hidden3[k][j];
						}
						delta_hidden2[j] *= relu_d(h2_out[j]);
					}
#pragma omp parallel for
					for (int j = 0; j < h1_size; j++) {
						delta_hidden1[j] = 0;
						for (int k = 0; k < h2_size; k++) {
							delta_hidden1[j] += delta_hidden2[k] * hidden2[k][j];
						}
						delta_hidden1[j] *= relu_d(h1_out[j]);
					}
					for (int k = 0; k < h3_size; ++k) {
						grad_output_layer[k] += delta_output * h3_out[k];
					}
#pragma omp parallel for
					for (int j = 0; j < h3_size; ++j) {
						for (int k = 0; k < h2_size; ++k) {
							grad_hidden3[j][k] += delta_hidden3[j] * h2_out[k];
						}
					}
#pragma omp parallel for
					for (int j = 0; j < h2_size; ++j) {
						for (int k = 0; k < h1_size; ++k) {
							grad_hidden2[j][k] += delta_hidden2[j] * h1_out[k];
						}
					}
#pragma omp parallel for
					for (int j = 0; j < h1_size; ++j) {
						for (int k = 0; k < input_size; ++k) {
							grad_hidden1[j][k] += delta_hidden1[j] * input[k];
						}
					}
					if (i == batch_start) clog << "\033[92m Grads count success\n";
				}
				clog << "\033[92m Cycle i finished successfully\n";
				for (int k = 0; k < h3_size; ++k) {
					grad_output_layer[k] /= current_batch_size;;
				}
#pragma omp parallel for
				for (int j = 0; j < h3_size; ++j) {
					for (int k = 0; k < h2_size; ++k) {
						grad_hidden3[j][k] /= current_batch_size;;
					}
				}
#pragma omp parallel for
				for (int j = 0; j < h2_size; ++j) {
					for (int k = 0; k < h1_size; ++k) {
						grad_hidden2[j][k] /= current_batch_size;;
					}
				}
#pragma omp parallel for
				for (int j = 0; j < h1_size; ++j) {
					for (int k = 0; k < input_size; ++k) {
						grad_hidden1[j][k] /= current_batch_size;
					}
				}
#pragma omp parallel for
				for (int j = 0; j < h1_size; ++j) {
					for (int k = 0; k < input_size; ++k) {
						hidden1[j][k] += lr * grad_hidden1[j][k];
					}
				}
#pragma omp parallel for
				for (int j = 0; j < h2_size; ++j) {
					for (int k = 0; k < h1_size; ++k) {
						hidden2[j][k] += lr * grad_hidden2[j][k];
					}
				}
				for (int j = 0; j < h3_size; ++j) {
					for (int k = 0; k < h2_size; ++k) {
						hidden3[j][k] += lr * grad_hidden3[j][k];
					}
				}
				for (int k = 0; k < h3_size; ++k) {
					output_weights[k] += lr * grad_output_layer[k];
				}
				clog << "\033[92m Weights correction success\n";
			}
			cout << "Epoch " << epoch << ". Loss: \033[0m" << total_error << endl;
			if (total_error < minloss && wait <= 0) {
				minloss = total_error;
				save_model();
				wait = 1 + epoch / 25;
			}
			wait--;
			if (!(epoch % 250)) {
				data_gen(data);
				clog << "\033[31m New dataset generated\n";
			}
			if (lr > 1.2) lr *= 0.999;
			else lr *= 0.9999;
		}
	}
};


int main(int argc, char* argv[]) {
	current_file = executable / argv[0];
	double** data = new double* [20000];
	for (int i = 0; i < 20000; i++) {
		data[i] = new double[5];
	}
	cout << "Load? Y/N\n";
	char ch;
	int m = 1;
	cin >> ch;
	if (ch == 'y' || ch == 'Y') {
		int m = 0;
	}
	else {
		m = 1;
	}
	data_gen(data);
	cout << "How much threads to use? (0 - auto)\n";
	int th;
	cin >> th;
	if(th) omp_set_num_threads(th);
	else omp_set_num_threads(thread::hardware_concurrency() / 1.8);
	NeuralMLP ps(m, 4, hhh1, hhh2, hhh3, 1);
	cout << "Train model before usage? \033[31m\nWrite a number of epochs. 0 if no training.\033[0m\n";
	cin >> th;
	ps.train(data, 0.000155, th, 20000);
	double x[4] = { 1, 2, 3, 4 };
	cout << ps.forward(x);
	double ndata[4] = {};
	while (1) {
		cout << "Enter nums\n";
		for (int i = 0; i < 4; i++) {
			cin >> ndata[i];
			ndata[i] = log10(ndata[i]);
		}
		if (ndata[0] == -2.65536) break;
		cout << "\nResult: " << pow(10, ps.forward(ndata)) << "\n";
	}
	delete[] data;
	delete[] ndata;
}
