#pragma once

#include "matrix.hpp"

#include <vector>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

class Dataset {
public:
  virtual int count() const = 0;
  virtual Matrix get_input(int index) const = 0;
  virtual Matrix get_output(int index) const = 0;
};


struct Layer {
  Matrix outputs;
  Matrix biased;
  Matrix weights;

  Layer(int neuron_count = 0);
  Layer(Layer&& other) noexcept;

  // Create the next layer from current updating the weights.
  Layer next_layer(int neuron_count);

  static void forward(Layer& curr, Layer& prev);
};


struct NN {

  matrix_t learn_rate = 0.01;
  std::vector<Layer> layers;
  std::vector<std::string> output_labels;

  int trained = 0;    // Number of times the model trained on the dataset.
  int data_index = 0; // Index in the dataset to the next training data.

  NN();
  NN(const std::vector<int>& config, const std::vector<std::string>& output_labels);

  Matrix& get_outputs();

  void forward(const Matrix& input);
  void backprop(const Matrix& expected);

  void save(const char* path) const;
  void load(const char* path);
};


float error(Matrix& out, Matrix& exp);


#ifdef SINGLE_SOURCE_IMPL


float error(Matrix& out, Matrix& exp) {
  return (out - exp).square().sum() / out.cols();
}


Layer::Layer(int neuron_count) {
  outputs.init(1, neuron_count);
  biased.init(1, neuron_count);
}


Layer Layer::next_layer(int neuron_count) {
  Layer next;
  next.outputs.init(1, neuron_count);
  next.biased.init(1, neuron_count);
  this->weights.init(this->outputs.cols(), next.outputs.cols());
  return next;
}


Layer::Layer(Layer&& other) noexcept :
  outputs(std::move(other.outputs)),
  biased(std::move(other.biased)),
  weights(std::move(other.weights))
{}


void Layer::forward(Layer& curr, Layer& prev) {
  curr.outputs = (
    (prev.outputs * prev.weights) += curr.biased
  ).sigmoid();
}


NN::NN() {}


NN::NN(const std::vector<int>& config, const std::vector<std::string>& output_labels)
  : output_labels(output_labels) {

  assert(config.size() >= 1);
  assert(output_labels.size() == config.at(config.size() - 1));

  for (size_t i = 0; i < config.size(); i++) {
    int neurons_count = config[i];
    if (i == 0) {
      layers.push_back(Layer(neurons_count));
    } else {
      Layer& prev = layers.at(layers.size() - 1);
      layers.push_back(prev.next_layer(neurons_count));
    }
  }

  for (Layer& layer : layers) {
    layer.weights.randomize(-.5, .5);
  }
}


Matrix& NN::get_outputs() {
  return layers[layers.size() - 1].outputs;
}


void NN::forward(const Matrix& input) {
  layers[0].outputs = input;
  for (size_t i = 1; i < layers.size(); i++) {
    Layer& curr = layers[i];
    Layer& prev = layers[i - 1];
    Layer::forward(curr, prev);
  }
}


void NN::backprop(const Matrix& expected) {
  Matrix& output = layers[layers.size() - 1].outputs;
  assert(expected.rows() == output.rows() &&
         expected.cols() == output.cols());

  // delta_out = out - exp
  // delta_hidden = w.trans() * next_delta x (a * (1-a))
  //
  // curr_b += -learn_rate * curr_delta
  // prev_w += -learn_rate * (curr_delta.trans() * prev_active)

  Matrix delta = output - expected;
  for (size_t i = layers.size() - 1; i > 0; i--) {
    Layer& curr = layers[i];
    Layer& prev = layers[i - 1];

    curr.biased += (delta * (-learn_rate));
    prev.weights += (prev.outputs.transpose() * delta) * (-learn_rate);

    // sigmoid_derivative = (a * (1 - a));
    Matrix one = Matrix(prev.outputs.rows(), prev.outputs.cols(), 1);
    Matrix sigmoid_derivative = prev.outputs.multiply(one - prev.outputs);

    // delta_next = (delta * prev.w.trans()) x (a * (1-a));
    delta = (delta * prev.weights.transpose()).multiply_inplace(sigmoid_derivative);
  }
}


static void write_matrix(std::ofstream& file, const Matrix& m) {
  int rows = m.rows(), cols = m.cols();
  assert(m.data().size() == rows * cols);

  file.write((const char*) &rows, sizeof rows);
  file.write((const char*) &cols, sizeof cols);

  for (matrix_t val : m.data()) {
    file.write((const char*)&val, sizeof val);
  }
}


static Matrix read_matrix(std::ifstream& file) {
  int rows, cols;
  file.read((char*)&rows, sizeof rows);
  file.read((char*)&cols, sizeof cols);
  assert(rows >= 0 && cols >= 0);

  Matrix m(rows, cols);
  std::vector<matrix_t>& data = m.data();
  for (size_t i = 0; i < rows * cols; i++) {
    matrix_t val;
    file.read((char*)&val, sizeof val);
    data[i] = val;
  }

  return m;
}


void NN::save(const char* path) const {

  std::ofstream file(path, std::ios::binary);
  assert(!!file);

  file.write((const char*)(&trained), sizeof trained);
  file.write((const char*)(&data_index), sizeof data_index);

  int layer_count = (int) layers.size();
  file.write((const char*)(&layer_count), sizeof layer_count);

  for (const Layer& layer : layers) {
    int activation_count = (int) layer.outputs.cols();

    assert(
      layer.outputs.rows() == 1 &&
      layer.biased.rows() == 1 &&
      activation_count == layer.biased.cols()
    );

    write_matrix(file, layer.biased);
    write_matrix(file, layer.weights);
  }

  file.close();
}


void NN::load(const char* path) {

  // TODO: What if it has layers already.
  //assert(layers.size() == 0 && "Cannot load to an already built nn.");
  layers.clear();

  std::ifstream file(path, std::ios::binary);
  assert(!!file && "Cannot open the nn file.");

  file.read((char*)(&trained), sizeof trained);
  file.read((char*)(&data_index), sizeof data_index);

  int layer_count;
  file.read((char*)&layer_count, sizeof layer_count);
  assert(layer_count >= 0);

  for (int i = 0; i < layer_count; i++) {

    Layer l;

    l.biased = read_matrix(file);
    assert(l.biased.rows() == 1);

    l.outputs.init(1, l.biased.cols());
    l.weights = read_matrix(file);

    layers.push_back(std::move(l));
  }

  // Assert the dimentions are valid.
  for (size_t i = 0; i < layers.size() - 1; i++) {
    const Layer& curr = layers[i];
    const Layer& next = layers[i + 1];
    assert(curr.weights.rows() == curr.outputs.cols());
    assert(curr.weights.cols() == next.outputs.cols());
  }

}

#endif // SINGLE_SOURCE_IMPL
