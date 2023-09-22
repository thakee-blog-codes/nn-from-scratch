#pragma once

#include <raylib.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

#include "matrix.hpp"
#include "nn.hpp"

typedef Image GrayImage;


class DsMinist : public Dataset {
public:
  DsMinist(const char* path_labels, const char* path_images);
  ~DsMinist();

  std::vector<uint8_t> labels;
  std::vector<GrayImage> images;

  int count() const override;
  Matrix get_input(int index) const override;
  Matrix get_output(int index) const override;

  static Matrix image_to_input(GrayImage* image);

private:
  static Matrix _image_to_input(const GrayImage* image);
  void _reserve(int size);
};


#ifdef SINGLE_SOURCE_IMPL
typedef unsigned char data_t;


DsMinist::~DsMinist() {
  for (size_t i = 0; i < images.size(); i++) {
    UnloadImage(images[i]);
  }
}


void DsMinist::_reserve(int size) {
  labels.reserve(size);
  images.reserve(size);
}


int DsMinist::count() const {
  return (int) labels.size();
}


GrayImage gen_image_gray(int width, int height, data_t* data) {
  data_t* pixels = (data_t*)RL_CALLOC(width * height, sizeof(data_t));
  for (int i = 0; i < width * height; i++) pixels[i] = data[i];
  Image image;
  image.data = pixels;
  image.width = width;
  image.height = height;
  image.format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE;
  image.mipmaps = 1;
  return image;
}


// https://www.kaggle.com/code/hojjatk/read-mnist-dataset
DsMinist::DsMinist(const char* path_labels, const char* path_images) {

#define READ_INT(dst, ptr)        \
  do {                            \
    (dst) = 0;                    \
    (dst) |= (*(ptr)++) << 8 * 3; \
    (dst) |= (*(ptr)++) << 8 * 2; \
    (dst) |= (*(ptr)++) << 8 * 1; \
    (dst) |= (*(ptr)++) << 8 * 0; \
  } while (0)

  // Load the labels.
  {
    unsigned int bytes_read;
    data_t* data = LoadFileData(path_labels, &bytes_read);
    data_t* ptr = data;

    uint32_t magic = 0;
    READ_INT(magic, ptr);
    assert(magic == 2049);

    uint32_t size = 0;
    READ_INT(size, ptr);
    _reserve(size);

    for (uint32_t i = 0; i < size; i++) {
      labels.push_back(ptr[i]);
    }

    UnloadFileData(data);
  }

  // Load the imges.
  {
    unsigned int bytes_read;
    data_t* data = LoadFileData(path_images, &bytes_read);
    data_t* ptr = data;

    uint32_t magic = 0;
    READ_INT(magic, ptr);
    assert(magic == 2051);

    uint32_t size, rows, cols;
    READ_INT(size, ptr);
    READ_INT(rows, ptr);
    READ_INT(cols, ptr);

    for (uint32_t i = 0; i < size; i++) {
      Image img = gen_image_gray(cols, rows, ptr);
      images.push_back(img);
      ptr += (cols * rows);
    }

    UnloadFileData(data);
  }

#undef READ_INT
}

Matrix DsMinist::get_input(int index) const {
  const GrayImage& image = images[index];
  return _image_to_input(&image);
}

Matrix DsMinist::get_output(int index) const {
  Matrix output(1, 10);
  output.set(0, labels[index], 1.f);
  return output;
}

Matrix DsMinist::image_to_input(GrayImage* image) {
  if (image->width != 28 || image->height != 28) {
    ImageResize(image, 28, 28);
  }
  return _image_to_input(image);
}


Matrix DsMinist::_image_to_input(const GrayImage* image) {
  assert(image != nullptr);
  assert(image->width == 28 && image->height == 28, "Resize image to 28 * 28 before calling this function.");

  Matrix m(1, image->height * image->width);
  std::vector<matrix_t>& data = m.data();
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = (matrix_t)(*((data_t*)(image->data) + i)) / 255.f;
  }
  return m;
}

#endif // SINGLE_SOURCE_IMPL
