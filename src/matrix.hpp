#pragma once

#include <vector>
typedef float matrix_t;


class Matrix {
public:
  Matrix(int rows=0, int cols=0, matrix_t val=0);

  Matrix& init(int rows, int cols, matrix_t val = 0);
  Matrix& fill(matrix_t val);

  void print() const;

  // Inplace operations.
  Matrix& operator+=(const Matrix& other);
  Matrix& operator*=(matrix_t value); // Dot product.
  Matrix& multiply_inplace(const Matrix& other); // Element by element.

  // Operators that'll return new matrix.
  Matrix operator-(const Matrix& other) const;
  Matrix operator*(const Matrix& other) const;
  Matrix operator*(matrix_t value) const;
  Matrix transpose() const;
  Matrix multiply(const Matrix& other) const;

  matrix_t at(int row, int col) const;
  void set(int row, int col, matrix_t value);

  matrix_t sum() const;
  Matrix& randomize(matrix_t min = 0, matrix_t max = 1);
  Matrix& sigmoid();
  Matrix& square();

  std::vector<matrix_t>& data();
  const std::vector<matrix_t>& data() const;

  int rows() const;
  int cols() const;

private:
  int _rows, _cols;
  std::vector<matrix_t> _data;
};


#ifdef SINGLE_SOURCE_IMPL

#include <stdlib.h>
#include <math.h>


Matrix::Matrix(int rows, int cols, matrix_t val)
  : _rows(rows), _cols(cols), _data(rows* cols, val)
{}


Matrix& Matrix::init(int rows, int cols, matrix_t val) {
  this->_rows = rows;
  this->_cols = cols;
  this->_data = std::vector<matrix_t>(rows * cols, val);
  return *this;
}


matrix_t Matrix::at(int row, int col) const {
  return _data[row * _cols + col];
}


void Matrix::set(int row, int col, matrix_t value) {
  _data[row * _cols + col] = value;
}


Matrix& Matrix::fill(matrix_t val) {
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] = val;
  }
  return *this;
}


Matrix& Matrix::randomize(matrix_t min, matrix_t max) {

  assert(max > min);
  for (size_t i = 0; i < _data.size(); i++) {
    matrix_t val = (matrix_t)((float)rand() / (float)RAND_MAX) * (max - min) + min;
    _data[i] = val;
  }
  return *this;
}


matrix_t Matrix::sum() const {
  matrix_t total = 0;
  for (size_t i = 0; i < _data.size(); i++) {
    total += _data[i];
  }
  return total;
}


static inline matrix_t sigmoid(matrix_t x) {
  return 1.f / (1.f + expf(-x));
}


Matrix& Matrix::sigmoid() {
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] = ::sigmoid(_data[i]);
  }
  return *this;
}


Matrix& Matrix::square() {
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] = _data[i] * _data[i];
  }
  return *this;
}


std::vector<matrix_t>& Matrix::data() {
  return _data;
}


const std::vector<matrix_t>& Matrix::data() const {
  return _data;
}


int Matrix::rows() const {
  return _rows;
}


int Matrix::cols() const {
  return _cols;
}


void Matrix::print() const {
  printf("[\n");
  for (int r = 0; r < _rows; r++) {
    printf("  ");
    for (int c = 0; c < _cols; c++) {
      if (c != 0) printf(", ");
      // negative number has extra '-' character at the start.
      matrix_t val = at(r, c);
      if (val >= 0) printf(" %.6f", val);
      else printf("%.6f", val);
    }
    printf("\n");
  }
  printf("]\n");
}


Matrix& Matrix::operator+=(const Matrix& other) {
  bool cond = (_rows == other._rows && _cols == other._cols);
  assert(_rows == other._rows && _cols == other._cols);
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] += other._data[i];
  }
  return *this;
}


Matrix& Matrix::operator*=(matrix_t value) {
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] *= value;
  }
  return *this;
}


Matrix& Matrix::multiply_inplace(const Matrix& other) {
  assert(_rows == other._rows && _cols == other._cols);
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] *= other._data[i];
  }
  return *this;
}


Matrix Matrix::operator-(const Matrix& other) const {
  assert(_rows == other._rows && _cols == other._cols);
  Matrix m(this->_rows, this->_cols);
  for (size_t i = 0; i < _data.size(); i++) {
    m._data[i] = this->_data[i] - other._data[i];
  }
  return m;
}


Matrix Matrix::operator*(const Matrix& other) const {

  // (r1 x c1) * (r2 x c2) =>
  //   assert(c1 == r2), result = (r1 x c2)
  assert(this->_cols == other._rows);

  Matrix m(this->_rows, other._cols);

  int n = _cols; // Width or a row.
  for (int r = 0; r < m._rows; r++) {
    for (int c = 0; c < m._cols; c++) {

      matrix_t val = 0;
      for (int i = 0; i < n; i++) {
        val += this->at(r, i) * other.at(i, c);
      }
      m.set(r, c, val);
    }
  }

  return m;
}


Matrix Matrix::operator*(matrix_t value) const {
  Matrix m(_rows, _cols);
  std::vector<matrix_t>& m_data = m.data();
  for (size_t i = 0; i < _data.size(); i++) {
    m_data[i] = _data[i] * value;
  }
  return m;
}


Matrix Matrix::transpose() const {
  Matrix m(_cols, _rows);
  for (int r = 0; r < _rows; r++) {
    for (int c = 0; c < _cols; c++) {
      m.set(c, r, at(r, c));
    }
  }
  return m;
}


Matrix Matrix::multiply(const Matrix& other) const {
  assert(_rows == other._rows && _cols == other._cols);
  Matrix m(_rows, _cols);
  for (size_t i = 0; i < _data.size(); i++) {
    m._data[i] = _data[i] * other._data[i];
  }
  return m;
}


#endif // SINGLE_SOURCE_IMPL
