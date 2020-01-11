//
// Created by Boyan Atanasov on 11/01/2020.
//

#ifndef SIMPLENEURALNETWORK_MATRIX_H
#define SIMPLENEURALNETWORK_MATRIX_H

#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <random>
#include <iostream>

#include "nn_config.h"

class matrix {
private:
    // outer vector is rows, inner vector is columns
    std::vector<std::vector<floating_type>> values;

public:
    matrix(int row_count, int col_count);

    void set_value_at(int row, int col, floating_type value);

    floating_type get_value_at(int row, int col);

    int get_row_count();

    int get_col_count();

    // relies constructor has been used and valid
    void fill_rand();

    // you must manually dispose of the object created by this function
    matrix *new_from_transpose();

    matrix *new_from_multiply(matrix *m);

    // squash the matrix to a linear vector (1 dimension)
    std::vector<floating_type> *new_vector_from_squash();

    std::string get_str();
};

#endif //SIMPLENEURALNETWORK_MATRIX_H
