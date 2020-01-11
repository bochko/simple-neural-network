//
// Created by Boyan Atanasov on 11/01/2020.
//

#include "matrix.h"

matrix::matrix(int row_count, int col_count) {
    for (int row_idx = 0; row_idx < row_count; row_idx++) {
        std::vector<floating_type> column_matrix_storage; // inner
        for (int col_idx = 0; col_idx < col_count; col_idx++) {
            column_matrix_storage.push_back(0.0f);
        }
        this->values.push_back(column_matrix_storage);
    }
}

void matrix::set_value_at(int row, int col, floating_type value) {
    values.at(row).at(col) = value;
}

floating_type matrix::get_value_at(int row, int col) {
    return values.at(row).at(col);
}

int matrix::get_row_count() {
    // assume first dimension is rows
    return (int) values.size();
}

int matrix::get_col_count() {
    // since vector of vectors, and all interior vectors
    // are identical in size, we can use the first one to get the
    // length of the second dimension
    return (int) values.at(0).size();
}

// relies constructor has been used and valid
void matrix::fill_rand() {
    std::random_device rdev;
    std::mt19937 rng(rdev());
    std::uniform_real_distribution<floating_type> distribution(0.0f, 1.0f);
    int row_size = this->get_row_count();
    int col_size = this->get_col_count();

    for (int row_idx = 0; row_idx < row_size; row_idx++) {
        for (int col_idx = 0; col_idx < col_size; col_idx++) {
            this->values.at(row_idx).at(col_idx) = distribution(rng);
        }
    }
}

// you must manually dispose of the object created by this function
matrix* matrix::new_from_transpose() {
    matrix *transposed = new matrix(this->get_col_count(), this->get_row_count());
    int row_size = transposed->get_row_count();
    int col_size = transposed->get_col_count();
    for (int row_idx = 0; row_idx < row_size; row_idx++) {
        for (int col_idx = 0; col_idx < col_size; col_idx++) {
            transposed->set_value_at(row_idx, col_idx, this->get_value_at(col_idx, row_idx));
        }
    }
    return transposed;
}

matrix* matrix::new_from_multiply(matrix *m) {
    if (this->get_col_count() != m->get_row_count()) {
        std::cerr << "origin col size " << this->get_col_count() << " != m row size " << m->get_row_count()
                  << std::endl;
        std::abort();
    }
    matrix *ret = new matrix(this->get_row_count(), m->get_col_count());
    for (int orow_idx = 0; orow_idx < this->get_row_count(); orow_idx++) {
        for (int mulcol_idx = 0; mulcol_idx < m->get_col_count(); mulcol_idx++) {
            for (int mulrow_idx = 0; mulrow_idx < m->get_row_count(); mulrow_idx++) {
                floating_type product =
                        this->get_value_at(orow_idx, mulrow_idx) * m->get_value_at(mulrow_idx, mulcol_idx);
                // add onto the value in the result matrix
                ret->set_value_at(orow_idx, mulcol_idx, (ret->get_value_at(orow_idx, mulcol_idx) + product));
            }
        }
    }
    return ret;
}

// squash the matrix to a linear vector (1 dimension)
std::vector<floating_type>* matrix::new_vector_from_squash() {
    std::vector<floating_type> *vtor = new std::vector<floating_type>();
    // fill in the newly created vector
    for (int row_idx = 0; row_idx < this->get_row_count(); row_idx++) {
        for (int col_idx = 0; col_idx < this->get_col_count(); col_idx++) {
            vtor->push_back(this->get_value_at(row_idx, col_idx));
        }
    }
    return vtor;
}

std::string matrix::get_str() {
    std::stringstream sstream;
    int row_size = this->get_row_count();
    int col_size = this->get_col_count();

    for (int row_idx = 0; row_idx < row_size; row_idx++) {
        for (int col_idx = 0; col_idx < col_size; col_idx++) {
            sstream << this->values.at(row_idx).at(col_idx)
                    << " \t";
        }
        sstream << "\r\n";
    }
    return sstream.str();
}
