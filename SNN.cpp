#include <iostream>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <algorithm>

#include <vector>

#include "snn_config.h"

class matrix {
public:
    matrix(dimension_size nrow, dimension_size ncol) {
        for (dimension_size row_idx = 0; row_idx < nrow; row_idx++) {
            std::vector<floating_type> column_matrix_storage; // inner
            for (dimension_size col_idx = 0; col_idx < ncol; col_idx++) {
                column_matrix_storage.push_back(0.0f);
            }
            this->values.push_back(column_matrix_storage);
        }
    }

    ~matrix() {
        // no dynamic allocations
    }

    void set_val(dimension_size row, dimension_size col, floating_type value) {
        values.at(row).at(col) = value;
    }

    floating_type get_val(dimension_size row, dimension_size col) {
        return values.at(row).at(col);
    }

    dimension_size row_count() {
        // assume first dimension is rows
        return (dimension_size) values.size();
    }

    dimension_size col_count() {
        // since vector of vectors, and all interior vectors
        // are identical in size, we can use the first one to get the
        // length of the second dimension
        return (dimension_size) values.at(0).size();
    }

    // relies constructor has been used and valid
    void fill_rand() {
        std::random_device rdev;
        std::mt19937 rng(rdev());
        std::uniform_real_distribution<floating_type> distribution(0.0f, 1.0f);
        dimension_size row_size = this->row_count();
        dimension_size col_size = this->col_count();

        for (dimension_size row_idx = 0; row_idx < row_size; row_idx++) {
            for (dimension_size col_idx = 0; col_idx < col_size; col_idx++) {
                this->values.at(row_idx).at(col_idx) = distribution(rng);
            }
        }
    }

    // you must manually dispose of the object created by this function
    matrix *transpose(void) {
        matrix *transposed = new matrix(this->col_count(), this->row_count());
        dimension_size row_size = transposed->row_count();
        dimension_size col_size = transposed->col_count();
        for (dimension_size row_idx = 0; row_idx < row_size; row_idx++) {
            for (dimension_size col_idx = 0; col_idx < col_size; col_idx++) {
                transposed->set_val(row_idx, col_idx, this->get_val(col_idx, row_idx));
            }
        }

        return transposed;
    }

    matrix *mul(matrix *&mmul) {
        if (this->col_count() != mmul->row_count()) {
            std::cerr << "origin col size " << this->col_count() << " != mmul row size " << mmul->row_count()
                      << std::endl;
            std::abort();
            return nullptr;
        }
        matrix *ret = new matrix(this->row_count(), mmul->col_count());
        for (dimension_size orow_idx = 0; orow_idx < this->row_count(); orow_idx++) {
            for (dimension_size mulcol_idx = 0; mulcol_idx < mmul->col_count(); mulcol_idx++) {
                for (dimension_size mulrow_idx = 0; mulrow_idx < mmul->row_count(); mulrow_idx++) {
                    floating_type product =
                            this->get_val(orow_idx, mulrow_idx) * mmul->get_val(mulrow_idx, mulcol_idx);
                    // add onto the value in the result matrix
                    ret->set_val(orow_idx, mulcol_idx, (ret->get_val(orow_idx, mulcol_idx) + product));
                }
            }
        }

        return ret;
    }

    // squash the matrix to a linear vector (1 dimension)
    std::vector<floating_type> *get_vtor() {
        std::vector<floating_type> *vtor = new std::vector<floating_type>();
        // fill in the newly created vector
        for (dimension_size row_idx = 0; row_idx < this->row_count(); row_idx++) {
            for (dimension_size col_idx = 0; col_idx < this->col_count(); col_idx++) {
                vtor->push_back(this->get_val(row_idx, col_idx));
            }
        }
        return vtor;
    }

    std::string get_str() {
        std::stringstream sstream;
        dimension_size row_size = this->row_count();
        dimension_size col_size = this->col_count();

        for (dimension_size row_idx = 0; row_idx < row_size; row_idx++) {
            for (dimension_size col_idx = 0; col_idx < col_size; col_idx++) {
                sstream << this->values.at(row_idx).at(col_idx)
                        << " \t";
            }
            sstream << "\r\n";
        }
        return sstream.str();
    }

private:
    // outer vector is rows, inner vector is columns
    std::vector<std::vector<floating_type>> values;
};

class layer {
public:
    // constructs neurons in a layer
    layer(dimension_size size) {
        for (dimension_size idx = 0; idx < size; idx++) {
            // create and allocate the neurons
            neuron *n = new neuron(0.0f);
            this->neurons.push_back(n);
        }
    }

    // destroys all neurons in a layer
    ~layer() {
        for (dimension_size idx = 0; idx < (dimension_size) this->neurons.size(); idx++) {
            if (this->neurons.at(idx) != nullptr) {
                delete this->neurons.at(idx);
            }
        }
    }

    // assigns input raw_value value to neuron
    void set_val(dimension_size at, floating_type val) {
        this->neurons.at(at)->set_raw(val);
    }

    // creates a matrix representation of the neuron layer raw_value values
    matrix *get_raw_matrix() {
        matrix *m = new matrix(1 /*always one dimension*/,
                               this->neurons.size());

        for (dimension_size idx = 0; idx < (dimension_size) this->neurons.size(); idx++) {
            m->set_val(0, idx, this->neurons.at(idx)->get_raw());
        }

        return m;
    }

    // fast sigmoid calculated values of neurons in layer
    matrix *get_fs_matrix() {
        matrix *m = new matrix(1 /*always one dimension*/,
                               this->neurons.size());

        for (dimension_size idx = 0; idx < (dimension_size) this->neurons.size(); idx++) {
            m->set_val(0, idx, this->neurons.at(idx)->get_fs());
        }

        return m;
    }

    dimension_size get_size() {
        return this->neurons.size();
    }

    // fast sigmoid derivative values of neurons in layer
    matrix *get_fsd_matrix() {
        matrix *m = new matrix(1 /*always one dimension*/,
                               this->neurons.size());

        for (dimension_size idx = 0; idx < (dimension_size) this->neurons.size(); idx++) {
            m->set_val(0, idx, this->neurons.at(idx)->get_fsd());
        }

        return m;
    }

private:
    std::vector<neuron *> neurons;
};


class network {
public:
    typedef std::vector<dimension_size> topology_vector;

    network(topology_vector topology) {
        this->topology = topology;
        for (dimension_size top_idx = 0; top_idx < (dimension_size) topology.size(); top_idx++) {
            layer *l = new layer(topology.at(top_idx));
            this->layers.push_back(l);
        }
        // there's topology size - 1 weight matrices
        for (dimension_size top_idx = 0; top_idx < (dimension_size) topology.size() - 1; top_idx++) {
            // in the weight matrix number of rows is the number of input neurons
            // and the number of columns is the number of feed-into output neurons.
            matrix *m = new matrix(topology.at(top_idx), topology.at(top_idx + 1));
            m->fill_rand(); // initial values are random
            this->weights.push_back(m);
        }

        // output errors vector must be as large as the last layer
        for (dimension_size evtor_idx = 0;
             evtor_idx < (this->topology.at(this->topology.size() - 1));
             evtor_idx++) {
            err_output.push_back(0.0f);
        }
    }

    // destroy all dynamically allocated instances
    ~network() {
        for (dimension_size layer_idx = 0; layer_idx < (dimension_size) this->layers.size(); layer_idx++) {
            if (this->layers.at(layer_idx) != nullptr) {
                delete this->layers.at(layer_idx);
            }
        }

        for (dimension_size wmatrix_idx = 0; wmatrix_idx < (dimension_size) this->layers.size(); wmatrix_idx++) {
            if (this->weights.at(wmatrix_idx) != nullptr) {
                delete this->weights.at(wmatrix_idx);
            }
        }
    }

    void set_input(std::vector<floating_type> input) {
        for (dimension_size idx = 0; idx < (dimension_size) input.size(); idx++) {
            this->layers.at(INPUT_LAYER_IDX)->set_val(idx, input.at(idx));
        }
        this->input = input;
    }

    void feed_forward() {
        // do this for each layer in the network
        for (dimension_size layer_idx = 0; layer_idx < (dimension_size) this->layers.size() - 1; layer_idx++) {
            matrix *neuron_matrix;
            matrix *weight_matrix;
            matrix *product_matrix;

            if (layer_idx == INPUT_LAYER_IDX) {
                // get the raw_value input values fed into the neuron
                neuron_matrix = this->layers.at(layer_idx)->get_raw_matrix();
            } else {
                // get the calculated fast sigmoids
                neuron_matrix = this->layers.at(layer_idx)->get_fs_matrix();
            }

            weight_matrix = this->weights.at(layer_idx);
            // get the product of the input (or left) neuron matrix
            // and weight matrix following it
            product_matrix = neuron_matrix->mul(weight_matrix);
            // potential optimisation here, no real need for intermediate vector?
            std::vector<floating_type> *product_vals = product_matrix->get_vtor();
            // feed the next layer
            for (dimension_size next_neuron = 0; next_neuron < (dimension_size) product_vals->size(); next_neuron++) {
                // assign as input to next layer
                this->layers.at(layer_idx + 1)->set_val(next_neuron, product_vals->at(next_neuron));
            }

            // dealloc
            delete neuron_matrix;
            delete product_matrix;
            delete product_vals;
            // we do not delete weight_matrix, as it just points to a pre-allocated matrix
        }
    }

    std::string get_str() {
        std::stringstream sstream;

        for (dimension_size layer_idx = 0; layer_idx < (dimension_size) this->layers.size(); layer_idx++) {
            if (layer_idx == INPUT_LAYER_IDX) {
                // if input layer (leftmost) - print raw_value values
                matrix *m = this->layers.at(layer_idx)->get_raw_matrix();
                sstream << "Network input layer: " << std::endl;
                sstream << m->get_str();
                delete m;
            } else {
                // print fast sigmoid calculated value otherwise
                matrix *m = this->layers.at(layer_idx)->get_fs_matrix();
                sstream << "Network Layer: " << std::endl;
                sstream << m->get_str();
                delete m;
            }
            if (layer_idx < (dimension_size) this->weights.size()) {
                sstream << FUNKY_DECORATOR << std::endl;
                sstream << "WEIGHTS at layer " << layer_idx << std::endl;
                sstream << this->weights.at(layer_idx)->get_str() << std::endl;
            }
        }
        return sstream.str();
    }

    void calc_err(std::vector<floating_type> targets) {
        // compare output layer size (last index) to targets size
        // and abort if mismatched
        dimension_size output_layer_size =
                this->layers.at(this->layers.size() - 1)->get_fs_matrix()->col_count();
        if (output_layer_size
            !=
            targets.size()) {
            std::cerr << "Output layer size " << output_layer_size <<
                      " != Targets size " << targets.size();
            abort();
        }

        matrix* m = this->layers.at(this->layers.size() - 1)->get_fs_matrix();
        for (dimension_size idx = 0; idx < (dimension_size) targets.size(); idx++) {
            floating_type err = m->get_val(0, idx);
            err -= targets.at(idx);
            err_output.at(idx) = err;
        }
        delete m;

        err_historical.push_back(get_err_total());
    }

    floating_type get_err_total() {
        floating_type total_error = 0.0f;
        for (dimension_size evtor_idx = 0; evtor_idx < (dimension_size) this->err_output.size(); evtor_idx++) {
            // taking the absolute value to make sure
            // negative and positive errors don't cancel each other

            // should we square each member to promote big values?
            total_error += std::abs(this->err_output.at(evtor_idx));
        }
        return total_error;
    }

    std::vector<floating_type> get_err_output() {
        return this->err_output;
    }

    std::vector<floating_type> get_err_historical() {
        return this->err_historical;
    }

    void bprop() {
        std::vector<matrix*> new_weights;
        matrix* gradients;
        // direction: output to hidden layer
        dimension_size output_layer_idx = (dimension_size) this->layers.size() - 1;
        // derived values for output layer
        matrix* derived_oth = this->layers.at(output_layer_idx)->get_fsd_matrix();
        // have a column for each neuron in the layer, always one row
        matrix* gradients_oth = new matrix(1, this->layers.at(output_layer_idx)->get_size());

        dimension_size idx;
        for(idx = 0; idx < (dimension_size) this->err_output.size(); idx++)
        {
            floating_type dfsval = derived_oth->get_val(0, idx);
            floating_type errval = this->err_output.at(idx);
            floating_type gradient = (dfsval * (errval * 0.01));
            // store it in the gradient matrix
            gradients_oth->set_val(0, idx, gradient);
        }

        dimension_size last_hidden_layer_idx = output_layer_idx - 1;
        layer* last_hidden_layer = this->layers.at(last_hidden_layer_idx);
        matrix* weights_oth = this->weights.at(output_layer_idx - 1);
        matrix* gradients_transpose = gradients_oth->transpose();
        matrix* activated_vals_last_hidden = last_hidden_layer->get_fs_matrix();
        // delta weights is the transpose of the gradient
        // multiplied with the activated values of the last hidden layer
        matrix* delta_weights_oth_t = gradients_transpose->mul(activated_vals_last_hidden);
        // now transpose so that you get it in the same format as the ACTUAL
        // weights_oth (output to hidden)
        matrix* delta_weights_oth = delta_weights_oth_t->transpose();
        // some size assertions need to be done here
        matrix* new_weights_oth = new matrix(delta_weights_oth->row_count(), delta_weights_oth->col_count());



        for(dimension_size r = 0; r < delta_weights_oth->row_count(); r++)
        {
            for(dimension_size c=0; c < delta_weights_oth->col_count(); c++)
            {
                floating_type current_weight = weights_oth->get_val(r, c);
                floating_type delta_weight = delta_weights_oth->get_val(r, c);
                // here we record the new weight value
                new_weights_oth->set_val(r, c, current_weight - delta_weight);
            }
        }

        new_weights.push_back(new_weights_oth);

        gradients = new matrix(gradients_oth->row_count(), gradients_oth->col_count());
        for(dimension_size r = 0; r < gradients_oth->row_count(); r++)
        {
            for(dimension_size c=0; c < gradients_oth->col_count(); c++)
            {
                gradients->set_val(r, c, gradients_oth->get_val(r, c));
            }
        }

//        std::cout <<"Output to Hidden new Weights:" << std::endl;
//        std::cout << new_weights_oth->get_str() << std::endl;

        // next hidden layers down to the input one
        for(idx = output_layer_idx - 1; idx > 0 /* skip input layer */; idx--)
        {
            layer* l = this->layers.at(idx);
            matrix* derived_hidden = l->get_fsd_matrix();
            // gradient size will always be the size of our current layer
            matrix* derived_gradients = new matrix(1, l->get_size());
            // weight matrix
            matrix* activated_hidden = l->get_fs_matrix();
            matrix* weight_matrix = this->weights.at(idx);
            matrix* original_weights = this->weights.at(idx - 1);
            // we are in the last hidden layer
            for (dimension_size r = 0; r < weight_matrix->row_count(); r++)
            {
                floating_type sum = 0.0f;
                for(dimension_size c = 0; c < weight_matrix->col_count(); c++)
                {
                    // gradients are same sizes as neurons so gradient row is 0
                    // and column is always the current weight matrix row number
                    // gradients or derived gradients?
                    floating_type p = gradients->get_val(0, c /*r?*/) * weight_matrix->get_val(r, c);
                    sum += p;
                }
                // rows in weight matrix is neurons,
                // each col is the connections to the neurons in the next layer
                double g = sum * activated_hidden->get_val(0, r);
                derived_gradients->set_val(0, r, g);
            }

            matrix* left_neurons;
            // if input layer is on the left
            if((idx - 1) == 0)
            {
                // input layer stays raw_value and never gets activated
                left_neurons = this->layers.at(idx -1)->get_raw_matrix();
            }
            else
            {
                // otherwise get the vals after activation
                left_neurons = this->layers.at(idx - 1)->get_fs_matrix();
            }

            matrix* derived_gradients_T = derived_gradients->transpose();
            matrix* delta_weights_T = derived_gradients_T->mul(left_neurons);
            matrix* delta_weights = delta_weights_T->transpose();

            matrix* new_weights_hidden = new matrix(delta_weights->row_count(), delta_weights->col_count());

            for(dimension_size r = 0; r < new_weights_hidden->row_count(); r++)
            {
                for(dimension_size c=0; c < new_weights_hidden->col_count(); c++)
                {
                    floating_type w = original_weights->get_val(r, c);
                    floating_type d = delta_weights->get_val(r, c);
                    floating_type new_weight = w - d;
                    new_weights_hidden->set_val(r, c, new_weight);
                }
            }
            new_weights.push_back(new_weights_hidden);

            gradients = new matrix(derived_gradients->row_count(), derived_gradients->col_count());
            for(dimension_size r = 0; r < gradients->row_count(); r++)
            {
                for(dimension_size c=0; c < gradients->col_count(); c++)
                {
                    gradients->set_val(r, c, derived_gradients->get_val(r, c));
                }
            }
        }

        // as the new weight matrices were pushed back in reverse,
        // we need to invert their order before we assign them to
        // be our current standard weight matrices

        // use algorithm reverse iterators?
        std::reverse(new_weights.begin(), new_weights.end());
        for(int i = 0; i < this->weights.size(); i++)
        {
            delete this->weights.at(i);
        }
        // replace weights with new ones
        this->weights = new_weights;

    }

private:
    topology_vector topology;
    std::vector<layer*> layers;
    std::vector<matrix*> weights;
    std::vector<matrix*> gradients;
    std::vector<floating_type> input;
    std::vector<floating_type> err_output;
    std::vector<floating_type> err_historical;
};

int main() {
//    // NEURON CREATED
//    neuron *n = new neuron(1.5f);
//    std::cout << "Neuron constructor test:" << std::endl;
//    std::cout << "raw_value: " << n->get_raw() << std::endl;
//    std::cout << "fast_sigmoid_value: " << n->get_fs() << std::endl;
//    std::cout << "dfs: " << n->get_fsd() << std::endl << std::endl;
//
//    // MATRIX CREATED
//    matrix *m = new matrix(3, 2);
//    std::cout << "Matrix constructor test:" << std::endl;
//    std::cout << m->get_str() << std::endl;
//
//    std::cout << "Matrix randomization test:" << std::endl;
//    m->fill_rand();
//    std::cout << m->get_str() << std::endl;
//
//    std::cout << "Matrix transpose test:" << std::endl;
//    matrix *mnew = m->transpose();
//    std::cout << mnew->get_str() << std::endl;
//
//    std::cout << "Test neural network class:" << std::endl;
//    network::topology_vector topology = {6, 10, 2};
//    network *nn = new network(topology);
//    nn->set_input(std::vector<floating_type>{0.7f, 3.1f, 5.0f, 0.7f, 3.1f, 5.0f});
//    std::cout << nn->get_str() << std::endl;
//
//    std::cout << "Test matrix multiplication:" << std::endl;
//    matrix *mmul1 = new matrix(1, 4);
//    mmul1->fill_rand();
//    matrix *mmul2 = new matrix(4, 2);
//    mmul2->fill_rand();
//    matrix *mmulr = mmul1->mul(mmul2);
//    std::cout << "Matrix 1: \n" << mmul1->get_str() << std::endl;
//    std::cout << "Matrix 2: \n" << mmul2->get_str() << std::endl;
//    std::cout << "Matrix Product: \n" << mmulr->get_str() << std::endl;
//
//    std::cout << "Test feeding forward:" << std::endl;
//    network::topology_vector big_topology = {3, 2, 3};
//    network *real_neural_network = new network(big_topology);
//
//    // setting input test (only after defining topology)
//    real_neural_network->set_input(std::vector<floating_type>{
//            1, 0, 1
//    });
//    std::cout << "Input layer set:" << std::endl;
//    std::cout << real_neural_network->get_str() << std::endl;
//
//    // feed forward test (only after setting input)
//    real_neural_network->feed_forward();
//    std::cout << "After feed-forward procedure" << std::endl;
//    std::cout << real_neural_network->get_str() << std::endl;
//
//    // net error test (only after feed-forward)
//    std::cout << "Net error" << std::endl;
//    real_neural_network->calc_err(std::vector<floating_type>{0.75, 0.75, 0.75});
//    std::cout << real_neural_network->get_err_total() << std::endl;
//
//    std::cout << "Back Propagation" << std::endl;
//    real_neural_network->bprop();

    std::cout << "========== REAL RUN ==========" << std::endl;
#define LEARNING_ITERATIONS 10000
        network::topology_vector big_topology = {3, 8, 3};
        network* real_neural_network = new network(big_topology);
        for(int i = 0; i < LEARNING_ITERATIONS; i++)
        {
            real_neural_network->set_input(std::vector<floating_type>{1, 1, 1});
            real_neural_network->feed_forward();
            real_neural_network->calc_err(std::vector<floating_type>{0, 1, 0});
            std::cout << "Net error: " << real_neural_network->get_err_total() << std::endl;
            real_neural_network->bprop();
        }
        std::cout << "=== FINAL ERROR ===" << std::endl;
        std::cout << real_neural_network->get_err_total() << std::endl;
        std::cout << real_neural_network->get_str();

    return 0;
}