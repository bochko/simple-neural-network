#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>

#include <vector>

#include "snn_config.h"
#include "neuron.h"
#include "matrix.h"
#include "layer.h"




class network {


private:
    topology_vector topology;
    std::vector<layer*> layers;
    std::vector<matrix*> weights;
    std::vector<floating_type> err_output;
    std::vector<floating_type> err_historical;

public:


    network(topology_vector topology) {
        this->topology = topology;
        for (int top_idx = 0; top_idx < (int) topology.size(); top_idx++) {
            layer *l = new layer(topology.at(top_idx));
            this->layers.push_back(l);
        }
        // there's topology size - 1 weight matrices
        for (int top_idx = 0; top_idx < (int) topology.size() - 1; top_idx++) {
            // in the weight matrix number of rows is the number of input neurons
            // and the number of columns is the number of feed-into output neurons.
            matrix *m = new matrix(topology.at(top_idx), topology.at(top_idx + 1));
            m->fill_rand(); // initial values are random
            this->weights.push_back(m);
        }

        // output errors vector must be as large as the last layer
        for (int evtor_idx = 0;
             evtor_idx < (this->topology.at(this->topology.size() - 1));
             evtor_idx++) {
            err_output.push_back(0.0f);
        }
    }

    // destroy all dynamically allocated instances
    ~network() {
        for (int layer_idx = 0; layer_idx < (int) this->layers.size(); layer_idx++) {
            if (this->layers.at(layer_idx) != nullptr) {
                delete this->layers.at(layer_idx);
            }
        }

        for (int wmatrix_idx = 0; wmatrix_idx < (int) this->layers.size(); wmatrix_idx++) {
            if (this->weights.at(wmatrix_idx) != nullptr) {
                delete this->weights.at(wmatrix_idx);
            }
        }
    }

    void set_input(std::vector<floating_type> input) {
        for (int idx = 0; idx < (int) input.size(); idx++) {
            this->layers.at(INPUT_LAYER_IDX)->set_val(idx, input.at(idx));
        }
    }

    void feed_forward() {
        // do this for each layer in the network
        for (int layer_idx = 0; layer_idx < (int) this->layers.size() - 1; layer_idx++) {
            matrix *neuron_matrix;
            matrix *weight_matrix;
            matrix *product_matrix;

            if (layer_idx == INPUT_LAYER_IDX) {
                // get the raw_value input values fed into the neuron
                neuron_matrix = this->layers.at(layer_idx)->new_from_raw_values();
            } else {
                // get the calculated fast sigmoids
                neuron_matrix = this->layers.at(layer_idx)->new_from_activated_values();
            }

            weight_matrix = this->weights.at(layer_idx);
            // get the product of the input (or left) neuron matrix
            // and weight matrix following it
            product_matrix = neuron_matrix->new_from_multiply(weight_matrix);
            // potential optimisation here, no real need for intermediate vector?
            std::vector<floating_type> *product_vals = product_matrix->new_vector_from_squash();
            // feed the next layer
            for (int next_neuron = 0; next_neuron < (int) product_vals->size(); next_neuron++) {
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

        for (int layer_idx = 0; layer_idx < (int) this->layers.size(); layer_idx++) {
            if (layer_idx == INPUT_LAYER_IDX) {
                // if input layer (leftmost) - print raw_value values
                matrix *m = this->layers.at(layer_idx)->new_from_raw_values();
                sstream << "Network input layer: " << std::endl;
                sstream << m->get_str();
                delete m;
            } else {
                // print fast sigmoid calculated value otherwise
                matrix *m = this->layers.at(layer_idx)->new_from_activated_values();
                sstream << "Network Layer: " << std::endl;
                sstream << m->get_str();
                delete m;
            }
            if (layer_idx < (int) this->weights.size()) {
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
        int output_layer_size =
                this->layers.at(this->layers.size() - 1)->new_from_activated_values()->get_col_count();
        if (output_layer_size
            !=
            targets.size()) {
            std::cerr << "Output layer size " << output_layer_size <<
                      " != Targets size " << targets.size();
            std::abort();
        }

        matrix* m = this->layers.at(this->layers.size() - 1)->new_from_activated_values();
        for (int idx = 0; idx < (int) targets.size(); idx++) {
            floating_type err = m->get_value_at(0, idx);
            err -= targets.at(idx);
            err_output.at(idx) = err;
        }
        delete m;

        err_historical.push_back(get_err_total());
    }

    floating_type get_err_total() {
        floating_type total_error = 0.0f;
        for (int evtor_idx = 0; evtor_idx < (int) this->err_output.size(); evtor_idx++) {
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
        // direction: output to hidden layer
        int output_layer_idx = (int) this->layers.size() - 1;
        // derived values for output layer
        matrix* derived_oth = this->layers.at(output_layer_idx)->new_from_derived_values();
        // have a column for each neuron in the layer, always one row
        matrix* gradients_oth = new matrix(1, this->layers.at(output_layer_idx)->get_size());

        int idx;
        for(idx = 0; idx < (int) this->err_output.size(); idx++)
        {
            floating_type dfsval = derived_oth->get_value_at(0, idx);
            floating_type errval = this->err_output.at(idx);
            floating_type gradient = (dfsval * (errval * 0.01f));
            // store it in the gradient matrix
            gradients_oth->set_value_at(0, idx, gradient);
        }

        int last_hidden_layer_idx = output_layer_idx - 1;
        layer* last_hidden_layer = this->layers.at(last_hidden_layer_idx);
        matrix* weights_oth = this->weights.at(output_layer_idx - 1);
        matrix* gradients_transpose = gradients_oth->new_from_transpose();
        matrix* activated_vals_last_hidden = last_hidden_layer->new_from_activated_values();
        // delta weights is the new_from_transpose of the gradient
        // multiplied with the activated values of the last hidden layer
        // ============== A*B != B*A
        matrix* delta_weights_oth_t = gradients_transpose->new_from_multiply(activated_vals_last_hidden);
        //matrix* delta_weights_oth_t = activated_vals_last_hidden->new_from_multiply(gradients_transpose);
        // now new_from_transpose so that you get it in the same format as the ACTUAL
        // weights_oth (output to hidden)
        matrix* delta_weights_oth = delta_weights_oth_t->new_from_transpose();
        // some size assertions need to be done here
        matrix* new_weights_oth = new matrix(delta_weights_oth->get_row_count(), delta_weights_oth->get_col_count());



        for(int r = 0; r < delta_weights_oth->get_row_count(); r++)
        {
            for(int c=0; c < delta_weights_oth->get_col_count(); c++)
            {
                floating_type current_weight = weights_oth->get_value_at(r, c);
                floating_type delta_weight = delta_weights_oth->get_value_at(r, c);
                // here we record the new weight value
                new_weights_oth->set_value_at(r, c, current_weight - delta_weight);
            }
        }

        new_weights.push_back(new_weights_oth);

        matrix* gradients = new matrix(gradients_oth->get_row_count(), gradients_oth->get_col_count());
        for(int r = 0; r < gradients_oth->get_row_count(); r++)
        {
            for(int c=0; c < gradients_oth->get_col_count(); c++)
            {
                gradients->set_value_at(r, c, gradients_oth->get_value_at(r, c));
            }
        }

//        std::cout <<"Output to Hidden new Weights:" << std::endl;
//        std::cout << new_weights_oth->get_str() << std::endl;

        // next hidden layers down to the input one
        for(idx = output_layer_idx - 1; idx > 0 /* skip input layer */; idx--)
        {
            layer* l = this->layers.at(idx);
            // gradient size will always be the size of our current layer
            matrix* derived_gradients = new matrix(1, l->get_size());
            // weight matrix
            matrix* activated_hidden = l->new_from_activated_values();
            matrix* weight_matrix = this->weights.at(idx);
            matrix* original_weights = this->weights.at(idx - 1);
            // we are in the last hidden layer
            for (int r = 0; r < weight_matrix->get_row_count(); r++)
            {
                floating_type sum = 0.0f;
                for(int c = 0; c < weight_matrix->get_col_count(); c++)
                {
                    // gradients are same sizes as neurons so gradient row is 0
                    // and column is always the current weight matrix row number
                    // gradients or derived gradients?
                    floating_type p = gradients->get_value_at(0, c /*r?*/) * weight_matrix->get_value_at(r, c);
                    sum += p;
                }
                // rows in weight matrix is neurons,
                // each col is the connections to the neurons in the next layer
                double g = sum * activated_hidden->get_value_at(0, r);
                derived_gradients->set_value_at(0, r, g);
            }

            matrix* left_neurons;
            // if input layer is on the left
            if((idx - 1) == 0)
            {
                // input layer stays raw_value and never gets activated
                left_neurons = this->layers.at(idx - 1)->new_from_raw_values();
            }
            else
            {
                // otherwise get the vals after activation
                left_neurons = this->layers.at(idx - 1)->new_from_activated_values();
            }

            matrix* derived_gradients_T = derived_gradients->new_from_transpose();
            matrix* delta_weights_T = derived_gradients_T->new_from_multiply(left_neurons);
            matrix* delta_weights = delta_weights_T->new_from_transpose();

            matrix* new_weights_hidden = new matrix(delta_weights->get_row_count(), delta_weights->get_col_count());

            for(int r = 0; r < new_weights_hidden->get_row_count(); r++)
            {
                for(int c=0; c < new_weights_hidden->get_col_count(); c++)
                {
                    floating_type w = original_weights->get_value_at(r, c);
                    floating_type d = delta_weights->get_value_at(r, c);
                    floating_type new_weight = w - d;
                    new_weights_hidden->set_value_at(r, c, new_weight);
                }
            }
            new_weights.push_back(new_weights_hidden);

//            delete gradients;
            gradients = new matrix(derived_gradients->get_row_count(), derived_gradients->get_col_count());
            for(int r = 0; r < gradients->get_row_count(); r++)
            {
                for(int c=0; c < gradients->get_col_count(); c++)
                {
                    gradients->set_value_at(r, c, derived_gradients->get_value_at(r, c));
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
};

int main() {
//    // NEURON CREATED
//    neuron *n = new neuron(1.5f);
//    std::cout << "Neuron constructor test:" << std::endl;
//    std::cout << "raw_value: " << n->get_raw() << std::endl;
//    std::cout << "sigmoid_value: " << n->get_fs() << std::endl;
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
//    std::cout << "Matrix new_from_transpose test:" << std::endl;
//    matrix *mnew = m->new_from_transpose();
//    std::cout << mnew->get_str() << std::endl;
//
//    std::cout << "Test neural network class:" << std::endl;
//    network::topology_vector topology = {6, 10, 10, 2};
//    network *nn = new network(topology);
//    nn->set_input(std::vector<floating_type>{0.7f, 3.1f, 5.0f, 0.7f, 3.1f, 5.0f});
//    std::cout << nn->get_str() << std::endl;
//
//    std::cout << "Test matrix multiplication:" << std::endl;
//    matrix *mmul1 = new matrix(1, 4);
//    mmul1->fill_rand();
//    matrix *mmul2 = new matrix(4, 2);
//    mmul2->fill_rand();
//    matrix *mmulr = mmul1->new_from_multiply(mmul2);
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
        topology_vector big_topology = {3, 4, 4, 3};
        network* real_neural_network = new network(big_topology);
        for(int i = 0; i < LEARNING_ITERATIONS; i++)
        {
            real_neural_network->set_input(std::vector<floating_type>{1, .75f, 0.13f});
            real_neural_network->feed_forward();
            real_neural_network->calc_err(std::vector<floating_type>{0, 1, 0});

            real_neural_network->bprop();
        }
        std::cout << "=== FINAL ERROR ===" << std::endl;
        std::cout << real_neural_network->get_err_total() << std::endl;
        std::cout << real_neural_network->get_str();

    return 0;
}