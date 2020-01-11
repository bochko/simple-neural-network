#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include <vector>

#include "nn.h"

    network::network(topology_vector topology) {
        this->topology = topology;
        for (int value_at_index : topology) {
            layer *l = new layer(value_at_index);
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
    network::~network() {
        for (auto & layer : this->layers) {
            if (layer != nullptr) {
                delete layer;
            }
        }

        for (int wmatrix_idx = 0; wmatrix_idx < (int) this->layers.size(); wmatrix_idx++) {
            if (this->weights.at(wmatrix_idx) != nullptr) {
                delete this->weights.at(wmatrix_idx);
            }
        }
    }

    void network::set_input(std::vector<floating_type> input) {
        for (int idx = 0; idx < (int) input.size(); idx++) {
            this->layers.at(INPUT_LAYER_IDX)->set_val(idx, input.at(idx));
        }
    }

    void network::feed_forward() {
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

    std::string network::get_str() {
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

    void network::calc_err(std::vector<floating_type> targets) {
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

    floating_type network::get_err_total() {
        floating_type total_error = 0.0f;
        for (double err : this->err_output) {
            // taking the absolute value to make sure
            // negative and positive errors don't cancel each other
            // should we square each member to promote big values?
            total_error += std::abs(err);
        }
        return total_error;
    }

    std::vector<floating_type> network::get_err_output() {
        return this->err_output;
    }

    std::vector<floating_type> network::get_err_historical() {
        return this->err_historical;
    }

    void network::bprop() {
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
        for(auto & weight : this->weights)
        {
            delete weight;
        }
        // replace weights with new ones
        this->weights = new_weights;

    }

