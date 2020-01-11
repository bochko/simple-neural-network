//
// Created by Boyan Atanasov on 11/01/2020.
//

#include "layer.h"

// constructs neurons in a layer
layer::layer(int size) {
    for (int idx = 0; idx < size; idx++) {
        // create and allocate the neurons
        neuron *n = new neuron(0.0f);
        this->neurons.push_back(n);
    }
}

// destroys all neurons in a layer
layer::~layer() {
    for (auto & neuron : this->neurons) {
        // deleting nullptr is okay
        delete neuron;
    }
}

// assigns input raw_value value to neuron
void layer::set_val(int at, floating_type val) {
    this->neurons.at(at)->set_raw(val);
}

// creates a matrix representation of the neuron layer raw_value values
matrix* layer::new_from_raw_values() {
    matrix *m = new matrix(1 /*always one dimension*/,
                           this->neurons.size());

    for (int idx = 0; idx < (int) this->neurons.size(); idx++) {
        m->set_value_at(0, idx, this->neurons.at(idx)->get_raw());
    }

    return m;
}

// fast sigmoid calculated values of neurons in layer
matrix* layer::new_from_activated_values() {
    matrix *m = new matrix(1 /*always one dimension*/,
                           this->neurons.size());

    for (int idx = 0; idx < (int) this->neurons.size(); idx++) {
        m->set_value_at(0, idx, this->neurons.at(idx)->get_fs());
    }

    return m;
}

int layer::get_size() {
    return this->neurons.size();
}

// fast sigmoid derivative values of neurons in layer
matrix* layer::new_from_derived_values() {
    matrix *m = new matrix(1 /*always one dimension*/,
                           this->neurons.size());

    for (int idx = 0; idx < (int) this->neurons.size(); idx++) {
        m->set_value_at(0, idx, this->neurons.at(idx)->get_fsd());
    }

    return m;
}