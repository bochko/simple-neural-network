//
// Created by Boyan Atanasov on 11/01/2020.
//

#ifndef SIMPLENEURALNETWORK_LAYER_H
#define SIMPLENEURALNETWORK_LAYER_H

#include <vector>
#include <sstream>
#include <string>

#include "nn_config.h"
#include "neuron.h"
#include "matrix.h"

class layer {
private:
    std::vector<neuron *> neurons;

public:
    // constructs neurons in a layer
    layer(int size);

    // destroys all neurons in a layer
    ~layer();

    // assigns input raw value to neuron
    void set_val(int at, floating_type val);

    // creates a matrix representation of the neuron layer raw values
    matrix *new_from_raw_values();

    // fast sigmoid calculated values of neurons in layer
    matrix *new_from_activated_values();

    int get_size();

    // fast sigmoid derivative values of neurons in layer
    matrix *new_from_derived_values();
};

#endif //SIMPLENEURALNETWORK_LAYER_H
