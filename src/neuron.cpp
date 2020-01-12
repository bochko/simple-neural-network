//
// Created by Boyan Atanasov on 23/11/2019.
//

#include "neuron.h"

#include <cmath>

neuron::neuron(floating_type val)  {
    this->raw = val;
    this->calculate_sigmoid();
    this->calculate_derivative();
}

void neuron::calculate_sigmoid() {
    // we will do the computation with the real logistic function
    // https://en.wikipedia.org/wiki/Logistic_function
    this->sigmoid = this->CURVE_MAX / (1 + std::exp((-1 * (this->LOGISTIC_GROWTH_RATE)) * (this->raw - this->SIGMOID_MIDPOINT)));
}

void neuron::calculate_derivative() {
    // the derivative is quite simple
    // https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
    this->derivative = this->sigmoid * (1.0f - this->sigmoid);
}

floating_type neuron::get_raw() {
    return this->raw;
}

floating_type neuron::get_sigmoid() {
    return this->sigmoid;
}

floating_type neuron::get_derivative() {
    return this->derivative;
}

void neuron::set_raw(floating_type new_raw) {
    this->raw = new_raw;
    this->calculate_sigmoid();
    this->calculate_derivative();
}
