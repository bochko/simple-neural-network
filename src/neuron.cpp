//
// Created by Boyan Atanasov on 23/11/2019.
//

#include "neuron.h"

#include <cmath>

neuron::neuron(floating_type val)  {
    this->raw_value = val;
    this->calc_sigmoid();
    this->calc_sigmoid_derivative();
}

void neuron::calc_sigmoid() {
    // we will do the computation with the real logistic function
    // https://en.wikipedia.org/wiki/Logistic_function
    this->sigmoid_value = this->CURVE_MAX / (1 + std::exp((-1*(this->LOGISTIC_GROWTH_RATE))*(this->raw_value - this->SIGMOID_MIDPOINT)));
}

void neuron::calc_sigmoid_derivative() {
    // the derivative is quite simple
    // https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
    this->sigmoid_derivative_value = this->sigmoid_value * (1.0f - this->sigmoid_value);
}

floating_type neuron::get_raw() {
    return this->raw_value;
}

floating_type neuron::get_fs() {
    return this->sigmoid_value;
}

floating_type neuron::get_fsd() {
    return this->sigmoid_derivative_value;
}

void neuron::set_raw(floating_type new_raw) {
    this->raw_value = new_raw;
    this->calc_sigmoid();
    this->calc_sigmoid_derivative();
}
