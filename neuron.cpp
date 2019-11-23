//
// Created by Boyan Atanasov on 23/11/2019.
//

#include "neuron.h"

#include <cmath>

neuron::neuron(floating_type val)  {
    this->raw_value = val;
    this->calc_fast_sigmoid();
    this->calc_fast_sigmoid_derivative();
}

void neuron::calc_fast_sigmoid() {
    // 1 + std::abs(x) is always positive, no chance of dividing by zero
    this->fast_sigmoid_value = this->raw_value / (1.0f + std::abs(this->raw_value));
}

void neuron::calc_fast_sigmoid_derivative() {
    this->fast_sigmoid_derivative_value = this->fast_sigmoid_value * (1.0f - this->fast_sigmoid_value);
}

floating_type neuron::get_raw() {
    return this->raw_value;
}

floating_type neuron::get_fs() {
    return this->fast_sigmoid_value;
}

floating_type neuron::get_fsd() {
    return this->fast_sigmoid_derivative_value;
}

void neuron::set_raw(floating_type new_raw) {
    this->raw_value = new_raw;
    this->calc_fast_sigmoid();
    this->calc_fast_sigmoid_derivative();
}
