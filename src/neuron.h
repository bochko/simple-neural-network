//
// Created by Boyan Atanasov on 23/11/2019.
//

#ifndef SIMPLENEURALNETWORK_NEURON_H
#define SIMPLENEURALNETWORK_NEURON_H

#include <cmath>
#include "nn_config.h"

/**
 * Implements a neuron building block containing three values:
 * the raw_value input value, the fast sigmoid result, and the derivative
 * of a fast sigmoid
 */
class neuron {
private:
    const floating_type SIGMOID_MIDPOINT = 0.0f;
    const floating_type CURVE_MAX = 1.0f;
    const floating_type LOGISTIC_GROWTH_RATE = 1.0f;

    // Value types are self documenting.
    // The approach of calculating them at constructor time
    // is more memory intensive, but calculating them each time
    // they are needed would introduce unnecessary processing
    // wasted.
    floating_type raw;
    floating_type sigmoid;
    floating_type derivative;
public:
    /**
     * Creates a neuron instance and automatically
     * calculates activation and derivative values.
     * @param val raw input for neuron
     */
    explicit neuron(floating_type val);

    /*
     * Fast sigmoid activation function (currently the only
     * supported activation function.
     * f(x) = x / (1 + abs(x))
     */
    void calculate_sigmoid();

    /**
     * Derivative of the fast sigmoid activation function.
     * f'(x) = f(x) * (1 - f(x))
     */
    void calculate_derivative();

    /*
     * Re-sets the neuron raw input and recalculates
     * all adjacent values.
     */
    void set_raw(floating_type new_raw);

    // Getters
    floating_type get_raw();
    floating_type get_sigmoid();
    floating_type get_derivative();
};

#endif // SIMPLENEURALNETWORK_NEURON_H
