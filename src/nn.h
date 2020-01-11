//
// Created by Boyan Atanasov on 11/01/2020.
//

#ifndef SIMPLENEURALNETWORK_NN_H
#define SIMPLENEURALNETWORK_NN_H

#include "nn_config.h"
#include "neuron.h"
#include "layer.h"
#include "matrix.h"

class network {

private:
    topology_vector topology;
    std::vector<layer*> layers;
    std::vector<matrix*> weights;
    std::vector<floating_type> err_output;
    std::vector<floating_type> err_historical;

public:

    network(topology_vector topology);

    // destroy all dynamically allocated instances
    ~network();

    void set_input(std::vector<floating_type> input);

    void feed_forward();

    std::string get_str();

    void calc_err(std::vector<floating_type> targets);

    floating_type get_err_total();

    std::vector<floating_type> get_err_output();

    std::vector<floating_type> get_err_historical();

    void bprop();
};

#endif //SIMPLENEURALNETWORK_NN_H
