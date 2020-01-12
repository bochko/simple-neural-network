#include "nn.h"

#define LEARNING_ITERATIONS 30000

int main(int argc, char** argv) {
    // NEURON CREATED
    neuron *n = new neuron(1.5f);
    std::cout << "Neuron constructor test:" << std::endl;
    std::cout << "raw_value: " << n->get_raw() << std::endl;
    std::cout << "sigmoid_value: " << n->get_fs() << std::endl;
    std::cout << "dfs: " << n->get_fsd() << std::endl << std::endl;

    // MATRIX CREATED
    matrix *m = new matrix(3, 2);
    std::cout << "Matrix constructor test:" << std::endl;
    std::cout << m->get_str() << std::endl;

    std::cout << "Matrix randomization test:" << std::endl;
    m->fill_rand();
    std::cout << m->get_str() << std::endl;

    std::cout << "Matrix new_from_transpose test:" << std::endl;
    matrix *mnew = m->new_from_transpose();
    std::cout << mnew->get_str() << std::endl;

    std::cout << "Test neural network class:" << std::endl;
    topology_vector topology = {6, 10, 10, 2};
    network *nn = new network(topology);
    nn->set_input(std::vector<floating_type>{0.7f, 3.1f, 5.0f, 0.7f, 3.1f, 5.0f});
    std::cout << nn->get_str() << std::endl;

    std::cout << "Test matrix multiplication:" << std::endl;
    matrix *mmul1 = new matrix(1, 4);
    mmul1->fill_rand();
    matrix *mmul2 = new matrix(4, 2);
    mmul2->fill_rand();
    matrix *mmulr = mmul1->new_from_multiply(mmul2);
    std::cout << "Matrix 1: \n" << mmul1->get_str() << std::endl;
    std::cout << "Matrix 2: \n" << mmul2->get_str() << std::endl;
    std::cout << "Matrix Product: \n" << mmulr->get_str() << std::endl;

    std::cout << "Test feeding forward:" << std::endl;
    topology_vector topology2 = {3, 2, 3};
    network *real_neural_network = new network(topology2);

    // setting input test (only after defining topology)
    real_neural_network->set_input(std::vector<floating_type>{
            1, 0, 1
    });
    std::cout << "Input layer set:" << std::endl;
    std::cout << real_neural_network->get_str() << std::endl;

    // feed forward test (only after setting input)
    real_neural_network->feed_forward();
    std::cout << "After feed-forward procedure" << std::endl;
    std::cout << real_neural_network->get_str() << std::endl;

    // net error test (only after feed-forward)
    std::cout << "Net error" << std::endl;
    real_neural_network->calc_err(std::vector<floating_type>{0.75, 0.75, 0.75});
    std::cout << real_neural_network->get_err_total() << std::endl;

    std::cout << "Back Propagation" << std::endl;
    real_neural_network->bprop();

    std::cout << "========== REAL RUN ==========" << std::endl;
    topology_vector big_topology = {2, 4, 3};
    network* nn3 = new network(big_topology);
    for(int i = 0; i < LEARNING_ITERATIONS; i++)
    {
        nn3->set_input(std::vector<floating_type>{1, .75f});
        nn3->feed_forward();
        nn3->calc_err(std::vector<floating_type>{0, 1, 0});

        nn3->bprop();
    }
    std::cout << "=== FINAL ERROR ===" << std::endl;
    std::cout << nn3->get_err_total() << std::endl;
    std::cout << nn3->get_str();

    return 0;
}
