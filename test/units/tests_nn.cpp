#include "gtest/gtest.h"

#include "../../src/nn.h"
#include <cstdlib>
#include <vector>

TEST(NeuralNetworkTests, random_topology_nn_create)
{
    for(int testrun = 0; testrun < 10000; testrun ++)
    {
        // generate random topology width between 3 and 10
        unsigned int topology_size = std::rand() % 8 + 3;
        topology_vector topology;
        for(int i = 0; i < topology_size; i++)
        {
            // generate random layer size between 2 and 10
            unsigned int layer_size = std::rand() % 9 + 2;
            topology.push_back(layer_size);
        }
        network* nn = new network(topology);
        ASSERT_NE(nn, nullptr);
        delete nn;
    }
}

TEST(NeuralNetworkTests, random_topology_nn_feed_input)
{
    for(int testrun = 0; testrun < 10000; testrun ++)
    {
        // generate random topology width between 3 and 10
        unsigned int topology_size = std::rand() % 8 + 3;
        topology_vector topology;
        for(int i = 0; i < topology_size; i++)
        {
            // generate random layer size between 2 and 10
            unsigned int layer_size = std::rand() % 9 + 2;
            topology.push_back(layer_size);
        }
        network* nn = new network(topology);
        ASSERT_NE(nn, nullptr);

        std::vector<floating_type> input;
        for(int i = 0; i < topology.at(0); i++)
        {
            input.push_back(std::rand() % 2); // 0 or 1
        }
        nn->set_input(input);
        delete nn;
    }
}

TEST(NeuralNetworkTests, random_topology_full_run)
{
    for(int testrun = 0; testrun < 100; testrun ++)
    {
        // generate random topology width between 3 and 10
        unsigned int topology_size = std::rand() % 8 + 3;
        topology_vector topology;
        for(int i = 0; i < topology_size; i++)
        {
            // generate random layer size between 2 and 10
            unsigned int layer_size = std::rand() % 9 + 2;
            topology.push_back(layer_size);
        }
        network* nn = new network(topology);
        ASSERT_NE(nn, nullptr);

        std::vector<floating_type> input;
        std::vector<floating_type> output;
        for(int i = 0; i < topology.at(0); i++)
        {
            input.push_back(std::rand() % 2); // 0 or 1
        }
        for(int i = 0; i < topology.at(topology_size-1); i++)
        {
            output.push_back(std::rand() % 2); // 0 or 1
        }
        for(int li = 0; li < 100; li++)
        {
            nn->set_input(input);
            nn->feed_forward();
            nn->calc_err(output);
            nn->bprop();
        }
        delete nn;
    }
}

TEST(NeuralNetworkTests, random_topology_validate_learning)
{
    for(int testrun = 0; testrun < 100; testrun ++)
    {
        // generate random topology width between 3 and 10
        unsigned int topology_size = std::rand() % 8 + 3;
        topology_vector topology;
        for(int i = 0; i < topology_size; i++)
        {
            // generate random layer size between 2 and 10
            unsigned int layer_size = std::rand() % 9 + 2;
            topology.push_back(layer_size);
        }
        network* nn = new network(topology);
        ASSERT_NE(nn, nullptr);

        std::vector<floating_type> input;
        std::vector<floating_type> output;
        for(int i = 0; i < topology.at(0); i++)
        {
            input.push_back(std::rand() % 2); // 0 or 1
        }
        for(int i = 0; i < topology.at(topology_size-1); i++)
        {
            output.push_back(std::rand() % 2); // 0 or 1
        }
        floating_type start_error = 0;
        for(int li = 0; li < 100; li++)
        {
            nn->set_input(input);
            nn->feed_forward();
            nn->calc_err(output);
            nn->bprop();
            if(li == 1)
            {
                // set benchmark
                start_error = nn->get_err_total();
            }
        }
        ASSERT_LT(nn->get_err_total(), start_error);
        delete nn;
    }
}
