#include "gtest/gtest.h"

#include "../../src/neuron.h"

TEST(NeuronTests, assert_raw_value)
{
    neuron n = neuron(1.0f);
    ASSERT_EQ(n.get_raw(), 1.0f);
}

TEST(NeuronTests, assert_sigmoid_value)
{
    neuron n = neuron(0.0f);
    // for 0 the value of any unmodified sigmoid should be 0.5
    ASSERT_EQ(n.get_sigmoid(), 0.5f);
}

TEST(NeuronTests, assert_sigmoid_derivative_value)
{
    neuron n = neuron(0.0f);
    // for 0 the value of any unmodified sigmoid derivative should be 0.25
    ASSERT_EQ(n.get_derivative(), 0.25f);
}