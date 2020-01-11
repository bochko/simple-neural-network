#include "gtest/gtest.h"

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(SampleTests, test_always_succeeds)
{
    EXPECT_EQ(0, 0);
}