#include "gtest/gtest.h"

#include "../../src/layer.h"

TEST(LayerTests, layer_creation)
{
    layer* l = new layer(400);
    ASSERT_EQ(l->get_size(), 400);
    delete l;
}

TEST(LayerTests, layer_get_matrix)
{
    layer* l = new layer(400);

    matrix* mraw = l->new_from_raw_values();
    ASSERT_EQ(mraw->get_row_count(), 1);
    ASSERT_EQ(mraw->get_col_count(), 400);

    matrix* mact = l->new_from_activated_values();
    ASSERT_EQ(mact->get_row_count(), 1);
    ASSERT_EQ(mact->get_col_count(), 400);

    matrix* mder = l->new_from_derived_values();
    ASSERT_EQ(mder->get_row_count(), 1);
    ASSERT_EQ(mder->get_col_count(), 400);

    delete l;
    delete mraw;
    delete mact;
    delete mder;
}

TEST(LayerTests, layer_set)
{
    layer* l = new layer(400);
    for(int i = 0; i < l->get_size(); i++)
    {
        l->set_val(i, 1.0f);
    }

    matrix* mraw = l->new_from_raw_values();
    matrix* mact = l->new_from_activated_values();
    matrix* mder = l->new_from_derived_values();

    for(int i = 0; i < l->get_size(); i++)
    {
        ASSERT_EQ(mraw->get_value_at(0, i), 1);
    }
    for(int i = 0; i < l->get_size(); i++)
    {
        ASSERT_NE(mact->get_value_at(0, i), 0);
    }
    for(int i = 0; i < l->get_size(); i++)
    {
        ASSERT_NE(mder->get_value_at(0, i), 0);
    }

    delete l;
    delete mraw;
    delete mact;
    delete mder;
}