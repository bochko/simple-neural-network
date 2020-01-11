#include "gtest/gtest.h"
#include "../../src/matrix.h"

TEST(MatrixTests, matrix_creation)
{
    matrix* m = new matrix(400, 100);

    ASSERT_EQ(m->get_row_count(), 400);
    ASSERT_EQ(m->get_col_count(), 100);

    for(int rx = 0; rx < m->get_row_count(); rx++)
    {
        for(int cx = 0; cx < m->get_col_count(); cx++)
        {
            ASSERT_EQ(m->get_value_at(rx, cx), (floating_type) 0.0);
        }
    }

    delete m;
}

TEST(MatrixTests, matrix_randomization)
{
    matrix* m = new matrix(400, 100);
    m->fill_rand();

    for(int rx = 0; rx < m->get_row_count(); rx++)
    {
        for(int cx = 0; cx < m->get_col_count(); cx++)
        {
            // this has a very very small chance to fail
            EXPECT_NE(m->get_value_at(rx, cx), (floating_type) 0.0);
        }
    }
    delete m;
}

TEST(MatrixTests, matrix_get_set)
{
    matrix* m = new matrix(400, 100);
    m->fill_rand();

    for(int rx = 0; rx < m->get_row_count(); rx++)
    {
        for(int cx = 0; cx < m->get_col_count(); cx++)
        {
            // get random value, change it and multiply it,
            // then get, set and test
            floating_type tmp = m->get_value_at(rx, cx);
            tmp = (tmp + (floating_type) 1)*(floating_type) 40;
            m->set_value_at(rx, cx, tmp);
            ASSERT_EQ(m->get_value_at(rx, cx), tmp);
        }
    }
    delete m;
}

TEST(MatrixTests, matrix_transpose)
{
    matrix* m = new matrix(400, 100);
    m->fill_rand();

    matrix* mt = m->new_from_transpose();

    ASSERT_EQ(m->get_row_count(), mt->get_col_count());
    ASSERT_EQ(m->get_col_count(), mt->get_row_count());

    for(int rx = 0; rx < m->get_row_count(); rx++)
    {
        for(int cx = 0; cx < m->get_col_count(); cx++)
        {
            ASSERT_EQ(m->get_value_at(rx, cx), mt->get_value_at(cx, rx));
        }
    }

    delete m;
    delete mt;
}

TEST(MatrixTests, matrix_dot_product)
{
    matrix* m1 = new matrix(400, 100);
    m1->fill_rand();
    matrix* m2 = new matrix(100, 400);
    m1->fill_rand();

    matrix* m3 = m1->new_from_multiply(m2);
    // make sure resulting matrix has the row number of the first term, and the
    // column number of the second term
    ASSERT_EQ(m3->get_row_count(), m1->get_row_count());
    ASSERT_EQ(m3->get_col_count(), m2->get_col_count());

    matrix* m4 = m2->new_from_multiply(m1);
    // make sure resulting matrix has the row number of the first term, and the
    // column number of the second term
    ASSERT_EQ(m4->get_row_count(), m2->get_row_count());
    ASSERT_EQ(m4->get_col_count(), m1->get_col_count());

    delete m1;
    delete m2;
    delete m3;
    delete m4;
}

TEST(MatrixTests, squash_to_vector)
{
    matrix* m = new matrix(400, 100);
    m->fill_rand();

    auto v = m->new_vector_from_squash();
    ASSERT_EQ(m->get_col_count()*m->get_row_count(), v->size());
}