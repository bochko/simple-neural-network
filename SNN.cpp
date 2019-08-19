// SNN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
// clib
#include <iostream>
#include <cmath>
#include <random>
#include <string>
#include <sstream>

// stl
#include <vector>

#define FPOINT				double
#define LDIM				int
#define INPUT_LAYER_IDX		0 // always 0
#define FUNKY_DECORATOR		"***"

// implements a neuron building block containing three values:
// the raw input value, the fast sigmoid result, and the derivative
// of a fast sigmoid
class neuron
{
public:

	neuron(FPOINT val)
	{
		this->raw = val;
		// activate via fast sigmoid
		this->calc_fs();
		// calculate fast sigmoid derivative
		this->calc_fsd();
	}
	// activation function
	// f(x) = x / (1 + abs(x))
	void calc_fs()
	{
		this->fs = this->raw / (1.0f + std::abs(this->raw));
		// 1 / std::abs(x) can never be 0 so no need to check on division
	}

	// derivative of fast sigmoid function
	// f'(x) = f(x) * (1 - f(x))
	void calc_fsd()
	{
		this->fsd = this->fs * (1.0f - this->fs);
	}

	// get/set
	FPOINT get_raw() { return this->raw; }
	FPOINT get_fs() { return this->fs; }
	FPOINT get_fsd() { return this->fsd; }
	void set_raw(FPOINT new_raw) 
	{
		this->raw = new_raw;
		this->calc_fs();
		this->calc_fsd();
	}
private:
	// raw
	FPOINT raw;
	// linearized
	FPOINT fs;
	// derivative
	FPOINT fsd;
};

class matrix
{
public:
	matrix(LDIM nrow, LDIM ncol)
	{
		for (LDIM row_idx = 0; row_idx < nrow; row_idx++)
		{
			std::vector<FPOINT> column_matrix_storage; // inner
			for (LDIM col_idx = 0; col_idx < ncol; col_idx++)
			{
				column_matrix_storage.push_back(0.0f);
			}
			this->values.push_back(column_matrix_storage);
		}
	}

	~matrix()
	{
		// no dynamic allocations
	}

	void set_val(LDIM row, LDIM col, FPOINT value)
	{
		values.at(row).at(col) = value;
	}

	FPOINT get_val(LDIM row, LDIM col)
	{
		return values.at(row).at(col);
	}

	LDIM row_count()
	{
		// assume first dimension is rows
		return (LDIM)  values.size();
	}

	LDIM col_count()
	{
		// since vector of vectors, and all interior vectors
		// are identical in size, we can use the first one to get the
		// length of the second dimension
		return (LDIM) values.at(0).size();
	}

	// relies constructor has been used and valid
	void fill_rand()
	{
		std::random_device rdev;
		std::mt19937 rng(rdev());
		std::uniform_real_distribution<FPOINT> distribution(0.0f, 1.0f);
		LDIM row_size = this->row_count();
		LDIM col_size = this->col_count();

		for (LDIM row_idx = 0; row_idx < row_size; row_idx++)
		{
			for (LDIM col_idx = 0; col_idx < col_size; col_idx++)
			{
				this->values.at(row_idx).at(col_idx) = distribution(rng);
			}
		}
	}

	// you must manually dispose of the object created by this function
	matrix* transpose(void)
	{
		matrix* transposed = new matrix(this->col_count(), this->row_count());
		LDIM row_size = transposed->row_count();
		LDIM col_size = transposed->col_count();
		for (LDIM row_idx = 0; row_idx < row_size; row_idx++)
		{
			for (LDIM col_idx = 0; col_idx < col_size; col_idx++)
			{
				transposed->set_val(row_idx, col_idx, this->get_val(col_idx, row_idx));
			}
		}

		return transposed;
	}

	matrix* mul(matrix*& mmul)
	{
		if (this->col_count() != mmul->row_count())
		{
			std::cerr << "origin col size " << this->col_count() << " != mmul row size " << mmul->row_count() << std::endl;
			std::abort();
			return nullptr;
		}
		matrix* ret = new matrix(this->row_count(), mmul->col_count());
		for (LDIM orow_idx = 0; orow_idx < this->row_count(); orow_idx++)
		{
			for (LDIM mulcol_idx = 0; mulcol_idx < mmul->col_count(); mulcol_idx++)
			{
				for (LDIM mulrow_idx = 0; mulrow_idx < mmul->row_count(); mulrow_idx++)
				{
					FPOINT product = 
						this->get_val(orow_idx, mulrow_idx) * mmul->get_val(mulrow_idx, mulcol_idx);
					// add onto the value in the result matrix
					ret->set_val(orow_idx, mulcol_idx, (ret->get_val(orow_idx, mulcol_idx) + product));
				}
			}
		}

		return ret;
	}

	// squash the matrix to a linear vector (1 dimension)
	std::vector<FPOINT>* get_vtor()
	{
		std::vector<FPOINT> * vtor = new std::vector<FPOINT>();
		// fill in the newly created vector
		for (LDIM row_idx = 0; row_idx < this->row_count(); row_idx++)
		{
			for (LDIM col_idx = 0; col_idx < this->col_count(); col_idx++)
			{
				vtor->push_back(this->get_val(row_idx, col_idx));
			}
		}
		return vtor;
	}

	std::string get_str()
	{
		std::stringstream sstream;
		LDIM row_size = this->row_count();
		LDIM col_size = this->col_count();

		for (LDIM row_idx = 0; row_idx < row_size; row_idx++)
		{
			for (LDIM col_idx = 0; col_idx < col_size; col_idx++)
			{
				sstream << this->values.at(row_idx).at(col_idx)
					<< " \t";
			}
			sstream << "\r\n";
		}
		return sstream.str();
	}

private:
	// outer vector is rows, inner vector is columns
	std::vector<std::vector<FPOINT>> values;
};

class layer
{
public:
	// constructs neurons in a layer
	layer(LDIM size)
	{
		for (LDIM idx = 0; idx < size; idx++)
		{
			// create and allocate the neurons
			neuron* n = new neuron(0.0f);
			this->neurons.push_back(n);
		}
	}

	// destroys all neurons in a layer
	~layer()
	{
		for (LDIM idx = 0; idx < (LDIM) this->neurons.size(); idx++)
		{
			if (this->neurons.at(idx) != nullptr)
			{
				delete this->neurons.at(idx);
			}
		}
	}

	// assigns input raw value to neuron
	void set_val(LDIM at, FPOINT val)
	{
		this->neurons.at(at)->set_raw(val);
	}

	// creates a matrix representation of the neuron layer raw values
	matrix* get_raw_matrix()
	{
		matrix* m = new matrix(1 /*always one dimension*/, 
			this->neurons.size());

		for (LDIM idx = 0; idx < (LDIM)this->neurons.size(); idx++)
		{
			m->set_val(0, idx, this->neurons.at(idx)->get_raw());
		}

		return m;
	}

	// fast sigmoid calculated values of neurons in layer
	matrix* get_fs_matrix()
	{
		matrix* m = new matrix(1 /*always one dimension*/,
			this->neurons.size());

		for (LDIM idx = 0; idx < (LDIM)this->neurons.size(); idx++)
		{
			m->set_val(0, idx, this->neurons.at(idx)->get_fs());
		}

		return m;
	}

	LDIM get_size()
	{
		return this->neurons.size();
	}

	// fast sigmoid derivative values of neurons in layer
	matrix* get_fsd_matrix()
	{
		matrix* m = new matrix(1 /*always one dimension*/,
			this->neurons.size());

		for (LDIM idx = 0; idx < (LDIM)this->neurons.size(); idx++)
		{
			m->set_val(0, idx, this->neurons.at(idx)->get_fsd());
		}

		return m;
	}
private:
	std::vector<neuron*> neurons;
};


class network
{
public:
	typedef std::vector<LDIM> topology_vector;

	network(topology_vector topology)
	{
		this->topology = topology;
		for (LDIM top_idx = 0; top_idx < (LDIM)topology.size(); top_idx++)
		{
			layer* l = new layer(topology.at(top_idx));
			this->layers.push_back(l);
		}
		// there's topology size - 1 weight matrices
		for (LDIM top_idx = 0; top_idx < (LDIM)topology.size() - 1; top_idx++)
		{
			// in the weight matrix number of rows is the number of input neurons
			// and the number of columns is the number of feed-into output neurons.
			matrix* m = new matrix(topology.at(top_idx), topology.at(top_idx + 1));
			m->fill_rand(); // initial values are random
			this->weights.push_back(m);
		}
		
		// output errors vector must be as large as the last layer
		for (LDIM evtor_idx = 0; 
			evtor_idx < (this->topology.at(this->topology.size() - 1));
			evtor_idx++)
		{
			err_output.push_back(0.0f);
		}
	}

	// destroy all dynamically allocated instances
	~network()
	{
		for (LDIM layer_idx = 0; layer_idx < (LDIM)this->layers .size(); layer_idx++)
		{
			if (this->layers.at(layer_idx) != nullptr)
			{
				delete this->layers.at(layer_idx);
			}
		}

		for (LDIM wmatrix_idx = 0; wmatrix_idx < (LDIM)this->layers.size(); wmatrix_idx++)
		{
			if (this->weights.at(wmatrix_idx) != nullptr)
			{
				delete this->weights.at(wmatrix_idx);
			}
		}
	}

	void set_input(std::vector<FPOINT> input)
	{
		for (LDIM idx = 0; idx < (LDIM)input.size(); idx++)
		{
			this->layers.at(INPUT_LAYER_IDX)->set_val(idx, input.at(idx));
		}
		this->input = input;
	}

	void feed_forward()
	{
		// do this for each layer in the network
		for (LDIM layer_idx = 0; layer_idx < (LDIM)this->layers.size() - 1; layer_idx++)
		{
			matrix* neuron_matrix;
			matrix* weight_matrix;
			matrix* product_matrix;
			
			if (layer_idx == INPUT_LAYER_IDX)
			{
				// get the raw input values fed into the neuron
				neuron_matrix = this->layers.at(layer_idx)->get_raw_matrix();
			}
			else
			{
				// get the calculated fast sigmoids
				neuron_matrix = this->layers.at(layer_idx)->get_fs_matrix();
			}

			weight_matrix = this->weights.at(layer_idx);
			// get the product of the input (or left) neuron matrix 
			// and weight matrix following it
			product_matrix = neuron_matrix->mul(weight_matrix);
			// potential optimisation here, no real need for intermediate vector?
			std::vector<FPOINT>* product_vals = product_matrix->get_vtor();
			// feed the next layer
			for (LDIM next_neuron = 0; next_neuron < (LDIM)product_vals->size(); next_neuron++)
			{
				// assign as input to next layer
				this->layers.at(layer_idx + 1)->set_val(next_neuron, product_vals->at(next_neuron));
			}
		}
	}

	std::string get_str()
	{
		std::stringstream sstream;

		for (LDIM layer_idx = 0; layer_idx < (LDIM)this->layers.size(); layer_idx++)
		{
			if (layer_idx == INPUT_LAYER_IDX)
			{
				// if input layer (leftmost) - print raw values
				matrix* m = this->layers.at(layer_idx)->get_raw_matrix();
				sstream << "Network input layer: " << std::endl;
				sstream << m->get_str();
			}
			else
			{
				// print fast sigmoid calculated value otherwise
				matrix* m = this->layers.at(layer_idx)->get_fs_matrix();
				sstream << "Network Layer: " << std::endl;
				sstream << m->get_str();
			}
			if (layer_idx < (LDIM)this->weights.size())
			{
				sstream << FUNKY_DECORATOR << std::endl;
				sstream << this->weights.at(layer_idx)->get_str() << std::endl;
			}
		}
		return sstream.str();
	}

	void calc_err(std::vector<FPOINT> targets)
	{
		// compare output layer size (last index) to targets size
		// and abort if mismatched
		LDIM output_layer_size =
			this->layers.at(this->layers.size() - 1)->get_fs_matrix()->col_count();
		if (output_layer_size
			!=
			targets.size())
		{
			std::cerr << "Output layer size " << output_layer_size <<
				" != Targets size " << targets.size();
			abort();
		}

		for (LDIM idx = 0; idx < (LDIM)targets.size(); idx++)
		{
			FPOINT err =
				this->layers.at(this->layers.size() - 1)->get_fs_matrix()->get_val(0, idx);
			err -= targets.at(idx);
			err_output.at(idx) = err;
		}

		err_historical.push_back(get_err_total());
	}

	FPOINT get_err_total()
	{
		FPOINT total_error = 0.0f;
		for (LDIM evtor_idx = 0; evtor_idx < (LDIM)this->err_output.size(); evtor_idx++)
		{
			// taking the absolute value to make sure
			// negative and positive errors don't cancel each other

			// should we square each member to promote big values?
			total_error += std::abs(this->err_output.at(evtor_idx));
		}
		return total_error;
	}

	std::vector<FPOINT> get_err_output()
	{
		return this->err_output;
	}

	std::vector<FPOINT> get_err_historical()
	{
		return this->err_historical;
	}

	void bprop()
	{
		// from the output layer to the hidden layers
		LDIM ol_index = this->layers.size() - 1; // output layer index
		matrix* output_to_hidden_dfs = this->layers.at(ol_index)->get_fsd_matrix();
		// construct matrix with dimension 1 and the length of neurons in output layer
		matrix* output_to_hidden_grads = 
			new matrix(1, (this->layers.at(ol_index)->get_size()));

		for (LDIM err_idx = 0; err_idx < (LDIM)this->err_output.size(); err_idx++)
		{
			// always row zero because single dimensional
			double dfsval = output_to_hidden_dfs->get_val(0, err_idx);
			double error = this->err_output[err_idx];
			double gradient = dfsval * error;

			// fill in our matrix of output to hidden error gradients
			// always row zero because single dimensional
			output_to_hidden_grads->set_val(0, err_idx, gradient);
		}

		LDIM last_hidden_layer_index = ol_index - 1U;
		// get the outer-most hidden layer
		layer* last_hidden_layer = this->layers.at(last_hidden_layer_index);
		// get the weights between the hidden layer and the output layer
		matrix* weights_output_to_hidden = this->weights.at(last_hidden_layer_index);
		// transposed gradients
		matrix* transposed_gradients = output_to_hidden_grads->transpose();
		// product of transposed gradients and activated (fs) values
		matrix* matrix_fs_values = last_hidden_layer->get_fs_matrix();
		matrix* delta_weights_output_to_hidden = 
			transposed_gradients->mul(matrix_fs_values);

		// now compute for each of the other layers
		// backwards because it makes logical sense
		// USING AN UNSIGNED VALUE AS LAYER_IDX CAUSES LOOP TO LOOP FOREVER BECAUSE OF OVERFLOW
		// WHEN SUBTRACT FROM 0 at layer_idx = 0U
		for (LDIM layer_idx = (ol_index - 1U); layer_idx > 0; layer_idx--)
		{
			// we are computing the difference that each weight makes

		}

	}

private:
	topology_vector topology;
	std::vector<layer*> layers;
	std::vector<matrix*> weights;
	std::vector<FPOINT> input;
	std::vector<FPOINT> err_output;
	std::vector<FPOINT> err_historical;
};

int main()
{
	// NEURON CREATED
	neuron *n = new neuron(1.5f);
	std::cout << "Neuron constructor test:" << std::endl;
	std::cout << "raw: " << n->get_raw() << std::endl;
	std::cout << "fs: " << n->get_fs() << std::endl;
	std::cout << "dfs: " << n->get_fsd() << std::endl << std::endl;

	// MATRIX CREATED
	matrix *m = new matrix(3, 2);
	std::cout << "Matrix constructor test:" << std::endl;
	std::cout << m->get_str() << std::endl;

	std::cout << "Matrix randomization test:" << std::endl;
	m->fill_rand();
	std::cout << m->get_str() << std::endl;

	std::cout << "Matrix transpose test:" << std::endl;
	matrix* mnew = m->transpose();
	std::cout << mnew->get_str() << std::endl;

	std::cout << "Test neural network class:" << std::endl;
	network::topology_vector topology = { 3, 2, 3 };
	network *nn = new network(topology);
	nn->set_input(std::vector<FPOINT>{0.7f, 3.1f, 5.0f});
	std::cout << nn->get_str() << std::endl;

	std::cout << "Test matrix multiplication:" << std::endl;
	matrix* mmul1 = new matrix(1, 4);
	mmul1->fill_rand();
	matrix* mmul2 = new matrix(4, 2);
	mmul2->fill_rand();
	matrix* mmulr = mmul1->mul(mmul2);
	std::cout << "Matrix 1: \n" << mmul1->get_str() << std::endl;
	std::cout << "Matrix 2: \n" << mmul2->get_str() << std::endl;
	std::cout << "Matrix Product: \n" << mmulr->get_str() << std::endl;

	std::cout << "Test feeding forward:" << std::endl;
	network::topology_vector big_topology = { 3, 2, 1 };
	network* real_neural_network = new network(big_topology);

	// setting input test (only after defining topology)
	real_neural_network->set_input(std::vector<FPOINT>{
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
	real_neural_network->calc_err(std::vector<FPOINT>{0.75});
	std::cout << real_neural_network->get_err_total() << std::endl;

	real_neural_network->bprop();

	return 0;
}