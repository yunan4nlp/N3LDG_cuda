#include <iostream>
#include <vector>
#include <chrono>

//#include "cuda_v3/cuda_impl.h"
#include "N3LDG.h"


using namespace std;

void test(void) {
	cout << "----------" << endl;
	LDG::Tensor x1, x2, y1, y2, r1;
	DEV->random_uniform(x1, Shape({100, 10}), 2, 2);
	DEV->random_uniform(x2, Shape({100, 1}), 2, 2);
	DEV->random_uniform(y1, Shape({100, 1}), 2, 2);
	DEV->random_uniform(y2, Shape({100, 1}), 2, 2);

	int d_size = 200;
	vector<dtype> data(d_size);
	vector<int> index;
	index.push_back(0);
	index.push_back(9);
	for(int idx = 0; idx < d_size; idx++) {
		data[idx] = idx;
	}

	DEV->set_cols(x1, index, data);

	vector<dtype> r = DEV->to_vector(x1);
	for(int idx = 0; idx < r.size(); idx++)
	{
		cout << r[idx] << " ";
	}

}

void test_Dadd(void) {
	LDG::Tensor loss, in_loss;
	DEV->random_uniform(loss, Shape({200, 5}), 1, 1);
	DEV->random_uniform(in_loss, Shape({200, 1}), 1, 1);
	DEV->Dadd_inplace(in_loss, loss);
	vector<dtype> data;
	data = DEV->to_vector(in_loss);
	for(int idx = 0; idx < data.size(); idx++) {
		cout << data[idx] << " ";
	}
	cout << endl;
}

/*
void test_new_concat() {
	int n = 5;
	int count = 10;
	LDG::Tensor x[n * count];	
	LDG::Tensor r[count];
	vector<vector<LDG::PTensor> > vec_vec_x;
	vector<LDG::PTensor> vec_r;

	for(int idx = 0; idx < count; idx++) {
		vector<LDG::PTensor> vec_x;
		for(int idy = 0; idy < n; idy++)
		{
			int i = idx * n + idy;
			DEV->random_uniform(x[i], Shape({1 + idx, 1}), i, i);
			vec_x.push_back(&x[i]);
		}
		vec_vec_x.push_back(vec_x);
		vec_r.push_back(&r[idx]);
	}
	
	DEV->concat(vec_vec_x, vec_r);

	for(int idx = 0; idx < count; idx++) {
		vector<dtype> data = DEV->to_vector(r[idx]);
		cout << r[idx].shape().to_string() << endl;
		for(int idy = 0; idy < data.size(); idy++) {
			cout << data[idy] << " ";
		}
		cout << endl;
	}
	
}
*/

void test_to_vector() {
	int size = 150;
	LDG::Tensor x[size];
	vector<dtype> data[size];

	vector<LDG::PTensor> vec_x;
	vector<vector<dtype>* > vec_d;
	for(int idx = 0; idx < size; idx++) {
		DEV->random_uniform(x[idx], Shape({idx + 1, 1}), idx, idx);
		vec_x.push_back(&x[idx]);
		vec_d.push_back(&data[idx]);
	}
	DEV->to_vector(vec_x, vec_d);
	for(int idx = 0; idx < size; idx++) {
		int max_size = vec_d[idx]->size();
		for(int idy = 0; idy < max_size; idy++) {
			cout << (*(vec_d[idx]))[idy] << " ";
		}
		cout << endl;
	}
}

void test_drop() {
	int size = 100;
	dtype drop_rate = 0.5;
	LDG::Tensor x[size], y[size];
	vector<LDG::PTensor> vec_x, vec_y;
	IndexPtr mask;
	DtypePtr mask_val;
	for(int idx = 0; idx < size; idx++) {
		DEV->random_uniform(x[idx], Shape({50, 1}), idx + 1, idx + 1);
		DEV->random_uniform(y[idx], Shape({50, 1}), idx + 1, idx + 1);
		vec_x.push_back(&x[idx]);
		vec_y.push_back(&y[idx]);
	}
	DEV->Fdropout(vec_x, drop_rate, mask, vec_y);

	for(int idx = 0; idx < size; idx++) {
		vector<dtype> val = DEV->to_vector(y[idx]);
		for(int idy = 0; idy < val.size(); idy++) {
			cout << val[idy] << " ";
		}
		cout << endl;
	}
}

void test_random() {
	LDG::Tensor x[3];
	DEV->random_uniform(x[0], Shape({50, 1}), -1, 1);
	DEV->random_uniform(x[1], Shape({50, 1}), -2, 2);
	DEV->random_uniform(x[2], Shape({50, 1}), -0.1, 0.1);
	vector<LDG::PTensor> vec_x;
	for(int idx = 0; idx < 3; idx++) {
		vec_x.push_back(&x[idx]);
	}
	//DEV->set(vec_x, 10);
	DEV->Fadd(x[0], x[1], x[2]);
	vector<vector<dtype> *> vec_val;
	vector<dtype> val[3];
	for(int idx = 0; idx < 3; idx++) {
		vec_val.push_back(&val[idx]);
	}
	DEV->to_vector(vec_x, vec_val);
	for(int idx = 0; idx < 3; idx++) {
		for(int idy = 0; idy < vec_val[idx]->size(); idy++) {
			cout << (*vec_val[idx])[idy] << " ";
		}
		cout << endl;
	}
}

void test_fadd() {
	LDG::Tensor x[3];
	DEV->random_uniform(x[0], Shape({50, 1}), 1, 1);
	DEV->random_uniform(x[1], Shape({50, 1}), 2, 2);
	DEV->random_uniform(x[2], Shape({50, 1}), 3, 3);
	vector<LDG::PTensor> vec_x;
	for(int idx = 0; idx < 3; idx++) {
		vec_x.push_back(&x[idx]);
	}
	vector<vector<LDG::PTensor> > vec_vec_x;
	vec_vec_x.push_back(vec_x);

	LDG::Tensor y;
	DEV->random_uniform(y, Shape({50, 1}), -1, 1);
	vector<LDG::PTensor> vec_y;
	vec_y.push_back(&y);
	DEV->Fadd(vec_vec_x, vec_y);
	
	vector<dtype> data = DEV->to_vector(y);
	for(int idx = 0; idx < data.size(); idx++) {
		cout << data[idx] << " ";
	}
	cout << endl;
} void test_set_col() {
	LDG::Tensor x;
	DEV->random_uniform(x, Shape({10, 3}), -0.1, 0.1);
	DEV->set_col(x, 0, 2);
}

void test_get_col() {
	LDG::Tensor x, y;
	DEV->random_uniform(x, Shape({10, 3}), -0.1, 0.1);
	//DEV->random_uniform(y, Shape({10, 1}), -0.1, 0.1);
	DEV->get_col(x, 0, y);
	vector<dtype> x_val = DEV->to_vector(x);
	for(int idx = 0; idx < x_val.size(); idx++)
		cout << x_val[idx] << " ";
	cout << endl;
	vector<dtype> y_val = DEV->to_vector(y);
	for(int idx = 0; idx < y_val.size(); idx++)
		cout << y_val[idx] << " ";
	cout << endl;
}

void test_get_cols() {
	LDG::Tensor x, y;
	DEV->random_uniform(x, Shape({10, 5}), -1, 1);
	//DEV->random_uniform(y, Shape({10, 1}), -0.1, 0.1);
	vector<int> id;
	id.push_back(1);
	id.push_back(0); 
	vector<dtype> re1 = DEV->to_vector(x);
	for(int idx = 0; idx < re1.size(); idx++)
		cout << re1[idx] << " ";
	cout << endl;

	vector<dtype> re2 = DEV->to_vector(x, id);
	for(int idx = 0; idx < re2.size(); idx++)
		cout << re2[idx] << " ";
	cout << endl;
}

void test_FLookup() {
	int dim0 = 10;
	int dim1 = 5;
	int size = dim0 * dim1;
	LDG::Tensor x;
	LDG::Tensor y[3];
	DEV->malloc(x, Shape({dim0, dim1}));
	DEV->malloc(y[0], Shape({dim0, 1}));
	DEV->malloc(y[1], Shape({dim0, 1}));
	DEV->malloc(y[2], Shape({dim0, 1}));
	vector<dtype> val(size);
	for(int idx = 0; idx < size; idx++) {
		val[idx] = idx;	
	}
	DEV->set(x, val);
	int num = 3;
	int id[num] = {4, 3, 2};
	vector<LDG::PTensor> vec_y;
	vec_y.push_back(&y[0]);
	vec_y.push_back(&y[1]);
	vec_y.push_back(&y[2]);

	DEV->FLookup(x, id, num, vec_y);
	for(int idx = 0; idx < num; idx++) {
		vector<dtype> data = DEV->to_vector(*vec_y[idx]);
		for(int idy = 0; idy < dim0; idy++) {
			cout << data[idy] << " ";
		}
		cout << endl;
	}
}

void test_tanh() {
	int dim0 = 10;
	int dim1 = 5;
	int size = dim0 * dim1;
	LDG::Tensor x, y;
	vector<dtype> val(size);
	for(int idx = 0; idx < size; idx++) {
		val[idx] = (double) idx / 10;	
	}
	DEV->malloc(x, Shape({dim0, dim1}));
	DEV->set(x, val);
	DEV->Ftanh(x, y);

	vector<dtype> x_val = DEV->to_vector(x);
	for(int idx = 0; idx < size; idx++) {
		cout << x_val[idx] << " ";
	}
	cout << endl;

	vector<dtype> y_val = DEV->to_vector(y);
	for(int idx = 0; idx < size; idx++) {
		cout << y_val[idx] << " ";
	}
	cout << endl;
}

void test_mat() {
	LDG::Tensor y[3];
	DEV->random_uniform(y[0], Shape({10, 1}), 10, 10);
	DEV->random_uniform(y[1], Shape({10, 1}), 2, 2);
	DEV->random_uniform(y[2], Shape({10, 1}), 1, 1);

	vector<LDG::PTensor> vec_y;
	vec_y.push_back(&y[0]);
	vec_y.push_back(&y[1]);
	vec_y.push_back(&y[2]);

	LDG::Tensor x[3];
	vector<LDG::PTensor> vec_x;
	vec_x.push_back(&x[0]);
	vec_x.push_back(&x[1]);
	vec_x.push_back(&x[2]);

	DEV->Ftanh(vec_y, vec_x);

	for(int i = 0; i < 3; i++) {
		vector<dtype> x_val = DEV->to_vector(x[i]);
		for(int idx = 0; idx < x_val.size(); idx++) {
			cout << x_val[idx] << " ";
		}
		cout << endl;
	}
}

void test_concat() {
	LDG::Tensor x[3];
	DEV->random_uniform(x[0], Shape({50, 1}), 1, 1);
	DEV->random_uniform(x[1], Shape({50, 1}), 2, 2);
	DEV->random_uniform(x[2], Shape({50, 1}), 3, 3);
	vector<LDG::PTensor> vec_x;
	for(int idx = 0; idx < 3; idx++) {
		vec_x.push_back(&x[idx]);
	}

	LDG::Tensor y;
	DEV->concat(vec_x, y);
	vector<dtype> val = DEV->to_vector(y);
	for(int idx = 0; idx < val.size(); idx++)
		cout << val[idx] << " ";
	cout << endl;
}

void test_unconcat() {
	LDG::Tensor x[3];
	DEV->random_uniform(x[0], Shape({50, 1}), 1, 1);
	DEV->random_uniform(x[1], Shape({50, 1}), 2, 2);
	DEV->random_uniform(x[2], Shape({50, 1}), 3, 3);
	vector<LDG::PTensor> vec_x;
	for(int idx = 0; idx < 3; idx++) {
		vec_x.push_back(&x[idx]);
	}

	LDG::Tensor y;
	DEV->random_uniform(y, Shape({50, 3}), 3, 3);
	DEV->unconcat(y, vec_x);
	vector<dtype> val = DEV->to_vector(x[0]);
	for(int idx = 0; idx < val.size(); idx++)
		cout << val[idx] << " ";
	cout << endl;
}

void test_matmaul() {
	LDG::Tensor x, y, r;
	DEV->random_uniform(x, Shape({10, 1}), 1, 1);
	DEV->random_uniform(y, Shape({1, 10}), 1, 1);

	DEV->Fmatmul(x, y, r, true, true);
	vector<dtype> val = DEV->to_vector(r);
	for(int idx = 0; idx < val.size(); idx++)
		cout << val[idx] << " ";
	cout << endl;
}

void test_mul() {
	LDG::Tensor x, y, r;
	DEV->random_uniform(x, Shape({1, 10}), 2, 2);
	DEV->random_uniform(y, Shape({1, 10}), 2, 2);

	DEV->Fmultiply(x, y, r);
	vector<dtype> val = DEV->to_vector(r);
	for(int idx = 0; idx < val.size(); idx++)
		cout << val[idx] << " ";
	cout << endl;
}

void test_add_scalar() {
	LDG::Tensor x, y, r;
	DEV->random_uniform(x, Shape({1, 10}), 2, 2);

	DEV->Fadd_scalar(x, 10, r);
	vector<dtype> val = DEV->to_vector(r);
	for(int idx = 0; idx < val.size(); idx++)
		cout << val[idx] << " ";
	cout << endl;
}

void test_add_scalar_inplace() {
	LDG::Tensor x, y, r;
	DEV->random_uniform(x, Shape({1, 10}), 2, 2);

	DEV->Fadd_scalar_inplace(x, 10);
	vector<dtype> val = DEV->to_vector(x);
	for(int idx = 0; idx < val.size(); idx++)
		cout << val[idx] << " ";
	cout << endl;
}

void test_fmultiply_scalar() {
	LDG::Tensor x, y, r;
	DEV->random_uniform(x, Shape({10, 2}), 2, 2);
	DEV->random_uniform(y, Shape({1, 1}), 3, 3);
	DEV->Fmultiply_scalar(x, y, r);

	vector<dtype> val = DEV->to_vector(r);
	for(int idx = 0; idx < val.size(); idx++)
		cout << val[idx] << " ";
	cout << endl;
}

void test_add_col() {
	LDG::Tensor x, y;
	DEV->random_uniform(x, Shape({10, 5}), 1, 1);
	DEV->random_uniform(y, Shape({10, 1}), 2, 2);
	DEV->Fadd_col(x, y, 4);
	vector<dtype> val = DEV->to_vector(x);
	for(int idx = 0; idx < val.size(); idx++)
		cout << val[idx] << " ";
	cout << endl;
}

void test_Fconcat() {
	vector<vector<LDG::PTensor> > vec_vec_val;
	vector<LDG::PTensor> vec_r;
	LDG::Tensor x[100];
	LDG::Tensor r[10];
	for(int idx = 0; idx < 10; idx++) {
		int dim = 0;
		vector<LDG::PTensor> vec_val;
		for(int idy = 0; idy < 10; idy++) {
			DEV->random_uniform(x[idx * 10 + idy], Shape({idy + 1 , 1}), idx + idy + 1, idx + idy + 1);
			vec_val.push_back(&x[idx * 10 + idy]);
			dim += (idy + 1);
		}
		DEV->malloc(r[idx], Shape({55, 1}));
		vec_r.push_back(&r[idx]);
		vec_vec_val.push_back(vec_val);
	}

	DEV->Fconcat(vec_vec_val, vec_r);

	for(int idx = 0; idx < 10; idx++) {
		vector<dtype> data = DEV->to_vector(r[idx]);
		for(int idx = 0; idx < data.size(); idx++)
			cout << data[idx] << " ";
		cout << endl;
	}

}

void test_fdropout() {
	LDG::Tensor x[3];
	DEV->random_uniform(x[0], Shape({50, 1}), 1, 1);
	DEV->random_uniform(x[1], Shape({50, 1}), 2, 2);
	DEV->random_uniform(x[2], Shape({50, 1}), 3, 3);
	vector<LDG::PTensor> vec_x;
	for(int idx = 0; idx < 3; idx++) {
		vec_x.push_back(&x[idx]);
	}

	LDG::Tensor y[3];
	DEV->random_uniform(y[0], Shape({50, 1}), 1, 1);
	DEV->random_uniform(y[1], Shape({50, 1}), 2, 2);
	DEV->random_uniform(y[2], Shape({50, 1}), 3, 3);
	vector<LDG::PTensor> vec_y;
	for(int idx = 0; idx < 3; idx++) {
		vec_y.push_back(&y[idx]);
	}
	
	IndexPtr mask;
	DEV->Fdropout(vec_x, 0.1, vec_y);

	for(int idx = 0; idx < 3; idx++) {
		vector<dtype> data = DEV->to_vector(*vec_y[idx]);
		for(int idy = 0; idy < data.size(); idy++)
			cout << data[idy] << " ";
		cout << endl;
	}
}

void test_avg() {
	LDG::Tensor x[3];
	DEV->random_uniform(x[0], Shape({50, 1}), -1, 1);
	DEV->random_uniform(x[1], Shape({50, 1}), -2, 2);
	DEV->random_uniform(x[2], Shape({50, 1}), -3, 3);
	vector<LDG::PTensor> vec_x;
	for(int idx = 0; idx < 3; idx++) {
		vec_x.push_back(&x[idx]);
	}

	LDG::Tensor y;
	DEV->random_uniform(y, Shape({50, 1}), 1, 1);
	DEV->FAvgPooling(vec_x, y);

	vector<dtype> val = DEV->to_vector(y);
	for(int idx = 0; idx < val.size(); idx++)
		cout << val[idx] << " ";
	cout << endl;
}

void test_max() {
	LDG::Tensor x[3];
	DEV->random_uniform(x[0], Shape({10, 1}), -1, 1);
	DEV->random_uniform(x[1], Shape({10, 1}), -2, 2);
	DEV->random_uniform(x[2], Shape({10, 1}), -3, 3);
	vector<LDG::PTensor> vec_x;
	for(int idx = 0; idx < 3; idx++) {
		vec_x.push_back(&x[idx]);
	}

	LDG::Tensor y;
	DEV->random_uniform(y, Shape({10, 1}), 1, 1);
	IndexPtr mask;
	DEV->init_index_ptr(mask, 10);
	DEV->FMaxPooling(vec_x, y, mask.get_ptr());

	for(int i = 0; i < 3; i++) {
		vector<dtype> val = DEV->to_vector(x[i]);
		for(int idx = 0; idx < 10; idx++)
			cout << val[idx] << ",";
		cout << endl;
	}

	vector<dtype> val = DEV->to_vector(y);
	for(int idx = 0; idx < 10; idx++)
		cout << val[idx] << " ";
	cout << endl;

	for(int idx = 0; idx < 10; idx++)
		cout << mask.get_ptr()[idx] << " ";
	cout << endl;
}

void test_min() {
	LDG::Tensor x[3];
	DEV->random_uniform(x[0], Shape({10, 1}), -1, 1);
	DEV->random_uniform(x[1], Shape({10, 1}), -2, 2);
	DEV->random_uniform(x[2], Shape({10, 1}), -3, 3);
	vector<LDG::PTensor> vec_x;
	for(int idx = 0; idx < 3; idx++) {
		vec_x.push_back(&x[idx]);
	}

	LDG::Tensor y;
	DEV->random_uniform(y, Shape({10, 1}), 1, 1);
	IndexPtr mask;
	DEV->init_index_ptr(mask, 10);
	DEV->FMinPooling(vec_x, y, mask.get_ptr());

	for(int i = 0; i < 3; i++) {
		vector<dtype> val = DEV->to_vector(x[i]);
		for(int idx = 0; idx < 10; idx++)
			cout << val[idx] << ",";
		cout << endl;
	}

	vector<dtype> val = DEV->to_vector(y);
	for(int idx = 0; idx < 10; idx++)
		cout << val[idx] << " ";
	cout << endl;

	for(int idx = 0; idx < 10; idx++)
		cout << mask.get_ptr()[idx] << " ";
	cout << endl;
}

int main(void) {
	test_min();
	return 0;
}
