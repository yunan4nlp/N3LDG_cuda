#include <iostream>
#include <vector>
#include "cuda/cuda_impl.h"

using namespace std;


int main(void) {
	CudaDevice cu;
	cout << "----------" << endl;
	LDG::Tensor x,y,z, gx, gy;
	cu.random_uniform(x, Shape({10, 1}), -100, 100);
	
	cu.random_uniform(y, Shape({10, 1}), -0.01, 0.01);

	cu.random_uniform(z, Shape({1, 1}), -0.01, 0.01);

	//cu.show_val(y);
	//cu.show_val(gx);
	//cu.show_val(gy);
	int index[10];
	dtype data[10];
	for(int idx = 0; idx < 10; idx++) {
		data[idx] = idx ;
	}
	cu.set(x, data, 10);
	cu.set(y, data, 10);
	cu.show_val(x);
	cu.show_val(y);
	cu.Fmatmul(x, y, z, true, false);
	//cout << "=============" << endl;
	//for(int idx = 0; idx < 200; idx++)
		//cout << index[idx] << ", ";
	//cout << endl;

	//cu.show_val(x);
	cu.show_val(z);
	//cu.show_val(gx);
	//cu.show_val(gy);

	//cout << "=============" << endl;
	//cu.DMaxPooling(x, y, gy, gx, index);
	//cu.show_val(x);
	//cu.show_val(y);
	//cu.show_val(gx);
	//cu.show_val(gy);

	return 0;
}
