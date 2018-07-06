/*
 * Param.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef PARAM_H_
#define PARAM_H_

#include "Eigen/Dense"
#include "BaseParam.h"
#include "cuda/cuda_impl.h"


// Notice: aux is an auxiliary variable to help parameter updating
class Param : public BaseParam {
  public:
	  LDG::Tensor aux_square;
	  LDG::Tensor aux_mean;
    int iter;

    // allow sparse and dense parameters have different parameter initialization methods
    inline void initial(int outDim, int inDim) {
        //val.init(outDim, inDim);
        //grad.init(outDim, inDim);
        //aux_square.init(outDim, inDim);
        //aux_mean.init(outDim, inDim);
		//device.malloc(val, Shape({outDim, inDim}));
		device.init(grad, Shape({outDim, inDim}));
		device.init(aux_square, Shape({outDim, inDim}));
		device.init(aux_mean, Shape({outDim, inDim}));

        dtype bound = sqrt(6.0 / (outDim + inDim + 1));
        //val.random(bound);
		device.random_uniform(val, Shape({outDim, inDim}), -bound, bound);
        iter = 0;
    }

    inline int outDim() {
        //return val.row;
        return val.shape().dims()[0];
    }

    inline int inDim() {
        //return val.col;
        return val.shape().dims()[1];
    }

    inline void clearGrad() {
        //grad.zero();
		device.zero(grad);
    }

    inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
        if (outDim() > 1 && inDim() > 1) {
			LDG::Tensor v_r;
			device.init(v_r, val.shape()); 
			device.Fmultiply_scalar(val, reg, v_r);
			device.Fadd(grad, v_r, grad);
			//grad.vec() = grad.vec() + val.vec() * reg;
		}
		LDG::Tensor grad_square;
		device.init(grad_square, grad.shape());
		device.Fsquare(grad, grad_square);
		device.Fadd(aux_square, grad_square, aux_square);
        //aux_square.vec() = aux_square.vec() + grad.vec().square();

		LDG::Tensor aux_eps;
		device.init(aux_eps, aux_square.shape());
		device.Fadd_scalar(aux_square, eps, aux_eps);
		LDG::Tensor aux_sqrt;
		device.init(aux_sqrt, aux_square.shape());
		device.Fsqrt(aux_eps, aux_sqrt);
		
		LDG::Tensor grad_alpha;
		device.init(grad_alpha, grad.shape());
		device.Fmultiply_scalar(grad, alpha, grad_alpha);

		LDG::Tensor grad_aux;
		device.init(grad_aux, grad.shape());
		device.Fdivide(grad_alpha, aux_sqrt, grad_aux);
		

		device.Fsubtract(val, grad_aux, val);
        //val.vec() = val.vec() - grad.vec() * alpha / (aux_square.vec() + eps).sqrt();
    }

    inline void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
		/*
        if (val.col > 1 && val.row > 1)grad.vec() = grad.vec() + val.vec() * reg;
        aux_mean.vec() = belta1 * aux_mean.vec() + (1 - belta1) * grad.vec();
        aux_square.vec() = belta2 * aux_square.vec() + (1 - belta2) * grad.vec().square();
        dtype lr_t = alpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));
        val.vec() = val.vec() - aux_mean.vec() * lr_t / (aux_square.vec() + eps).sqrt();
        iter++;
		*/
    }

    inline void randpoint(int& idx, int &idy) {
        //select indexes randomly
        std::vector<int> idRows, idCols;
        idRows.clear();
        idCols.clear();
		int dim0 = val.shape().dims()[0];
		int dim1 = val.shape().dims()[1];
        for (int i = 0; i < dim0; i++)
            idRows.push_back(i);
        for (int i = 0; i < dim1; i++)
            idCols.push_back(i);

        random_shuffle(idRows.begin(), idRows.end());
        random_shuffle(idCols.begin(), idCols.end());

        idy = idRows[0];
        idx = idCols[0];
    }

    inline dtype squareGradNorm() {
        dtype sumNorm = 0.0;
		LDG::Tensor cpu_grad;
		cpu_grad.shape_ = grad.shape();
		int size = grad.shape().size();
		cpu_grad.v = new dtype[size];
		device.to_cpu(grad, cpu_grad);
        for (int i = 0; i < size; i++) {
            sumNorm += cpu_grad.v[i] * cpu_grad.v[i];
        }
        return sumNorm;
    }

    inline void rescaleGrad(dtype scale) {
        //grad.vec() = grad.vec() * scale;
		device.Fmultiply_scalar(grad, scale, grad);
    }

    inline void save(std::ofstream &os)const {
		/*
        val.save(os);
        aux_square.save(os);
        aux_mean.save(os);
        os << iter << endl;
		*/
    }

    inline void load(std::ifstream &is) {
		/*
        val.load(is);
        aux_square.load(is);
        aux_mean.load(is);
        is >> iter;
		*/
    }
};

#endif /* PARAM_H_ */
