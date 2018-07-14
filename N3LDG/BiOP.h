#ifndef BIOP_H_
#define BIOP_H_

/*
*  BiOP.h:
*  a simple feed forward neural operation, binary input.
*
*  Created on: June 11, 2017
*      Author: mszhang
*/


#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

class BiParams {
  public:
    Param W1;
    Param W2;
    Param b;

    bool bUseB;

  public:
    BiParams() {
        bUseB = true;
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W1);
        ada.addParam(&W2);
        if (bUseB) {
            ada.addParam(&b);
        }
    }

    inline void initial(int nOSize, int nISize1, int nISize2, bool useB = true) {
        W1.initial(nOSize, nISize1);
        W2.initial(nOSize, nISize2);
        bUseB = useB;
        if (bUseB) {
            b.initial(nOSize, 1);
        }
    }

    inline void save(std::ofstream &os) const {
        os << bUseB << std::endl;
        W1.save(os);
        W2.save(os);
        if (bUseB) {
            b.save(os);
        }
    }

    inline void load(std::ifstream &is) {
        is >> bUseB;
        W1.load(is);
        W2.load(is);
        if (bUseB) {
            b.load(is);
        }
    }

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class BiNode : public Node {
  public:
    PNode in1, in2;
    BiParams* param;
	void (Device::*activate)(const LDG::Tensor&, LDG::Tensor&);// activation function
	void (Device::*derivate)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&);// derivation function of activation function
    //dtype(*activate)(const dtype&);   // activation function
    //dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function


  public:
    BiNode() : Node() {
        in1 = in2 = NULL;
		activate = &Device::Ftanh;
		derivate = &Device::Dtanh;
        //activate = ftanh;
        //derivate = dtanh;
        param = NULL;
        node_type = "bi";
    }

    ~BiNode() {
        in1 = in2 = NULL;
    }


    inline void setParam(BiParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in1 = in2 = NULL;
    }

    // define the activate function and its derivation form
    inline void setFunctions(void (Device::*f)(const LDG::Tensor&, LDG::Tensor&),
			void (Device::*f_deri)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&)) {
        activate = f;
        derivate = f_deri;
    }


  public:
    void forward(Graph *cg, PNode x1, PNode x2) {
        in1 = x1;
        in2 = x2;
        degree = 0;
        in1->addParent(this);
        in2->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute(LDG::Tensor& ty) {
		LDG::Tensor ty1, ty2;
		device.init(ty1, ty.shape());
		device.init(ty2, ty.shape());
		device.Fmatmul(param->W1.val, in1->val, ty1);
		device.Fmatmul(param->W2.val, in2->val, ty2);
		device.Fadd(ty1, ty2, ty);

        //ty.mat() = param->W1.val.mat() * in1->val.mat() + param->W2.val.mat() * in2->val.mat();
        if (param->bUseB) {
			device.Fadd(ty, param->b.val, ty);
            //ty.vec() += param->b.val.vec();
        }
		device.unaryExp(ty, val, &device, activate);
        //val.vec() = ty.vec().unaryExpr(ptr_fun(activate));
    }

    inline void backward(LDG::Tensor& ty, LDG::Tensor& lty) {
		LDG::Tensor v_d;
		device.init(v_d, val.shape());
		device.binaryExp(ty, val, v_d, &device, derivate);
		device.Fmultiply(loss, v_d, lty);
        //lty.vec() = loss.vec() * ty.vec().binaryExpr(val.vec(), ptr_fun(derivate));

		LDG::Tensor wg1, wg2;
		device.init(wg1, param->W1.grad.shape());
		device.init(wg2, param->W2.grad.shape());

		device.Fmatmul(lty, in1->val, wg1, false, true);
		device.Fadd(param->W1.grad, wg1, param->W1.grad);

		device.Fmatmul(lty, in2->val, wg2, false, true);
		device.Fadd(param->W2.grad, wg2, param->W2.grad);

        //param->W1.grad.mat() += lty.mat() * in1->val.tmat();
        //param->W2.grad.mat() += lty.mat() * in2->val.tmat();

        if (param->bUseB) {
			device.Fadd(param->b.grad, lty, param->b.grad);
            //param->b.grad.vec() += lty.vec();
        }

		LDG::Tensor l_tmp1;
		device.init(l_tmp1, in1->loss.shape());
		device.Fmatmul(param->W1.val, lty, l_tmp1, true, false);
		device.Fadd(in1->loss, l_tmp1, in1->loss);

		LDG::Tensor l_tmp2;
		device.init(l_tmp2, in2->loss.shape());
		device.Fmatmul(param->W2.val, lty, l_tmp2, true, false);
		device.Fadd(in2->loss, l_tmp2, in2->loss);

        //in1->loss.mat() += param->W1.val.mat().transpose() * lty.mat();
        //in2->loss.mat() += param->W2.val.mat().transpose() * lty.mat();
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        BiNode* conv_other = (BiNode*)other;
        if (param != conv_other->param) {
            return false;
        }
        if (activate != conv_other->activate || derivate != conv_other->derivate) {
            return false;
        }

        return true;
    }

};



class BiExecute :public Execute {
  public:
	  LDG::Tensor x1, x2, ty, y, b;
	  int inDim1, inDim2, outDim;
	  BiParams* param;

	  void (Device::*activate)(const LDG::Tensor&, LDG::Tensor&);// activation function
	  void (Device::*derivate)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&);// derivation function of activation function
    //dtype(*activate)(const dtype&);   // activation function
    //dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    bool bTrain;

  public:
    ~BiExecute() {
        param = NULL;
        activate = NULL;
        derivate = NULL;
        inDim1 = inDim2 = outDim = 0;
    }


  public:
    inline void  forward() {
        int count = batch.size();
		device.init(x1, Shape({inDim1, count}));
		device.init(x2, Shape({inDim2, count}));
		device.init(b, Shape({outDim, count}));
		device.init(ty, Shape({outDim, count}));
		device.init(y, Shape({outDim, count}));
        //x1.init(inDim1, count);
       	//x2.init(inDim2, count);
        //b.init(outDim, count);
        //ty.init(outDim, count);
        //y.init(outDim, count);

		vector<LDG::PTensor> vec_x1;
		vector<LDG::PTensor> vec_x2;
		vector<LDG::PTensor> vec_b;
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
			vec_x1.push_back(&ptr->in1->val);
			vec_x2.push_back(&ptr->in2->val);
            if (param->bUseB) {
				vec_b.push_back(&param->b.val);
			}
		}

		device.concat(vec_x1, x1);
		device.concat(vec_x2, x2);
		if (param->bUseB) {
			device.concat(vec_b, b);
		}
		/*
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                x1[idx][idy] = ptr->in1->val[idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                x2[idx][idy] = ptr->in2->val[idy];
            }
            if (param->bUseB) {
                for (int idy = 0; idy < outDim; idy++) {
                    b[idx][idy] = param->b.val.v[idy];
                }
            }
        }
		*/
		LDG::Tensor ty1, ty2;
		device.init(ty1, ty.shape());
		device.init(ty2, ty.shape());
		device.Fmatmul(param->W1.val, x1, ty1);
		device.Fmatmul(param->W2.val, x2, ty2);
		device.Fadd(ty1, ty2, ty);

        //ty.mat() = param->W1.val.mat() * x1.mat() + param->W2.val.mat() * x2.mat();

        if (param->bUseB) {
			device.Fadd(ty, b, ty);
            //ty.vec() = ty.vec() + b.vec();
        }

		device.unaryExp(ty, y, &device, activate);
        //y.vec() = ty.vec().unaryExpr(ptr_fun(activate));

		vector<LDG::PTensor> vec_val;
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
			vec_val.push_back(&ptr->val);
		}
		device.unconcat(y, vec_val);

		/*
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idx][idy];
            }
            ptr->forward_drop(bTrain);
        }
		*/
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();

		LDG::Tensor lx1, lx2, lty, ly;
        device.init(lx1, Shape({inDim1, count}));
        device.init(lx2, Shape({inDim2, count}));
        device.init(lty, Shape({outDim, count}));
        device.init(ly, Shape({outDim, count}));

        //Tensor2D lx1, lx2, lty, ly;
        //lx1.init(inDim1, count);
        //lx2.init(inDim2, count);
        //lty.init(outDim, count);
        //ly.init(outDim, count);

		vector<LDG::PTensor> vec_loss;
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            ptr->backward_drop();
			vec_loss.push_back(&ptr->loss);
            //for (int idy = 0; idy < outDim; idy++) {
                //ly[idx][idy] = ptr->loss[idy];
            //}
        }
		device.concat(vec_loss, ly);

		LDG::Tensor dy;
		device.init(dy, y.shape());
		device.binaryExp(ty, y, dy, &device, derivate);
		device.Fmultiply(ly, dy, lty);
        //lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));

		LDG::Tensor wg1;
		device.init(wg1, param->W1.grad.shape());
		device.Fmatmul(lty, x1, wg1, false, true);
		device.Fadd(param->W1.grad, wg1, param->W1.grad);

		LDG::Tensor wg2;
		device.init(wg2, param->W2.grad.shape());
		device.Fmatmul(lty, x2, wg2, false, true);
		device.Fadd(param->W2.grad, wg2, param->W2.grad);
        //param->W1.grad.mat() += lty.mat() * x1.mat().transpose();
        //param->W2.grad.mat() += lty.mat() * x2.mat().transpose();

		vector<LDG::PTensor> vec_b;
		LDG::Tensor array_bgrad[count];
        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
				device.init(array_bgrad[idx], param->b.grad.shape());
				vec_b.push_back(&array_bgrad[idx]);
            }

			device.unconcat(lty, vec_b);
            for (int idx = 0; idx < count; idx++) {
				device.Fadd(param->b.grad, *vec_b[idx], param->b.grad);
			}
        }

		/*
        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += lty[idx][idy];
                }
            }
        }
		*/


		LDG::Tensor lxg1;
		device.init(lxg1, lx1.shape());
		device.Fmatmul(param->W1.val, lty, lxg1, true, false);
		device.Fadd(lx1, lxg1, lx1);

		LDG::Tensor lxg2;
		device.init(lxg2, lx2.shape());
		device.Fmatmul(param->W2.val, lty, lxg2, true, false);
		device.Fadd(lx2, lxg2, lx2);

        //lx1.mat() += param->W1.val.mat().transpose() * lty.mat();
        //lx2.mat() += param->W2.val.mat().transpose() * lty.mat();

		LDG::Tensor loss_array1[count];
		LDG::Tensor loss_array2[count];
		vector<LDG::PTensor> vec_in_loss1, vec_in_loss2;
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
			device.init(loss_array1[idx], ptr->in1->loss.shape());
			vec_in_loss1.push_back(&loss_array1[idx]);

			device.init(loss_array2[idx], ptr->in2->loss.shape());
			vec_in_loss2.push_back(&loss_array2[idx]);
        }
		device.unconcat(lx1, vec_in_loss1);
		device.unconcat(lx2, vec_in_loss2);
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
			device.Fadd(ptr->in1->loss, *vec_in_loss1[idx], ptr->in1->loss);

			device.Fadd(ptr->in2->loss, *vec_in_loss2[idx], ptr->in2->loss);
		}

		/*
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            for (int idy = 0; idy < inDim1; idy++) {
                ptr->in1->loss[idy] += lx1[idx][idy];
            }
            for (int idy = 0; idy < inDim2; idy++) {
                ptr->in2->loss[idy] += lx2[idx][idy];
            }
        }
		*/
    }
};

inline PExecute BiNode::generate(bool bTrain) {
    BiExecute* exec = new BiExecute();
    exec->batch.push_back(this);
    exec->inDim1 = param->W1.inDim();
    exec->inDim2 = param->W2.inDim();
    exec->outDim = param->W1.outDim();
    exec->param = param;
    exec->activate = activate;
    exec->derivate = derivate;
    exec->bTrain = bTrain;
    return exec;
}

#endif /* BIOP_H_ */
