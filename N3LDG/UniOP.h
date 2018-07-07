#ifndef UNIOP_H_
#define UNIOP_H_

/*
*  UniOP.h:
*  a simple feed forward neural operation, unary input.
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/


#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

class UniParams {
  public:
    Param W;
    Param b;
    bool bUseB;

  public:
    UniParams() {
        bUseB = true;
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W);
        if (bUseB) {
            ada.addParam(&b);
        }
    }

    inline void initial(int nOSize, int nISize, bool useB = true) {
        W.initial(nOSize, nISize);
        bUseB = useB;
        if (bUseB) {
            b.initial(nOSize, 1);
        }
    }

    inline void save(std::ofstream &os) const {
        os << bUseB << std::endl;
        W.save(os);
        if (bUseB) {
            b.save(os);
        }
    }

    inline void load(std::ifstream &is) {
        is >> bUseB;
        W.load(is);
        if (bUseB) {
            b.load(is);
        }
    }

};


// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class UniNode : public Node {
  public:
    PNode in;
    UniParams* param;
	void (Device::*activate)(const LDG::Tensor&, LDG::Tensor&);// activation function
	void (Device::*derivate)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&);// derivation function of activation function
    //dtype(*activate)(const dtype&);   // activation function
    //dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function


  public:
    UniNode() : Node() {
        in = NULL;
        //activate = ftanh;
        //derivate = dtanh;
		activate = &Device::Ftanh;
		derivate = &Device::Dtanh;
        param = NULL;
        node_type = "uni";
    }

    ~UniNode() {
        in = NULL;
    }


    inline void setParam(UniParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in = NULL;
    }

    // define the activate function and its derivation form
    inline void setFunctions(void (Device::*f)(const LDG::Tensor&, LDG::Tensor&),
			void (Device::*f_deri)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&)) {
        activate = f;
        derivate = f_deri;
    }

  public:
    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute(LDG::Tensor& ty) {
        //ty.mat() = param->W.val.mat() * in->val.mat();
		device.Fmatmul(param->W.val, in->val, ty);
        if (param->bUseB) {
			device.Fadd(ty, param->b.val, ty);
            //ty.vec() += param->b.val.vec();
        }
		device.unaryExp(ty, val, &device, activate);
        //val.vec() = ty.vec().unaryExpr(ptr_fun(activate));
    }

    inline void backward(LDG::Tensor& ty, LDG::Tensor& lty) {
		LDG::Tensor v_d;
		device.malloc(v_d, val.shape());
		device.binaryExp(ty, val, v_d, &device, derivate);
		device.Fmultiply(loss, v_d, lty);
        //lty.vec() = loss.vec() * ty.vec().binaryExpr(val.vec(), ptr_fun(derivate));

		LDG::Tensor wg;
		device.malloc(wg, param->W.grad.shape());
		device.Fmatmul(lty, in->val, wg, false, true);
		device.Fadd(param->W.grad, wg, param->W.grad);
        //param->W.grad.mat() += lty.mat() * in->val.tmat();
        if (param->bUseB) {
			device.Fadd(param->b.grad, lty, param->b.grad);
            //param->b.grad.vec() += lty.vec();
        }
		LDG::Tensor l_tmp;
		device.malloc(l_tmp, in->loss.shape());
		device.Fmatmul(param->W.val, lty, l_tmp, true, false);
		device.Fadd(in->loss, l_tmp, in->loss);

        //in->loss.mat() += param->W.val.mat().transpose() * lty.mat();
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        UniNode* conv_other = (UniNode*)other;
        if (param != conv_other->param) {
            return false;
        }
        if (activate != conv_other->activate || derivate != conv_other->derivate) {
            return false;
        }

        return true;
    }

};


class UniExecute :public Execute {
  public:
    //Tensor2D x, ty, y, b;
	LDG::Tensor x, ty, y, b;
    int inDim, outDim;
    UniParams* param;
	void (Device::*activate)(const LDG::Tensor&, LDG::Tensor&);// activation function
	void (Device::*derivate)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&);// derivation function of activation function
    //dtype(*activate)(const dtype&);   // activation function
    //dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    bool bTrain;

public:
    void  forward() {
        int count = batch.size();
        device.malloc(x, Shape({inDim, count}));
        device.malloc(b, Shape({outDim, count}));
        device.malloc(ty, Shape({outDim, count}));
        device.malloc(y, Shape({outDim, count}));
        //x.init(inDim, count);
        //b.init(outDim, count);
        //ty.init(outDim, count);
        //y.init(outDim, count);


        vector<LDG::PTensor> vec_x;
        vector<LDG::PTensor> vec_b;
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            vec_x.push_back(&ptr->in->val);
            if (param->bUseB) {
                vec_b.push_back(&param->b.val);
            }
        }
        //cout << x.shape().to_string() << endl;
        device.concat(vec_x, x);
        if (param->bUseB) {
            device.concat(vec_b, b);
            //device.show_val(b);
        }

        /*
           for (int idx = 0; idx < count; idx++) {
           UniNode* ptr = (UniNode*)batch[idx];
           for (int idy = 0; idy < inDim; idy++) {
           x[idx][idy] = ptr->in->val[idy];
           }
           if (param->bUseB) {
           for (int idy = 0; idy < outDim; idy++) {
           b[idx][idy] = param->b.val.v[idy];
           }
           }
           }
           */
        //cout << x.shape().size() << endl;
        //cout << x.shape().to_string() << endl;
        //device.show_val(x);
        device.Fmatmul(param->W.val, x, ty);
        //device.show_val(param->W.val);
        //cout << param->W.val.shape().to_string() << endl;
        //ty.mat() = param->W.val.mat() * x.mat();

        if (param->bUseB) {
            device.Fadd(ty, b, ty);
            //ty.vec() = ty.vec() + b.vec();
        }

        //y.vec() = ty.vec().unaryExpr(ptr_fun(activate));
        device.unaryExp(ty, y, &device, activate);

        vector<LDG::PTensor> vec_val;
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            vec_val.push_back(&ptr->val);
        }
        device.unconcat(y, vec_val);
        //for (int idx = 0; idx < count; idx++) {
        //UniNode* ptr = (UniNode*)batch[idx];
        //device.show_val(ptr->val);
        //}

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();

		LDG::Tensor lx, lty, ly;
        device.malloc(lx, Shape({inDim, count}));
        device.malloc(lty, Shape({outDim, count}));
        device.malloc(ly, Shape({outDim, count}));
		
        //Tensor2D lx, lty, ly;
        //lx.init(inDim, count);
        //lty.init(outDim, count);
        //ly.init(outDim, count);

		vector<LDG::PTensor> pl;
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            ptr->backward_drop();
			pl.push_back(&ptr->loss);
            //for (int idy = 0; idy < outDim; idy++) {
                //ly[idx][idy] = ptr->loss[idy];
            //}
        }
		device.concat(pl, ly);

		LDG::Tensor dy;
		device.malloc(dy, y.shape());
		device.binaryExp(ty, y, dy, &device, derivate);
		device.Fmultiply(ly, dy, lty);


        //lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));

		
		LDG::Tensor wg;
		device.malloc(wg, param->W.grad.shape());
		device.Fmatmul(lty, x, wg, false, true);
		device.Fadd(param->W.grad, wg, param->W.grad);

        //param->W.grad.mat() += lty.mat() * x.mat().transpose();
		vector<LDG::PTensor> vec_b;
        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
				LDG::Tensor b_grad;
				device.malloc(b_grad, param->b.grad.shape());
				vec_b.push_back(&b_grad);
                //for (int idy = 0; idy < outDim; idy++) {
                    //param->b.grad.v[idy] += lty[idx][idy];
                //}
            }

			device.unconcat(lty, vec_b);
            for (int idx = 0; idx < count; idx++) {
				device.Fadd(param->b.grad, *vec_b[idx], param->b.grad);
			}
        }

		//LDG::Tensor wt;
		//device.malloc(wt, param->W.val.shape());
		//device.Ftranspose(param->W.val, wt);
		LDG::Tensor lxg;
		device.malloc(lxg, lx.shape());
		device.Fmatmul(param->W.val, lty, lxg, true, false);
		//device.show_val(param->W.val);
		device.Fadd(lx, lxg, lx);
        //lx.mat() += param->W.val.mat().transpose() * lty.mat();

		vector<LDG::PTensor> vec_x;
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
			vec_x.push_back(&ptr->in->loss);
            //for (int idy = 0; idy < inDim; idy++) {
                //ptr->in->loss[idy] += lx[idx][idy];
            //}
        }
		device.unconcat(lx, vec_x);
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
			device.Fadd(ptr->in->loss, *vec_x[idx], ptr->in->loss);
		}
    }
};


inline PExecute UniNode::generate(bool bTrain) {
    UniExecute* exec = new UniExecute();
    exec->batch.push_back(this);
    exec->inDim = param->W.inDim();
    exec->outDim = param->W.outDim();
    exec->param = param;
    exec->activate = activate;
    exec->derivate = derivate;
    exec->bTrain = bTrain;
    return exec;
}


#endif /* UNIOP_H_ */
