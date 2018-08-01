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
	LDG::Tensor ty1, ty2, ty, y, dy, lty;
	LDG::Tensor lx1, lx2;
    BiParams* param;
	void (Device::*activate)(const vector<LDG::PTensor>&, vector<LDG::PTensor>&);// activation function
	void (Device::*derivate)(const vector<LDG::PTensor>&, const vector<LDG::PTensor>&, vector<LDG::PTensor>&);// derivation function of activation function
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

	void init(int ndim, dtype dropout) {
		Node::init(ndim, dropout);
		device.init(ty1, Shape({ndim, 1}));
		device.init(ty2, Shape({ndim, 1}));
		device.init(ty, Shape({ndim, 1}));
		device.init(y, Shape({ndim, 1}));
		device.init(lty, Shape({ndim, 1}));
		device.init(dy, Shape({ndim, 1}));

		int inDim1 = param->W1.val.shape().dims()[1];
		device.init(lx1, Shape({inDim1, 1}));

		int inDim2 = param->W2.val.shape().dims()[1];
		device.init(lx2, Shape({inDim2, 1}));
	}

    inline void setParam(BiParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in1 = in2 = NULL;
    }

    // define the activate function and its derivation form
    inline void setFunctions(void (Device::*f)(const vector<LDG::PTensor>&, vector<LDG::PTensor>&),
			void (Device::*f_deri)(const vector<LDG::PTensor>&, const vector<LDG::PTensor>&, vector<LDG::PTensor>&)) {
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
	  vector<LDG::PTensor> vec_x1, vec_x2, vec_b, vec_ty1, vec_ty2, vec_ty, vec_val, vec_dy, vec_lty;

	  int inDim1, inDim2, outDim;
	  BiParams* param;

	  void (Device::*activate)(const vector<LDG::PTensor>&, vector<LDG::PTensor>&);// activation function
	  void (Device::*derivate)(const vector<LDG::PTensor>&, const vector<LDG::PTensor>&, vector<LDG::PTensor>&);// derivation function of activation function
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
		vec_x1.clear();
		vec_x2.clear();
		vec_b.clear();

		vec_ty1.clear();
		vec_ty2.clear();
		vec_ty.clear();
		vec_val.clear();

        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
			vec_x1.push_back(&ptr->in1->val);
			vec_x2.push_back(&ptr->in2->val);
			vec_ty1.push_back(&ptr->ty1);
			vec_ty2.push_back(&ptr->ty2);
			vec_ty.push_back(&ptr->ty);
			vec_val.push_back(&ptr->val);
            if (param->bUseB) {
				vec_b.push_back(&param->b.val);
			}
		}

		device.Fmatmul(param->W1.val, vec_x1, vec_ty1);
		device.Fmatmul(param->W2.val, vec_x2, vec_ty2);
		device.Fadd(vec_ty1, vec_ty2, vec_ty);

        if (param->bUseB) {
			device.Fadd(vec_ty, vec_b, vec_ty);
        }


		device.unaryExp(vec_ty, vec_val, &device, activate);
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();

		vector<LDG::PTensor> vec_loss;
		vec_dy.clear();
		vec_lty.clear();
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
            ptr->backward_drop();
			vec_loss.push_back(&ptr->loss);
			vec_dy.push_back(&ptr->dy);
			vec_lty.push_back(&ptr->lty);
        }

		device.binaryExp(vec_ty, vec_val, vec_dy, &device, derivate);
		device.Fmultiply(vec_loss, vec_dy, vec_lty);

		LDG::Tensor wg1;
		device.init(wg1, param->W1.grad.shape());
		device.Fmatmul(vec_lty, vec_x1, wg1, false, true);
		device.Fadd(param->W1.grad, wg1, param->W1.grad);

		LDG::Tensor wg2;
		device.init(wg2, param->W2.grad.shape());
		device.Fmatmul(vec_lty, vec_x2, wg2, false, true);
		device.Fadd(param->W2.grad, wg2, param->W2.grad);

        if (param->bUseB) {
			device.Fadd(param->b.grad, vec_lty, param->b.grad);
        }

		vector<LDG::PTensor> vec_lx1, vec_lx2;
        for (int idx = 0; idx < count; idx++) {
            BiNode* ptr = (BiNode*)batch[idx];
			vec_lx1.push_back(&ptr->lx1);
			vec_lx2.push_back(&ptr->lx2);
		}

		device.Fmatmul(param->W1.val, vec_lty, vec_lx1, true, false);
		device.Fmatmul(param->W2.val, vec_lty, vec_lx2, true, false);

		BiNode* ptr = (BiNode*)batch[0];
		device.Fadd(ptr->in1->loss, vec_lx1, ptr->in1->loss);
		device.Fadd(ptr->in2->loss, vec_lx2, ptr->in2->loss);
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
