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
		LDG::Tensor ty, lty, dy, lx;
		UniParams* param;
		void (Device::*activate)(const vector<LDG::PTensor>&, vector<LDG::PTensor>&);// activation function
		void (Device::*derivate)(const vector<LDG::PTensor>&, const vector<LDG::PTensor>&, vector<LDG::PTensor>&);// derivation function of activation function
		//void (Device::*derivate)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor&);// derivation function of activation function
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

		void init(int ndim, dtype dropout) {
			Node::init(ndim, dropout);
			device.init(ty, Shape({ndim, 1}));
			device.init(lty, Shape({ndim, 1}));
			device.init(dy, Shape({ndim, 1}));

			int inDim = param->W.val.shape().dims()[1];
			device.init(lx, Shape({inDim, 1}));
		}

		inline void clearValue() {
			Node::clearValue();
			in = NULL;
		}

		// define the activate function and its derivation form
		inline void setFunctions(void (Device::*f)(const vector<LDG::PTensor>&, vector<LDG::PTensor>&),
				void (Device::*f_deri)(const vector<LDG::PTensor>&, const vector<LDG::PTensor>&, vector<LDG::PTensor>&)) {
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

		vector<LDG::PTensor> vec_x;
		vector<LDG::PTensor> vec_b;
		vector<LDG::PTensor> vec_ty;
		vector<LDG::PTensor> vec_val;

		int inDim, outDim;
		UniParams* param;
		void (Device::*activate)(const vector<LDG::PTensor>&, vector<LDG::PTensor>&);// activation function
		void (Device::*derivate)(const vector<LDG::PTensor>&, const vector<LDG::PTensor>&, vector<LDG::PTensor>&);// derivation function of activation function
		//dtype(*activate)(const dtype&);   // activation function
		//dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
		bool bTrain;

	public:
		inline void  forward() {
			int count = batch.size();

			vec_x.clear();
			vec_b.clear();
			vec_ty.clear();
			for (int idx = 0; idx < count; idx++) {
				UniNode* ptr = (UniNode*)batch[idx];
				vec_x.push_back(&ptr->in->val);
				if (param->bUseB) {
					vec_b.push_back(&param->b.val);
				}
				vec_ty.push_back(&ptr->ty);
			}

			device.Fmatmul(param->W.val, vec_x, vec_ty);
			//cout << param->W.val.shape().to_string() << endl;
			//ty.mat() = param->W.val.mat() * x.mat();

			if (param->bUseB) {
				device.Fadd(vec_ty, vec_b, vec_ty);
				//ty.vec() = ty.vec() + b.vec();
			}

			//y.vec() = ty.vec().unaryExpr(ptr_fun(activate));

			vec_val.clear();
			for (int idx = 0; idx < count; idx++) {
				UniNode* ptr = (UniNode*)batch[idx];
				vec_val.push_back(&ptr->val);
			}

			device.unaryExp(vec_ty, vec_val, &device, activate);

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

			vector<LDG::PTensor> vec_ly;
			vector<LDG::PTensor> vec_dy;
			vector<LDG::PTensor> vec_lty;
			vector<LDG::PTensor> vec_lx;
			for (int idx = 0; idx < count; idx++) {
				UniNode* ptr = (UniNode*)batch[idx];
				ptr->backward_drop();

				vec_ly.push_back(&ptr->loss);
				vec_dy.push_back(&ptr->dy);
				vec_lty.push_back(&ptr->lty);
				vec_lx.push_back(&ptr->lx);
			}

			device.binaryExp(vec_ty, vec_val, vec_dy, &device, derivate);

			device.Fmultiply(vec_ly, vec_dy, vec_lty);

			//lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));

			LDG::Tensor wg;
			device.init(wg, param->W.grad.shape());
			device.Fmatmul(vec_lty, vec_x, wg, false, true);
			device.Fadd(param->W.grad, wg, param->W.grad);

			//param->W.grad.mat() += lty.mat() * x.mat().transpose();
			if (param->bUseB) {
				device.Fadd(param->b.grad, vec_lty, param->b.grad);
			}

			device.Fmatmul(param->W.val, vec_lty, vec_lx, true, false);

			vector<LDG::PTensor> vec_in_loss;
			for (int idx = 0; idx < count; idx++) {
				UniNode* ptr = (UniNode*)batch[idx];
				vec_in_loss.push_back(&ptr->in->loss);
			}

			device.Fadd(vec_in_loss, vec_lx, vec_in_loss);
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

class LinearNode : public Node {
	public:
		PNode in;
		UniParams* param;
		LDG::Tensor lx;

	public:
		LinearNode() : Node() {
			in = NULL;
			param = NULL;
			node_type = "linear";
		}


		inline void setParam(UniParams* paramInit) {
			param = paramInit;
		}

		inline void clearValue() {
			Node::clearValue();
			in = NULL;
		}


		void init(int ndim, dtype dropout) {
			Node::init(ndim, dropout);
			int inDim = param->W.val.shape().dims()[1];

			device.init(lx, Shape({inDim, 1}));
		}


	public:
		void forward(Graph *cg, PNode x) {
			in = x;
			degree = 0;
			in->addParent(this);
			cg->addNode(this);
		}

	public:
		inline PExecute generate(bool bTrain);

		// better to rewrite for deep understanding
		inline bool typeEqual(PNode other) {
			bool result = Node::typeEqual(other);
			if (!result) return false;
			LinearNode* conv_other = (LinearNode*)other;
			if (param != conv_other->param) {
				return false;
			}

			return true;
		}

};

class LinearExecute :public Execute {
	public:
		vector<LDG::PTensor> vec_x;
		vector<LDG::PTensor> vec_val;
		//	Tensor2D x, y;
		int inDim, outDim, count;
		UniParams* param;
		bool bTrain;

	public:
		inline void  forward() {
			count = batch.size();
			//x.init(inDim, count);
			//y.init(outDim, count);

			vec_x.clear();
			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				vec_x.push_back(&ptr->in->val);
			}
			vec_val.clear();
			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				vec_val.push_back(&ptr->val);
			}
			device.Fmatmul(param->W.val, vec_x, vec_val);
			//y.mat() = param->W.val.mat() * x.mat();

			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				ptr->forward_drop(bTrain);
			}
		}

		inline void backward() {
			//	Tensor2D lx, ly;
			//	lx.init(inDim, count);
			//	ly.init(outDim, count);

			vector<LDG::PTensor> vec_loss;
			vector<LDG::PTensor> vec_lx;
			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				ptr->backward_drop();
				vec_loss.push_back(&ptr->loss);
				vec_lx.push_back(&ptr->lx);
				//for (int idy = 0; idy < outDim; idy++) {
				//ly[idx][idy] = ptr->loss[idy];
				//}
			}

			LDG::Tensor wg;
			device.init(wg, param->W.grad.shape());
			device.Fmatmul(vec_loss, vec_x, wg, false, true);
			device.Fadd(param->W.grad, wg, param->W.grad);
			//param->W.grad.mat() += ly.mat() * x.mat().transpose();

			device.Fmatmul(param->W.val, vec_loss, vec_lx, true, false);
			//lx.mat() += param->W.val.mat().transpose() * ly.mat();

			vector<LDG::PTensor> vec_in_loss;

			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				vec_in_loss.push_back(&ptr->in->loss);
			}

			device.Fadd(vec_in_loss, vec_lx, vec_in_loss);
		}
};

inline PExecute LinearNode::generate(bool bTrain) {
	LinearExecute* exec = new LinearExecute();
	exec->batch.push_back(this);
	exec->inDim = param->W.inDim();
	exec->outDim = param->W.outDim();
	exec->param = param;
	exec->bTrain = bTrain;
	return exec;
}

#endif /* UNIOP_H_ */
