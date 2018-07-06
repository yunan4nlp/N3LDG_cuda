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
		device.init(v_d, val.shape());
		device.binaryExp(ty, val, v_d, &device, derivate);
		device.Fmultiply(loss, v_d, lty);
        //lty.vec() = loss.vec() * ty.vec().binaryExpr(val.vec(), ptr_fun(derivate));

		LDG::Tensor wg;
		device.init(wg, param->W.grad.shape());
		device.Fmatmul(lty, in->val, wg, false, true);
		device.Fadd(param->W.grad, wg, param->W.grad);
        //param->W.grad.mat() += lty.mat() * in->val.tmat();
        if (param->bUseB) {
			device.Fadd(param->b.grad, lty, param->b.grad);
            //param->b.grad.vec() += lty.vec();
        }
		LDG::Tensor l_tmp;
		device.init(l_tmp, in->loss.shape());
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
    inline void  forward() {
        int count = batch.size();
		device.init(x, Shape({inDim, count}));
		device.init(b, Shape({outDim, count}));
		device.init(ty, Shape({outDim, count}));
		device.init(y, Shape({outDim, count}));

		//device.zero(x);
		//device.zero(b);
		//device.zero(ty);
		//device.zero(y);


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
        device.init(lx, Shape({inDim, count}));
        device.init(lty, Shape({outDim, count}));
        device.init(ly, Shape({outDim, count}));

		//device.zero(lx);
		//device.zero(lty);
		//device.zero(ly);
		
        //Tensor2D lx, lty, ly;
        //lx.init(inDim, count);
        //lty.init(outDim, count);
        //ly.init(outDim, count);

		vector<LDG::PTensor> vec_loss;
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
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

		
		LDG::Tensor wg;
		device.init(wg, param->W.grad.shape());
		device.Fmatmul(lty, x, wg, false, true);
		device.Fadd(param->W.grad, wg, param->W.grad);

        //param->W.grad.mat() += lty.mat() * x.mat().transpose();
		vector<LDG::PTensor> vec_b;
		LDG::Tensor array_bgrad[count];
        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
				device.init(array_bgrad[idx], param->b.grad.shape());

				vec_b.push_back(&array_bgrad[idx]);
                //for (int idy = 0; idy < outDim; idy++) {
                    //param->b.grad.v[idy] += lty[idx][idy];
                //}
            }

			device.unconcat(lty, vec_b);
            for (int idx = 0; idx < count; idx++) {
				device.Fadd(param->b.grad, *vec_b[idx], param->b.grad);
			}
        }

		LDG::Tensor lxg;
		device.init(lxg, lx.shape());
		device.Fmatmul(param->W.val, lty, lxg, true, false);
		device.Fadd(lx, lxg, lx);
        //lx.mat() += param->W.val.mat().transpose() * lty.mat();

		LDG::Tensor loss_array[count];
		vector<LDG::PTensor> vec_in_loss;
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
			device.init(loss_array[idx], ptr->in->loss.shape());
			vec_in_loss.push_back(&loss_array[idx]);
            //for (int idy = 0; idy < inDim; idy++) {
                //ptr->in->loss[idy] += lx[idx][idy];
            //}
        }
		device.unconcat(lx, vec_in_loss);
		//cout << vec_in_loss[0]->shape().to_string() << endl;
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
			device.Fadd(ptr->in->loss, *vec_in_loss[idx], ptr->in->loss);
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

class LinearNode : public Node {
	public:
		PNode in;
		UniParams* param;

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


	public:
		void forward(Graph *cg, PNode x) {
			in = x;
			degree = 0;
			in->addParent(this);
			cg->addNode(this);
		}

	public:
		inline void compute() {
			device.Fmatmul(param->W.val, in->val, val);
			//val.mat() = param->W.val.mat() * in->val.mat();
		}

		inline void backward() {
			LDG::Tensor wg;
			device.init(wg, param->W.grad.shape());
			device.Fmatmul(loss, in->val, wg, false, true);
			device.Fadd(param->W.grad, wg, param->W.grad);
			//param->W.grad.mat() += loss.mat() * in->val.tmat();

			LDG::Tensor in_loss;
			device.init(in_loss, in->loss.shape());
			device.Fmatmul(param->W.val, loss, in_loss, true, false);
			device.Fadd(in->loss, in_loss, in->loss);
			//in->loss.mat() += param->W.val.mat().transpose() * loss.mat();
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
		LDG::Tensor x, y;
	//	Tensor2D x, y;
		int inDim, outDim, count;
		UniParams* param;
		bool bTrain;

	public:
		inline void  forward() {
			count = batch.size();
			device.init(x, Shape({inDim, count}));
			device.init(y, Shape({outDim, count}));
			//x.init(inDim, count);
			//y.init(outDim, count);
	
			vector<LDG::PTensor> vec_x;
			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				vec_x.push_back(&ptr->in->val);
			}
			device.concat(vec_x, x);
			device.Fmatmul(param->W.val, x, y);
			//y.mat() = param->W.val.mat() * x.mat();

			vector<LDG::PTensor> vec_val;
			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				vec_val.push_back(&ptr->val);
			}
			device.unconcat(y, vec_val);

			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				ptr->forward_drop(bTrain);
			}
		}

		inline void backward() {
			LDG::Tensor lx, ly;
			device.init(lx, Shape({inDim, count}));
			device.init(ly, Shape({outDim, count}));

		//	Tensor2D lx, ly;
		//	lx.init(inDim, count);
		//	ly.init(outDim, count);

			vector<LDG::PTensor> vec_loss;
			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				ptr->backward_drop();
				vec_loss.push_back(&ptr->loss);
				//for (int idy = 0; idy < outDim; idy++) {
					//ly[idx][idy] = ptr->loss[idy];
				//}
			}
			device.concat(vec_loss, ly);

			LDG::Tensor wg;
			device.init(wg, param->W.grad.shape());
			device.Fmatmul(ly, x, wg, false, true);
			device.Fadd(param->W.grad, wg, param->W.grad);
			//param->W.grad.mat() += ly.mat() * x.mat().transpose();

			LDG::Tensor lxg;
			device.init(lxg, lx.shape());
			device.Fmatmul(param->W.val, ly, lxg, true, false);
			device.Fadd(lx, lxg, lx);
			//lx.mat() += param->W.val.mat().transpose() * ly.mat();

			LDG::Tensor loss_array[count];
			vector<LDG::PTensor> vec_in_loss;

			for (int idx = 0; idx < count; idx++) {
				LinearNode* ptr = (LinearNode*)batch[idx];
				device.init(loss_array[idx], ptr->in->loss.shape());
				vec_in_loss.push_back(&loss_array[idx]);

				//for (int idy = 0; idy < inDim; idy++) {
					//ptr->in->loss[idy] += lx[idx][idy];
				//}
			}
			device.unconcat(lx, vec_in_loss);

			for (int idx = 0; idx < count; idx++) {
				UniNode* ptr = (UniNode*)batch[idx];
				device.Fadd(ptr->in->loss, *vec_in_loss[idx], ptr->in->loss);
			}

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
