#ifndef ATOMICIOP_H_
#define ATOMICIOP_H_

/*
*  AtomicOP.h:
*  a list of atomic operations
*
*  Created on: June 11, 2017
*      Author: yue_zhang(suda), mszhang
*/

/*
ActivateNode
TanhNode
SigmoidNode
ReluNode
IndexNode
PSubNode
PDotNode
*/

#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"

class TanhNode :public Node {
  public:
    PNode in;

  public:
    TanhNode() : Node() {
        in = NULL;
        node_type = "tanh";
    }

    ~TanhNode() {
        in = NULL;
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
		device.unaryExp(in->val, val, &device, &Device::Ftanh);
        //val.vec() = in->val.vec().unaryExpr(ptr_fun(ftanh));
    }

    void backward() {
		LDG::Tensor v_d;
		device.init(v_d, in->val.shape());
		device.binaryExp(in->val, val, v_d, &device, &Device::Dtanh);

		LDG::Tensor temp_loss;
		device.init(temp_loss, in->loss.shape());
		device.Fmatmul(loss, v_d, temp_loss);

		device.Fadd(in->loss, temp_loss, in->loss);

        //in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(dtanh));
    }

  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        return result;
    }
};

class TanhExecute :public Execute {
	public:
		vector<LDG::PTensor> vec_x, vec_val;
		bool bTrain;

	public:
		~TanhExecute() {
		}

	public:
		inline void  forward() {
			int count = batch.size();

			vec_x.clear();
		   	vec_val.clear();
			for (int idx = 0; idx < count; idx++) {
				TanhNode* ptr = (TanhNode*)batch[idx];
				vec_x.push_back(&ptr->in->val);
				vec_val.push_back(&ptr->val);
			}

			device.unaryExp(vec_x, vec_val, &device, &Device::Ftanh);

			for (int idx = 0; idx < count; idx++) {
				TanhNode* ptr = (TanhNode*)batch[idx];
				ptr->forward_drop(bTrain);
			}
		}


		inline void backward() {
			int count = batch.size();
			vector<LDG::PTensor> vec_loss, vec_in_loss;

			LDG::Tensor array_lx[count];
			vector<LDG::PTensor> vec_lx;
			for (int idx = 0; idx < count; idx++) {
				TanhNode* ptr = (TanhNode*)batch[idx];
				ptr->backward_drop();
				vec_loss.push_back(&ptr->loss);
				vec_in_loss.push_back(&ptr->in->loss);

				device.init(array_lx[idx], ptr->in->loss.shape());
				vec_lx.push_back(&array_lx[idx]);
			}

			device.binaryExp(vec_x, vec_val, vec_lx, &device, &Device::Dtanh);
			device.Fmultiply(vec_loss, vec_lx, vec_lx);
			device.Fadd(vec_in_loss, vec_lx, vec_in_loss);
		}
};

inline PExecute TanhNode::generate(bool bTrain) {
    TanhExecute* exec = new TanhExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
};

class PDotNode : public Node {
	public:
		PNode in1, in2;
	public:
		PDotNode() : Node() {
			in1 = NULL;
			in2 = NULL;
			dim = 1;
			node_type = "point-dot";
		}
	public:
		virtual inline void clearValue() {
			Node::clearValue();
			in1 = NULL;
			in2 = NULL;
		}

		//can not be dropped since the output is a scalar
		inline void init(int ndim, dtype dropout) {
			dim = 1;
			Node::init(dim, -1);
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
		inline void compute() {
			device.zero(val);	
			device.Fmatmul(in1->val, in2->val, val, true, false);
			//val[0] = 0.0;
			//for (int idx = 0; idx < in1->dim; idx++) {
				//val[0] += in1->val[idx] * in2->val[idx];
			//}
		}

		void backward() {
			LDG::Tensor cpu_loss;
			cpu_loss.device_type == CPU;
			cpu_loss.shape_ = loss.shape();
			cpu_loss.v = new dtype[loss.shape().size()];
			device.to_cpu(loss, cpu_loss);

			LDG::Tensor temp_loss1, temp_loss2;
			device.init(temp_loss1, in1->loss.shape());
			device.init(temp_loss2, in2->loss.shape());
			device.Fmultiply_scalar(in2->val, cpu_loss.v[0], temp_loss1);
			device.Fmultiply_scalar(in1->val, cpu_loss.v[0], temp_loss2);
			device.Fadd(in1->loss, temp_loss1, in1->loss);
			device.Fadd(in2->loss, temp_loss2, in2->loss);
		}

	public:
		// better to rewrite for deep understanding
		inline bool typeEqual(PNode other) {
			return Node::typeEqual(other);
		}

		inline PExecute generate(bool bTrain);
};

class PDotExecute :public Execute {
	public:
		bool bTrain;
	public:
		inline void  forward() {
			int count = batch.size();
			//#pragma omp parallel for schedule(static,1)
			for (int idx = 0; idx < count; idx++) {
				PDotNode* ptr = (PDotNode*)batch[idx];
				ptr->compute();
				ptr->forward_drop(bTrain);
			}
		}

		inline void backward() {
			int count = batch.size();
			//#pragma omp parallel for schedule(static,1)
			for (int idx = 0; idx < count; idx++) {
				PDotNode* ptr = (PDotNode*)batch[idx];
				ptr->backward_drop();
				ptr->backward();
			}
		}
};


inline PExecute PDotNode::generate(bool bTrain) {
	PDotExecute* exec = new PDotExecute();
	exec->batch.push_back(this);
	exec->bTrain = bTrain;
	return exec;
}

#endif
