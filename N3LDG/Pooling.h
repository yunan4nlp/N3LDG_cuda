#ifndef POOLING
#define POOLING

/*
*  Pooling.h:
*  pool operation, max, min, average and sum pooling
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"


class PoolNode : public Node {
  public:
    vector<PNode> ins;

  public:
    PoolNode() : Node() {
        ins.clear();
    }

    ~PoolNode() {
        ins.clear();
    }

    inline void clearValue() {
        Node::clearValue();
        ins.clear();
    }

    inline void setParam(int maxsize) {
    }


    inline void init(int ndim, dtype dropout) {
        Node::init(ndim, dropout);
    }

  public:
    void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for max|min|sum|avg pooling" << std::endl;
            return;
        }
        int nSize = x.size();
        ins.clear();
        for (int i = 0; i < nSize; i++) {
			int val_dim0 = x[i]->val.shape()[0];
            if (val_dim0 != dim) {
                std::cout << "input matrixes are not matched" << std::endl;
                clearValue();
                return;
            }
            ins.push_back(x[i]);
        }

        degree = 0;
        for (int i = 0; i < nSize; i++) {
            ins[i]->addParent(this);
        }

        cg->addNode(this);
    }


  public:
    inline PExecute generate(bool bTrain);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

  public:

    virtual void compute() = 0;

	virtual void backward() = 0;
};

class MaxPoolNode : public PoolNode {
  public:
	int* index;
    MaxPoolNode() : PoolNode() {
        node_type = "max-pooling";
    }

    void init(int ndim, dtype dropout) {
		Node::init(ndim, dropout);
		index = new int[ndim];
	}
	~MaxPoolNode(){
		delete []index;
	}

  public:
    //Be careful that the row is the dim of input vector, and the col is the number of input vectors
    //Another point is that we change the input vectors directly.
    inline void compute() {
        int nSize = ins.size();
		LDG::Tensor in_x;
		device.init(in_x, Shape({dim, nSize}));
		vector<LDG::PTensor> vec_ins;
        for (int i = 0; i < nSize; ++i) {
			vec_ins.push_back(&ins[i]->val);
		}
		device.concat(vec_ins, in_x);
		device.FMaxPooling(in_x, val, index);
    }

    inline void backward() {
        int nSize = ins.size();

		LDG::Tensor in_x;
		device.init(in_x, Shape({dim, nSize}));
		vector<LDG::PTensor> vec_ins;
        for (int i = 0; i < nSize; ++i) {
			vec_ins.push_back(&ins[i]->val);
		}
		device.concat(vec_ins, in_x);

		LDG::Tensor in_loss;
		device.init(in_loss, Shape({dim, nSize}));

		device.DMaxPooling(in_x, val, loss, in_loss, index);
		LDG::Tensor array_in_loss[nSize];
		vector<LDG::PTensor> vec_in_loss;
        for (int i = 0; i < nSize; i++) {
			device.init(array_in_loss[i], Shape({dim, 1}));
			vec_in_loss.push_back(&array_in_loss[i]);
		}
		device.unconcat(in_loss, vec_in_loss);

        for (int i = 0; i < nSize; i++) {
			device.Fadd(ins[i]->loss, *vec_in_loss[i], ins[i]->loss);
            //ins[i]->loss.vec() += loss.vec() * masks[i].vec();
        }

    }

};

class AvgPoolNode : public PoolNode {
	public:
		AvgPoolNode() : PoolNode() {
			node_type = "avg-pooling";
		}

	public:
		//Be careful that the row is the dim of input vector, and the col is the number of input vectors
		//Another point is that we change the input vectors directly.
		inline void compute() {
			int nSize = ins.size();
			LDG::Tensor in_x;
			device.init(in_x, Shape({dim, nSize}));
			vector<LDG::PTensor> vec_ins;
			for (int i = 0; i < nSize; ++i) {
				vec_ins.push_back(&ins[i]->val);
			}
			device.concat(vec_ins, in_x);
			device.FAvgPooling(in_x, val);
		}

		inline void backward() {
			int nSize = ins.size();
			LDG::Tensor p_loss;
			device.init(p_loss, loss.shape());
			dtype scalar = (dtype) 1 / nSize;
			device.Fmultiply_scalar(loss, scalar, p_loss);
			for (int i = 0; i < nSize; i++) {
				device.Fadd(ins[i]->loss, p_loss, ins[i]->loss);
				//ins[i]->loss.vec() += loss.vec() * masks[i].vec();
			}
		}
};

class SumPoolNode : public PoolNode {
  public:
    SumPoolNode() : PoolNode() {
        node_type = "sum-pooling";
    }

  public:
    //Be careful that the row is the dim of input vector, and the col is the number of input vectors
    //Another point is that we change the input vectors directly.
    inline void compute() {
        int nSize = ins.size();
		LDG::Tensor in_x;
		device.init(in_x, Shape({dim, nSize}));
		vector<LDG::PTensor> vec_ins;
        for (int i = 0; i < nSize; ++i) {
			vec_ins.push_back(&ins[i]->val);
		}
		device.concat(vec_ins, in_x);
		device.FSumPooling(in_x, val);
    }

    inline void backward() {
        int nSize = ins.size();
        for (int i = 0; i < nSize; i++) {
			device.Fadd(ins[i]->loss, loss, ins[i]->loss);
            //ins[i]->loss.vec() += loss.vec() * masks[i].vec();
        }
    }
};

class MinPoolNode : public PoolNode {
  public:
	int* index;
    MinPoolNode() : PoolNode() {
        node_type = "min-pooling";
    }

    void init(int ndim, dtype dropout) {
		Node::init(ndim, dropout);
		index = new int[ndim];
	}
	~MinPoolNode(){
		delete []index;
	}

  public:
    //Be careful that the row is the dim of input vector, and the col is the number of input vectors
    //Another point is that we change the input vectors directly.

    inline void compute() {
        int nSize = ins.size();
		LDG::Tensor in_x;
		device.init(in_x, Shape({dim, nSize}));
		vector<LDG::PTensor> vec_ins;
        for (int i = 0; i < nSize; ++i) {
			vec_ins.push_back(&ins[i]->val);
		}
		device.concat(vec_ins, in_x);
		device.FMinPooling(in_x, val, index);
		cout << "=============" << endl;
		for(int idx = 0; idx < dim; idx++)
			cout << "index: " << index[idx] << " ";
		cout << endl;
		cout << "=============" << endl;
    }

    inline void backward() {
        int nSize = ins.size();

		LDG::Tensor in_x;
		device.init(in_x, Shape({dim, nSize}));
		vector<LDG::PTensor> vec_ins;
        for (int i = 0; i < nSize; ++i) {
			vec_ins.push_back(&ins[i]->val);
		}
		device.concat(vec_ins, in_x);

		LDG::Tensor in_loss;
		device.init(in_loss, Shape({dim, nSize}));

		device.DMinPooling(in_x, val, loss, in_loss, index);
		LDG::Tensor array_in_loss[nSize];
		vector<LDG::PTensor> vec_in_loss;
        for (int i = 0; i < nSize; i++) {
			device.init(array_in_loss[i], Shape({dim, 1}));
			vec_in_loss.push_back(&array_in_loss[i]);
		}
		device.unconcat(in_loss, vec_in_loss);

        for (int i = 0; i < nSize; i++) {
			device.Fadd(ins[i]->loss, *vec_in_loss[i], ins[i]->loss);
            //ins[i]->loss.vec() += loss.vec() * masks[i].vec();
        }

    }
};


//#if USE_GPU
//class PoolExecute : public Execute {
//public:
//  bool bTrain;
//public:
//  inline void  forward() {
//    int count = batch.size();
//    for (int idx = 0; idx < count; idx++) {
//      PoolNode* ptr = (PoolNode*)batch[idx];
//      ptr->compute();
//      ptr->forward_drop(bTrain);
//    }
//  }
//
//  inline void backward() {
//    int count = batch.size();
//    for (int idx = 0; idx < count; idx++) {
//      PoolNode* ptr = (PoolNode*)batch[idx];
//      ptr->backward_drop();
//      ptr->backward();
//    }
//  }
//};
//
//inline PExecute PoolNode::generate(bool bTrain) {
//  PoolExecute* exec = new PoolExecute();
//  exec->batch.push_back(this);
//  exec->bTrain = bTrain;
//  return exec;
//}
//#else

class PoolExecute : public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            PoolNode* ptr = (PoolNode*)batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    inline void backward() {
        int count = batch.size();
//#pragma omp parallel for schedule(static,1)
        for (int idx = 0; idx < count; idx++) {
            PoolNode* ptr = (PoolNode*)batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }
};

inline PExecute PoolNode::generate(bool bTrain) {
    PoolExecute* exec = new PoolExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}
//#endif

#endif
