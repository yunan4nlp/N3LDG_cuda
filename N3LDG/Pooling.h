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
		//LDG::Tensor in_x;
		//device.init(in_x, Shape({dim, nSize}));
		vector<LDG::PTensor> vec_ins;
        for (int i = 0; i < nSize; ++i) {
			vec_ins.push_back(&ins[i]->val);
		}
		//device.concat(vec_ins, in_x);
		device.FMaxPooling(vec_ins, val, index);
    }

    inline void backward() {
        int nSize = ins.size();

		vector<LDG::PTensor> vec_in_loss;
        for (int i = 0; i < nSize; i++) {
			vec_in_loss.push_back(&ins[i]->loss);
		}

		device.DMaxPooling(loss, vec_in_loss, index);

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

			vector<LDG::PTensor> vec_ins;
			for (int i = 0; i < nSize; ++i) {
				vec_ins.push_back(&ins[i]->val);
			}

			device.FAvgPooling(vec_ins, val);
		}

		inline void backward() {
			int nSize = ins.size();
			vector<LDG::PTensor> vec_ins_loss;
			for (int i = 0; i < nSize; i++) {
				vec_ins_loss.push_back(&ins[i]->loss);
			}
			device.DAvgPooling(loss, vec_ins_loss);
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
		vector<LDG::PTensor> vec_ins;
        for (int i = 0; i < nSize; ++i) {
			vec_ins.push_back(&ins[i]->val);
		}
		device.FSumPooling(vec_ins, val);
    }

    inline void backward() {
        int nSize = ins.size();
		vector<LDG::PTensor> vec_ins_loss;
        for (int i = 0; i < nSize; i++) {
			vec_ins_loss.push_back(&ins[i]->loss);
        }
		device.DSumPooling(loss, vec_ins_loss);
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

		vector<LDG::PTensor> vec_ins;
        for (int i = 0; i < nSize; ++i) {
			vec_ins.push_back(&ins[i]->val);
		}

		device.FMinPooling(vec_ins, val, index);
    }

    inline void backward() {
        int nSize = ins.size();

		vector<LDG::PTensor> vec_in_loss;
        for (int i = 0; i < nSize; i++) {
			vec_in_loss.push_back(&ins[i]->loss);
		}

		device.DMinPooling(loss, vec_in_loss, index);

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
