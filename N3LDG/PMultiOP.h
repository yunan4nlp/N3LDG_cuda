#ifndef PMultiOP
#define PMultiOP

/*
*  PMultiOP.h:
*  pointwise multiplication
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class PMultiNode : public Node {
  public:
    PNode in1, in2;
  public:
    PMultiNode() : Node() {
        in1 = NULL;
        in2 = NULL;
        node_type = "point-multiply";
    }
  public:
    virtual inline void clearValue() {
        Node::clearValue();
        in1 = NULL;
        in2 = NULL;
    }

  public:
    void forward(Graph *cg, PNode x1, PNode x2) {
        in1 = x1;
        in2 = x2;
        degree = 0;
        x1->addParent(this);
        x2->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        device.Fmultiply(in1->val, in2->val, val);
        //val.vec() = in1->val.vec() * in2->val.vec();
    }

    void backward() {
		LDG::Tensor temp_loss1, temp_loss2; 
		device.Fmultiply(loss, in2->val, temp_loss1);
		device.Fadd(in1->loss, temp_loss1, in1->loss);

		device.Fmultiply(loss, in1->val, temp_loss2);
		device.Fadd(in2->loss, temp_loss2, in2->loss);

        //in1->loss.vec() += loss.vec() * in2->val.vec();
        //in2->loss.vec() += loss.vec() * in1->val.vec();
    }

  public:
    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        return Node::typeEqual(other);
    }

    inline PExecute generate(bool bTrain);
};


class PMultiExecute :public Execute {
  public:
	LDG::Tensor y, x1, x2;
    int sumDim;
    bool bTrain;

  public:
    inline void  forward() {
        int count = batch.size();
        sumDim = 0;
        for (int idx = 0; idx < count; idx++) {
			int dim0 = batch[idx]->val.shape().dims()[0];
            sumDim += dim0;
        }
		device.init(y, Shape({sumDim, 1}));
		device.init(x1, Shape({sumDim, 1}));
		device.init(x2, Shape({sumDim, 1}));

        //y.init(sumDim);
        //x1.init(sumDim);
        //x2.init(sumDim);

		vector<LDG::PTensor> vec_x1, vec_x2;
        for (int idx = 0; idx < count; idx++) {
			PMultiNode* ptr = (PMultiNode*)batch[idx];
			vec_x1.push_back(&ptr->in1->val);
			vec_x2.push_back(&ptr->in2->val);
		}
		device.concat(vec_x1, x1);
		device.concat(vec_x2, x2);

		/*
        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                x1[offset + idy] = ptr->in1->val[idy];
                x2[offset + idy] = ptr->in2->val[idy];
            }
            offset += ptr->dim;
        }
		*/

		device.Fmultiply(x1, x2, y);
		//y.vec() = x1.vec() * x2.vec();

		vector<LDG::PTensor> vec_val;
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
			vec_val.push_back(&ptr->val);
		}
		device.unconcat(y, vec_val);

        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
            ptr->forward_drop(bTrain);
        }
		/*
        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->val[idy] = y[offset + idy];
            }
            offset += ptr->dim;
            ptr->forward_drop(bTrain);
        }
		*/
    }

    inline void  backward() {
		LDG::Tensor ly, lx1, lx2;
        device.init(ly, Shape({sumDim, 1}));
        device.init(lx1, Shape({sumDim, 1}));
        device.init(lx2, Shape({sumDim, 1}));

        //Tensor1D ly, lx1, lx2;
        //ly.init(sumDim);
        //lx1.init(sumDim);
        //lx2.init(sumDim);

        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
            ptr->backward_drop();
		}

		vector<LDG::PTensor> vec_loss;
        for (int idx = 0; idx < count; idx++) {
			PMultiNode* ptr = (PMultiNode*)batch[idx];
			vec_loss.push_back(&ptr->loss);
		}
		device.concat(vec_loss, ly);
		
		/*
        int offset = 0;
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < ptr->dim; idy++) {
                ly[offset + idy] = ptr->loss[idy];
            }
            offset += ptr->dim;
        }
		*/

		device.Fmultiply(ly, x2, lx1);
		device.Fmultiply(ly, x1, lx2);

        //lx1.vec() = ly.vec() * x2.vec();
        //lx2.vec() = ly.vec() * x1.vec();

		LDG::Tensor array_lx1[count], array_lx2[count];
		vector<LDG::PTensor> vec_lx1, vec_lx2;
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
			device.init(array_lx1[idx], ptr->in1->loss.shape());
			device.init(array_lx2[idx], ptr->in2->loss.shape());
			vec_lx1.push_back(&array_lx1[idx]);
			vec_lx2.push_back(&array_lx2[idx]);
		}

		device.unconcat(lx1, vec_lx1);
		device.unconcat(lx2, vec_lx2);

        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
			device.Fadd(ptr->in1->loss, *vec_lx1[idx], ptr->in1->loss);
			device.Fadd(ptr->in2->loss, *vec_lx2[idx], ptr->in2->loss);
		}
		/*
        offset = 0;
        for (int idx = 0; idx < count; idx++) {
            PMultiNode* ptr = (PMultiNode*)batch[idx];
            for (int idy = 0; idy < ptr->dim; idy++) {
                ptr->in1->loss[idy] += lx1[offset + idy];
                ptr->in2->loss[idy] += lx2[offset + idy];
            }
            offset += ptr->dim;
        }
		*/
    }

};

inline PExecute PMultiNode::generate(bool bTrain) {
    PMultiExecute* exec = new PMultiExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    return exec;
}
#endif
