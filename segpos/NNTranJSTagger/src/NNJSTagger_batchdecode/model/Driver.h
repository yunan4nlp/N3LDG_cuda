/*
 * Driver.h
 *
 *  Created on: June 15, 2017
 *      Author: mszhang
 */

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include "N3LDG.h"
#include "State.h"
#include "ActionedNodes.h"
#include "Action.h"
#include "BeamGraph.h"
#include "GreedyGraph.h"

class Driver {
	public:
		Driver() {
			_batch = 0;
			_clip = 10.0;
		}

		~Driver() {
			_batch = 0;
			_clip = 10.0;
			_beam_builders.clear();
			_greedy_builders.clear();
		}

	public:
		Graph _cg;  // build neural graphs
		Graph _dcg;
		vector<Graph> _decode_cgs;
		vector<BeamGraphBuilder> _beam_builders;
		vector<GreedyGraphBuilder> _greedy_builders;
		ModelParams _modelparams;  // model parameters
		HyperParams _hyperparams;

		Metric _eval;
		ModelUpdate _ada;  // model update
		ModelUpdate _beam_ada;  // model update

		int _batch;
		bool _useBeam;
		dtype _clip;

	public:

		inline void initial() {
			if (!_hyperparams.bValid()) {
				std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
				return;
			}
			if (!_modelparams.initial(_hyperparams)) {
				std::cout << "model parameter initialization Error, Please check!" << std::endl;
				return;
			}
			_hyperparams.print();

			_beam_builders.resize(_hyperparams.batch);
			_greedy_builders.resize(_hyperparams.batch);
			_decode_cgs.resize(_hyperparams.batch);

			dtype dropout_value = _hyperparams.dropProb;
			_hyperparams.dropProb = -1.0;
			for (int idx = 0; idx < _hyperparams.batch; idx++) {
				_beam_builders[idx].initial(_modelparams, _hyperparams);
			}

			_hyperparams.dropProb = dropout_value;
			for (int idx = 0; idx < _hyperparams.batch; idx++) {
				_greedy_builders[idx].initial(_modelparams, _hyperparams);
			}

			setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
			_batch = 0;
			_useBeam = false;
		}


	public:
		dtype train(const std::vector<std::vector<string> >& sentences, const vector<vector<CAction> >& goldACs) {
			_eval.reset();
			dtype cost = 0.0;
			int num = sentences.size();
			if (_useBeam) {
				if (num > _beam_builders.size()) {
					std::cout << "input example number is larger than predefined batch number" << std::endl;
					return -1;
				}

				_cg.clearValue(true);
				for (int idx = 0; idx < num; idx++) {
					_beam_builders[idx].encode(&_cg, &sentences[idx]);
				}
				_cg.compute();

#pragma omp parallel for schedule(static,1)
				for (int idx = 0; idx < num; idx++) {
					_decode_cgs[idx].clearValue(true);
					_beam_builders[idx].decode(&(_decode_cgs[idx]), &sentences[idx], &goldACs[idx]);
					_eval.overall_label_count += goldACs[idx].size();
					cost += loss_google(_beam_builders[idx], num);
					_decode_cgs[idx].backward();
				}

				_cg.backward();
			} else {
				if (num > _greedy_builders.size()) {
					std::cout << "input example number is larger than predefined batch number" << std::endl;
					return -1;
				}

				_cg.clearValue(true);
				for (int idx = 0; idx < num; idx++) {
					_greedy_builders[idx].encode(&_cg, &sentences[idx]);
				}
				_cg.compute();

				_dcg.clearValue(true);
				//#pragma omp parallel for schedule(static,1)
				for (int idx = 0; idx < num; idx++) {
					_greedy_builders[idx].decode_prepare(&sentences[idx]);
				}
				bool decode_finish = false;
				while (!decode_finish) {
					decode_finish = true;
					for (int idx = 0; idx < num; idx++) {
						if (!_greedy_builders[idx].is_finish)
							decode_finish = false;
					}
					if (!decode_finish) {
						for (int idx = 0; idx < num; idx++) {
							_greedy_builders[idx].decode_forward(&_dcg, &sentences[idx], &goldACs[idx]);
						}
						_dcg.compute();
						for (int idx = 0; idx < num; idx++) {
							_greedy_builders[idx].decode_apply_action(&_dcg, &sentences[idx], &goldACs[idx]);
						}
					}
				}
				for (int idx = 0; idx < num; idx++) {
					//_decode_cgs[idx].clearValue(true);
					//_greedy_builders[idx].decode(&(_decode_cgs[idx]), &sentences[idx], &goldACs[idx]);
					_eval.overall_label_count += goldACs[idx].size();
					cost += loss_google(_greedy_builders[idx], num);
					//_decode_cgs[idx].backward();
				}
				_dcg.backward();
				_cg.backward();

				/*
#pragma omp parallel for schedule(static,1)
				for (int idx = 0; idx < num; idx++) {
					_decode_cgs[idx].clearValue(true);
					_greedy_builders[idx].decode(&(_decode_cgs[idx]), &sentences[idx], &goldACs[idx]);
					_eval.overall_label_count += goldACs[idx].size();
					cost += loss_google(_greedy_builders[idx], num);
					_decode_cgs[idx].backward();
				}
				_cg.backward();
				*/

			}
			return cost;
		}

		void decode(const std::vector<std::vector<string> >& sentences, vector<vector<string> >& seg_results, vector<vector<string> >& tag_results) {
			int num = sentences.size();
			if (_useBeam) {
				if (num > _beam_builders.size()) {
					std::cout << "input example number is larger than predefined batch number" << std::endl;
					return;
				}
				_cg.clearValue();
				for (int idx = 0; idx < num; idx++) {
					_beam_builders[idx].encode(&_cg, &sentences[idx]);
				}
				_cg.compute();

				seg_results.resize(num);
				tag_results.resize(num);
#pragma omp parallel for schedule(static,1)
				for (int idx = 0; idx < num; idx++) {
					_decode_cgs[idx].clearValue();
					_beam_builders[idx].decode(&(_decode_cgs[idx]), &sentences[idx]);
					int step = _beam_builders[idx].outputs.size();
					_beam_builders[idx].states[step - 1][0].getResults(seg_results[idx], tag_results[idx], &_hyperparams);
				}
			} else {
				if (num > _greedy_builders.size()) {
					std::cout << "input example number is larger than predefined batch number" << std::endl;
					return;
				}
				_cg.clearValue();
				for (int idx = 0; idx < num; idx++) {
					_greedy_builders[idx].encode(&_cg, &sentences[idx]);
				}
				_cg.compute();

				_dcg.clearValue();
				for (int idx = 0; idx < num; idx++) {
					_greedy_builders[idx].decode_prepare(&sentences[idx]);
				}
				bool decode_finish = false;
				while (!decode_finish) {
					decode_finish = true;
					for (int idx = 0; idx < num; idx++) {
						if (!_greedy_builders[idx].is_finish)
							decode_finish = false;
					}
					if (!decode_finish) {
						for (int idx = 0; idx < num; idx++) {
							_greedy_builders[idx].decode_forward(&_dcg, &sentences[idx]);
						}
						_dcg.compute();
						for (int idx = 0; idx < num; idx++) {
							_greedy_builders[idx].decode_apply_action(&_dcg, &sentences[idx]);
						}
					}
				}

				seg_results.resize(num);
				tag_results.resize(num);

				//#pragma omp parallel for schedule(static,1)
				for (int idx = 0; idx < num; idx++) {
					//_decode_cgs[idx].clearValue();
					//_greedy_builders[idx].decode(&(_decode_cgs[idx]), &sentences[idx]);
					int step = _greedy_builders[idx].outputs.size();
					_greedy_builders[idx].states[step - 1].getResults(seg_results[idx], tag_results[idx], &_hyperparams);
				}

				/*
				seg_results.resize(num);
				tag_results.resize(num);
#pragma omp parallel for schedule(static,1)
				for (int idx = 0; idx < num; idx++) {
					_decode_cgs[idx].clearValue();
					_greedy_builders[idx].decode(&(_decode_cgs[idx]), &sentences[idx]);
					int step = _greedy_builders[idx].outputs.size();
					_greedy_builders[idx].states[step - 1].getResults(seg_results[idx], tag_results[idx], &_hyperparams);
				}
				*/
			}
		}

		void updateModel() {
			//if (_batch <= 0) return;
			if (!_useBeam) {
				if (_ada._params.empty()) {
					_modelparams.exportModelParams(_ada);
				}
				//_ada.rescaleGrad(1.0 / _batch);
				//_ada.update(10);
				_ada.updateAdam(_clip);
				_batch = 0;
			} else {
				if (_beam_ada._params.empty()) {
					_modelparams.exportModelBeamParams(_beam_ada);
				}
				//_beam_ada.rescaleGrad(1.0 / _batch);
				//_beam_ada.update(10);
				_beam_ada.updateAdam(_clip);
				_batch = 0;
			}
		}

		void writeModel();

		void loadModel();

	private:
		dtype loss_google(GreedyGraphBuilder& builder, int num) {
			int maxstep = builder.outputs.size();
			if (maxstep == 0) return 1.0;
			//_eval.correct_label_count += maxstep;
			PNode pBestNode = NULL;
			PNode pGoldNode = NULL;
			PNode pCurNode;
			dtype sum, max;
			int curcount, goldIndex;
			vector<dtype> scores;
			dtype cost = 0.0;

			for (int step = 0; step < maxstep; step++) {
				curcount = builder.outputs[step].size();
				if (curcount == 1) {
					_eval.correct_label_count++;
					continue;
				}
				max = 0.0;
				goldIndex = -1;
				pBestNode = pGoldNode = NULL;
				for (int idx = 0; idx < curcount; idx++) {
					pCurNode = builder.outputs[step][idx].in;
					if(pBestNode == NULL)
						pBestNode = pCurNode;

					//LDG::Tensor cpu_cur_val, cpu_best_val;

					//cpu_cur_val.device_type = CPU;
					//cpu_cur_val.shape_ = pCurNode->val.shape();
					//cpu_cur_val.v = new dtype[pCurNode->val.shape().size()];
					//DEV->to_cpu(pCurNode->val, cpu_cur_val);
					vector<dtype> cpu_cur_val = DEV->to_vector(pCurNode->val);

					//cpu_best_val.device_type = CPU;
					//cpu_best_val.shape_ = pBestNode->val.shape();
					//cpu_best_val.v = new dtype[pBestNode->val.shape().size()];
					//DEV->to_cpu(pBestNode->val, cpu_best_val);
					vector<dtype> cpu_best_val = DEV->to_vector(pBestNode->val);


					if (cpu_cur_val[0] > cpu_best_val[0]) {
						pBestNode = pCurNode;
						max = cpu_best_val[0];
					}

					if (builder.outputs[step][idx].bGold) {
						pGoldNode = pCurNode;
						goldIndex = idx;
					}
				}

				if (goldIndex == -1) {
					std::cout << "impossible" << std::endl;
				}
				dtype data[1] = {-1.0 / num};
				DEV->set(pGoldNode->loss, data, 1);
				//pGoldNode->loss[0] = -1.0 / num;

				//LDG::Tensor cpu_best_val;
				//cpu_best_val.device_type = CPU;
				//cpu_best_val.shape_ = pBestNode->val.shape();
				//cpu_best_val.v = new dtype[pBestNode->val.shape().size()];
				//DEV->to_cpu(pBestNode->val, cpu_best_val);
				vector<dtype> cpu_best_val = DEV->to_vector(pBestNode->val);

				//max = pBestNode->val[0];
				sum = 0.0;
				scores.resize(curcount);
				for (int idx = 0; idx < curcount; idx++) {
					pCurNode = builder.outputs[step][idx].in;

					//LDG::Tensor cpu_cur_val;
					//cpu_cur_val.shape_ = pCurNode->val.shape();
					//cpu_cur_val.v = new dtype[pCurNode->val.shape().size()];
					//DEV->to_cpu(pCurNode->val, cpu_cur_val);
					vector<dtype> cpu_cur_val = DEV->to_vector(pCurNode->val);

					//scores[idx] = exp(pCurNode->val[0] - max);
					scores[idx] = exp(cpu_cur_val[0] - max);
					sum += scores[idx];
				}

				for (int idx = 0; idx < curcount; idx++) {
					pCurNode = builder.outputs[step][idx].in;
					DEV->Fadd_scalar_inplace(pCurNode->loss, scores[idx] / (sum * num));
					//DEV->Fadd_scalar(pCurNode->loss, scores[idx] / (sum * num),  pCurNode->loss);
					//pCurNode->loss[0] += scores[idx] / (sum * num);
				}

				if (pBestNode == pGoldNode)_eval.correct_label_count++;
				//_eval.overall_label_count++;

				cost += -log(scores[goldIndex] / sum);

				if (std::isnan(cost)) {
					std::cout << "std::isnan(cost), google loss,  debug" << std::endl;
				}

				_batch++;
			}

			return cost;
		}

		dtype loss_google(BeamGraphBuilder& builder, int num) {
			int maxstep = builder.outputs.size();
			if (maxstep == 0) return 1.0;
			//_eval.correct_label_count += maxstep;
			PNode pBestNode = NULL;
			PNode pGoldNode = NULL;
			PNode pCurNode;
			dtype sum, max;
			int curcount, goldIndex;
			vector<dtype> scores;
			dtype cost = 0.0;

			for (int step = maxstep-1; step < maxstep; step++) {
				curcount = builder.outputs[step].size();
				if (curcount == 1) {
					_eval.correct_label_count++;
					continue;
				}
				max = 0.0;
				goldIndex = -1;
				pBestNode = pGoldNode = NULL;
				for (int idx = 0; idx < curcount; idx++) {
					pCurNode = builder.outputs[step][idx].in;

					vector<dtype> cpu_cur_val = DEV->to_vector(pCurNode->val);
					vector<dtype> cpu_best_val = DEV->to_vector(pBestNode->val);
					/*
					LDG::Tensor cpu_cur_val, cpu_best_val;
					cpu_cur_val.device_type = CPU;
					cpu_best_val.device_type = CPU;

					cpu_cur_val.shape_ = pCurNode->val.shape();
					cpu_best_val.shape_ = pBestNode->val.shape();

					cpu_cur_val.v = new dtype[pCurNode->val.shape().size()];
					cpu_best_val.v = new dtype[pBestNode->val.shape().size()];

					DEV->to_cpu(pCurNode->val, cpu_cur_val);
					DEV->to_cpu(pBestNode->val, cpu_best_val);
					*/

					//if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
					if (pBestNode == NULL || cpu_cur_val[0] > cpu_best_val[0]) {
						pBestNode = pCurNode;
						max = cpu_best_val[0];
					}
					if (builder.outputs[step][idx].bGold) {
						pGoldNode = pCurNode;
						goldIndex = idx;
					}
				}

				if (goldIndex == -1) {
					std::cout << "impossible" << std::endl;
				}
				dtype data[1] = {-1.0 / num};
				DEV->set(pGoldNode->loss, data, 1);
				//pGoldNode->loss[0] = -1.0 / num;

				//max = pBestNode->val[0];
				sum = 0.0;
				scores.resize(curcount);
				for (int idx = 0; idx < curcount; idx++) {
					pCurNode = builder.outputs[step][idx].in;

					vector<dtype> cpu_cur_val = DEV->to_vector(pCurNode->val);
					/*
					LDG::Tensor cpu_cur_val;
					cpu_cur_val.shape_ = pCurNode->val.shape();
					cpu_cur_val.v = new dtype[pCurNode->val.shape().size()];
					DEV->to_cpu(pCurNode->val, cpu_cur_val);
					*/

					//scores[idx] = exp(pCurNode->val[0] - max);
					scores[idx] = exp(cpu_cur_val[0] - max);
					sum += scores[idx];
				}

				for (int idx = 0; idx < curcount; idx++) {
					pCurNode = builder.outputs[step][idx].in;
					DEV->Fadd_scalar_inplace(pCurNode->loss, scores[idx] / (sum * num));
					//DEV->Fadd_scalar(pCurNode->loss, scores[idx] / (sum * num),  pCurNode->loss);
					//pCurNode->loss[0] += scores[idx] / (sum * num);
				}

				if (pBestNode == pGoldNode)_eval.correct_label_count++;
				//_eval.overall_label_count++;

				cost += -log(scores[goldIndex] / sum);

				if (std::isnan(cost)) {
					std::cout << "std::isnan(cost), google loss,  debug" << std::endl;
				}

				_batch++;
				}

				return cost;
			}

			public:
			inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
				_ada._alpha = adaAlpha;
				_ada._eps = adaEps;
				_ada._reg = nnRegular;

				_beam_ada._alpha = adaAlpha * 0.1;
				_beam_ada._eps = adaEps;
				_beam_ada._reg = nnRegular;
			}

			//useBeam = true, beam searcher
			inline void setGraph(bool useBeam) {
				_useBeam = useBeam;
			}

			inline void setClip(dtype clip) {
				_clip = clip;
			}

		};

#endif /* SRC_Driver_H_ */
