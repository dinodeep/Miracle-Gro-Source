#include <iostream>
#include <optional>
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <tuple>
#include <immintrin.h> 
#include <nmmintrin.h>
#include <omp.h>

#include <Eigen/Dense>
#include "utils/timing.cpp"

#ifndef TREE_CPP
#define TREE_CPP

#define NUM_SPLITS_PER_FEATURE 3 // was 10 before
#define DEBUG false

Eigen::MatrixXd create_decision(Eigen::MatrixXd probs) {
    int nrows = probs.rows();
    int ncols = probs.cols();
    Eigen::MatrixXd labels(nrows, 1);

    for (int i = 0; i < nrows; i++) {
        int col;
        probs.row(i).maxCoeff(&col);
        labels(i,0) = col;
    }

    return labels;
}

int count_num_labels(Eigen::MatrixXd labels) {
    std::unordered_set<int> unique_labels;
    int nrows = labels.rows();
    for (int i = 0; i < nrows; i++) {
        unique_labels.insert(labels(i, 0));
    }
    return unique_labels.size();
}

std::unordered_map<int, int> get_label_counts(Eigen::MatrixXd labels) {
    std::unordered_map<int, int> counts;
    float n = labels.rows();
    int label;
    
    for (int i = 0; i < n; i++) {
        label = labels(i, 0);
        if (counts.count(label) == 0) {
            counts[label] = 0;
        }
        counts[label] += 1;
    }

    return counts;
}

std::unordered_map<int, float> get_label_probs(Eigen::MatrixXd labels) {
    std::unordered_map<int, float> probs;
    std::unordered_map<int, int> counts = get_label_counts(labels);
    float nrows = labels.rows();

    for (auto &pair: counts) {
        probs[pair.first] = ((float) pair.second) / ((float) nrows);
    }

    return probs;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<Eigen::MatrixXd>> split_data(Eigen::MatrixXd data, int split_index, float split_value) {
    std::vector<int> lt;
    std::vector<int> g;
    int nsamples = data.rows();
    int nfeatures = data.cols();

    // calculate the indices
    for (int i = 0; i < nsamples; i++) {
        if (data(i, split_index) <= split_value) {
            lt.push_back(i);
        } else {
            g.push_back(i);
        }
    }

    int nleft = lt.size();
    int nright = g.size();
    Eigen::MatrixXd left(nleft, nfeatures);
    Eigen::MatrixXd right(nright, nfeatures);

    // split the dataset
    int idx;
    for (int i = 0; i < nleft; i++) {
        idx = lt[i];
        left.row(i) = data.row(idx);
    }

    for (int i = 0; i < nright; i++) {
        idx = g[i];
        right.row(i) = data.row(idx);
    }

    // add as result
    std::vector<Eigen::MatrixXd> sdata = { left, right };
    return { lt, g, sdata };

}

std::vector<Eigen::MatrixXd> split(Eigen::MatrixXd data, Eigen::MatrixXd labels, int split_index, float split_value) {
    std::vector<int> lt;
    std::vector<int> g;
    int nsamples = data.rows();
    int nfeatures = data.cols();

    for (int i = 0; i < nsamples; i++) {
        if (data(i, split_index) <= split_value) {
            lt.push_back(i);
        } else {
            g.push_back(i);
        }
    }
    
    int nleft = lt.size();
    int nright = g.size();
    Eigen::MatrixXd left(nleft, nfeatures);
    Eigen::MatrixXd left_labels(nleft, 1);
    Eigen::MatrixXd right(nright, nfeatures);
    Eigen::MatrixXd right_labels(nright, 1);

    int idx;
    for (int i = 0; i < nleft; i++) {
        idx = lt[i];
        left.row(i) = data.row(idx);
        left_labels.row(i) = labels.row(idx);
    }

    for (int i = 0; i < nright; i++) {
        idx = g[i];
        right.row(i) = data.row(idx);
        right_labels.row(i) = labels.row(idx);
    }
    
    std::vector<Eigen::MatrixXd> result;
    result.push_back(left);
    result.push_back(left_labels);
    result.push_back(right);
    result.push_back(right_labels);
    return result;
}

float gini_index(Eigen::MatrixXd labels) {
    std::unordered_map<int, float> probs = get_label_probs(labels);

    float prob_sum = 0.0;
    for (auto &it : probs) {
        prob_sum += it.second * it.second;
    }

    return 1 - prob_sum;
}

float calculate_gini_split(Eigen::MatrixXd labels0, Eigen::MatrixXd labels1) {
    float n0 = labels0.rows();
    float n1 = labels1.rows();
    float n = n0 + n1;
    float gini0 = gini_index(labels0);
    float gini1 = gini_index(labels1);

    return gini0 * (n0 / n) + gini1 * (n1 / n);
}

float estimate_split(Eigen::MatrixXd data, Eigen::MatrixXd labels, int split_index, float split_value) {
    std::vector<Eigen::MatrixXd> splits = split(data, labels, split_index, split_index);
    return calculate_gini_split(splits[1], splits[3]);
}

std::vector<float> calculate_potential_splits(Eigen::MatrixXd data, int split_index, int numSplitsPerFeature) {
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::min();
    int nrows = data.rows();
    int ncols = data.cols();

    for (int r = 0; r < nrows; r++) {
        float currVal = data(r, split_index);
        minVal = std::min(minVal, currVal);
        maxVal = std::max(maxVal, currVal);
    }

    std::vector<float> result;
    float step = (maxVal - minVal) / (numSplitsPerFeature + 1);
    float currSplit = 0;
    for (int i = 0; i < numSplitsPerFeature; i++) {
        currSplit += step;
        result.push_back(currSplit);
    }

    return result;
}

std::tuple<int, float, float> find_best_split(Eigen::MatrixXd data, Eigen::MatrixXd labels, int numSplitsPerFeature) {

    /*
        approaches for parallelizing this function
            1. parallelize along dimensions
            2. parallelize along all possible splits
    */

    int best_split_index = 0;
    float best_split_value = 0;
    float best_gini = 1;

    int nrows = data.rows();
    int ncols = data.cols();

    Timer t_find_best_split;

    // #pragma omp parallel for schedule(guided)
    for (int split_index = 0; split_index < ncols; split_index++) {
        Timer t;

        // calculate splits values
        std::vector<float> splits = calculate_potential_splits(data, split_index, numSplitsPerFeature);
        for (float split_value : splits) {
            float gini = estimate_split(data, labels, split_index, split_value);
            // #pragma omp critical
            if (gini < best_gini) {
                best_gini = gini;
                best_split_index = split_index;
                best_split_value = split_value;
            }
        }

        // printf("\t[dim=%d] parallel-time-to-calc-best=%.5fs\n", split_index, t.elapsed());
    }
    // printf("Find Best Split takes: %.3fs\n", t_find_best_split.elapsed());
    

    std::tuple<int, float, float> result = {best_split_index, best_split_value, best_gini};
    return result;
}

std::tuple<int, float, float> find_best_split_parallel_dim(Eigen::MatrixXd data, Eigen::MatrixXd labels, int numSplitsPerFeature) {
    int best_split_index = 0;
    float best_split_value = 0;
    float best_gini = 1;

    int nrows = data.rows();
    int ncols = data.cols();

    Timer t_find_best_split;

    std::vector<std::tuple<int, float, float>> best_splits(ncols);

    // #pragma omp parallel for
    for (int split_index = 0; split_index < ncols; split_index++) {
        #pragma omp task
        {
            Timer t;

            // calculate splits values
            int best_split_index = 0;
            float best_split_value = 0;
            float best_gini = 1;

            std::vector<float> splits = calculate_potential_splits(data, split_index, numSplitsPerFeature);
            for (float split_value : splits) {
                float gini = estimate_split(data, labels, split_index, split_value);
                if (gini < best_gini) {
                    best_gini = gini;
                    best_split_index = split_index;
                    best_split_value = split_value;
                }
            }
            best_splits[split_index] = { best_split_index, best_split_value, best_gini };

            // printf("\t[thread=dim=%d] parallel-time-to-calc-best=%.5fs\n", split_index, t.elapsed());
        }
    }

    best_split_index = 0;
    best_split_value = 0;
    best_gini = 1;

    for (int split_index = 0; split_index < ncols; split_index++) {

        std::tuple<int, float, float> tup = best_splits[split_index];
        float split_value = std::get<1>(tup);
        float gini = std::get<2>(tup);

        if (gini < best_gini) {
            best_gini = gini;
            best_split_index = split_index;
            best_split_value = split_value;
        }
    }

    // printf("Find Best Split (parallel dim) takes: %.3fs\n", t_find_best_split.elapsed());

    std::tuple<int, float, float> result = {best_split_index, best_split_value, best_gini};
    return result;
}

////////////////////////////////////////////////
////////////////////////////////////////////////
////////////////////////////////////////////////

class Node {
public:
    // default values
    std::string criterion = "gini";
    std::string splitter = "best";
    bool isLeaf = false;

    // values determined at initialization
    int depth;
    int num_labels;
    int max_depth;
    std::string path;

    // values determined at training time
    int decision_index;
    float decision_value;
    std::unique_ptr<Node> left_node;
    std::unique_ptr<Node> right_node;
    Eigen::MatrixXd proba_decision;

    Node(int depth, int max_depth, int num_labels, std::string path) {
        this->depth = depth;
        this->num_labels = num_labels;
        this->max_depth = max_depth;
        this->path = path;
    }

    void makeLeaf(std::unordered_map<int, float> probs) {
        this->isLeaf = true;
        this->proba_decision = Eigen::MatrixXd::Zero(1, this->num_labels);
        for (auto &pair : probs) {
            this->proba_decision(0, pair.first) = pair.second;
        }
    }

    void fit(Eigen::MatrixXd data, Eigen::MatrixXd labels) {

        Timer t;

        // check if labels are all pure
        std::unordered_map<int, float> probs = get_label_probs(labels);
        int num_curr_labels = probs.size();
        bool is_pure = num_curr_labels == 1;

        // base case check: reached pure labels or max depth
        if (this->depth == this->max_depth || is_pure) {
            this->makeLeaf(probs);

            if (DEBUG) {
                std::cout << "[node=" << this->path << "] turned into an immediate leaf :o" << "\n";
                std::cout << data << "\n";
                std::cout << labels << "\n";
                std::cout << "Probability decision" << "\n";
                for (int i = 0; i < this->proba_decision.rows(); i++) {
                    std::cout << "\tc = " << i << " p = " << proba_decision(0, i) << "\n";
                }
            }
        } else {

            Timer t_find_split;

            // finding the best split
            // std::tuple<int, float, float> best_split = find_best_split(data, labels, NUM_SPLITS_PER_FEATURE);
            std::tuple<int, float, float> best_split = find_best_split_parallel_dim(data, labels, NUM_SPLITS_PER_FEATURE);


            // assert that results are the same
            int split_index = std::get<0>(best_split);
            float split_value = std::get<1>(best_split);
            float gini = std::get<2>(best_split);

            // save the decision made at this ndoe
            this->decision_index = split_index;
            this->decision_value = split_value;

            float elapsed_find_split = t_find_split.elapsed();

            if (DEBUG) {
                std::cout << "[node=" << this->path << "] split-idx=" << split_index << " split-val=" << split_value << " gini=" << gini << "\n";
                std::cout << data << "\n";
                std::cout << labels << "\n";
            }

            Timer tsplit;

            // splitting the dataset on it
            std::vector<Eigen::MatrixXd> split_result = split(data, labels, split_index, split_value);

            Eigen::MatrixXd left = split_result[0];
            Eigen::MatrixXd left_labels = split_result[1];
            Eigen::MatrixXd right = split_result[2];
            Eigen::MatrixXd right_labels = split_result[3];

            float elapsed_split = tsplit.elapsed();

            // if the best split doesn't separate the dataset, then make it a leaf
            if (left.rows() == 0 || right.rows() == 0) {
                this->makeLeaf(probs);

                // debug the leaf node result
                if (DEBUG) {
                    std::cout << "[node=" << this->path << "] turned into an unsplittable leaf :o" << "\n";
                    std::cout << "Probability decision" << "\n";
                    for (int i = 0; i < this->proba_decision.rows(); i++) {
                        std::cout << "\tc = " << i << " p = " << proba_decision(0, i) << "\n";
                    }
                }
            } else {
                this->isLeaf = false;
                this->left_node = std::make_unique<Node>(this->depth + 1, this->max_depth, this->num_labels, this->path + ",L");
                this->right_node = std::make_unique<Node>(this->depth + 1, this->max_depth, this->num_labels, this->path + ",R");
                #pragma omp task 
                {
                    this->left_node->fit(left, left_labels);
                }
                #pragma omp task 
                {
                    this->right_node->fit(right, right_labels);
                }
            }
            printf("[node=%s] time=%.4f\n", this->path.c_str(), t.elapsed());
            printf("\t[node=%s] elapsed-find-split=%.4f\n", this->path.c_str(), elapsed_find_split);
            printf("\t[node=%s] elapsed-split=%.4f\n", this->path.c_str(), elapsed_split);
        }
        return;
    }

    Eigen::MatrixXd predict_proba(Eigen::MatrixXd data) {
        // predict_proba on the root node
        if (this->isLeaf) {
            Eigen::MatrixXd result(data.rows(), this->num_labels);
            for (int i = 0; i < data.rows(); i++) {
                result.row(i) = this->proba_decision;
            }

            if (DEBUG) {
                std::cout << "[node=" << this->path << "] Reached base case: " << std::endl << result << std::endl;
            }
            return result;
        }

        // otherwise recurse the results downwards
        auto result = split_data(data, this->decision_index, this->decision_value);
        std::vector<int> left_indices = std::get<0>(result);
        std::vector<int> right_indices = std::get<1>(result);
        int nleft = left_indices.size();
        int nright = right_indices.size();

        
        std::vector<Eigen::MatrixXd> splits = std::get<2>(result);
        Eigen::MatrixXd left_data = splits[0];
        Eigen::MatrixXd right_data = splits[1];

        // get the results of the sub-nodes
        Eigen::MatrixXd probs_left = this->left_node->predict_proba(left_data);
        Eigen::MatrixXd probs_right = this->right_node->predict_proba(right_data);

        // reformat the results into one matrix
        Eigen::MatrixXd probs_result(data.rows(), this->num_labels);
        for (int i = 0; i < nleft; i++) {
            probs_result.row(left_indices[i]) = probs_left.row(i);
        }
        for (int i = 0; i < nright; i++) {
            probs_result.row(right_indices[i]) = probs_right.row(i);
        }
        
        return probs_result;
    }
};

////////////////////////////////////////////////
////////////////////////////////////////////////
////////////////////////////////////////////////


class DecisionTreeClassifier {
public:

    std::string criterion = "gini";
    std::string splitter = "best";
    int max_depth;
    int num_labels;
    std::unique_ptr<Node> root;

    DecisionTreeClassifier(int max_depth) {
        this->max_depth = max_depth;
        this->root = std::make_unique<Node>(0, max_depth, -1, "\t<root>");
        this->num_labels = -1;
    }

    DecisionTreeClassifier(int max_depth, int num_labels) {
        this->max_depth = max_depth;
        this->root = std::make_unique<Node>(0, max_depth, -1, "\t<root>");
        this->num_labels = num_labels;
    }

    void fit(Eigen::MatrixXd data, Eigen::MatrixXd labels) {
        // define the number of labels if not given already
        if (this->num_labels == -1) {
            this->num_labels = count_num_labels(labels);
        }
        this->root->num_labels = this->num_labels;

        // fit the root node
        this->root->fit(data, labels);
    }
    Eigen::MatrixXd predict(Eigen::MatrixXd data) {
        // predict_proba and make decisions
        Eigen::MatrixXd probs = this->predict_proba(data);
        return create_decision(probs);
    }
    Eigen::MatrixXd predict_proba(Eigen::MatrixXd data) {
        // predict_proba on the root node
        return this->root->predict_proba(data);
    }
    float score(Eigen::MatrixXd data, Eigen::MatrixXd labels) {
        // predict and then compare to labels
        Eigen::MatrixXd predictions = this->predict(data);
        int ncorrect = 0;
        int nrows = data.rows();

        for (int i = 0; i < nrows; i++) {
            if (labels(i,0) == predictions(i, 0)) {
                ncorrect += 1;
            }
        }

        float acc = ((float) ncorrect / (float) nrows);

        return acc;
    }
};

#endif