#include <iostream>
#include <memory>
#include <random>

#include <Eigen/Dense>
#include <omp.h>

#include "tree/tree.cpp"
#include "utils/timing.cpp"

#ifndef FOREST_CPP
#define FOREST_CPP

#define NUM_THREADS 16

class RandomForestClassifier {
public: 

    // variables determined after intiailization
    int ntrees;
    int max_depth;

    // variables filled after calling fit
    int num_labels;
    std::vector<std::unique_ptr<DecisionTreeClassifier>> trees;

    RandomForestClassifier(int ntrees, int max_depth) {
        this->ntrees = ntrees;
        this->max_depth = max_depth;
    }

    std::vector<Eigen::MatrixXd> bootstrap(Eigen::MatrixXd data, Eigen::MatrixXd labels) {
        // create a bootstrapped dataset the same size as the original datset
        int nrows = data.rows();
        int ncols = data.cols();
        Eigen::MatrixXd data_bs(nrows, ncols);
        Eigen::MatrixXd labels_bs(nrows, 1);

        // intiailize a random number generator
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev());
        std::uniform_int_distribution<int> distr(0, nrows - 1);
        for (int i = 0; i < nrows; i++) {
            int didx = distr(generator);
            data_bs.row(i) = data.row(didx);
            labels_bs.row(i) = labels.row(didx);
        }

        std::vector<Eigen::MatrixXd> result;
        result.push_back(data_bs);
        result.push_back(labels_bs);
        return result;
    }

    void fit(Eigen::MatrixXd data, Eigen::MatrixXd labels) {

        // count the number of labels
        this->num_labels = count_num_labels(labels);

        // for each tree, bootstrap the dataset and then init and train tree
        int ntrees = this->ntrees;
        this->trees.resize(ntrees);

        // #pragma omp parallel for schedule(guided) num_threads(NUM_THREADS)
        #pragma omp parallel 
        {
            #pragma omp master 
            {
                for (int i = 0; i < ntrees; i++) {
                    #pragma omp task 
                    {

                        Timer t;

                        std::vector<Eigen::MatrixXd> bootstrapped_dataset = bootstrap(data, labels);
                        Eigen::MatrixXd data_bs = bootstrapped_dataset[0];
                        Eigen::MatrixXd labels_bs = bootstrapped_dataset[1];

                        std::unique_ptr<DecisionTreeClassifier> tree = std::make_unique<DecisionTreeClassifier>(this->max_depth, this->num_labels);
                        tree->fit(data_bs, labels_bs);

                        this->trees[i] = std::move(tree);

                        printf("Done Training Tree: %d (%.5fs)\n", i, t.elapsed());
                    }
                }
            }
        }
    }
    Eigen::MatrixXd predict_proba(Eigen::MatrixXd data) {
        // perform probability predictions on each tree and average results
        int nrows = data.rows();
        Eigen::MatrixXd probs = Eigen::MatrixXd::Zero(nrows, this->num_labels);

        int ntrees = this->ntrees;
        for (int i = 0; i < ntrees; i++) {
            Eigen::MatrixXd subresult = this->trees[i]->predict_proba(data);
            probs += subresult;
        }
        probs /= ntrees;

        return probs;
    }
    Eigen::MatrixXd predict(Eigen::MatrixXd data) {
        Eigen::MatrixXd probs = this->predict_proba(data);
        return create_decision(probs);
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
