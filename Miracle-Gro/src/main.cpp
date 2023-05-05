#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <omp.h>

#include "tree/tree.cpp"
#include "forest/forest.cpp"
#include "data/data.cpp"
#include "utils/timing.cpp"

/*
TODO LIST:
    1. profile code to figure out what portions are very expensive
        - finding the optimal split takes a significant amount of time at the root
        - tree calculations are independent of each other
    2. record data regarding that information
    3. parallelize
        - create trees in parallel
            - examine speedup changes as dataset size changes?
        - creating a single tree in parallel
            - splitting the creation of various child nodes
            - look at speedups under different policies, OpenMP
        - fine-grained paralellism on some for loops
            - finding the best split
*/

typedef unsigned index_t;

int main() {

    std::vector<Eigen::MatrixXd> results = load_data();
    Eigen::MatrixXd m = results[0];
    Eigen::MatrixXd labels = results[1];
    RandomForestClassifier rfc(10, 3);
    rfc.fit(m, labels);

    // std::cout << "m:" << std::endl << m.block(0, 0, 5, 11) << std::endl;
    // std::cout << "labels:" << std::endl << labels.block(0, 0, 5, 1) << std::endl;
    // std::cout << "=========================================================" << std::endl;
    // DecisionTreeClassifier *tree = new DecisionTreeClassifier(2);
    // tree->fit(m, labels);

    // std::cout << "=========================================================" << std::endl;
    // Eigen::MatrixXd result = tree->predict_proba(m);
    // std::cout << result << std::endl;
    // std::cout << "=========================================================" << std::endl;
    // result = tree->predict(m);
    // std::cout << result << std::endl;
    // std::cout << "=========================================================" << std::endl;
    // float acc = tree->score(m, labels);
    // std::cout << "Training Accuracy = " << acc << std::endl;


    #pragma omp parallel for
    for (index_t i = 0; i < 10; i++) {
        printf("[t=%d] Hello world!\n", i);
    }


    // std::cout << "Done Training ==============================================" << std::endl;
    // Eigen::MatrixXd probs = rfc.predict_proba(m);
    // std::cout << "Done Prediction Probabilities ===========================================" << std::endl;
    // std::cout << probs << std::endl;
    // std::cout << rfc.predict(m) << std::endl;
    // std::cout << rfc.score(m, labels) << std::endl;
    

    return 0;
}
