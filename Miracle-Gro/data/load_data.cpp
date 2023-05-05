#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <Eigen/Dense>

#include "tree/tree.cpp"
#include "forest/forest.cpp"

#include <vector>

void load_data() {

    // open the file & allocate the matrix
    std::Vector<std::string> lines;
    std::ifstream file("data/log2.csv");
    std::string line;
    int num_lines = 0;

    // extract the data
    while (std::getline(file, line)) {
        lines.push_back(line);
        num_lines++;
    }

    // return the dataset

}