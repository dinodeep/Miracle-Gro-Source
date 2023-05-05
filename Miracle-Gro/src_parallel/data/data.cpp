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

#ifndef DATA_CPP
#define DATA_CPP

// original dataset size is 65K samples
#define DATA_LIMIT 1024

std::vector<Eigen::MatrixXd> load_data() {

    // open the file & allocate the matrix
    std::vector<std::string> lines;
    std::ifstream file("src/data/log2.csv");
    std::string line;
    int num_lines = 0;

    // extract the data
    while (std::getline(file, line)) {
        lines.push_back(line);
        num_lines++;
        if (num_lines >= DATA_LIMIT) {
            break;
        }
    }

    // return the dataset
    int num_samples = lines.size();
    int num_features = 11;

    // Source Port	Destination Port	NAT Source Port	NAT Destination Port	Action	Bytes	Bytes Sent	Bytes Received	Packets	Elapsed Time (sec)	pkts_sent	pkts_received
    Eigen::MatrixXd data(num_samples, num_features);
    Eigen::MatrixXd labels(num_samples, 1);


    for (int i = 0; i < num_samples; i++) {
        int src_port, dst_port, nat_src_port, nat_dst_port, action, bytes, bytes_sent, bytes_recv, packets, elapsed_time, packets_sent, packets_recv;
        sscanf(lines[i].c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", &src_port, &dst_port, &nat_src_port, &nat_dst_port, &action, &bytes, &bytes_sent, &bytes_recv, &packets, &elapsed_time, &packets_sent, &packets_recv);
        
        data(i, 0) = src_port;
        data(i, 1) = dst_port;
        data(i, 2) = nat_src_port;
        data(i, 3) = nat_dst_port;
        data(i, 4) = bytes;
        data(i, 5) = bytes_sent;
        data(i, 6) = bytes_recv;
        data(i, 7) = packets;
        data(i, 8) = elapsed_time;
        data(i, 9) = packets_sent;
        data(i, 10) = packets_recv;

        labels(i, 0) = action;
    }

    // std::cout << data.block(0, 0, 5, 11) << std::endl;
    // std::cout << labels.block(0,0, 5, 1) << std::endl;
    std::vector<Eigen::MatrixXd> results;
    results.push_back(data);
    results.push_back(labels);

    return results;
}

#endif