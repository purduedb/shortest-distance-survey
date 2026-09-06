// Called from: src/data_preprocess/generate_query_data.py (_hl_distances)
// Compile:     make compile  (in non_ml_index/Hierarchical-Cut-Labelling/)
//
// Protocol: inference <index.hl> <pairs.txt>
//   pairs.txt : one "src dst" per line, 1-indexed; lines starting with '#' skipped
//   stdout    : "src dst dist\n" per pair (src/dst echoed unchanged, 1-indexed)
//   Caller is responsible for 0↔1-indexed conversion before/after calling this binary.
#include "road_network.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
using namespace std;
using namespace road_network;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <index.hl> <pairs.txt>\n";
        return 1;
    }

    ifstream index_in(argv[1], ios::in);
    if (!index_in) {
        cerr << "Cannot open index: " << argv[1] << "\n";
        return 1;
    }
    ContractionIndex idx(index_in);
    index_in.close();

    ifstream pairs_in(argv[2]);
    if (!pairs_in) {
        cerr << "Cannot open pairs file: " << argv[2] << "\n";
        return 1;
    }

    string line;
    while (getline(pairs_in, line)) {
        if (line.empty() || line[0] == '#') continue;
        istringstream iss(line);
        NodeID s, d;
        if (!(iss >> s >> d)) continue;
        distance_t dist = idx.get_distance(s, d);
        cout << s << ' ' << d << ' ' << dist << '\n';
    }
    return 0;
}
