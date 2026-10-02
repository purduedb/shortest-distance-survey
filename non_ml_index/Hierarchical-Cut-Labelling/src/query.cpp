#include "road_network.h"
#include "util.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
using namespace std;
using namespace road_network;

const size_t nr_queries = 1000000;
const size_t MB = 1024 * 1024;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <index_file.hl>" << endl;
        return 1;
    }
#ifdef NDEBUG
    srand(time(nullptr));
#endif
    // read index
    util::start_timer();
    ifstream index_file_in(argv[1], ios::in);
    if (!index_file_in.is_open())
    {
        cerr << "Failed to open index file: " << argv[1] << endl;
        return 1;
    }
    ContractionIndex con_index(index_file_in);
    index_file_in.close();
    double read_index_time = util::stop_timer();
    cout << "read index in " << read_index_time << "s (" << con_index.size() / MB << " MB)" << endl;
    // test query speed
    vector<pair<NodeID,NodeID>> queries;
    for (size_t i = 0; i < nr_queries; i++)
        queries.push_back(con_index.random_query());
    cout << "OMP threads: " << omp_get_max_threads() << endl;
    util::start_timer();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < queries.size(); i++)
        con_index.get_distance(queries[i].first, queries[i].second);
    double random_query_time = util::stop_timer();
    double random_hoplinks = con_index.avg_hoplinks(queries);
    cout << "ran " << queries.size() << " random queries in " << random_query_time << "s (hoplinks=" << random_hoplinks << ")" << endl;
    cout << fixed << setprecision(3) << "query latency: " << (random_query_time / queries.size()) * 1e6 << " us" << endl;
    return 0;
}
