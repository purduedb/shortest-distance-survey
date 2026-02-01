#include "road_network.h"
#include "util.h"

#include <iostream>
#include <cstring>
#include <fstream>
#include <filesystem>
using namespace std;
using namespace road_network;
const size_t MB = 1024 * 1024;
struct ResultData
{
    size_t label_count;
    size_t max_label_count;
    size_t index_size;
    size_t index_height;
    double index_time;
    double avg_cut_size;
    size_t max_cut_size;
    size_t pruning_2hop;
    size_t pruning_3hop;
    size_t pruning_tail;
    double random_query_time;
    double random_hoplinks;
    vector<double> bucket_query_times;
    vector<double> bucket_hoplinks;
};
void preprocess_json(std::string filename,std::string output_name){
    Graph g;
    cout << endl << "reading graph from " << filename << endl;
    fstream fs(filename.c_str());
    //read_graph(g, std::cin);
    read_graph(g,fs);
    fs.close();
    vector<Edge> redundant_edges;
    g.get_redundant_edges(redundant_edges);
    for (Edge e : redundant_edges)
        g.remove_edge(e.a, e.b);
    vector<Neighbor> closest;
    g.contract(closest);
#ifdef NDEBUG
    srand(time(nullptr));
    g.randomize();
#endif
    // construct index
    vector<CutIndex> ci;
    g.create_cut_index(ci, 0.2);
    ContractionIndex con_index(ci, closest);
    // write index
    //json ? con_index.write_json(std::cout) : con_index.write(std::cout);
    ofstream index_file;
    index_file.open (output_name, ios::out | ios::app );
    //con_index.write_json(index_file);
    con_index.write_json_v2(index_file);
    index_file.close();
}

void preprocess_gilbreth(std::string graph_path,std::string output_name){
    Graph g;
    cout << endl << "reading graph from " << graph_path << endl;
    fstream fs(graph_path.c_str());
    read_dense_graph(g, fs,true);
    fs.close();
    cout << "read " << g.node_count() << " vertices and " << g.edge_count() << " edges" << std::endl;

    util::start_timer();
    ResultData result = {};
    vector<Neighbor> closest;
    g.contract(closest);
#ifdef NDEBUG
    srand(time(nullptr));
    g.randomize();
#endif
    // construct index
    vector<CutIndex> ci;
    g.create_cut_index(ci, 0.2);
    ContractionIndex con_index(ci, closest);
    result.index_time = util::stop_timer();
    result.index_size = con_index.size() / MB;
    std::cout<<"index build"<<std::endl;
    cout << "created index of size " << result.index_size << " MB in " << result.index_time << "s" << endl;
    // write index
    //json ? con_index.write_json(std::cout) : con_index.write(std::cout);
    if(std::filesystem::exists(output_name)){
        if(std::filesystem::remove(output_name)){
            std::cout<<"remove old index ile"<<std::endl;
        }else{
            std::cerr<<"fail to remove the old file"<<std::endl;
        }
    }
    ofstream index_file;
    index_file.open (output_name, ios::out | ios::app );
    //con_index.write_json(index_file);
    con_index.write_json_v2(index_file);
    index_file.close();
}

void compute_ground_truth(std::string graph_path, std::string query_path,std::string output_name){
    Graph g;
    cout << endl << "reading graph from " << graph_path << endl;
    fstream fs(graph_path.c_str());
    read_dense_graph(g, fs,true);
    fs.close();
    cout << "read " << g.node_count() << " vertices and " << g.edge_count() << " edges" << std::endl;
    vector<Neighbor> closest;
    g.contract(closest);
#ifdef NDEBUG
    srand(time(nullptr));
    g.randomize();
#endif
    // construct index
    vector<CutIndex> ci;
    g.create_cut_index(ci, 0.2);
    ContractionIndex con_index(ci, closest);
    std::cout<<"index build"<<std::endl;
    // read queries
    ifstream query_file(query_path);
    if (!query_file.is_open()) {
        cerr << "Failed to open query file: " << query_path << endl;
        return;
    }
    if(std::filesystem::exists(output_name)){
        if(std::filesystem::remove(output_name)){
            std::cout<<"remove old ground truth file"<<std::endl;
        }else{
            std::cerr<<"fail to remove the old file"<<std::endl;
        }
    }
    std::ofstream output_file(output_name);
    if (!output_file.is_open()) {
        cerr << "Failed to open output file: " << output_name << endl;
        return;
    }
    
    NodeID v, w;
    std::string line;
    while (std::getline(query_file, line)) {
        v = std::stoi(line.substr(0, line.find(',')));
        w =  std::stoi(line.substr(line.find(',') + 1));
        distance_t dist = con_index.get_distance(v, w);
        output_file << v << "," << w << "," << dist << std::endl;
    }
    
    query_file.close();
    output_file.close();
}

int main(int argc, char** argv)
{
    std::cout<<"run index"<<std::endl;
    if(argc<3){
        cerr << "Usage: " << argv[0] << " <graph_file> <output_file>" << endl;
        return 1;
    }
    preprocess_gilbreth(argv[1], argv[2]);
    //preprocess_gilbreth("/home/zhou822/roads/W_Beijing/W_Beijing.edges","/home/zhou822/roads/W_Beijing/Beijing.hl");
    //compute_ground_truth("/scratch1/zhou822/Beijing_Cleaned/adjacency_list.txt","/scratch1/zhou822/Beijing_Cleaned/queries.txt","/scratch1/zhou822/Beijing_Cleaned/ground_truth.txt");
    //for new york
    //compute_ground_truth("/scratch1/zhou822/NewYork_v1/NewYork.edgelist","/scratch1/zhou822/NewYork_v1/NewYork.queries","/scratch1/zhou822/NewYork_v1/NewYork.groundtruth");
    return 0;
    /*bool json = false;
    if (argc > 1)
    {
        if (strcmp(argv[1], "-json") == 0)
            json = true;
        else
        {
            cerr << "Syntax: " << argv[0] << " [-json]" << endl;
            exit(0);
        }
    }

    if(json){
        std::string filename="/scratch1/zhou822/road_source/USA-road-d.NY.gr";
        std::string outputname ="NY_Label_json.hl";
        preprocess_json(filename,outputname);
    }*/
    // read graph
/*
    Graph g;
    const char* filename = "/scratch1/zhou822/road_source/USA-road-d.NY.gr";
    cout << endl << "reading graph from " << filename << endl;
    fstream fs(filename);
    //read_graph(g, std::cin);
    read_graph(g,fs);
    fs.close();
    vector<Edge> redundant_edges;
    g.get_redundant_edges(redundant_edges);
    for (Edge e : redundant_edges)
        g.remove_edge(e.a, e.b);
    vector<Neighbor> closest;
    g.contract(closest);
#ifdef NDEBUG
    srand(time(nullptr));
    g.randomize();
#endif
    // construct index
    vector<CutIndex> ci;
    g.create_cut_index(ci, 0.2);
    ContractionIndex con_index(ci, closest);
    // write index
    //json ? con_index.write_json(std::cout) : con_index.write(std::cout);
    ofstream index_file;
    index_file.open ("NY_Label_test.hl", ios::out | ios::app );
    //con_index.write_json(std::cout);
    //con_index.write_json(index_file);
    con_index.write(index_file);
    index_file.close();
    */

    /*
    ifstream index_file_in;
    index_file_in.open("NY_Label.hl", ios::in);
    ContractionIndex ic2(index_file_in);
    index_file_in.close();*/
    return 0;
}
