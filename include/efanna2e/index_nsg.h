#ifndef EFANNA2E_INDEX_NSG_H
#define EFANNA2E_INDEX_NSG_H

#include "util.h"
#include "parameters.h"
#include "neighbor.h"
#include "index.h"
#include <cassert>
#include <unordered_map>
#include <string>
#include <sstream>
#include <boost/dynamic_bitset.hpp>
#include <stack>

namespace efanna2e {

class IndexNSG : public Index {
 public:
  explicit IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexNSG();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;
  void SearchWithOptGraph(
      const float *query,
      size_t K,
      const Parameters &parameters,
      unsigned *indices,
      int& com_num);
  void OptimizeGraph(float* data);
  float eval_recall(std::vector<std::vector<unsigned> > query_res, std::vector<std::vector<int> > gts, int K);
  void Save_part_point(const char *filename, const char *in_graph_file);
  void Load_part_point(const char *filename, const char *in_graph_file);
  std::vector<std::vector<int> > load_ground_truth(const char* filename, unsigned k);
  void Write_disk(unsigned R, const char *filename, const char *in_graph_file, float aerfa);
  void Search_write_disk(unsigned del_id, const char *filename, efanna2e::Parameters paras);
  void prune_result(float* data, unsigned R, float aerfa, std::vector<unsigned>& res, unsigned add_node_id, unsigned resize, unsigned total_size, unsigned k_num);
  void save_opt(const char *filename, const char *in_graph);
  void com_degree();
  bool add_nebor(unsigned cur_id, unsigned new_id, unsigned R);
  void Compute_gt(const float *query_load, const char *gt_file, unsigned K, unsigned query_num);
  void my_realloc(unsigned total_size);
  unsigned* in_graph_ = NULL;

  protected:
    typedef std::vector<std::vector<unsigned > > CompactGraph;
    typedef std::vector<SimpleNeighbors > LockGraph;
    typedef std::vector<nhood> KNNGraph;

    CompactGraph final_graph_;

    Index *initializer_;
    void init_graph(const Parameters &parameters);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(const float *query, const Parameters &parameter,
        boost::dynamic_bitset<> &flags,
        std::vector<Neighbor> &retset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<Neighbor> &fullset);
    void get_neighbors(
        const float *query,
        const Parameters &parameter,
        boost::dynamic_bitset<>& flags,
        std::vector<Neighbor> &retset,
        std::vector<unsigned> &fullset,
        int path_num);
    //void add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph& cut_graph_);
    void InterInsert(unsigned range, std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_, const Parameters &parameter);
    unsigned sync_prune(unsigned q, std::vector<Neighbor> pool, const Parameters &parameter, boost::dynamic_bitset<>& flags, SimpleNeighbor* cut_graph_, std::vector<Neighbor> path_node, std::vector<std::mutex> &locks);
    void Link(const Parameters &parameters, SimpleNeighbor* cut_graph_);
    void Load_nn_graph(const char *filename);
    void tree_grow(const Parameters &parameter);
    void DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt);
    void findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter);



  private:
    unsigned width;
    unsigned ep_;
    std::vector<std::mutex> locks;
    char* opt_graph_;
    size_t node_size;
    size_t data_len;
    size_t neighbor_len;
    KNNGraph nnd_graph;
};
}

#endif //EFANNA2E_INDEX_NSG_H
