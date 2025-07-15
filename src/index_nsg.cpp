#include "efanna2e/index_nsg.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <boost/dynamic_bitset.hpp>
#include <set>
#include <iomanip>

#include "efanna2e/exceptions.h"
#include "efanna2e/parameters.h"
#include <xmmintrin.h>
#include <immintrin.h> 

namespace efanna2e {
#define _CONTROL_NUM 100
IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m,
                   Index *initializer)
    : Index(dimension, n, m), initializer_{initializer} {}

IndexNSG::~IndexNSG() {}

void IndexNSG::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *)&width, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  constexpr size_t ALIGNMENT = 32;
  constexpr size_t FLOATS_PER_ALIGNMENT = ALIGNMENT / sizeof(float);
  data_padding = (FLOATS_PER_ALIGNMENT - (dimension_ % FLOATS_PER_ALIGNMENT)) % FLOATS_PER_ALIGNMENT;
  neighbor_padding = (FLOATS_PER_ALIGNMENT - ((width + 1) % FLOATS_PER_ALIGNMENT)) % FLOATS_PER_ALIGNMENT;
  out.write((char *)&data_padding, sizeof(unsigned));
  out.write((char *)&neighbor_padding, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    out.write(reinterpret_cast<const char*>(data_ + i * dimension_), dimension_ * sizeof(float));
    unsigned GK = (unsigned)final_graph_[i].size();
    final_graph_[i].resize(width, -1);
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)final_graph_[i].data(), width * sizeof(unsigned));
  }
  out.close();
}

void IndexNSG::Save_part_point(const char *filename, const char *in_graph_file) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *)&width, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  std::cout << "质心点：" << ep_ << std::endl;
  constexpr size_t ALIGNMENT = 32;
  constexpr size_t FLOATS_PER_ALIGNMENT = ALIGNMENT / sizeof(float);
  data_padding = (FLOATS_PER_ALIGNMENT - (dimension_ % FLOATS_PER_ALIGNMENT)) % FLOATS_PER_ALIGNMENT;
  neighbor_padding = (FLOATS_PER_ALIGNMENT - ((width + 1) % FLOATS_PER_ALIGNMENT)) % FLOATS_PER_ALIGNMENT;
  out.write((char *)&data_padding, sizeof(unsigned));
  out.write((char *)&neighbor_padding, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    out.write(reinterpret_cast<const char*>(data_ + i * dimension_), dimension_ * sizeof(float));
    // 写入填充的0
    if (data_padding > 0) {
        static const std::vector<float> padding_zeros(data_padding, 0.0f);
        out.write(reinterpret_cast<const char*>(padding_zeros.data()), data_padding * sizeof(float));
    }
    unsigned GK = (unsigned)final_graph_[i].size();
    final_graph_[i].resize(width, -1);
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)final_graph_[i].data(), width * sizeof(unsigned));
    if (neighbor_padding > 0) {
        static const std::vector<float> neighbor_padding_zeros(neighbor_padding, 0);
        out.write(reinterpret_cast<const char*>(neighbor_padding_zeros.data()), neighbor_padding * sizeof(float));
    }
  }
  out.close();
  std::ofstream out_graph(in_graph_file, std::ios::binary | std::ios::out);
  std::vector<unsigned> vec(nd_, 1u);
  out_graph.write((char *)vec.data(), nd_ * sizeof(unsigned));
  out_graph.close();
}

void IndexNSG::Load(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&width, sizeof(unsigned));
  in.read((char *)&ep_, sizeof(unsigned));
  in.read((char *)&data_padding, sizeof(unsigned));
  in.read((char *)&neighbor_padding, sizeof(unsigned));
  data_len = (dimension_ + data_padding) * sizeof(float);
  neighbor_len = (width + 1 + neighbor_padding) * sizeof(unsigned);

  node_size = data_len + neighbor_len;
  // opt_graph_ = (char *)malloc(node_size * nd_);
  opt_graph_ = static_cast<char*>(_mm_malloc(node_size * nd_, 32));
  in.read((char *)opt_graph_, node_size * nd_);
  in.close();
  // unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * 6 + data_len);
  // unsigned MaxM = *neighbors;
  // neighbors++;
  // std::cout << MaxM << std::endl;
  // std::cout << "id:" << 6 << std::endl;
  // for(unsigned i = 0; i < MaxM; ++i){
  //   std::cout << neighbors[i] << ' ';
  // }
  // std::cout << std::endl;
}

void IndexNSG::Load_part_point(const char *filename, const char *in_graph_file) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&width, sizeof(unsigned));
  in.read((char *)&ep_, sizeof(unsigned));
  in.read((char *)&data_padding, sizeof(unsigned));
  in.read((char *)&neighbor_padding, sizeof(unsigned));
  data_len = (dimension_ + data_padding) * sizeof(float);
  neighbor_len = (width + 1 + neighbor_padding) * sizeof(unsigned);

  node_size = data_len + neighbor_len;
  // opt_graph_ = (char *)malloc(node_size * nd_);
  opt_graph_ = static_cast<char*>(_mm_malloc(node_size * nd_, 32));
  in.read((char *)opt_graph_, node_size * nd_);
  in.close();
  std::ifstream in_graph(in_graph_file, std::ios::binary);
  in_graph_ = static_cast<unsigned*>(malloc(nd_ * sizeof(unsigned)));
  in_graph.read(reinterpret_cast<char*>(in_graph_), nd_ * sizeof(unsigned));
  in_graph.close();
}


void IndexNSG::Load_nn_graph(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  in.read((char *)&k, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t num = (unsigned)(fsize / (k + 1) / 4);
  in.seekg(0, std::ios::beg);

  final_graph_.resize(num);
  final_graph_.reserve(num);
  unsigned kk = (k + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(k);
    final_graph_[i].reserve(kk);
    in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
  }
  in.close();
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  boost::dynamic_bitset<> flags{nd_, 0};
  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
  boost::dynamic_bitset<> &flags,
  std::vector<Neighbor> &retset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);

  L = 0;
  unsigned *ep_neighbors = (unsigned *)(opt_graph_ + node_size * ep_ + data_len);
  unsigned ep_MaxM = *ep_neighbors;
  ep_neighbors++;
  for (unsigned i = 0; i < init_ids.size() && i < ep_MaxM; i++) {
    init_ids[i] = ep_neighbors[i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }
  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float dist = distance_->compare((float *)(opt_graph_ + node_size * id + 4), query,
            (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;
    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);
      unsigned MaxM = *neighbors;
      neighbors++;
      // std::cout << MaxM << std::endl;
      // std::cout << "id:" << n << std::endl;
      // for(unsigned i = 0; i < MaxM; ++i){
      //   std::cout << neighbors[i] << ' ';
      // }
      // std::cout << std::endl;
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, (float *)(opt_graph_ + node_size * id + 4),
                    (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             boost::dynamic_bitset<> &flags,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    fullset.push_back(retset[i]);
    // flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexNSG::get_neighbors(const float *query, const Parameters &parameter,
                             boost::dynamic_bitset<> &flags,
                             std::vector<Neighbor> &retset,
                             std::vector<unsigned> &fullset,
                             int path_num) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  L = 0;
  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;
      fullset.push_back(n);
      if(fullset.size() >= 10)
        return;
      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if(path_num != -1 && fullset.size() >= (unsigned)path_num)
      return;
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}



void IndexNSG::init_graph(const Parameters &parameters) {
  float *center = new float[dimension_];
  for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
  for (unsigned i = 0; i < nd_; i++) {
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] += data_[i * dimension_ + j];
    }
  }
  for (unsigned j = 0; j < dimension_; j++) {
    center[j] /= nd_;
  }
  std::vector<Neighbor> tmp, pool;
  ep_ = rand() % nd_;  // random initialize navigating point
  get_neighbors(center, parameters, tmp, pool);
  ep_ = tmp[0].id;
}

unsigned IndexNSG::sync_prune(unsigned q, std::vector<Neighbor> pool,
                          const Parameters &parameter,
                          boost::dynamic_bitset<> &flags,
                          SimpleNeighbor *cut_graph_,
                          std::vector<Neighbor> path_node, 
                          std::vector<std::mutex> &locks) {
  unsigned range = parameter.Get<unsigned>("R");
  unsigned maxc = parameter.Get<unsigned>("C");
  unsigned start = 0;
  float ratio1 = 1;

  for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
    unsigned id = final_graph_[q][nn];
    if (flags[id]) continue;
    float dist =
        distance_->compare(data_ + dimension_ * (size_t)q,
                           data_ + dimension_ * (size_t)id, (unsigned)dimension_);
    pool.push_back(Neighbor(id, dist, true));
  }
  

  std::sort(pool.begin(), pool.end());
  std::vector<Neighbor> result;
  if (pool[start].id == q) start++;
  result.push_back(pool[start]);
  unsigned diff_num = 0;
  while (result.size() < range && (++start) < pool.size() && start < maxc) {
    ratio1 = float(range - result.size()) / range;
    auto &p = pool[start];
    bool occlude = false;
    for (unsigned t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                     data_ + dimension_ * (size_t)p.id,
                                     (unsigned)dimension_);   
      if ( djk < p.distance /* dik */) {
        if(p.distance * p.distance > djk * djk + ratio1 * djk * result[t].distance){  //c² > a² + ab
          occlude = true;
          break;
        }
        diff_num++;
      }
    }
    
    if (!occlude) result.push_back(p);
  }

  SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
  {
    LockGuard guard(locks[q]);
    size_t t = 0;
    while(t < range && des_pool[t].distance != -1)
      ++t;
    for (size_t tt = 0; tt < result.size() && t < range; tt++) {
      des_pool[t].id = result[tt].id;
      des_pool[t].distance = result[tt].distance;
      t++;
    }
    if (t < range) {
      des_pool[t].distance = -1;
    }
  }
  return diff_num;
  
}


void IndexNSG::InterInsert(unsigned range,
                           std::vector<std::mutex> &locks,
                           SimpleNeighbor *cut_graph_, const Parameters &parameter) {   //对于某个点a，在它的邻居b上尝试将a插入b的邻居中
  std::vector<std::vector<SimpleNeighbor>> nn_list(nd_); //要进行筛选的节点集合
  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel
  {
    #pragma omp for schedule(dynamic, 2048)
    for(unsigned j = 0; j < nd_; ++j){
      SimpleNeighbor *src_pool = cut_graph_ + (size_t)j * (size_t)range;  //j的邻居
      for (size_t i = 0; i < range; i++) {
        if (src_pool[i].distance == -1) break;

        size_t des = src_pool[i].id;  //当前邻居的编号
        {
          LockGuard guard(locks[j]);
          nn_list[j].push_back(SimpleNeighbor(des, src_pool[i].distance));
        }
        {
          LockGuard guard(locks[des]);
          nn_list[des].push_back(SimpleNeighbor(j, src_pool[i].distance));
        }
      }
    }
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "nn time:" << diff.count()  << "\n";

  s = std::chrono::high_resolution_clock::now();
#pragma omp parallel
  {
    #pragma omp for schedule(dynamic, 2048)
    for(unsigned i = 0; i < nd_; ++i){
      float ratio1 = 1;
      std::sort(nn_list[i].begin(), nn_list[i].end());
      cut_graph_[i*range].distance = -1;
      std::vector<SimpleNeighbor> result;
      unsigned start = 0;
      if (nn_list[i][start].id == i) start++;
      result.push_back(nn_list[i][start]);
      while (result.size() < range && (++start) < nn_list[i].size()) {
        ratio1 = float(range - result.size()) / range;
        auto &p = nn_list[i][start]; //第二邻居
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
          float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                        data_ + dimension_ * (size_t)p.id,
                                        (unsigned)dimension_);   
          if ( djk < p.distance /* dik */) {
            if(p.distance * p.distance > djk * djk + ratio1 * djk * result[t].distance){  //c² > a² + ab
              occlude = true;
              break;
            }
          }
        }
        if (!occlude) result.push_back(p);
      }

      SimpleNeighbor *des_pool = cut_graph_ + (size_t)i * (size_t)range;
      {
        LockGuard guard(locks[i]);
        size_t t = 0;
        for (size_t tt = 0; tt < result.size() && t < range; tt++) {
          des_pool[t].id = result[tt].id;
          des_pool[t].distance = result[tt].distance;
          t++;
        }
        if (t < range) {
          des_pool[t].distance = -1;
        }
      }
    }
  }
  e = std::chrono::high_resolution_clock::now();
  diff = e - s;
  std::cout << "prune time:" << diff.count()  << "\n";

}

void IndexNSG::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
  /*
  std::cout << " graph link" << std::endl;
  unsigned progress=0;
  unsigned percent = 100;
  unsigned step_size = nd_/percent;
  std::mutex progress_lock;
  */
  unsigned range = parameters.Get<unsigned>("R");
  std::vector<std::mutex> locks(nd_);
  std::vector<std::mutex> mylocks(1);
  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel
  {
    // unsigned cnt = 0;
    std::vector<Neighbor> pool, tmp;
    boost::dynamic_bitset<> flags{nd_, 0};
#pragma omp for schedule(dynamic, 2048)
    for (unsigned n = 0; n < nd_; ++n) {
      pool.clear();
      tmp.clear();
      flags.reset();
      get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
      sync_prune(n, pool, parameters, flags, cut_graph_, tmp, locks);
    }
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "sync_prune time:" << diff.count()  << "\n";
  /*for (size_t i = 0; i < nd_; i++) {
    SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
    unsigned pool_size = 0;
    for (unsigned j = 0; j < range; j++) {
      if (pool[j].distance == -1) break;
      pool_size = j;
    }
    pool_size++;
    final_graph_[i].resize(pool_size);
    for (unsigned j = 0; j < pool_size; j++) {
      final_graph_[i][j] = pool[j].id;
    }
  }*/

  // unsigned max = 0, min = 1e6, avg = 0;
  // for (size_t i = 0; i < nd_; i++) {
  //   SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
  //   unsigned size = 0;
  //   for (unsigned j = 0; j < range; j++) {
  //     if (pool[j].distance == -1) break;
  //     size = j + 1;
  //   }
  //   max = max < size ? size : max;
  //   min = min > size ? size : min;
  //   avg += size;
  // }
  // avg /= 1.0 * nd_;
  // printf("sync_prune之后: Max = %d, Min = %d, Avg = %d\n", max, min, avg);

  s = std::chrono::high_resolution_clock::now();
  InterInsert(range, locks, cut_graph_, parameters);
  e = std::chrono::high_resolution_clock::now();
  diff = e - s;
  std::cout << "InterInsert time:" << diff.count()  << "\n";
}

unsigned IndexNSG::my_Link(const Parameters &parameters, SimpleNeighbor *cut_graph_, size_t test_nd) {
  unsigned range = parameters.Get<unsigned>("R");
  std::vector<std::mutex> locks(nd_);
  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel
  {
    // unsigned cnt = 0;
    std::vector<Neighbor> pool, tmp;
    boost::dynamic_bitset<> flags{nd_, 0};
#pragma omp for schedule(dynamic, 2048)
    for (unsigned n = 0; n < test_nd; ++n) {
      pool.clear();
      tmp.clear();
      flags.reset();
      get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);
      sync_prune(n, pool, parameters, flags, cut_graph_, tmp, locks);
    }
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "sync_prune time:" << diff.count()  << "\n";
  
  unsigned avg = 0;
  for (size_t i = 0; i < test_nd; i++) {
    SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
    unsigned pool_size = 0;
    for (unsigned j = 0; j < range; j++) {
      if (pool[j].distance == -1) break;
      pool_size = j + 1;
    }
    avg += pool_size;
  }
  avg /= 1.0 * test_nd;
  printf("sync_prune之后: Avg = %d\n", avg);
  return avg;
}

void IndexNSG::Build(size_t n, const float *data, const Parameters &parameters) {
  std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
  unsigned range = parameters.Get<unsigned>("R");
  unsigned k_num = parameters.Get<unsigned>("k_num");
  width = range;
  Load_nn_graph(nn_graph_path.c_str());
  data_ = data;
  init_graph(parameters);

  // ratio = 1;
  // size_t test_nd = nd_/100;
  // if(test_nd > 10000)
  //   test_nd = 10000;
  // SimpleNeighbor *test_cut_graph_ = new SimpleNeighbor[test_nd * (size_t)range];
  // for (size_t i = 0; i < test_nd; i++) {
  //   test_cut_graph_[i*range].distance = -1;
  // }
  // unsigned max_r = my_Link(parameters, test_cut_graph_, test_nd);

  // ratio = 0;
  // for (size_t i = 0; i < test_nd; i++) {
  //   test_cut_graph_[i*range].distance = -1;
  // }
  // unsigned min_r = my_Link(parameters, test_cut_graph_, test_nd);

  // ratio = (range * 0.35 - min_r) / (max_r - min_r);
  // if(ratio > 1)
  //   ratio = 1;
  // else if(ratio < 0.25)
  //   ratio = 0.25;
  // // ratio = 1;
  // std::cout << "ratio = " << ratio << std::endl;

  std::vector<std::mutex> locks(nd_);
  SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
  for (size_t i = 0; i < nd_; i++) {
    cut_graph_[i*range].distance = -1;
  }
  Link(parameters, cut_graph_);
  final_graph_.resize(nd_);

  
  #pragma omp parallel
  {
  #pragma omp for schedule(dynamic, 2048)
    for (size_t i = 0; i < nd_; i++) { // 插入k_num个邻居
      SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
      unsigned pool_size = 0;
      std::vector<unsigned> temp = final_graph_[i]; //保留原本的k个邻居
      for (unsigned j = 0; j < range; j++) {
        if (pool[j].distance == -1) break;
        pool_size = j + 1;
      }
      unsigned min_size = pool_size;
      min_size = (pool_size + k_num) < range ? pool_size + k_num : range;
      final_graph_[i].resize(min_size);
      for (unsigned j = 0; j < pool_size; j++) {
        final_graph_[i][j] = pool[j].id;
      }
      unsigned cur_k = 0;
      for(unsigned j = pool_size; j < min_size; ++j){  //插入k个近邻，判断是否存在于筛选后的邻居
        bool is_exist = false;
        unsigned jj = 0;
        while(jj < pool_size){
          if(temp[cur_k] == final_graph_[i][jj]){
            is_exist = true;
            break;
          }
          jj++;
        }
        if(!is_exist)
          final_graph_[i][j] = temp[cur_k];
        if(temp[cur_k] >= nd_)
          std::cout << cur_k << ' ' << temp[cur_k] << std::endl;
        cur_k ++;
      }
    } // 插入k_num个邻居
  }
  
  auto s = std::chrono::high_resolution_clock::now();
//下面这一部分是插入高速公路
#pragma omp parallel
  {
    // unsigned cnt = 0;
    std::vector<Neighbor> tmp;
    std::vector<unsigned> pool;
    boost::dynamic_bitset<> flags{nd_, 0};
#pragma omp for schedule(dynamic, 100)
    for (unsigned n = 0; n < nd_; ++n) {
      // if(final_graph_[n].size() >= 5) continue;
      pool.clear();
      tmp.clear();
      flags.reset();
      get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool, 10);
      for(unsigned i = 0; i < pool.size(); ++i){
        unsigned path_id = pool[i];
        {
          LockGuard guard(locks[path_id]);
          auto it = std::find(final_graph_[path_id].begin(), final_graph_[path_id].end(), n);
          if(it != final_graph_[path_id].end())
            break;
          if(final_graph_[path_id].size() < range){
            final_graph_[path_id].push_back(n);
            break;
          }
        }
      }

    }
  }

  // tree_grow(parameters);
  // std::cout << "建立树完成" << std::endl;
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  std::cout << "连通性保证用时:" << diff.count()  << "\n";
  

  unsigned max = 0, min = 1e6, avg = 0;
  for (size_t i = 0; i < nd_; i++) {
    auto size = final_graph_[i].size();
    max = max < size ? size : max;
    min = min > size ? size : min;
    avg += size;
  }
  avg /= 1.0 * nd_;
  printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);

  has_built = true;
}

void IndexNSG::Search(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  data_ = x;
  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  unsigned tmp_l = 0;
  for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
    init_ids[tmp_l] = final_graph_[ep_][tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    float dist =
        distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    // flags[id] = true;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;
        float dist =
            distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexNSG::SearchWithOptGraph(const float *query, size_t K,
                                  const Parameters &parameters, unsigned *indices, int& com_num) {
  unsigned L = parameters.Get<unsigned>("L_search");
  // DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
  std::vector<Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  // std::mt19937 rng(rand());
  // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned tmp_l = 0;
  unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * ep_ + data_len);
  unsigned MaxM_ep = *neighbors;
  neighbors++;
  // std::cout << "质心点:" << ep_ << std::endl;

  for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
    init_ids[tmp_l] = neighbors[tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while (tmp_l < L) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
  }
  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float *x = reinterpret_cast<float*>(opt_graph_ + node_size * id);
    float dist =
        distance_->compare(x, query, (unsigned)dimension_);
    com_num ++;
    retset[i] = Neighbor(id, dist, true);
    flags[id] = true;
    L++;
  }
  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * n + data_len);
      unsigned MaxM = *neighbors;
      neighbors++;
      // std::cout << MaxM << std::endl;
      // std::cout << "id:" << n << std::endl;
      // for(unsigned i = 0; i < MaxM; ++i){
      //   std::cout << neighbors[i] << ' ';
      // }
      // std::cout << std::endl;
      for (unsigned m = 0; m < MaxM; ++m)
        _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
        if (flags[id]) continue;
        flags[id] = 1;
        float *data = reinterpret_cast<float*>(opt_graph_ + node_size * id);
        float dist =
          distance_->compare(data, query, (unsigned)dimension_);
        com_num ++;
        if (dist >= retset[L - 1].distance) continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        // if(L+1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  unsigned cur_kk = 0;
  for (size_t i = 0; i < L; i++) {
    unsigned temp_id = retset[i].id;
    if(in_graph_ != NULL && in_graph_[temp_id] == 0){
      continue;
    }
    indices[cur_kk] = temp_id;
    cur_kk++;
    if(cur_kk == K)
      break;
  }
}

void IndexNSG::OptimizeGraph(float *data) {  // use after build or load

  data_ = data;
  data_len = (dimension_ + 1) * sizeof(float);
  neighbor_len = (width + 1) * sizeof(unsigned);
  node_size = data_len + neighbor_len;
  opt_graph_ = (char *)malloc(node_size * nd_);
  DistanceFastL2 *dist_fast = (DistanceFastL2 *)distance_;
  for (unsigned i = 0; i < nd_; i++) {
    char *cur_node_offset = opt_graph_ + i * node_size;
    float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
    std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
    std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_,
                data_len - sizeof(float));

    cur_node_offset += data_len;
    unsigned k = final_graph_[i].size();
    std::memcpy(cur_node_offset, &k, sizeof(unsigned));
    std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(),
                k * sizeof(unsigned));
    std::vector<unsigned>().swap(final_graph_[i]);
  }
  CompactGraph().swap(final_graph_);
}

void IndexNSG::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
  unsigned tmp = root;
  std::stack<unsigned> s;
  s.push(root);
  if (!flag[root]) cnt++;
  flag[root] = true;
  while (!s.empty()) {
    unsigned next = nd_ + 1;
    for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
      if (flag[final_graph_[tmp][i]] == false) {
        next = final_graph_[tmp][i];
        break;
      }
    }
    // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
    if (next == (nd_ + 1)) {
      s.pop();
      if (s.empty()) break;
      tmp = s.top();
      continue;
    }
    tmp = next;
    flag[tmp] = true;
    s.push(tmp);
    cnt++;
  }
}

void IndexNSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                        const Parameters &parameters) {
  unsigned range = parameters.Get<unsigned>("R");
  // unsigned k_num = parameters.Get<unsigned>("k_num");
  unsigned id = nd_;
  for (unsigned i = 0; i < nd_; i++) {
    if (flag[i] == false) {
      id = i;
      break;
    }
  }

  if (id == nd_) return;  // No Unlinked Node

  std::vector<Neighbor> tmp, pool;
  get_neighbors(data_ + dimension_ * id, parameters, tmp, pool);
  std::sort(pool.begin(), pool.end());

  unsigned found = 0;
  for (unsigned i = 0; i < pool.size(); i++) {
    unsigned tmp_id = pool[i].id;
    if (flag[tmp_id] && final_graph_[tmp_id].size() < range) {
      // std::cout << pool[i].id << '\n';
      root = tmp_id;
      found = 1;
      break;
    }
  }
  if (found == 0) {
    while (true) {
      unsigned rid = rand() % nd_;
      if (flag[rid]) {
        root = rid;
        break;
      }
    }
  }
  final_graph_[root].push_back(id);
}
void IndexNSG::tree_grow(const Parameters &parameters) {
  unsigned root = ep_;
  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned unlinked_cnt = 0;
  while (unlinked_cnt < nd_) {
    DFS(flags, root, unlinked_cnt);
    // std::cout << unlinked_cnt << '\n';
    if (unlinked_cnt >= nd_) break;
    findroot(flags, root, parameters);
    // std::cout << "new root"<<":"<<root << '\n';
  }
  for (size_t i = 0; i < nd_; ++i) {
    if (final_graph_[i].size() > width) {
      width = final_graph_[i].size();
    }
  }
}

float IndexNSG::eval_recall(std::vector<std::vector<unsigned> > query_res, std::vector<std::vector<int> > gts, int K) {
    float mean_recall = 0;
    for (unsigned i = 0; i < query_res.size(); i++) {
        // assert(query_res[i].size() <= gts[i].size());

        // if(i == 0){
        //   for(unsigned j = 0; j < K; ++j){
        //     std::cout << query_res[i][j] << ' ';
        //   }
        //   std::cout << std::endl;
        //   for(unsigned j = 0; j < K; ++j){
        //     std::cout << gts[i][j] << ' ';
        //   }
        //   std::cout << std::endl;
        // }

        float recall = 0;
        std::set<unsigned> cur_query_res_set(query_res[i].begin(), query_res[i].begin() + K);
        std::set<int> cur_query_gt(gts[i].begin(), gts[i].begin() + K);

        for (std::set<unsigned>::iterator x = cur_query_res_set.begin(); x != cur_query_res_set.end(); x++) {
            std::set<int>::iterator iter = cur_query_gt.find(*x);
            if (iter != cur_query_gt.end()) {
                recall++;
            }
        }
        recall = recall / K; 
        mean_recall += recall;
    }
    mean_recall = (mean_recall / query_res.size());

    return mean_recall;
}

std::vector<std::vector<int> > IndexNSG::load_ground_truth(const char* filename, unsigned k) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
      std::cout << "尝试打开文件: " << filename << std::endl;
      std::cout << "open file error (in load_ground_truth)" << std::endl;
      exit(-1);
  }

  unsigned dim, num;

  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);

  int* data = new int[num * dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
      in.seekg(4, std::ios::cur);
      in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();

  std::vector<std::vector<int> > res;
  for (unsigned i = 0; i < num; i++) {
      std::vector<int> a;
      unsigned cur_kk = 0;
      for (unsigned j = i * dim; j < (i + 1) * dim; j++) {
          unsigned temp_id = data[j];
          if(in_graph_[temp_id] == 0)
            continue;
          a.push_back(data[j]);
          cur_kk++;
          if(cur_kk == k)
            break;
      }
      res.push_back(a);
  }

  return res;
}

void IndexNSG::Write_disk(unsigned R, const char *filename, const char *in_graph_file, float aerfa){
  std::vector<std::vector<unsigned>> nn_list(nd_); //反向图
  std::vector<std::vector<unsigned>> prune_line(nd_); //需要筛选的边
  std::vector<std::mutex> locks(nd_);
  
  // unsigned* reverse_list = new unsigned[nd_ * (R+1)];
  // std::ifstream in(re_graph_file, std::ios::binary);
  // in.read((char *)reverse_list, nd_ * (R + 1) * sizeof(unsigned));
#pragma omp parallel
{
  #pragma omp for schedule(dynamic, 2048)
	for(unsigned i = 0; i < nd_; ++i){
    unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * i + data_len);
    unsigned MaxM = *neighbors;
    neighbors++;
    for(unsigned j = 0; j < MaxM; ++j){
      unsigned id = neighbors[j];
      LockGuard guard(locks[id]);
      nn_list[id].push_back(i);
    }
  }
}

#pragma omp parallel
{
  #pragma omp for schedule(dynamic, 2048)
  for(unsigned i = 0; i < nd_; ++i){
      if(in_graph_[i] == 1)  //只处理被删除的节点
        continue;
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * i + data_len);
      unsigned MaxM = *neighbors;
      neighbors++;
			for(unsigned j = 0; j < nn_list[i].size(); ++j){  //处理连接被删除点的点
        unsigned cur_id = nn_list[i][j];
        {
          LockGuard guard(locks[cur_id]);
          if(in_graph_[cur_id] == 0)
          continue;
          for(unsigned z = 0; z < MaxM; ++z){  //添加被删除点的邻居
            if(in_graph_[neighbors[z]] == 0)
              continue;
            prune_line[cur_id].push_back(neighbors[z]);
          }
        }
      }
  }
}

#pragma omp parallel
{
  #pragma omp for schedule(dynamic, 2048)
  for(unsigned i = 0; i < nd_; ++i){  //对边进行筛选
    if(in_graph_[i] == 0)  //如果节点已被删除则不处理
        continue;
    unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * i + data_len);
    unsigned MaxM = *neighbors;
    neighbors++;
    std::vector<Neighbor> result;
    for(unsigned j = 0; j < MaxM; ++j){
      unsigned id = neighbors[j];
      if(in_graph_[id] == 0)
        continue;
      float dist =
        distance_->compare( (float*)(opt_graph_ + node_size * i),
        (float*)(opt_graph_ + node_size * id), (unsigned)dimension_);
      // std::cout << i << std::endl;
      result.push_back(Neighbor(id, dist, true));
    }
    unsigned start = 0;
    std::sort(result.begin(), result.end());
    while(result.size() < R && start < prune_line[i].size()){
      float ratio1 = float(R - result.size()) / R;
      unsigned id = prune_line[i][start];
      bool occlude = false;
      float p_disk = distance_->compare((float*)(opt_graph_ + node_size * i),
                                        (float*)(opt_graph_ + node_size * id),
                                        (unsigned)dimension_);
      for (unsigned t = 0; t < result.size(); t++) {
        if (id == result[t].id) {
          occlude = true;
          break;
        }
        if(p_disk < result[t].distance)
          break;
        float djk = distance_->compare((float*)(opt_graph_ + node_size * result[t].id),
                                      (float*)(opt_graph_ + node_size * id),
                                       (unsigned)dimension_);   
        if ( djk < p_disk /* dik */) {
          if(p_disk * p_disk > djk * djk + ratio1 * djk * result[t].distance){  //c² > a² + ab
            occlude = true;
            break;
          }
        }
      }
      if (!occlude) {
        result.push_back(Neighbor(id, p_disk, true));
        sort(result.begin(), result.end());
      }
      ++start;
    }
    unsigned *res_id = (unsigned *)(opt_graph_ + node_size * i + data_len);
    *res_id = result.size();
    for(unsigned j = 0; j < result.size(); ++j){
      neighbors[j] = result[j].id;
    }
    
  }
}
  std::cout << "筛选边成功" << std::endl;

  std::vector<unsigned> change_id(nd_, 0);
  unsigned total_delete = 0;
  for(unsigned i = 0; i < nd_; ++i){
    if(in_graph_[i] == 0){
      total_delete ++;
    }
    change_id[i] = total_delete;
  }
#pragma omp parallel
{
  #pragma omp for schedule(dynamic, 2048)
  for(unsigned i = 0; i < nd_; ++i){
    if(in_graph_[i] == 0){
      continue;
    }
    unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * i + data_len);
    unsigned MaxM = *neighbors;
    neighbors++;
    for(unsigned j = 0; j < MaxM; ++j){
      neighbors[j] -= change_id[neighbors[j]];
    }
  }
}

  unsigned af_dele_num = nd_ - total_delete;
  ep_ -= change_id[ep_];
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  out.write((char *)&R, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  out.write((char *)&data_padding, sizeof(unsigned));
  out.write((char *)&neighbor_padding, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    if(in_graph_[i] == 0){
      continue;
    }
    out.write(reinterpret_cast<char*>(opt_graph_ + node_size * i), node_size);
  }
  out.close();
  std::ofstream out_graph(in_graph_file, std::ios::binary | std::ios::out);
  std::vector<unsigned> vec(af_dele_num, 1u);
  out_graph.write((char *)vec.data(), af_dele_num * sizeof(unsigned));

  std::cout << "保存文件成功，总共删除了" << total_delete << "个点；" << std::endl;
  std::cout << "索引文件大小为" << af_dele_num * node_size + 16 << "字节" << std::endl;
}

void IndexNSG::Search_write_disk(unsigned del_id, const char *filename, efanna2e::Parameters paras){
  unsigned R = paras.Get<unsigned>("R");
  std::vector<std::vector<unsigned>> nn_list(nd_); //反向图
  std::vector<std::mutex> locks(nd_);
  
#pragma omp parallel
{
  #pragma omp for schedule(dynamic, 2048)
	for(unsigned i = 0; i < nd_; ++i){
    unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * i + data_len);
    unsigned MaxM = *neighbors;
    neighbors++;
    for(unsigned j = 0; j < MaxM; ++j){
      unsigned id = neighbors[j];
      LockGuard guard(locks[id]);
      nn_list[id].push_back(i);
    }
  }
}


  for(unsigned zz = 0; zz < del_id; ++zz){

    for(unsigned i = 0; i < nn_list[zz].size(); ++i){  //对边进行筛选
      unsigned nei_id = nn_list[zz][i];
      unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * nei_id + data_len);
      neighbors++;
      std::vector<Neighbor> result;

      //图中查找最近邻
      std::vector<Neighbor> pool;
      boost::dynamic_bitset<> flags{nd_, 0};
      pool.clear();
      flags.reset();
      get_neighbors((float *)(opt_graph_ + node_size * nei_id), paras, flags, pool);
      unsigned start = 0;
      
        
      while(result.size() < R && start < pool.size()){
        float ratio1 = float(R - result.size()) / R;
        unsigned id = pool[start].id;
        ++start;
        if(id < del_id || id == nei_id)
          continue;
        bool occlude = false;
        float p_disk = distance_->compare((float*)(opt_graph_ + node_size * i),
                                          (float*)(opt_graph_ + node_size * id),
                                          (unsigned)dimension_);
        for (unsigned t = 0; t < result.size(); t++) {
          if (id == result[t].id) {
            occlude = true;
            break;
          }
          if(p_disk < result[t].distance)
            break;
          float djk = distance_->compare((float*)(opt_graph_ + node_size * result[t].id),
                                        (float*)(opt_graph_ + node_size * id),
                                        (unsigned)dimension_);   
          if ( djk < p_disk /* dik */) {
            if(p_disk * p_disk > djk * djk + ratio1 * djk * result[t].distance){  //c² > a² + ab
              occlude = true;
              break;
            }
          }
        }
        if (!occlude) {
          result.push_back(Neighbor(id, p_disk, true));
          sort(result.begin(), result.end());
        }
      }
      unsigned *res_id = (unsigned *)(opt_graph_ + node_size * nei_id + data_len);
      res_id[0] = result.size();
      for(unsigned j = 0; j < result.size(); ++j){
        neighbors[j] = result[j].id;
      }
      
    }

    

  }
  std::cout << "筛选边成功；" << std::endl;

  for(unsigned i = 0; i < nd_; ++i){
    if(i < del_id){
      continue;
    }
    unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * i + data_len);
    unsigned MaxM = *neighbors;
    neighbors++;
    for(unsigned j = 0; j < MaxM; ++j){
      neighbors[j] -= del_id;
    }
  }

  unsigned af_dele_num = nd_ - del_id;
  ep_ -= del_id;
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  out.write((char *)&R, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  out.write((char *)&data_padding, sizeof(unsigned));
  out.write((char *)&neighbor_padding, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    if(i < del_id){
      continue;
    }
    out.write(reinterpret_cast<char*>(opt_graph_ + node_size * i), node_size);
  }
  out.close();

  memmove(
    opt_graph_ + del_id * node_size,        // 目标地址（被删除行的起始位置）
    opt_graph_ + (del_id + 1) * node_size,  // 源地址（后续数据的起始位置）
    (nd_ - del_id - 1) * node_size  // 要移动的字节数
  );
  nd_ = af_dele_num;
}

void IndexNSG::prune_result(float* data, unsigned R, float aerfa, std::vector<unsigned>& res, unsigned add_node_id, unsigned resize, unsigned total_size, unsigned k_num){
  std::vector<Neighbor> pool;
  for (unsigned nn = 0; nn < res.size(); nn++) {
    unsigned id = res[nn];
    float dist =
        distance_->compare( data,
        (float*)(opt_graph_ + node_size * id), (unsigned)dimension_);
    pool.push_back(Neighbor(id, dist, true));
  }

  //筛选边
  unsigned start = 0;
  std::sort(pool.begin(), pool.end());
  std::vector<Neighbor> result;
  result.push_back(pool[start]);
  while (result.size() < R && (++start) < pool.size()) {
    float ratio1 = float(R - result.size()) / R;
    auto &p = pool[start];
    bool occlude = false;
    for (unsigned t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = distance_->compare((float*)(opt_graph_ + node_size * (size_t)result[t].id),
                                      (float*)(opt_graph_ + node_size * (size_t)p.id),
                                     (unsigned)dimension_);   
      if ( djk < p.distance /* dik */) {
        if(p.distance * p.distance > djk * djk + ratio1 * djk * result[t].distance){  //c² > a² + ab
          occlude = true;
          break;
        }
      }
    }
    
    if (!occlude) result.push_back(p);
  }
  

  unsigned cur_pool = 0;
  unsigned cur_k = 0;
  
  if(result.size() < R){
    for(; cur_pool < pool.size(); ++cur_pool){
      bool is_insert = true;
      for(unsigned i = 0; i < result.size(); ++i){
        if(pool[cur_pool].id == result[i].id){
          is_insert = false;
          break;
        }
      }
      if(is_insert){
        cur_k ++;
        result.push_back(pool[cur_pool]);
        if(cur_k >= k_num || result.size() >= R)
          break;
      }
    }
  }

  std::vector<unsigned> res_id;
  for(unsigned i = 0; i < result.size(); ++i){
    res_id.push_back(result[i].id);
  }

  //插入反向边

  for(unsigned i = 0; i < result.size(); ++i){
    size_t cur_id = result[i].id;  //当前点id
    unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * cur_id + data_len);
    unsigned *change_M = (unsigned *)(opt_graph_ + node_size * cur_id + data_len);
    unsigned MaxM = *neighbors;
    neighbors++;
    if(MaxM < R){
      neighbors[MaxM] = add_node_id;
      change_M[0] ++;
    }
    else{
      pool.clear();
      for (unsigned nn = 0; nn < MaxM; nn++) {
        unsigned id = neighbors[nn];  //邻居id
        float dist =
            distance_->compare( (float*)(opt_graph_ + node_size * cur_id),
            (float*)(opt_graph_ + node_size * id), (unsigned)dimension_);
        pool.push_back(Neighbor(id, dist, true));
      }
      pool.push_back(Neighbor(add_node_id, result[i].distance, true));
      std::sort(pool.begin(), pool.end());
      std::vector<Neighbor> cur_result;
      unsigned start = 0;
      cur_result.push_back(pool[start]);
      while (cur_result.size() < R && (++start) < pool.size()) {
        if(cur_result.size() < k_num){
          cur_result.push_back(pool[start]);
          continue;
        }
        float ratio1 = float(R - result.size()) / R;
        auto &p = pool[start]; //第二邻居
        bool occlude = false;
        for (unsigned t = 0; t < cur_result.size(); t++) {
          if (p.id == cur_result[t].id) {
            occlude = true;
            break;
          }
          float djk = distance_->compare((float*)(opt_graph_ + node_size * (size_t)cur_result[t].id),
                                        (float*)(opt_graph_ + node_size * (size_t)p.id),
                                        (unsigned)dimension_);   
          if ( djk < p.distance /* dik */) {
            if(p.distance * p.distance > djk * djk + ratio1 * djk * result[t].distance){  //c² > a² + ab
              if(p.id == add_node_id && add_nebor(cur_result[t].id, add_node_id, R)){  //如果被遮挡了，则将新节点插入遮挡点的邻居
                occlude = true;
                break;
              }
              if(cur_result[t].id == add_node_id){  //如果新节点未被遮挡，则将被遮挡的原节点插入新节点邻居中
                for(unsigned ri = 0; ri < res_id.size(); ++ri){
                  if(p.id == res_id[ri]){
                    occlude = true;
                    break;
                  }
                }
              }
              if(occlude)
                break;
            }
          }
        }
        if (!occlude) cur_result.push_back(p);
      }
      change_M[0] = cur_result.size();
      for(unsigned nn = 0; nn < change_M[0]; ++nn){
        neighbors[nn] = cur_result[nn].id;
      }
    }
  }

  unsigned GK = (unsigned)res_id.size();
  res_id.resize(R, -1);
  memcpy(opt_graph_ + add_node_id * node_size, data, dimension_ * sizeof(float));
  memcpy(opt_graph_ + add_node_id * node_size + data_len, &GK, sizeof(unsigned));
  memcpy(opt_graph_ + add_node_id * node_size + data_len + 4, res_id.data(), R * sizeof(unsigned));
  nd_ ++;
}

void IndexNSG::save_opt(const char *filename, const char *in_graph){
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  out.write((char *)&width, sizeof(unsigned));
  out.write((char *)&ep_, sizeof(unsigned));
  out.write((char *)&data_padding, sizeof(unsigned));
  out.write((char *)&neighbor_padding, sizeof(unsigned));
  out.write(opt_graph_, nd_ * node_size);
  out.close();
  std::ofstream out_graph(in_graph, std::ios::binary | std::ios::out);
  std::vector<unsigned> vec(nd_, 1u);
  out_graph.write((char *)vec.data(), nd_ * sizeof(unsigned));
  out_graph.close();
}

void IndexNSG::com_degree(){
  unsigned max = 0, min = 1e6, avg = 0;
  for (size_t i = 0; i < nd_; i++) {
    unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * i + data_len);
    unsigned size = *neighbors;
    max = max < size ? size : max;
    min = min > size ? size : min;
    avg += size;
  }
  avg /= 1.0 * nd_;
  printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);
}

void IndexNSG::my_realloc(unsigned total_size){
  std::cout << "分配的点数：" << total_size << std::endl;
  char* new_opt_graph = (char*)realloc(opt_graph_, total_size * node_size);
  if (!new_opt_graph) {
    // 严谨的错误处理
    free(opt_graph_);
    opt_graph_ = nullptr;  // 避免悬空指针
    throw std::bad_alloc(); // 或自定义异常
  }
  opt_graph_ = new_opt_graph;
}

bool IndexNSG::add_nebor(unsigned cur_id, unsigned new_id, unsigned R){
  unsigned *neighbors = (unsigned *)(opt_graph_ + node_size * cur_id + data_len);
  unsigned *change_M = (unsigned *)(opt_graph_ + node_size * cur_id + data_len);
  unsigned MaxM = *neighbors;
  neighbors++;
  if(MaxM < R){
    neighbors[MaxM] = new_id;
    change_M[0] ++;
    return true;
  }
  return false;  //没有被插入
}

void IndexNSG::Compute_gt(const float *query_load, const char *gt_file, unsigned K, unsigned query_num){
  std::ofstream out_gt(gt_file, std::ios::binary | std::ios::out);
  std::vector<std::vector<unsigned>> res(query_num);

#pragma omp parallel
{
#pragma omp for schedule(dynamic, 2048)
  for(unsigned i = 0; i < query_num; ++i){
    float* query = (float*)query_load + i * dimension_;
    std::vector<Neighbor> result;
    for(unsigned j = 0; j < nd_; ++j){
      float disk = distance_->compare((float*)(opt_graph_ + node_size * j),
                                      query,
                                      (unsigned)dimension_);   
      result.push_back(Neighbor(j, disk, true));
    }
    sort(result.begin(), result.end());
    
    for(unsigned j = 0; j < K; j++){
      res[i].push_back(result[j].id);
    }
    // float percentage = static_cast<float>(i + 1) / query_num * 100;
    // // 输出百分比，保留两位小数
    // std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << percentage << "%";
    // std::cout.flush();  // 刷新输出缓冲区
  }
}
  for(unsigned i = 0; i < query_num; ++i){
    out_gt.write((char *)&K, sizeof(unsigned));
    out_gt.write((char *)res[i].data(), K * sizeof(unsigned));
  }
  out_gt.close();
}

}
