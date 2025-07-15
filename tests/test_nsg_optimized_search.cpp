#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <chrono>
#include <string>
#include <set>
#include <algorithm>

void load_data(char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  // std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

void load_id(char* filename, float*& data,
  unsigned& dim, unsigned offset, unsigned total_num) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  data = new float[(size_t)total_num * (size_t)dim];

  in.seekg(4 * (dim + 1) * offset, std::ios::beg);
  for (size_t i = 0; i < total_num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

void real_size(char* filename, unsigned& num, unsigned dim) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
  std::cout << "open file error" << std::endl;
  exit(-1);
  }
  unsigned width, temp;
  in.read((char *)&width, sizeof(unsigned));
  in.read((char *)&temp, sizeof(unsigned));
  unsigned data_pad, nei_pad;
  in.read((char *)&data_pad, sizeof(unsigned));
  in.read((char *)&nei_pad, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t node_size = (dim + data_pad) * sizeof(float) + sizeof(unsigned) + (width + nei_pad) * sizeof(unsigned); 
  num = (unsigned)(fsize / node_size);
  in.close();
}

void save_result(const char* filename,
                 std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);

  for (unsigned i = 0; i < results.size(); i++) {
    unsigned GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

std::vector<std::vector<int> > load_ground_truth(const char* filename) {
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

    int* data = new int[num * dim * sizeof(int)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();

    std::vector<std::vector<int> > res;
    for (unsigned i = 0; i < num; i++) {
        std::vector<int> a;
        for (unsigned j = i * dim; j < (i + 1) * dim; j++) {
            a.push_back(data[j]);
        }
        res.push_back(a);
    }

    return res;
}

void Delete_id(char* filename, unsigned& start_id, unsigned& end_id){
  std::fstream file(filename, 
  std::ios::binary |  // 二进制模式
  std::ios::in  |     // 可读
  std::ios::out);     // 可写
  std::streampos pos = start_id * sizeof(unsigned);
  file.seekp(pos);
  unsigned zero = 0;
  for(unsigned i = start_id; i < end_id; ++i)
    file.write(reinterpret_cast<char*>(&zero), sizeof(unsigned));
  // std::cout << "成功标记节点" << move_id << "为删除" << std::endl;
  // file.seekp(0);
  // char* buffer = new char[1000000 * sizeof(unsigned)];
  // file.read(buffer, 1000000 * sizeof(unsigned));
  // unsigned* data = reinterpret_cast<unsigned*>(buffer); // 后续使用data
  // unsigned totalde = 0;
  // for(unsigned i = 0; i < 1000000; ++i){
  //   if(data[i] == 0)
  //     totalde ++;
  // }
  // std::cout << "已有删除点数为：" << totalde << std::endl;
  file.close();
}


int main(int argc, char** argv) {
  if (argc < 2) {
    printf("错误：未提供参数。\n");
    return 1;
  }

  if(strcmp(argv[1], "search") == 0){
    if (argc != 10) {
      std::cout << argv[0]
                << " <search> <query_file> <nsg_path> <search_L> <search_K> <result_path> <true> <k_num> <in_graph_file>"
                << std::endl;
      exit(-1);
    }
  }
  else if(strcmp(argv[1], "delete") == 0){
    if (argc != 5) {
      std::cout << argv[0]
                << " <delete> <in_graph_file> <start_id> <end_id>"
                << std::endl;
      exit(-1);
    }
    unsigned start_id = (unsigned)atoi(argv[3]);
    unsigned end_id = (unsigned)atoi(argv[4]);
    std::cout << end_id << std::endl;
    Delete_id(argv[2], start_id, end_id);
    return 0;
  }
  else if(strcmp(argv[1], "write_disk") == 0){
    if (argc != 8) {
      std::cout << argv[0]
                << " <write_disk> <nsg_path> <in_graph_file> <true> <dim> <R> <α>"
                << std::endl;
      exit(-1);
    }
    unsigned dim = (unsigned)atoi(argv[5]);
    unsigned R = (unsigned)atoi(argv[6]);
    float aerfa = (float)atof(argv[7]);
    unsigned points_num;
    real_size(argv[2], points_num, dim);
    efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
    std::cout << "in_graph_file: " << argv[3] << std::endl;
    index.Load_part_point(argv[2], argv[3]);
    index.Write_disk(R, argv[2], argv[3], aerfa);
    index.com_degree();
    return 0;
  }
  else if(strcmp(argv[1], "global_del") == 0){
    if (argc != 8) {
      std::cout << argv[0]
                << " <global_del> <nsg_path> <id> <true> <dim> <R> <α>"
                << std::endl;
      exit(-1);
    }
    unsigned del_id = (unsigned)atoi(argv[3]);
    unsigned dim = (unsigned)atoi(argv[5]);
    unsigned R = (unsigned)atoi(argv[6]);
    float aerfa = (float)atof(argv[7]);
    unsigned points_num;
    real_size(argv[2], points_num, dim);
    efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
    index.Load(argv[2]);
    efanna2e::Parameters paras;
    unsigned L = R * 2;
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("C", L + R);
    paras.Set<unsigned>("aerfa", aerfa);
    index.Search_write_disk(del_id, argv[2], paras);
    return 0;
  }
  else if(strcmp(argv[1], "compute_gt") == 0){
    if (argc != 7) {
      std::cout << argv[0]
                << " <compute_gt> <nsg_path> <in_graph_file> <query_file> <true> <K>"
                << std::endl;
      exit(-1);
    }
    
    float* query_load = NULL;
    unsigned query_num, query_dim, dim;
    load_data(argv[4], query_load, query_num, query_dim);
    unsigned K = (unsigned)atoi(argv[6]);
    unsigned points_num;
    dim = query_dim;
    real_size(argv[2], points_num, dim);
    efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
    index.Load_part_point(argv[2], argv[3]);
    index.Compute_gt(query_load, argv[5], K, query_num);
    return 0;
  }
  else if(strcmp(argv[1], "add_node") == 0){
    if (argc != 11) {
      std::cout << argv[0]
                << " <add_node> <nsg_path> <data_file> <total_num> <R> <aerfa> <dim> <offset> <k_num> <in_graph>"
                << std::endl;
      exit(-1);
    }
    
    unsigned query_dim;
    unsigned total_num = (unsigned)atoi(argv[4]);
    unsigned R = (unsigned)atoi(argv[5]);
    float aerfa = (float)atof(argv[6]);
    unsigned dim = (unsigned)atoi(argv[7]);
    unsigned offset = (unsigned)atoi(argv[8]);
    unsigned k_num = (unsigned)atoi(argv[9]);
    
    unsigned points_num;
    real_size(argv[2], points_num, dim);
    efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
    auto s = std::chrono::high_resolution_clock::now();
    index.Load(argv[2]);
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "索引加载用时：" << diff.count() <<"\n";
    unsigned L = 200;
    int com_num = 0;

    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);
    paras.Set<unsigned>("k_num", 0);
    index.my_realloc(total_num + points_num);
    float* query_load = NULL;
    load_id(argv[3], query_load, query_dim, offset, total_num);
    std::vector<std::mutex> locks(total_num + points_num);
    for(unsigned i = 0; i < total_num; i++){
      std::vector<unsigned> res(L);
      index.SearchWithOptGraph(query_load + i * dim, L, paras, res.data(), com_num);
      index.prune_result(query_load + i * dim, R, aerfa, res, points_num + i, i, total_num, k_num);
    }

    index.com_degree();
    index.save_opt(argv[2], argv[10]);
    return 0;
  }
  else if(strcmp(argv[1], "help") == 0){
    std::cout << argv[0] << " <mode>" << std::endl;
    std::cout << "其中mode可选search、delete、write_disk" << std::endl;
    std::cout << "节点编号从0开始" << std::endl;
    return 0;
  }
  else{
    printf("错误输入参数\n");
    return 1;
  }
  
  unsigned points_num, dim;
  
  float* query_load = NULL;
  unsigned query_num, query_dim;
  load_data(argv[2], query_load, query_num, query_dim);
  dim = query_dim;
  real_size(argv[3], points_num, dim);

  unsigned L = (unsigned)atoi(argv[4]);
  unsigned K = (unsigned)atoi(argv[5]);
  unsigned k_num = (unsigned)atoi(argv[8]);
  

  if (L < K) {
    std::cout << "search_L cannot be smaller than search_K!" << std::endl;
    exit(-1);
  }

  efanna2e::IndexNSG index(dim, points_num, efanna2e::FAST_L2, nullptr);

  index.Load_part_point(argv[3], argv[9]);
  

  std::vector<std::vector<int> > gts = index.load_ground_truth(argv[7], K);
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L_search", L);
  paras.Set<unsigned>("P_search", L);
  paras.Set<unsigned>("k_num", k_num);

  std::vector<std::vector<unsigned> > res(query_num);
  for (unsigned i = 0; i < query_num; i++) res[i].resize(K);

  int com_num = 0;
  auto s = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < query_num; i++) {
    index.SearchWithOptGraph(query_load + i * dim, K, paras, res[i].data(), com_num);
  }
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  float recall = index.eval_recall(res, gts, K);
  std::cout << recall << ' ' << diff.count() << ' ' << com_num/query_num <<"\n";

  save_result(argv[6], res);

  return 0;
}
