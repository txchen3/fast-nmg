#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

void load_data(char* filename, float*& data, unsigned& num,unsigned& dim){// load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
  in.read((char*)&dim,4);
  std::cout<<"data dimension: "<<dim<<std::endl;
  in.seekg(0,std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim+1) / 4);
  std::cout << "num: " << num << std::endl;
  long float_size = (long)num * dim;
  data = new (std::nothrow) float[float_size];
  std::cout << "理论分配: " 
          << float_size * 4 / (1024.0 * 1024 * 1024) 
          << " GB" << std::endl;

  in.seekg(0,std::ios::beg);
  for(size_t i = 0; i < num; i++){
    in.seekg(4,std::ios::cur);
    in.read((char*)(data+i*dim),dim*4);
  }
  in.close();
}

int main(int argc, char** argv) {
  if (argc != 10) {
    std::cout << argv[0] << " data_file nn_graph_path L R C save_graph_file r k_num out_graph_file"
              << std::endl;
    exit(-1);
  }
  float* data_load = NULL;
  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  std::cout << "数据加载成功" << std::endl;

  std::string nn_graph_path(argv[2]);
  unsigned L = (unsigned)atoi(argv[3]);
  unsigned R = (unsigned)atoi(argv[4]);
  unsigned C = (unsigned)atoi(argv[5]);
  float aerfa = (float)atof(argv[7]);
  unsigned k_num = (unsigned)atoi(argv[8]);

  // data_load = efanna2e::data_align(data_load, points_num, dim);//one must
  // align the data before build
  efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);

  auto s = std::chrono::high_resolution_clock::now();
  efanna2e::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("aerfa", aerfa);
  paras.Set<unsigned>("k_num", k_num);
  paras.Set<std::string>("nn_graph_path", nn_graph_path);
  index.Build(points_num, data_load, paras);
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;

  std::cout << "indexing time: " << diff.count() << "\n";
  // index.Save(argv[6]);
  index.Save_part_point(argv[6], argv[9]);

  return 0;
}
