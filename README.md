```shell
$ cd build/tests/
$ ./test_nsg_optimized_search DATA_PATH QUERY_PATH NSG_PATH SEARCH_L SEARCH_K RESULT_PATH
```

输入help
```shell
$ ./test_nsg_optimized_search help
```
可以查看所有功能

输入某一功能
```shell
$ ./test_nsg_optimized_search search
```
可以查看具体需要什么参数

编译代码：
```shell
$ cd nmg/
$ mkdir build/ && cd build/
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j
```

具体示例：
构建索引：
```shell
$ tests/test_nsg_index /data/sift/data.fvecs ../../index/sift.knng 40 50 100 test.nsg 0.23 0 ingraph.bin
```

标记数据集中编号为0-9999的节点为删除：
```shell
$ tests/test_nsg_optimized_search delete ingraph.bin 0 10000
```

将标记为删除的节点从索引中删除：
```shell
$ tests/test_nsg_optimized_search write_disk test.nsg ingraph.bin true.ivecs 128 50 0.23
```

这是一个对比方法，将索引中前10000个节点从索引中删除：
```shell
$ tests/test_nsg_optimized_search global_del test.nsg 10000 true.ivecs 128 50 0.21
```

计算现有数据的精确top100：
```shell
$ tests/test_nsg_optimized_search compute_gt test.nsg ingraph.bin /data/sift/query.fvecs true.ivecs 100
```

搜索最近邻：
```shell
$ tests/test_nsg_optimized_search search /data/sift/query.fvecs test.nsg 100 100 result.ivecs true.ivecs 0 ingraph.bin
```

添加节点：
```shell
$ tests/test_nsg_optimized_search add_node test.nsg /data/sift/data.fvecs 10000 50 0.23 128 0 5 ingraph.bin
```

