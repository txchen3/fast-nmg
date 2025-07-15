# #!/bin/bash
datasets=("crawl")
# datasets=("gist" "crawl" "glove" "deep" "gauss")
delete_num=20000
dim=300
aerfa=0.27
R=50
k_num=5
C=$((R * 2))


# tests/test_nsg_index "/data/crawl/data.fvecs" "../../index/crawl.knng" $R $R $C test1.nsg $aerfa 5 ingraph1.bin
# echo "增删前搜索性能："
# for i in {100..200..10}; do
#     tests/test_nsg_optimized_search search "/data/crawl/query.fvecs" test1.nsg $i 100 result.ivecs "/data/crawl/true.ivecs" 0 ingraph1.bin
# done


for dataset in "${datasets[@]}"; do
    start11=$(date +%s.%N)  # 开始时间戳
    offset=0
    cp "${dataset}1.nsg" test.nsg
    cp "${dataset}_reverse_graph" reverse_graph
    cp "${dataset}_ingraph.bin" ingraph.bin
    # tests/test_nsg_index "/data/${dataset}/data.fvecs" "../../index/${dataset}.knng" $R $R $C test.nsg $aerfa $k_num ingraph.bin reverse_graph
    # echo "增删前搜索性能："
    # for i in {100..200..10}; do
    #     tests/test_nsg_optimized_search search "/data/${dataset}/query.fvecs" test.nsg $i 100 result.ivecs "/data/${dataset}/true.ivecs" 0 ingraph.bin
    # done

    for((iter=0; iter<10; ++iter)); do
        #=================================================删除操作==============================================================
        start=$(date +%s.%N)  # 开始时间戳
        tests/test_nsg_optimized_search delete ingraph.bin 0 $delete_num
        end=$(date +%s.%N)    # 结束时间戳

        tests/test_nsg_optimized_search write_disk test.nsg ingraph.bin "/data/${dataset}/true.ivecs" $dim $R $aerfa
        end1=$(date +%s.%N)    # 结束时间戳
        duration=$(echo "$end1 - $start" | bc -l | awk '{printf "%.4f", $0}')
        echo "删除总共耗时: ${duration}秒"

        # #===============================================计算groundtrue===========================================================
        # tests/test_nsg_optimized_search compute_gt test.nsg ingraph.bin "/data/${dataset}/query.fvecs" "${dataset}.ivecs" 100  #生成groundtrue

        #===================================================搜索=================================================================
        if (( iter == 0 )); then
            echo "删除后搜索性能："
            for i in {100..200..10}; do
                tests/test_nsg_optimized_search search "/data/${dataset}/query.fvecs" test.nsg $i 100 result.ivecs "${dataset}_global.ivecs" 0 ingraph.bin
            done
        fi

        #===============================================重新插入删除的节点=========================================================
        start=$(date +%s.%N)  # 开始时间戳

        tests/test_nsg_optimized_search add_node test.nsg "/data/${dataset}/data.fvecs" $delete_num $R $aerfa $dim $offset $k_num ingraph.bin

        end=$(date +%s.%N)    # 结束时间戳
        duration=$(echo "$end - $start" | bc -l | awk '{printf "%.4f", $0}')
        echo "添加节点耗时: ${duration}秒"

        
        ((offset+=delete_num))
    done
    end11=$(date +%s.%N)    # 结束时间戳
    duration=$(echo "$end11 - $start11" | bc -l | awk '{printf "%.4f", $0}')
    echo "删增总共耗时: ${duration}秒"

    # #===============================================计算groundtrue===========================================================
    # tests/test_nsg_optimized_search compute_gt test.nsg ingraph.bin "/data/${dataset}/query.fvecs" "${dataset}_a10_insert.ivecs" 100  #生成groundtrue

    # ================================================插入节点后搜索============================================================
    echo "删增后搜索性能："
    for i in {100..200..10}; do
        tests/test_nsg_optimized_search search "/data/${dataset}/query.fvecs" test.nsg $i 100 result.ivecs "${dataset}_a10_insert.ivecs" 0 ingraph.bin
    done

done

# echo "删增后搜索性能："
# for i in {100..200..10}; do
#     tests/test_nsg_optimized_search search "/data/sift/query.fvecs" test.nsg $i 100 result.ivecs "sift_a_insert.ivecs" 0 ingraph.bin
# done

# tests/test_nsg_optimized_search add_node test.nsg "/data/crawl/data.fvecs" 1 50 0.27 300 0 0 ingraph.bin