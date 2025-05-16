dataset="gauss"

delete_num=10000
dim=128
aerfa=0.05
R=50
k_num=5
C=$((R * 2))

tests/test_nsg_index "/data/${dataset}/data.fvecs" "../../index/${dataset}.knng" $R $R $C "${dataset}1.nsg" $aerfa $k_num "${dataset}_ingraph.bin" "${dataset}_reverse_graph"
echo "增删前搜索性能："
for i in {100..200..10}; do
    tests/test_nsg_optimized_search search "/data/${dataset}/query.fvecs" "${dataset}1.nsg" $i 100 result.ivecs "/data/${dataset}/true.ivecs" 0 "${dataset}_ingraph.bin"
done

cp "${dataset}1.nsg" test.nsg
cp "${dataset}_reverse_graph" reverse_graph
cp "${dataset}_ingraph.bin" ingraph.bin

start11=$(date +%s.%N)  # 开始时间戳
tests/test_nsg_optimized_search global_del test.nsg $delete_num true.ivecs $dim $R $aerfa reverse_graph
end11=$(date +%s.%N)    # 结束时间戳
duration=$(echo "$end11 - $start11" | bc -l | awk '{printf "%.4f", $0}')
echo "删除总共耗时: ${duration}秒"
tests/test_nsg_optimized_search compute_gt test.nsg ingraph.bin "/data/${dataset}/query.fvecs" "${dataset}_global.ivecs" 100  #生成groundtrue

for i in {200..400..20}; do
    tests/test_nsg_optimized_search search "/data/${dataset}/query.fvecs" test.nsg $i 100 result.ivecs "${dataset}_global.ivecs" 0 ingraph.bin
done