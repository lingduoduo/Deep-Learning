cd .
export PYTHONPATH=:.

table_name=code_test_split
models=model1,model2
# log_file_name="logs/${table_name}_${online_model_names}.log"
b=`echo $models |awk -F ',' '{for(i=1;i<=NF;i++){print $i}}'`
# 进行循环并打印
for model in $b
do
echo $model
echo ---------------------------------
# problem_file="/mnt/SuperCLUE/SuperCLUE-eval/human_eval/data/problem_${table_name}_${model}.jsonl"
# samples_file="/mnt/SuperCLUE/SuperCLUE-eval/human_eval/data/samples_${table_name}_${model}.jsonl"
problem_file="./data/${table_name}/problem_${table_name}_${model}.jsonl"
samples_file="./data/${table_name}/samples_${table_name}_${model}.jsonl"
python ./human_eval/evaluate_functional_correctness.py \
--n_workers 4 \
--problem_file $problem_file \
--sample_file $samples_file

echo "$model done"
done
