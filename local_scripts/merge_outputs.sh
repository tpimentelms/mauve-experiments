

for model in 'gpt2' 'gpt2-medium' 'gpt2-large' 'gpt2-xl'
do
    for top_p in .9 .95
    do
        python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit test --seed 0
        # python merge_samples.py --data_dir webtext --model gpt2-xl --seed 0 --datasplit test --top_p .95
    done
done