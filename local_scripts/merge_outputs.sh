datasplit=test
for seed in {5..14}
do
    for model in 'gpt2' 'gpt2-medium' 'gpt2-large' 'gpt2-xl'
    do
        for top_p in 1.0
        do
            python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit ${datasplit} --seed ${seed}
        done
    done
    for model in 'gpt2' 'gpt2-medium'
    do
        for top_p in .9
        do
            python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit ${datasplit} --seed ${seed}
        done
    done
    
    for model in 'gpt2-large' 'gpt2-xl'
    do
        for top_p in .95
        do
            python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit ${datasplit} --seed ${seed}
        done
    done
done
