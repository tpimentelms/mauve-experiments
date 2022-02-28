for seed in 0 1 2
do
    for model in 'gpt2' 'gpt2-medium' 'gpt2-large' 'gpt2-xl'
    do
        for top_p in 1.0
        do
            # python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit train --seed 0
            # python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit valid --seed 0
            python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit test --seed ${seed}
        done
    done
    for model in 'gpt2' 'gpt2-medium'
    do
        for top_p in .9
        do
            # python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit train --seed 0
            # python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit valid --seed 0
            python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit test --seed ${seed}
        done
    done
    
    for model in 'gpt2-large' 'gpt2-xl'
    do
        for top_p in .95
        do
            # python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit train --seed 0
            # python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit valid --seed 0
            python merge_samples.py --data_dir webtext --model ${model} --top_p ${top_p} --datasplit test --seed ${seed}
        done
    done
done
