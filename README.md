This codebase is developed on top of https://github.com/ContextualAI/gritlm
Carefully install the requirements and follow the instructions from gritlm.
Specifically, you need to use a modified modelling_mistral.py in your transformers library to enable bidirectional attention.


#using mteb script
```
python ./eval_mteb.py --model_name_or_path raghavlight/TDTE  --instruction_set e5 --instruction_format mistral --task_names ArguAna --attn bbcc --attn_implementation sdpa --batch_size 32
```

#example usage
```
python example_usage.py
```