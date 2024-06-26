This codebase is developed on top of https://github.com/ContextualAI/gritlm .
Carefully install the requirements and follow the instructions from gritlm.
Specifically, you need to use a modified **modelling_mistral_gritlm.py** in your transformers library to enable bidirectional attention. Module requirements are same as gritlm.

## Model Details
- TDTE model uses a base bidirectional Mistral7Bv0.2 + Lora adapter.
- It is trained using transformed versions of publically available retrieval datasets. TDTE dataset will be released soon.
- More details coming soon

## using mteb evaluation script
```
python ./eval_mteb.py --model_name_or_path <HUGGINFACE_MODEL_PATH>  --instruction_set e5 --instruction_format mistral --task_names ArguAna --attn bbcc --attn_implementation sdpa --batch_size 32
```

## example usage
```
python example_usage.py
```