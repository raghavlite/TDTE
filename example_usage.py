from tdte import TDTE
import torch
# Loads the model for both capabilities; If you only need embedding pass `mode="embedding"` to save memory (no lm head)


kwargs = {
    "model_name_or_path": "<HUGGINFACE_MODEL_PATH>",
    # Normalizing embeddings here. Not needed inside eval_mteb.
    "normalized": True,
    "mode": "embedding",
    "pooling_method": "mean",
    "attn": "bbcc",
    "embed_eos":"\n[/INST]",
    # base model parameters
    "torch_dtype": torch.bfloat16,
    "attn_implementation": "sdpa",
}




model = TDTE(**kwargs)
# Needs atleast 24gb of GPU memory

### Embedding/Representation ###
instruction = "Given a scientific paper title, retrieve the paper's abstract"
queries = ['Bitcoin: A Peer-to-Peer Electronic Cash System', 'Generative Representational Instruction Tuning']
documents = [
    "A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution. Digital signatures provide part of the solution, but the main benefits are lost if a trusted third party is still required to prevent double-spending. We propose a solution to the double-spending problem using a peer-to-peer network. The network timestamps transactions by hashing them into an ongoing chain of hash-based proof-of-work, forming a record that cannot be changed without redoing the proof-of-work. The longest chain not only serves as proof of the sequence of events witnessed, but proof that it came from the largest pool of CPU power. As long as a majority of CPU power is controlled by nodes that are not cooperating to attack the network, they'll generate the longest chain and outpace attackers. The network itself requires minimal structure. Messages are broadcast on a best effort basis, and nodes can leave and rejoin the network at will, accepting the longest proof-of-work chain as proof of what happened while they were gone.",
    "All text-based language problems can be reduced to either generation or embedding. Current models only perform well at one or the other. We introduce generative representational instruction tuning (GRIT) whereby a large language model is trained to handle both generative and embedding tasks by distinguishing between them through instructions. Compared to other open models, our resulting GritLM 7B sets a new state of the art on the Massive Text Embedding Benchmark (MTEB) and outperforms all models up to its size on a range of generative tasks. By scaling up further, GritLM 8X7B outperforms all open generative language models that we tried while still being among the best embedding models. Notably, we find that GRIT matches training on only generative or embedding data, thus we can unify both at no performance loss. Among other benefits, the unification via GRIT speeds up Retrieval-Augmented Generation (RAG) by > 60% for long documents, by no longer requiring separate retrieval and generation models. Models, code, etc. are freely available at https://github.com/ContextualAI/gritlm."
]

def mistral_instruction_format(instruction):
    return "<s>[INST]\n" + instruction + "\n<|embed|>\n" if instruction else "<s>[INST]\n<|embed|>\n"


# No need to add instruction for retrieval documents
d_rep = model.encode(documents, instruction=mistral_instruction_format(""))
q_rep = model.encode(queries, instruction=mistral_instruction_format(instruction))


from scipy.spatial.distance import cosine
cosine_sim_q0_d0 = 1 - cosine(q_rep[0], d_rep[0])
cosine_sim_q0_d1 = 1 - cosine(q_rep[0], d_rep[1])
cosine_sim_q1_d0 = 1 - cosine(q_rep[1], d_rep[0])
cosine_sim_q1_d1 = 1 - cosine(q_rep[1], d_rep[1])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (queries[0][:15], documents[0][:15], cosine_sim_q0_d0))
# Cosine similarity between "Bitcoin: A Peer" and "A purely peer-t" is: 0.744
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (queries[0][:15], documents[1][:15], cosine_sim_q0_d1))
# Cosine similarity between "Bitcoin: A Peer" and "All text-based " is: 0.423
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (queries[1][:15], documents[0][:15], cosine_sim_q1_d0))
# Cosine similarity between "Generative Repr" and "A purely peer-t" is: 0.355
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (queries[1][:15], documents[1][:15], cosine_sim_q1_d1))
# Cosine similarity between "Generative Repr" and "All text-based " is: 0.676

# import ipdb; ipdb.set_trace()