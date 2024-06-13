import argparse
import os
from functools import partial

from mteb import MTEB
import torch
from importlib import reload
import tdte.tdte as tdte
reload(tdte)
from tdte.tdte import TDTE
import time
from peft import PeftModel, PeftConfig
import time


SET_TO_TASK_TO_DS_TO_PROMPT = {
    # https://github.com/microsoft/unilm/blob/16da2f193b9c1dab0a692c6e4380bd43e70a40cd/e5/utils.py#L93
    'e5': {
        "Classification": {
            'AmazonCounterfactualClassification': 'Classify a given Amazon customer review text as either counterfactual or not-counterfactual',
            'AmazonPolarityClassification': 'Classify Amazon reviews into positive or negative sentiment',
            'AmazonReviewsClassification': 'Classify the given Amazon review into its appropriate rating category',
            'Banking77Classification': 'Given a online banking query, find the corresponding intents',
            'EmotionClassification': 'Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise',
            'ImdbClassification': 'Classify the sentiment expressed in the given movie review text from the IMDB dataset',
            'MassiveIntentClassification': 'Given a user utterance as query, find the user intents',
            'MassiveScenarioClassification': 'Given a user utterance as query, find the user scenarios',
            'MTOPDomainClassification': 'Classify the intent domain of the given utterance in task-oriented conversation',
            'MTOPIntentClassification': 'Classify the intent of the given utterance in task-oriented conversation',
            'ToxicConversationsClassification': 'Classify the given comments as either toxic or not toxic',
            'TweetSentimentExtractionClassification': 'Classify the sentiment of a given tweet as either positive, negative, or neutral',
        },
        "Clustering": {
            'ArxivClusteringP2P': 'Identify the main and secondary category of Arxiv papers based on the titles and abstracts',
            'ArxivClusteringS2S': 'Identify the main and secondary category of Arxiv papers based on the titles',
            'BiorxivClusteringP2P': 'Identify the main category of Biorxiv papers based on the titles and abstracts',
            'BiorxivClusteringS2S': 'Identify the main category of Biorxiv papers based on the titles',
            'MedrxivClusteringP2P': 'Identify the main category of Medrxiv papers based on the titles and abstracts',
            'MedrxivClusteringS2S': 'Identify the main category of Medrxiv papers based on the titles',
            'RedditClustering': 'Identify the topic or theme of Reddit posts based on the titles',
            'RedditClusteringP2P': 'Identify the topic or theme of Reddit posts based on the titles and posts',
            'StackExchangeClustering': 'Identify the topic or theme of StackExchange posts based on the titles',
            'StackExchangeClusteringP2P': 'Identify the topic or theme of StackExchange posts based on the given paragraphs',
            'TwentyNewsgroupsClustering': 'Identify the topic or theme of the given news articles',
        },
        "PairClassification": {
            'SprintDuplicateQuestions': 'Retrieve duplicate questions from Sprint forum',
            'TwitterSemEval2015': 'Retrieve tweets that are semantically similar to the given tweet',
            'TwitterURLCorpus': 'Retrieve tweets that are semantically similar to the given tweet',
        },
        "Reranking": {
            'AskUbuntuDupQuestions': {
                'query': 'Retrieve duplicate questions from AskUbuntu forum',
                'corpus': 'Retrieve duplicate questions from AskUbuntu forum',
            },
            'MindSmallReranking': {
                'query': 'Retrieve relevant news articles based on user browsing history',
                'corpus': 'Retrieve relevant news articles based on user browsing history',
            },
            'SciDocsRR': {
                'query': 'Given a title of a scientific paper, retrieve the titles of other relevant papers',
                'corpus': 'Given a title of a scientific paper, retrieve the titles of other relevant papers',
            },
            'StackOverflowDupQuestions': {
                'query': 'Retrieve duplicate questions from StackOverflow forum',
                'corpus': 'Retrieve duplicate questions from StackOverflow forum',
            },
        },
        'Retrieval': {
            'ArguAna': {
                'query': 'Given a claim, find documents that refute the claim',
                'corpus': '',
            },
            'ClimateFEVER': {
                'query': 'Given a claim about climate change, retrieve documents that support or refute the claim',
                'corpus': '',
            },
            'CQADupstackRetrieval': {
                'query': 'Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question',
                'corpus': '',
            },
            'DBPedia': {
                'query': 'Given a query, retrieve relevant entity descriptions from DBPedia',
                'corpus': '',
            },
            'FEVER': {
                'query': 'Given a claim, retrieve documents that support or refute the claim',
                'corpus': '',
            },
            'FiQA2018': {
                'query': 'Given a financial question, retrieve user replies that best answer the question',
                'corpus': '',
            },
            'HotpotQA': {
                'query': 'Given a multi-hop question, retrieve documents that can help answer the question',
                'corpus': '',
            },
            'MSMARCO': {
                'query': 'Given a web search query, retrieve relevant passages that answer the query',
                'corpus': '',
            },
            'NFCorpus': {
                'query': 'Given a question, retrieve relevant documents that best answer the question',
                'corpus': '',
            },
            'NQ': {
                'query': 'Given a question, retrieve Wikipedia passages that answer the question',
                'corpus': '',
            },
            'QuoraRetrieval': {
                'query': 'Given a question, retrieve questions that are semantically equivalent to the given question',
                'corpus': '',
            },
            'SCIDOCS': {
                'query': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper',
                'corpus': '',
            },
            'SciFact': {
                'query': 'Given a scientific claim, retrieve documents that support or refute the claim',
                'corpus': '',
            },
            'Touche2020': {
                'query': 'Given a question, retrieve detailed and persuasive arguments that answer the question',
                'corpus': '',
            },
            'TRECCOVID': {
                'query': 'Given a query on COVID-19, retrieve documents that answer the query',
                'corpus': '',
            },
        },
        'STS': {
            'STS12': "Retrieve semantically similar text.",
            'STS13': "Retrieve semantically similar text.",
            'STS14': "Retrieve semantically similar text.",
            'STS15': "Retrieve semantically similar text.",
            'STS16': "Retrieve semantically similar text.",
            'STS17': "Retrieve semantically similar text.",
            'STS22': "Retrieve semantically similar text.",
            'BIOSSES': "Retrieve semantically similar text.",
            'SICK-R': "Retrieve semantically similar text.",
            'STSBenchmark': "Retrieve semantically similar text.",
        },            
        'Summarization': {
            'SummEval': "Given a news summary, retrieve other semantically similar summaries",
        },
    },
    }

SET_TO_TASK_TO_DS_TO_SHOTS = {
    "e5": {
        "Classification": {
            "Banking77Classification": [
                "I am still waiting on my card?", 
                "card_arrival"
            ],
            "EmotionClassification": [
                "ive been feeling a little burdened lately wasnt sure why that was", 
                "sadness"
            ],
            "ImdbClassification": [
                "If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />One might feel virtuous for sitting thru it because it touches on so many IMPORTANT issues but it does so without any discernable motive. The viewer comes away with no new perspectives (unless one comes up with one while one's mind wanders, as it will invariably do during this pointless film).<br /><br />One might better spend one's time staring out a window at a tree growing.<br /><br />", 
                "negative"
            ],
        },
        "Clustering": {
            "BiorxivClusteringS2S": [
                "Association of CDH11 with ASD revealed by matched-gene co-expression analysis and mouse behavioral studies",
                "neuroscience",
            ],
        },
        "PairClassification": {
            "SprintDuplicateQuestions": [
                "Why is it impossible for me to find a easy way to send a picture with text on my Kyocera DuraCore ?",
                "Send or receive a picture with text - Kyocera DuraCore",
            ],
            "TwitterSemEval2015": [
                "The Ending to 8 Mile is my fav part of the whole movie",
                "Those last 3 battles in 8 Mile are THE shit",
            ],
            "TwitterURLCorpus": [
                "Liberals , dont let Donald Trump tarnish L.L. Beans sterling brand reputation ",
                "Liberals, Don&rsquo;t Let Donald Trump Tarnish L.L. Bean&rsquo;s Sterling Brand Reputation",
            ],            
        },
        "Reranking": {
            "AskUbuntuDupQuestions": {
                "query": [
                    "what is a short cut i can use to switch applications ?",
                    "keyboard short cut for switching between two or more instances of the same application ?",
                ],
                "corpus": [
                    "keyboard short cut for switching between two or more instances of the same application ?",
                    "what is a short cut i can use to switch applications ?",
                ],
            },
        },
        "Retrieval": {
            'ArguAna': {
                'query': [
                    "People will die if we don’t do animal testing Every year, 23 new drugs are introduced in the UK alone.[13] Almost all will be tested on animals. A new drug will be used for a long time. Think of all the people saved by the use of penicillin. If drugs cost more to test, that means drug companies will develop less. This means more people suffering and dying",
                    "animals science science general ban animal testing junior" + " " +  "Many of these drugs are “me too” drugs – ones with a slight change that doesn’t make much difference to an existing drug. [14] So often the benefits from animal testing are marginal, and even if there was a slight increase in human suffering, it would be worth it based on the animal suffering saved.",
                ],
                'corpus': [
                    "animals science science general ban animal testing junior" + " " +  "Many of these drugs are “me too” drugs – ones with a slight change that doesn’t make much difference to an existing drug. [14] So often the benefits from animal testing are marginal, and even if there was a slight increase in human suffering, it would be worth it based on the animal suffering saved.",
                    "People will die if we don’t do animal testing Every year, 23 new drugs are introduced in the UK alone.[13] Almost all will be tested on animals. A new drug will be used for a long time. Think of all the people saved by the use of penicillin. If drugs cost more to test, that means drug companies will develop less. This means more people suffering and dying",
                ],
            },
            'SCIDOCS': {
                'query': [
                    "A Direct Search Method to solve Economic Dispatch Problem with Valve-Point Effect",
                    "A Hybrid EP and SQP for Dynamic Economic Dispatch with Nonsmooth Fuel Cost Function" + " " + "Dynamic economic dispatch (DED) is one of the main functions of power generation operation and control. It determines the optimal settings of generator units with predicted load demand over a certain period of time. The objective is to operate an electric power system most economically while the system is operating within its security limits. This paper proposes a new hybrid methodology for solving DED. The proposed method is developed in such a way that a simple evolutionary programming (EP) is applied as a based level search, which can give a good direction to the optimal global region, and a local search sequential quadratic programming (SQP) is used as a fine tuning to determine the optimal solution at the final. Ten units test system with nonsmooth fuel cost function is used to illustrate the effectiveness of the proposed method compared with those obtained from EP and SQP alone.",
                ],
                'corpus': [
                    "A Hybrid EP and SQP for Dynamic Economic Dispatch with Nonsmooth Fuel Cost Function" + " " + "Dynamic economic dispatch (DED) is one of the main functions of power generation operation and control. It determines the optimal settings of generator units with predicted load demand over a certain period of time. The objective is to operate an electric power system most economically while the system is operating within its security limits. This paper proposes a new hybrid methodology for solving DED. The proposed method is developed in such a way that a simple evolutionary programming (EP) is applied as a based level search, which can give a good direction to the optimal global region, and a local search sequential quadratic programming (SQP) is used as a fine tuning to determine the optimal solution at the final. Ten units test system with nonsmooth fuel cost function is used to illustrate the effectiveness of the proposed method compared with those obtained from EP and SQP alone.",
                    "A Direct Search Method to solve Economic Dispatch Problem with Valve-Point Effect",
                ],
            },            
        },
        "STS": {
            'STS12': [	
                "Counties with population declines will be Vermillion, Posey and Madison.",
                "Vermillion, Posey and Madison County populations will decline.",
            ],
        },
        "Summarization": {
            "SummEval": [
                "Mexican restaurant has decided to tap into $70 billion food delivery market. Fast-casual chain will work with the Postmates app to allow mobile orders. App works in similar way to Uber, using hired drivers to deliver the food. But the chain will add a 9% service charge - on top of Postmates' $5 rate.",
                "chipotle has decided to tap into the $ 70 billion food delivery market by teaming up with an app to bring burritos straight to customers ' doors . the fast-casual chain will work with the postmates app to begin offering delivery for online and mobile orders in 67 cities . the restaurant plans to add a nine per cent service charge - with the delivery fees for postmates beginning at $ 5 and up depending on distance and demand .",
            ],
        },
    },
    }

QUICK_EVAL = [
    # Classification
    "Banking77Classification",
    "EmotionClassification",
    # Clustering
    "MedrxivClusteringS2S",
    # PairClassification
    "TwitterSemEval2015",
    # Reranking
    "AskUbuntuDupQuestions",
    # Retrieval
    "ArguAna",
    "NFCorpus",
    "SciFact",
    # STS
    "BIOSSES",
    "STS17",
    "STSBenchmark",
    # Summarization
    "SummEval",
]




TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]

ALL_TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)




DTYPE_TO_TORCH_DTYPE = {
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    'float16': torch.float16,
}

def get_gpus_max_memory(max_memory):
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    return max_memory

def gritlm_instruction_format(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

def zephyr_instruction_format(instruction):
    return "<|user|>\n" + instruction + "</s>\n<|assistant|>\n"

def tulu_instruction_format(instruction):
    return "<|user|>\n" + instruction + "\n<|assistant|>\n"

def mistral_instruction_format(instruction):
    return "<s>[INST]\n" + instruction + "\n<|embed|>\n" if instruction else "<s>[INST]\n<|embed|>\n"



NAME_TO_FUNC = {
    "gritlm": gritlm_instruction_format,
    "zephyr": zephyr_instruction_format,
    "tulu": tulu_instruction_format,
    "mistral": mistral_instruction_format,
}

EMBED_EOS_MODEL = {
    "gritlm": "",
    "mistral": "\n[/INST]",
}


SET_TO_FEWSHOT_PROMPT = {
    "e5": {
        "Retrieval": '\n\nFor example given "{}", you should retrieve "{}"',
        "Other": '\n\nFor example given "{}", it would match with "{}"',
    },
    "medi2": {
        "Retrieval": '\n\nThe provided query could be "{}" and the positive "{}"',
        "Other": '\n\nThe provided query could be "{}" and the positive "{}"',
    },
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="GritLM/GritLM-7B", type=str)
    parser.add_argument('--attn_implementation', default='sdpa', type=str, help="eager/sdpa/flash_attention_2")
    parser.add_argument('--attn', default='bbcc', type=str, help="only first two letters matter for embedding")
    parser.add_argument('--task_types', default=None, help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument('--task_names', default=None, help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument('--instruction_set', default="e5", type=str, help="Instructions to use")
    parser.add_argument('--instruction_format', default="gritlm", type=str, help="Formatting to use")
    parser.add_argument('--no_instruction', action='store_true', help="Do not use instructions")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_length', default=None, type=int)
    parser.add_argument('--num_shots', default=None, type=int)
    parser.add_argument('--dtype', default='bfloat16', type=str)
    parser.add_argument('--output_folder', default=None, type=str)
    parser.add_argument('--overwrite_results', action='store_true')
    parser.add_argument('--pipeline_parallel', action='store_true')
    parser.add_argument('--embedding_head', default=None, type=str)
    parser.add_argument('--pooling_method', default='mean', type=str)
    # parser.add_argument('--embed_eos', default="\n[/INST]", type=str)
    parser.add_argument('--save_qrels', action='store_true')
    parser.add_argument('--lora', action='store_true', default=False)
    parser.add_argument('--qlora', action='store_true', default=False)
    parser.add_argument('--quick_mode', action='store_true', default=False)
    parser.add_argument('--top_k', default=10, type=int)    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    print(args, flush=True)
    # Quick skip if exists

    model_name = args.model_name_or_path.strip("/").split("/")[-1]


    output_folder = args.output_folder if args.output_folder else f"./results/{model_name}"
    if (args.task_names is not None) and (len(args.task_names.split(",")) == 1) and os.path.exists(f"{output_folder}/{args.task_names.split(',')[0]}.json"):
        print(f"Skipping {args.task_names.split(',')[0]}")
        exit()
    


    kwargs = {
        "model_name_or_path": args.model_name_or_path,
        # Normalizing embeddings will harm the performance of classification task
        # as it does not use the cosine similarity
        # For other tasks, cosine similarity is used in the evaluation, 
        # so embeddings are automatically normalized
        "normalized": False,
        "torch_dtype": DTYPE_TO_TORCH_DTYPE.get(args.dtype, torch.bfloat16),
        "mode": "embedding",
        "pooling_method": args.pooling_method,
        "attn_implementation": args.attn_implementation,
        "attn": args.attn,
        "embed_eos":EMBED_EOS_MODEL[args.instruction_format]
    }

    if args.pipeline_parallel:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = get_gpus_max_memory("50GB")
        kwargs["offload_folder"] = "offload"

    if any([x in args.model_name_or_path for x in ["instructor"]]):
        assert kwargs["pooling_method"] == "mean"
    elif any([x in args.model_name_or_path for x in ["bge"]]):
        assert kwargs["pooling_method"] == "cls"
    
    if args.pooling_method == "lasttoken":
        kwargs["embed_eos"] = "</e>"
    if args.embedding_head:
        kwargs["projection"] = args.embedding_head

    model = TDTE(**kwargs)

    if args.embedding_head:
        model.load_state_dict(
            torch.load(args.model_name_or_path + "/embedding_head.bin"), strict=False,
        )
        model.projection.to(model.device)

    if os.getenv("BIDIRECTIONAL_ATTN", False):
        model.model.padding_idx = model.tokenizer.pad_token_id
        if hasattr(model.model, "model"):
            model.model.model.padding_idx = model.tokenizer.pad_token_id
        if hasattr(model.model, "module"):
            model.model.module.padding_idx = model.tokenizer.pad_token_id            

    kwargs = {"task_langs": ['en']}
    if args.task_names:
        kwargs["tasks"] = args.task_names.split(",")
    elif args.task_types:
        kwargs["task_types"] = args.task_types.split(",")
    elif(args.quick_mode):
        kwargs["tasks"] = QUICK_EVAL
    else:
        kwargs["tasks"] = ALL_TASK_LIST
    
    tasks = [(t.metadata.name, t.metadata.type) for t in MTEB(**kwargs).tasks]

    if args.max_length is not None:
        model.encode = partial(model.encode, max_length=args.max_length)

    print(tasks, flush=True)

    for (task_name, task_type) in tasks:
        st = time.time()
        if task_name in ['MSMARCOv2', 'BigPatentClustering']:
            print('Skipping task: ' + task_name)
            continue
        if not args.no_instruction:
            if task_name.startswith("CQADupstack") and \
                "CQADupstackRetrieval" in SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set][task_type]:
                instruction = SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set][task_type]["CQADupstackRetrieval"]
            else:
                if task_name not in SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set][task_type]:
                    print('Skipping task: ' + task_name)
                    continue
                instruction = SET_TO_TASK_TO_DS_TO_PROMPT[args.instruction_set][task_type][task_name]
            if isinstance(instruction, dict):
                if args.num_shots is not None:
                    instruction = {
                        k: v + SET_TO_FEWSHOT_PROMPT[args.instruction_set]["Retrieval"].format(
                            *SET_TO_TASK_TO_DS_TO_SHOTS[args.instruction_set][task_type][task_name][k]
                        ) if v else v for k, v in instruction.items()
                    }
                instruction = {k: NAME_TO_FUNC[args.instruction_format](v.strip(": \n")) for k, v in instruction.items()}
            else:
                if args.num_shots is not None:
                    instruction = instruction + SET_TO_FEWSHOT_PROMPT[args.instruction_set]["Other"].format(
                        *SET_TO_TASK_TO_DS_TO_SHOTS[args.instruction_set][task_type][task_name]
                    )
                instruction = NAME_TO_FUNC[args.instruction_format](instruction.strip(": \n"))
            print(f"{model_name} instruction for {task_name}: ", instruction)
            if isinstance(instruction, dict):
                model.encode_queries = partial(model.encode_queries, instruction=instruction['query'])
                model.encode_corpus = partial(model.encode_corpus, instruction=instruction['corpus'])
            else:
                model.encode = partial(model.encode, instruction=instruction)
        eval_splits = ["test" if task_name not in ['MSMARCO'] else 'dev']

        # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        evaluation = MTEB(tasks=[task_name], task_langs=['en'])
        evaluation.run(
            model,
            output_folder=output_folder,
            eval_splits=eval_splits,
            batch_size=args.batch_size,
            save_qrels=args.save_qrels,
            top_k=args.top_k,
            overwrite_results=args.overwrite_results,
        )
        print(task_name, task_type, flush=True)
        et = time.time()
        print("Total time (hours):", (et-st)/3600)

