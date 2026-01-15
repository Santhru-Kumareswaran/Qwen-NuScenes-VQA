
import numpy as np
import torch
from collections import defaultdict
import math

def calculate_bleu(references, hypothesis):
    """
    Simple BLEU-4 implementation using NLTK
    args:
        references: list of reference sentences
        hypothesis: generated sentence
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        chencherry = SmoothingFunction()
        # NLTK expects tokenized list of strings
        ref_tokens = [r.split() for r in references]
        hyp_tokens = hypothesis.split()
        return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=chencherry.method1)
    except ImportError:
        print("nltk not installed, skipping BLEU")
        return 0.0

def compute_metrics(predictions, ground_truths):
    """
    Compute BLEU-4 and CIDr (approximation or placeholder if pycocoevalcap missing)
    """
    scores = {"bleu4": [], "cider": []} # Placeholder
    
    for pred, refs in zip(predictions, ground_truths):
        # Allow single ref or list of refs
        if isinstance(refs, str): refs = [refs]
        
        # BLEU-4
        b4 = calculate_bleu(refs, pred)
        scores["bleu4"].append(b4)
        
    avg_bleu4 = np.mean(scores["bleu4"]) if scores["bleu4"] else 0.0
    
    return {
        "bleu4": avg_bleu4,
        "cider": 0.0 # CIDr requires complex IDF weighting, usually external lib
    }
