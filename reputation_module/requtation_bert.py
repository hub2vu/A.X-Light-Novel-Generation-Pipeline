from pathlib import Path
import re, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMB = SentenceTransformer("jhgan/ko-sroberta-multitask")  # 로컬 임베딩

def sent_split(text):
    return re.split(r'(?<=[.!?…])\s+|\n+', text.strip())

def embed_mean(texts):
    embs = EMB.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embs

def outline_adherence_score(outline_text, draft_text):
    o_sents = sent_split(outline_text)
    d_sents = sent_split(draft_text)
    if not o_sents or not d_sents: return 0.0
    Eo = embed_mean(o_sents)
    Ed = embed_mean(d_sents)
    sim = cosine_similarity(Eo, Ed).max(axis=1).mean()  # outline 각 문장에 가장 가까운 문장 유사도 평균
    return float(sim)

def repetition_score(text, n=3):
    toks = list(text)
    ngrams = ["".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
    total = len(ngrams)
    dup = total - len(set(ngrams))
    return dup / (total+1e-6)

def readability_signals(text):
    sents = [s for s in sent_split(text) if s.strip()]
    lengths = [len(s) for s in sents] or [0]
    avg = np.mean(lengths); std = np.std(lengths)
    dialog_ratio = sum(1 for s in sents if s.strip().startswith(("“","\"","'","-"))) / max(1,len(sents))
    return {"avg_sent_len": float(avg), "std_sent_len": float(std), "dialog_ratio": float(dialog_ratio)}

def novelty_vs_source(gen_text, src_text):
    Eg = embed_mean([gen_text])
    Es = embed_mean([src_text])
    sim = cosine_similarity(Eg, Es)[0,0]
    return 1.0 - float(sim)   # 높을수록 새로움

# 사용 예
outline = Path("./rag_corpus/outline.txt").read_text(encoding="utf-8").splitlines()[-1]
draft   = Path("./rag_corpus/new_story.txt").read_text(encoding="utf-8")

scores = {
    "outline_adherence": outline_adherence_score(outline, draft),
    "repetition_3gram": repetition_score(draft, n=3),
    "novelty_vs_last":  novelty_vs_source(draft, Path("./rag_corpus/last_story.txt").read_text(encoding="utf-8")),
    **readability_signals(draft)
}
print(scores)
