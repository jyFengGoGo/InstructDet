import os, json
import argparse
import numpy as np
import torch
from PIL import Image
import yaml
import copy
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import io
from transformers import BertTokenizer, BertModel


def dbscan_culster(image_features, eps=10):
    dbscan = DBSCAN(eps=eps, min_samples=2)
    labels = dbscan.fit_predict(image_features)
    cluster_num = len(set(labels))
    clusters = []
    for i in range(max(labels)+1):
        cluster = np.argwhere(labels==i)[:,0]
        clusters.append(cluster)
    # print(f"Noise: {list(image_features[labels==-1])}")
    # if len(clusters)>0:
    #     print(f"Noise: {list(labels==-1)}")
    return clusters


def PCA_downsample(image_features, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(image_features)
    return pca_result


def expression_clustering(args, samples=None, debug=False):
    def get_expall(bbox):
        expall = []
        for exptype in ["expressions", "expressions_vlm", "expressions_llm"]:
            if exptype in bbox:
                expall += bbox[exptype]
        expall = list(set(expall))
        return expall
    
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertModel.from_pretrained(args.model_path)
    
    eps = args.eps
    samples_clustered = []
    for line in tqdm(samples):
        # clustering on expressions
        bboxes = line["bboxes"]
        expressions = [", ".join(get_expall(bbox)) for bbox in bboxes]
        bert_input = tokenizer(expressions, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        with torch.no_grad(), torch.cuda.amp.autocast():
            _, pooled_output = model(input_ids=bert_input["input_ids"], attention_mask=bert_input["attention_mask"], return_dict=False)

        text_features = pooled_output
        clusters = dbscan_culster(text_features, eps=eps)

        line["clusters"] = []
        if len(clusters) > 0:
            line["clusters"] = [dict(bbox_ids=[int(n) for n in _]) for _ in clusters]
        samples_clustered.append(line)
        if debug:
            for cluster in clusters:
                for idx in cluster:
                    print(idx, expressions[idx])
    return samples_clustered
