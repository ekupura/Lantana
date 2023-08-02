#!/usr/bin/env python
# coding: utf-8

import os
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import datasets
import pickle
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import Ridge
from fire import Fire
from captum.attr import LimeBase
import random

os.makedirs("./data", exist_ok=True)

def preprocess():
    # Load the IMDB dataset
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load the IMDB dataset
    train_dataset, test_dataset = datasets.load_dataset('imdb', split=['train', 'test'])

    # slice
    train_dataset = train_dataset.shuffle(seed=42)[:1000]
    test_dataset = test_dataset.shuffle(seed=42)[:1000]

    # Tokenize the texts
    train_encodings = tokenizer(train_dataset['text'], truncation=True, padding=True)
    test_encodings = tokenizer(test_dataset['text'], truncation=True, padding=True)

    # Convert labels to tensors
    train_labels = torch.tensor(train_dataset['label'])
    test_labels = torch.tensor(test_dataset['label'])

    # Create TensorDatasets
    train_dataset = TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        train_labels
    )

    test_dataset = TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask']),
        test_labels
    )

    # dump
    pickle.dump(train_dataset, open("./data/train_dataset.pkl", "wb"))
    pickle.dump(test_dataset, open("./data/test_dataset.pkl", "wb"))


def train():
    train_dataset = pickle.load(open("./data/train_dataset.pkl", "rb"))

    # BERTモデルを読み込む
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.cuda()

    # データローダーを作成する
    batch_size = 32  # 適宜調整する
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # オプティマイザーを設定する
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # モデルを訓練する
    num_epochs = 10  # 適宜調整する

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader):
            input_ids = batch[0].to(model.device).cuda()
            attention_masks = batch[1].to(model.device).cuda()
            labels = batch[2].to(model.device).cuda()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(avg_loss)

    # dump
    model.save_pretrained('./data/bert')


class ScratchLime:
    def __init__(self, predict_fn, perturb_fn, kernel_width=0.75):
        self.predict_fn = predict_fn
        self.perturb_fn = perturb_fn
        self.kernel_width = kernel_width

    def explain_instance(self, instance, target, use_weights=True):
        # Step 1: Generate random samples around the instance
        samples = self.perturb_fn(instance)

        # Step 2: Get predictions for the samples using the black-box model
        predictions = [self.predict_fn(sample)[target] for sample in tqdm(samples)]

        if use_weights:
            # Step 3: Compute distances between the instance and the samples
            distances = self._compute_distances(instance, samples)
            # Step 4: Compute weights using kernel function
            weights = self._kernel(distances)
        else:
            weights = np.ones(samples.shape[0])

        # Step 5: Fit a linear model using the samples and predictions
        model = Ridge(alpha=1.0)
        model.fit(samples, predictions, sample_weight=weights)

        return model.coef_

    def _compute_distances(self, instance, samples):
        dot_product = np.dot(samples, instance)
        instance_norm = np.linalg.norm(instance)
        sample_norm = np.linalg.norm(samples, axis=1)
        cosine_similarity = dot_product / (instance_norm * sample_norm)
        return cosine_similarity

    def _kernel(self, distances):
        return np.exp(-distances / self.kernel_width)


class Perturb:
    @staticmethod
    def additive_fn(instance, num_samples=100):
        samples = np.random.normal(0, 1, (num_samples, instance.shape[0]))
        return samples + instance


    @staticmethod
    def mask_fn(num_samples=100, perturb_size=1):
        def func(instance):
            mask_array = []
            for _ in range(num_samples):
                row = [1.0 if idx >= perturb_size else 0.0 for idx in range(instance.shape[0])]
                mask_array.append(np.random.permutation(row))
            mask_array = np.array(mask_array)
            return mask_array * instance
        return func


# Define a black-box model
def black_box_model(array):
    predictions = array[0] + array[1] * 2 + array[2] * 3
    predictions += array[3] ** 2 + array[4] ** 3
    return predictions


def test_lime():
    # Explain an instance
    instance = np.array([1, 1, 1, 1, 1])
    # Create an instance of LIME
    lime = Lime(predict_fn=black_box_model, perturb_fn=Perturb.additive_fn)
    explanation = lime.explain_instance(instance, 0, use_weights=True)
    print(explanation)

    # Explain an instance
    instance = np.array([1, 1, 1, 1, 1])
    # Create an instance of LIME
    comp_fn = Perturb.mask_fn(perturb_size=1)
    lime = Lime(predict_fn=black_box_model, perturb_fn=comp_fn)
    explanation = lime.explain_instance(instance, 0, use_weights=True)
    print(explanation)


class TestBert:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained("./data/bert").cuda()
        self.train_dataset = pickle.load(open("./data/train_dataset.pkl", "rb"))

    def to_embedding(self, sample):
        ids, mask = sample[0].unsqueeze(0).cuda(), sample[1].unsqueeze(0).cuda()
        embedding = self.model.bert.embeddings(input_ids=ids)
        return embedding.cpu().squeeze(0).detach().numpy()

    def predict_target(self, sample):
        ids = torch.tensor(sample[0]).unsqueeze(0).cuda()
        mask = torch.tensor(sample[1]).unsqueeze(0).cuda()
        prediction = self.model(input_ids=ids, attention_mask=mask).logits
        return torch.argmax(prediction.squeeze(0).cpu()).detach().numpy()

    def reshape(self, array):
        array = array.reshape(-1, 768)
        return array

    def bert_wrapper(self, instance):
        reshaped = self.reshape(instance)
        with torch.no_grad():
            reshaped = torch.tensor(reshaped).unsqueeze(0).cuda().to(torch.float32)
            prediction = self.model(inputs_embeds=reshaped)
        return prediction.logits.cpu().squeeze(0).detach().numpy()

    def bernoulli_fn(self, instance, num_samples=100):
        instance = self.reshape(instance)
        samples = np.random.binomial(1, p=0.5, size=(num_samples, instance.shape[0]))
        return [(instance.T * s).T.flatten() for s in samples]

    def test_bert(self,):
        target = self.predict_target(self.train_dataset[0])
        instance = self.to_embedding(self.train_dataset[0]).flatten()
        lime = ScratchLime(predict_fn=self.bert_wrapper, perturb_fn=self.bernoulli_fn)
        explanation = lime.explain_instance(instance, target, use_weights=True)
        explanation = np.sum(self.reshape(explanation), axis=1)
        print(np.argsort(explanation))


class CaptumLime:
    # remove the batch dimension for the embedding-bag model
    def forward_func(text, offsets):
        return eb_model(text.squeeze(0), offsets)

    # encode text indices into latent representations & calculate cosine similarity
    def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
        original_emb = eb_model.embedding(original_inp, None)
        perturbed_emb = eb_model.embedding(perturbed_inp, None)
        distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=1)
        return torch.exp(-1 * (distance ** 2) / 2)

    # binary vector where each word is selected independently and uniformly at random
    def bernoulli_perturb(text, **kwargs):
        probs = torch.ones_like(text) * 0.5
        return torch.bernoulli(probs).long()

    # remove absenst token based on the intepretable representation sample
    def interp_to_input(interp_sample, original_input, **kwargs):
        return original_input[interp_sample.bool()].view(original_input.size(0), -1)

    lasso_lime_base = LimeBase(
        forward_func,
        interpretable_model=SkLearnLasso(alpha=0.08),
        similarity_func=exp_embedding_cosine_distance,
        perturb_func=bernoulli_perturb,
        perturb_interpretable_space=True,
        from_interp_rep_transform=interp_to_input,
        to_interp_rep_transform=None
    )

class Console:
    def __init__(self):
        pass

    def preprocess(self):
        preprocess()

    def train(self):
        train()

    def toy_test(self):
        test_lime()

    def bert_test(self):
        TestBert().test_bert()


if __name__ == "__main__":
    Fire(Console)



