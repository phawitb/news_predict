# import configure as conf
# import dbx as db
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import sys
from thai2transformers.preprocess import process_transformers

MAX_LENGTH = 416

model_fake = AutoModelForSequenceClassification.from_pretrained('models/wisesight_sentiment_wangchanberta_antifake')
tokenizer_fake = AutoTokenizer.from_pretrained('models/wisesight_sentiment_wangchanberta_antifake')
model_useful = AutoModelForSequenceClassification.from_pretrained('models/wisesight_sentiment_wangchanberta_useful')
tokenizer_useful = AutoTokenizer.from_pretrained('models/wisesight_sentiment_wangchanberta_useful')
model_opinion = AutoModelForSequenceClassification.from_pretrained('models/wisesight_sentiment_wangchanberta_opinion')
tokenizer_opinion = AutoTokenizer.from_pretrained('models/wisesight_sentiment_wangchanberta_opinion')
model_domain = AutoModelForSequenceClassification.from_pretrained('models/wisesight_sentiment_wangchanberta_domain')
tokenizer_domain = AutoTokenizer.from_pretrained('models/wisesight_sentiment_wangchanberta_domain')

def predict_fake(content_list):
    batch_fake = tokenizer_fake(content_list, padding=True, truncation=True,max_length=MAX_LENGTH, return_tensors="pt")
    class_fake = {0: 1, 1: 2}
    with torch.no_grad():
        output_test = model_fake(**batch_fake)
        pred_test = F.softmax(output_test.logits, dim=1)
        labels_fake_num = torch.argmax(pred_test, dim=1)
        labels_fake = [class_fake[label] for label in labels_fake_num.numpy()]
        percen = [max(x) for x in pred_test.tolist()]
        return (labels_fake,percen)
    
def predict_useful(content_list):
    batch_useful = tokenizer_useful(content_list, padding=True, truncation=True,max_length=MAX_LENGTH, return_tensors="pt")
    class_useful = {0: 1, 1: 2}
    with torch.no_grad():
        output_test = model_useful(**batch_useful)
        pred_test = F.softmax(output_test.logits, dim=1)
        labels_useful_num = torch.argmax(pred_test, dim=1)
        labels_useful = [class_useful[label]
                         for label in labels_useful_num.numpy()]

        percen = [max(x) for x in pred_test.tolist()]
        return (labels_useful,percen)

def predict_opinion(content_list):
    batch_opinion = tokenizer_opinion(content_list, padding=True, truncation=True,max_length=MAX_LENGTH, return_tensors="pt")
    class_opinion = {0: 1, 1: 2}
    with torch.no_grad():
        output_test = model_opinion(**batch_opinion)
        pred_test = F.softmax(output_test.logits, dim=1)
        labels_opinion_num = torch.argmax(pred_test, dim=1)
        labels_opinion = [class_opinion[label]
                          for label in labels_opinion_num.numpy()]
        percen = [max(x) for x in pred_test.tolist()]
        return (labels_opinion,percen)

def predict_domain(content_list):
    batch_domain = tokenizer_domain(content_list, padding=True, truncation=True,max_length=MAX_LENGTH, return_tensors="pt")
    class_domain = {0: "OT", 1: "DC", 2: "SP", 3: "HC", 4: "ML"}
    with torch.no_grad():
        output_test = model_fake(**batch_domain)
        pred_test = F.softmax(output_test.logits, dim=1)
        labels_domain_num = torch.argmax(pred_test, dim=1)
        labels_domain = [class_domain[label]
                         for label in labels_domain_num.numpy()]

        percen = [max(x) for x in pred_test.tolist()]
        return (labels_domain,percen)

def preprocess(text):
    if isinstance(text, str):
        text = [text]
    return list(map(process_transformers, text))



##########################################################################################

# text = 'ทอสอบระบบการทำนายโดเมนจากข่าวสาร./,//'
# # text = ['ทอสอบระบบการทำนายโดเมนจากข่าวสาร./,//','หดิกดิำดิกดิกดิกดิกดิกดิกดิ']

# text = preprocess(text)
# (ai_oganic_news,per_ai_oganic_news) = predict_fake(text)
# (ai_useful_pct,per_ai_useful_pct) = predict_useful(text)
# (ai_opinion_pct,per_ai_opinion_pct) = predict_opinion(text)
# (ai_domain,per_ai_domain) = predict_domain(text)

# print('ai_oganic_news',ai_oganic_news,per_ai_oganic_news)
# print('ai_useful_pct',ai_useful_pct,per_ai_useful_pct)
# print('ai_opinion_pct',ai_opinion_pct,per_ai_opinion_pct)
# print('ai_domain',ai_domain,per_ai_domain)
