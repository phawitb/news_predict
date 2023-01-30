import configure as conf
import dbx as db
import model_predict as model
import time
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
import torch.nn.functional as F     
import sys
import datetime
import csv

# Batch_size = int(sys.argv[1])
# print('Batch_size',Batch_size)
Batch_size = 100   #-----------------------------

def update_db_list(id, ai_useful_pct, ai_creative_pct, ai_domain, ai_oganic_news):
    V = []
    for i in range(len(id)):
        sql = "UPDATE newsai SET  ai_useful_pct=%s,ai_opinion_pct=%s,ai_domain=%s,ai_oganic_news=%s,percen_ai_useful_pct=%s,percen_ai_opinion_pct=%s,percen_ai_domain=%s,percen_ai_oganic_news=%s where id=%s"
        val = (ai_useful_pct[0][i], ai_creative_pct[0][i],ai_domain[0][i], ai_oganic_news[0][i],round(ai_useful_pct[1][i], 4), round(ai_creative_pct[1][i], 4),round(ai_domain[1][i], 4), round(ai_oganic_news[1][i], 4), id[i])
        V.append(val)

    xcount = db.updatepara_multi(sql, V)
    print(xcount, "record updated.")
        

# sqlx="SELECT id,news_title,news_content FROM newsai ORDER BY id desc "
sqlx = "SELECT id,news_title,news_content FROM newsai WHERE percen_ai_useful_pct IS NULL ORDER BY id desc"
myresult = db.query(sqlx)
print('Total Data unlabeled',len(myresult))

ID = []
content_list = []
i = 0
t = time.time()
for x in myresult:
    ID.append(str(x[0]))
    content_list.append(str(x[1]))  #news_title
    if len(ID) == Batch_size:

        ai_oganic_news = model.predict_fake(content_list)
        ai_useful_pct = model.predict_useful(content_list)
        ai_creative_pct = model.predict_opinion(content_list)
        ai_domain = model.predict_domain(content_list)

        print('-'*10)
        print(i,time.time()-t,datetime.datetime.now(),ID)    #,ai_useful_pct,ai_creative_pct,ai_domain,ai_oganic_news)
        with open('/home/agentai/phawit/news_predict/log.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([i,time.time()-t,datetime.datetime.now()])
        i += 1
        t = time.time()

        update_db_list(ID,ai_useful_pct,ai_creative_pct,ai_domain,ai_oganic_news)

        
        ID = []
        content_list = []
