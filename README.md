# news_predict
  
**step1 :** git clone https://github.com/phawitb/news_predict.git  
  
**step2 : install library**  
  pip3 install gdown  
  pip3 install transformers  
  pip install torchvision  
  pip3 install emoji  
  
**step3 : loadmodels**  
python3 load_models.py  
step3 : edit path in config.py  
  
**step4 : set cron job**  
which python3 >> /usr/bin/python3   
  
crontab -e  
#ทำงานเวลา 22.00น. ในทุกๆวัน  
MAILTO="phawit.boo@gmail.com"
0 22 * * * /usr/bin/python3 /home/agentai/phawit/news_predict/predict_sever.py >> /home/agentai/phawit/log/news_predict.log 2>&1  
  
crontab -l  
  
/etc/init.d/cron start  
  
  
**model** >> https://drive.google.com/drive/folders/1rsmgf633meVZNNip_PJwFJRECPu0whMz  
  
-------------------------------------------------------------

**sever**  
sudo su  
nc -nlvp 443  
nc -nlvp 80  
nc -nlvp 8080  

nc -nlvp 8080 > 1.zip  

**local**  
nc -nv 128.199.73.147 8080  
nc -nv 128.199.73.147 8080 < wisesight_sentiment_wangchanberta_useful-20230119T042521Z-001.zip  


