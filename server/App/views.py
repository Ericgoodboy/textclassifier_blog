from flask import Blueprint, request, render_template, make_response, Response, abort, session
blue = Blueprint('blue', __name__)
import pandas as pd

def init_blue(app):
    app.register_blueprint(blue)


@blue.route("/")
def index():
    return "hello my flask"


from bs4 import BeautifulSoup
import json
import os
from settings import model_path, ROOT_DIR
svmpath = os.path.join(model_path,"svm.pkl")
import pickle as pkl
import jieba

svm = pkl.load(open(svmpath,"rb"))
stopwords = []

@blue.route("/tag/predict",methods=["POST"])
def tag_info(tag="moren"):
    text = json.loads(request.data.decode("utf8"))["data"]
    soup = BeautifulSoup(text,"lxml")
    text = soup.text
    text = jieba.cut(text)
    text = [" ".join(text)]
    res = svm.predict(text)


    return res[0]
