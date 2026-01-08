# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "vit_cat_dog.pth")

TESTSET_PATH = os.path.join(BASE_DIR, "dataset", "catdog","test_set")

TRAININGSET_PATH = os.path.join(BASE_DIR, "dataset", "catdog","training_set")

LOG_PATH = os.path.join(BASE_DIR, "CatDog_Classifier", "vitlog")
