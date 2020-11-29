# -*- coding: utf-8 -*-
# I think this file will not be needed


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time
import os

DATADIR = 'C:/Users/Pollen/project/dataset'  #change the last letter after finishing the data collection "C:/Users/Light/project/dataset/Y"#and also changed the name Light to Pollen after windows setup
CATEGORIES = ["SNEEZE", "MEDICINE", "INFECTION", "INHALER", "BROKE", "BLOOD PRESSURE", \
              "FORGOT", "WHEN", "BAD", "GOOD", "FATHER", "MOTHER", "NEVER", "GUILT", \
              "SHOWER", "BETTER", "WAKE UP", "GLASS", "DEATH", "LIFE", "DAY", "BALL", \
              "ADULT", "STREET OR STRAIGHT", "AIRPLANE", "FEAR", "READ", "THANK YOU", \
              "TRUST", "LOVE", "DOCTOR", "BOOK", "TREE", "EYE", "SMILE", "PICTURE", "PAIN", \
              "DONT CARE", "DONT KNOW", "HELP", "MOON", "SHOES", "LEARN", "CAT", "DOG", "WORK", \
              "PHONE", "WATER", "BABY", "FOOD", "BED", "PERSON", "DEAF", "CAR", "BEAUTIFUL", "AGAIN", \
              "WHAT", "UNDERSTAND", "PLEASE", "NO", "YES", "HELLO" ]

CATEGORIES2 = ["ADULT", "AGAIN", "AIRPLANE", "BABY", "BAD", "BALL", \
               "BEAUTIFUL", "BED", "BETTER", "BLOOD PRESSURE", "BOOK", "BROKE", \
               "CAR", "CAT", "DAY", "DEAF", "DEATH", "DOCTOR", \
               "DONT CARE", "DONT KNOW", "EYE", "FATHER", "FEAR", "FOOD", \
               "FORGOT", "GLASS", "GOOD", "GOOD BYE", "GUILT", "HELLO", \
               "HELP", "INFECTION", "INHALER", "LEARN", "LIFE", "LOVE", \
               "MEDICINE", "MOON", "MOTHER", "NEVER", "NO", "PAIN", \
               "PERSON", "PHONE", "PICTURE", "PLEASE", "READ", "SHOES", \
               "SHOWER", "SMILE", "SNEEZE", "STREET", "THANK YOU", "TREE", \
               "TRUST", "UNDERSTAND", "WAKE UP", "WATER", "WHAT", "WHEN", \
               "WORK", "YES"]

print(len(CATEGORIES))
print(len(CATEGORIES2))

