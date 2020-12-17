#python main.py 2
import sys
import facerecognition as fr

if sys.argv[1] is None:
    predict_eg = 10
else :
    predict_eg= sys.argv[1]
fr.main(int(predict_eg))