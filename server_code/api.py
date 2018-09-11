__author__ = 'chirag'

from flask import Flask, request
from flask import jsonify

import ast
import text_detection_v8

app = Flask(__name__)

min_confidence = 0.2
min_area = 0
adjustment_factor_x = 0.3
adjustment_factor_y = 0.6

@app.route('/scan',methods=['POST'])
def scan_process():
    global min_confidence,min_area,adjustment_factor_x,adjustment_factor_y
    if request.method == 'POST':
        data = ast.literal_eval(request.data)
        encoded_byte   = data['encoded_byte']
        if('min_confidence' in data):
            min_confidence = float(data['min_confidence'])
        if('min_area' in data):
            min_area       = float(data['min_area'])
        if('adjustment_factor_x' in data):
            adjustment_factor_x = float(data['adjustment_factor_x'])
        if('adjustment_factor_y' in data):
            adjustment_factor_y = float(data['adjustment_factor_y'])

        try:
            boxes = text_detection_v8.imageProcessor(encoded_byte, min_confidence, min_area, adjustment_factor_x, adjustment_factor_y)
            
            # boxes = text_detection_v8.imageProcessor(encoded_byte, 0.2, 200, 0.2, 0.02)
            if(len(boxes)>0):
                output = boxes.tolist()
            else: 
                output = boxes

            res = {
                "error" : {
                    "status" : False
                },
                "response" : output
            }

        except Exception as e:

            print("Exception:")
            print(str(e))

            res = {
                "error" : {
                    "status" : True,
                    "message" : str(e)
                }
            }

        return jsonify(**res)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)
    app.run()