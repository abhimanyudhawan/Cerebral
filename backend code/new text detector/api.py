__author__ = 'chirag'

from flask import Flask, request
from flask import jsonify

import ast
import text_detection_video_v7

app = Flask(__name__)


@app.route('/scan',methods=['POST'])
def scan_process():
    if request.method == 'POST':

        data = ast.literal_eval(request.data)
        
        encoded_byte   = data['encoded_byte'];

        # min_confidence = data['min_confidence'];
        # min_Area       = data['min_Area'];
        # adjustment_Factor_x = data['adjustment_Factor_x'];
        # adjustment_Factor_y = data['adjustment_Factor_y'];

        try:
            # boxes = text_detection_video_v7.imageProcessor(encoded_byte, min_Confidence, min_area, adjustment_factor_x, adjustment_factor_y)
            
            boxes = text_detection_video_v7.imageProcessor(encoded_byte, 0.2, 200, 0.2, 0.02)
            

            res = {
                "error" : {
                    "status" : False
                },
                "response" : boxes
            }

        except Exception as e:

            print("Exception:");
            print(str(e));

            res = {
                "error" : {
                    "status" : True,
                    "message" : str(e)
                }
            }

        return jsonify(**{'res': res})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)
    app.run()
