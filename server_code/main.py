from flask import Flask, request
from flask import jsonify

import ast
import text_detection_v10

app = Flask(__name__)

min_confidence = 0.2
min_area = 20
adjustment_factor_x = 0.3
adjustment_factor_y = 0.6
offline_detection = False
x_coordinate = 0 
y_coordinate = 0
z_coordinate = 0
accessToken = '0'

@app.route('/scan',methods=['POST'])
def scan_process():
    global min_confidence,min_area,adjustment_factor_x,adjustment_factor_y,offline_detection,x_coordinate,y_coordinate,z_coordinate,authorization_token
    if request.method == 'POST':
        data = ast.literal_eval(request.data)
        authorization_token = request.headers.get('Authorization')
        encoded_byte   = data['encoded_byte']
        if('min_confidence' in data):
            min_confidence = float(data['min_confidence'])
        if('min_area' in data):
            min_area       = float(data['min_area'])
        if('adjustment_factor_x' in data):
            adjustment_factor_x = float(data['adjustment_factor_x'])
        if('adjustment_factor_y' in data):
            adjustment_factor_y = float(data['adjustment_factor_y'])
        if('offline_detection' in data):
            offline_detection = data['offline_detection']
        if('x_coordinate' in data):
            x_coordinate = float(data['x_coordinate'])
        if('y_coordinate' in data):
            y_coordinate = float(data['y_coordinate'])
        if('z_coordinate' in data):
            z_coordinate = float(data['z_coordinate'])

        try:
            boxes,recognised_text = text_detection_v10.imageProcessor(encoded_byte, min_confidence, min_area, adjustment_factor_x, adjustment_factor_y, offline_detection)
            
            # boxes = text_detection_v8.imageProcessor(encoded_byte, 0.2, 200, 0.2, 0.02)
            if(len(boxes)>0):
                output = boxes.tolist()
            else: 
                output = boxes

            res = {
                "error" : {
                    "status" : False
                },
                "response" : {
                    "boxes" :   output,
                    "recognised_text"   :   recognised_text
                }
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
    # app.run(host='127.0.0.1', port=8080, debug=True)
    app.run()
