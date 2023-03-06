config = {'sample_data': {'RI': 1.52081, 'NA': 13.78, 'MG': 2.28, 'AL': 1.43, 'SI': 71.99, 'K': 0.49, 'CA': 9.85, 'BA': 0.0, 'FE': 0.17}, 'target_data': {'TYPE': 2}, 'model_name': 'GLASS', 'dataset_name': 'glass'}



import requests
import json


req_sample = config["sample_data"]

def test_ml_service(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, json=json.loads(json.dumps(req_sample)), headers=headers)
    assert resp.status_code == requests.codes["ok"]
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0

#TEST
def test_prediction(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, json=json.loads(json.dumps(req_sample)), headers=headers)
    resp_json = json.loads(resp.text)
    assert resp_json['output']['predicted_'+list(config["target_data"].keys())[0].lower()] == str(list(config["target_data"].values())[0])