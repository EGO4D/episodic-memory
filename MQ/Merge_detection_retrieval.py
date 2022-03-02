import json

result_det = 'output/detections_postNMS.json'
result_rev = 'output/retreival_postNMS.json'


with open(result_det, 'r') as fobj:
    data_det = json.load(fobj)

with open(result_rev, 'r') as fobj:
    data_rev = json.load(fobj)

data_submission = data_det
data_submission['detection'] = data_submission.pop('results')
data_submission['retrieval'] = data_rev['results']

with open("output/submission.json", "w") as fp:
    json.dump(data_submission,fp)