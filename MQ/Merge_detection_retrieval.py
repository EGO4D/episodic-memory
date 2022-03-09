import json

result_det = 'output/detections_postNMS.json'
result_rev = 'output/retrieval_postNMS.json'

with open(result_det, 'r') as fobj:
    data_det = json.load(fobj)

with open(result_rev, 'r') as fobj:
    data_rev = json.load(fobj)

data_submission = {"version": "1.0", "challenge": "ego4d_moment_queries"}

data_submission['detect_results'] = data_det['results']
data_submission['retrieve_results'] = data_rev['results']

with open("output/submission.json", "w") as fp:
    json.dump(data_submission,fp)