import json
import Utils.opts as opts
import os

opt = opts.parse_opt()
opt = vars(opt)
result_det = os.path.join(opt['output_path'], opt['detect_result_file'])
result_rev = os.path.join(opt['output_path'], opt['retrieval_result_file'])
submission_file = opt['output_path'] + "/submission.json"

with open(result_det, 'r') as fobj:
    data_det = json.load(fobj)

with open(result_rev, 'r') as fobj:
    data_rev = json.load(fobj)

data_submission = {"version": "1.0", "challenge": "ego4d_moment_queries"}

data_submission['detect_results'] = data_det['results']
data_submission['retrieve_results'] = data_rev['results']

with open(submission_file, "w") as fp:
    json.dump(data_submission, fp)
