**[NEW!] Detailed examples for VQ Challenge submission available here: [VQ2D README](./VQ2D/README.md)**

**[NEW!] 2022 [Ego4D Challenges](https://ego4d-data.org/docs/challenge/) now open for Episodic Memory**
- [Natural Language Queries](https://eval.ai/web/challenges/challenge-page/1629/overview)
- [Visual Queries 2D](https://eval.ai/web/challenges/challenge-page/1843/overview)
- [Moments queries](https://eval.ai/web/challenges/challenge-page/1626/overview)
- [Visual Queries 3D](https://eval.ai/web/challenges/challenge-page/1646/overview)

Please note that:
- VQ test annotations for challenge submissions are now available: [Ego4D Challenges](https://ego4d-data.org/docs/challenge/)
- NLQ annotations have a known issue where ~14% of annotations have a near-0 query window and will result in under reported performance for the challenge (which will be corrected with a future dataset update): [NLQ Forum Post](https://discuss.ego4d-data.org/t/nlq-annotation-zero-temporal-windows/36)

# Ego4D Episodic Memory Benchmark

[EGO4D](https://ego4d-data.org/docs/) is the world's largest egocentric (first person) video ML dataset and benchmark suite.

For more information on Ego4D or to download the dataset, read: [Start Here](https://ego4d-data.org/docs/start-here/).

The [Episodic Memory Benchmark](https://ego4d-data.org/docs/benchmarks/episodic-memory/) aims to make past video queryable and requires localizing where the answer can be seen within the user’s past video.  The repository contains the code needed to reproduce the results in the [Ego4D: Around the World in 3,000 Hours of Egocentric Video](https://arxiv.org/abs/2110.07058).

There are 4 related tasks within a benchmark. Please see the README within each benchmark for details on setting up the codebase.

# [VQ2D](./VQ2D/README.md): *Visual Queries with 2D Localization*

This task asks: “When did I last see [this]?”  Given an egocentric video clip and an image crop depicting the query object, the goal is to return the last occurrence of the object in the input video, in terms of the tracked bounding box (2D + temporal localization).  The novelty of this task is to upgrade traditional object instance recognition to deal with video, and particularly ego-video with challenging view transformations.

# [VQ3D](./VQ3D/README.md): *Visual Queries with 3D Localization*

This task asks, “Where did I last see [this]?”  Given an egocentric video clip and an image crop depicting the query object, the goal is to localize the last time it was seen in the video and return a 3D displacement vector from the camera center of the query frame to the center of the object in 3D.  Hence, this task builds on the 2D localization above, expanding it to require localization in the 3D environment.  The task is novel in how it requires both video object instance recognition and 3D reasoning.

# [NLQ](./NLQ/README.md): *Natural Language Queries*

This task asks, "What/when/where....?" -- general natural language questions about the video past.    Given a video clip and a query expressed in natural language, the goal is to localize the temporal window within all the video history where the answer to the question is evident.  The task is novel because it requires searching through video to answer flexible linguistic queries.  For brevity, these example clips illustrate the video surrounding the ground truth (whereas the original input videos are each ~8 min). 

# [MQ](./MQ/README.md): *Moments Queries*

This task asks, "When did I do X?”  Given an egocentric video and an activity name (i.e., a "moment"), the goal is to localize all instances of that activity in the past video.  The task is activity detection, but specifically for the egocentric activity of the camera wearer who is largely out of view.


License

Ego4D is released under the MIT License.
