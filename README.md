# Replication Package for Paper "What Helps a New GitHub Project Achieve Sustained Activity?"

This is the replication package for the paper "What Helps a New GitHub Project Achieve Sustained Activity?". It contains: 1) a dataset of 299,556 repositories; and 2) scripts to train different models and reproduce results, as described in the paper.

## Required Environment

We recommend to manually setup the required environment in a commodity Linux machine with at least 1 CPU Core, 8GB Memory and 100GB empty storage space. We conduct development and execute all our experiments on a Ubuntu 20.04 server with two Intel Xeon Gold CPUs, 320GB memory, and 36TB RAID 5 Storage.

## Files and Replicating Results

We use GHTorrent to restore histarical states of 299,556 repositories with more than 57 commits, 4 PRs, 1 issue, 1 fork and 2 stars. Because the data is too large for git, please download data file separately from [Zenodo](https://zenodo.org/record/6529038#.YngErDVBwlJ) and put it under `Project-Sustained-Activity/`. The raw data of repositories is stored in `Project-Sustained-Activity/data/projectdata`, and the contribution of features resulting from LIME model is stored in `Project-Sustained-Activity/data/LIMEdata`. You can run `Project-Sustained-Activity/models/fitdata.py` to get the results for prediction performance, run `Project-Sustained-Activity/models/feature_statistics.py` to get results for Features at Individual Level and Features at project level and run `Project-Sustained-Activity/models/draw_diff_contexts.py` to get results for Feature Contribution under Different Project Contexts. To get the interpretable results from LIME model, you can run `Project-Sustained-Activity/interpretation/limemodel.py`.

## Additinal results

Though we use AUC as the primary metric to evaluate the performance of models, we also present accuracy for reference.
![image](https://user-images.githubusercontent.com/105106761/167373626-9d1176fd-ed5a-440c-98c7-3640a95fd1d0.png)
