# Youtube-Comedy-Comparison

# AUTHOR = Satish Palaniappan.

# Dataset INFO below

https://archive.ics.uci.edu/ml/datasets/YouTube+Comedy+Slam+Preference+Data

Abstract: This dataset provides user vote data on which video from a pair of videos is funnier collected on YouTube Comedy Slam. The task is to automatically predict this preference based on video metadata.

Source:

Provided by Google, Inc. Please contact duhadway '@' google.com if you have any questions.

Data Set Information:

YouTube Comedy Slam ([Web Link]) is a video discovery experiment running on YouTube's version of labs (called TestTube) for a few months in 2011 and 2012. In this experiment, a pair of videos were shown to the user and the user was asked to vote for the video that they found funnier. Left/right positions of the videos were randomly selected before being presented to the user to eliminate position bias. Videos were selected from a large pool of weekly updated sets of videos. 

One of the outcomes of this experiment is a training dataset for automatically predicting which video would be deemed funnier by users using a variety of features. For example, uploader supplied metadata and/or user comments on watch pages of these videos could be used as features. See [Web Link] for more detail. The attached dataset includes roughly 1.7 million preference votes. The votes were recorded chronologically. The first 80% are provided here as the training dataset and the remaining 20% as the testing dataset. Each line in this dataset corresponds to one vote over a pair of YouTube videos. Each video is represented by its YouTube video ID. For example, the watch page URL for video ID 'txqiwrbYGrs' is [Web Link]. User preference over a pair of videos is presented in the form of string â€œleftâ€ if the left video was deemed funnier, and â€œrightâ€ otherwise. This user vote should be used as ground truth for both training and testing. Evaluation should be based on average accuracy in predicting this preference over the testing partition (provided). Any other information about the vote (e.g. ID of the user) is not provided.

Attribute Information:

Each row in this text file represents one anonymous user vote. Each line contains three comma-separated fields. The first two fields are YouTube video IDs. The third field is either 'left' or 'right'. Left indicates the first video from the pair was voted to be funnier than the second. Right indicates the opposite preference.

Citation Request:

Sanketh Shetty, 'Quantifying comedy on YouTube: why the number of oâ€™s in your LOL matter,' Google Research Blog.
###########################################

