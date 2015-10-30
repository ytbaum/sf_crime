train <- read.csv("data/train.csv")



sample.submission <- read.csv("data/sampleSubmission.csv")
sample.submission.head <- head(sample.submission)
test <- read.csv("data/test.csv")
test.head <- head(test)

offenses <- colnames(sample.submission.head)
offenses <- offenses[!(offenses == "Id")] # remove "Id" from this list, since it's not an offense