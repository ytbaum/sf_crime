source("helpers.R")
source("bayesian_grid.R")

if (is.null(train)) {
  loginfo("Reading training data...")
  train <- read.csv("data/train.csv")
}

loginfo("Getting all possible pairs of num.x.buckets and num.y.buckets...")
bucket.pairs <- get.bucket.pairs(10)

loginfo("All possible pairs of num.x.buckets and num.y.buckets:")
loginfo(bucket.pairs)

loginfo("Getting the model constructor function for each pair of model parameters...")
bgrid.constructors <- lapply(bucket.pairs, get.constructor)

loginfo("The first few model constructor functions:")
loginfo(head(bgrid.constructors))

loginfo("Running parameter search through cross-validation...")
scores <- sapply(bgrid.constructors, run.samples, train, 10)

loginfo("Results are available in the vector called 'scores'.")

### Write results to a file ###

filename <- "bgrid_param_scores.txt"
loginfo(paste0("Writing scores to ", filename, "..."))
output <- as.matrix(sort(scores))
sectors <- rownames(output)

# helper function to convert a space-separated string of two digits into a one-row data.frame
str.to.df <- function(str) {
  coords <- as.numeric(unlist(strsplit(str, " ")))
  data.frame(x.buckets = coords[1], y.buckets = coords[2])}

bucket.ct.cols <- do.call(rbind, lapply(sectors, str.to.df))

colnames(output) <- "score"
output <- cbind(bucket.ct.cols, output)

write.table(output, filename, sep = "\t", row.names = FALSE)

loginfo("Done.")