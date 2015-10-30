# helper function to mult.log.loss
row.log.loss <- function(row)
{
  label <- row["Category"]
  prob <- as.numeric(row[label])
  return(log(prob))
}

# compute the multiclass log loss over a set of predictions
mult.log.loss <- function(pred, raw.data)
{
  pred <- as.data.frame(pred)
  pred$Category <- raw.data$Category

  row.log.losses <- apply(pred, 1, row.log.loss)
  return(-mean(row.log.losses))
}

# do all the prep work to prep data for training and prediction
build.model <- function(train)
{
  train$WoY <- as.integer(strftime(train$Dates, "%W"))
  train$DoM <- as.integer(strftime(train$Dates, "%d"))
  train$Month <- as.integer(strftime(train$Dates, "%m"))
  train$Year <- as.integer(strftime(train$Dates, "%Y")) - 2000
  train$Hour <- as.integer(strftime(train$Dates, "%H"))

  return(train)
}

# grab a small sample of the training data
sample.train <- function(train, n = 10000)
{
  # until I find a way to compute this in a reasonable amount of time, only deal with 10000 rows
  sample.rows <- sample(nrow(train), 10000)
  smpl <- train[sample.rows,]

  return(smpl)
}

# get an "inclusive sample"
# i.e., get a sample of train such that at least one example is included
# for each kind of class label
incl.sample <- function(train, n = 10000)
{
  categories <- unique(as.character(train$Category))

  find.row <- function(category, train) {
    indices <- which(train$Category == category)
    index <- sample(indices, 1)
    return(index)
  }

  # to begin with, get a sample consisting of one row for each category
  first.sample <- sapply(categories, find.row, train)

  # then sample the rest of the data in order to obtain the desired total sample size
  remaining.indices <- 1:nrow(train)
  remaining.indices <- remaining.indices[!(remaining.indices %in% first.sample)]
  second.sample.size <- n - length(first.sample)
  second.sample <- sample(remaining.indices, second.sample.size)

  combined.sample <- c(first.sample, second.sample)

  output <- train[combined.sample,]

  return(output)
}

# assign each row to a fold for cross-validation
# (using only the row indices, not the data itself)
create.folds <- function(num.folds, num.rows)
{
  labels <- 1:num.folds

  # randomly assign each row index a label
  rep.times <- ceiling(num.rows / num.folds)
  labels.raw <- rep(labels, rep.times)
  labels.raw <- labels.raw[1:num.rows]
  smpl <- sample(labels.raw)

  create.fold <- function(label, smpl) {
    fold.test <- which(smpl == label)
    fold.train <- which(smpl != label)

    fold <- list(train = fold.train, test = fold.test)
  }

  folds <- lapply(labels, create.fold, smpl)

  return(folds)
}

# run one fold of cross-validation
cv.fold <- function(fold, train)
{
  train.indices <- fold$train
  test.indices <- fold$test

  cv.train <- train[train.indices,]
  cv.test <- train[-train.indices,]

  # only include categories in the test set that appear in the train set
  categories <- as.character(unique(cv.train$Category))
  cv.test.rows <- which(cv.test$Category %in% categories)
  cv.test <- cv.test[cv.test.rows,]

  m <- multinom(Category ~ DayOfWeek + X + Y + WoY + DoM + Month + Year + Hour,
                data = cv.train,
                entropy = TRUE)

  pred <- predict(m, newdata = cv.test, type = "probs")

  mll <- mult.log.loss(pred, cv.test)

  return(mll)
}

# run cross-validation over a small sample of training data
run.sample <- function(iter, train)
{
  print(paste("Now on sample", iter))

  train <- sample.train(train)
  train <- build.model(train)

  folds <- create.folds(5, nrow(train))

  fold.scores <- sapply(folds, cv.fold, train)
  sample.mean <- mean(fold.scores)

  return(sample.mean)
}

# run cross-validation over multiple samples from training data
run.samples <- function(train, num.samples = 10)
{
  sample.scores <- sapply(1:num.samples, run.sample, train)
  return(mean(sample.scores))
}