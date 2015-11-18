require(logging)
basicConfig()

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
build.model <- function(data, mode = 'train')
{
  data$WoY <- as.integer(strftime(data$Dates, "%W"))
  data$DoM <- as.integer(strftime(data$Dates, "%d"))
  data$Month <- as.integer(strftime(data$Dates, "%m"))
  data$Year <- as.integer(strftime(data$Dates, "%Y")) - 2000
  data$Hour <- as.integer(strftime(data$Dates, "%H"))

  day.num <- sapply(data$DayOfWeek, day.to.num)
  data$DayOfWeek <- day.num

  PdDistrict <- as.character(data$PdDistrict)
  PdDistrict.mat <- model.matrix( ~ PdDistrict - 1, data)

  columns.to.keep <- c("Category", "X", "Y", "DayOfWeek", "WoY", "DoM", "Month", "Year", "Hour")
  if (mode == 'test') {
    columns.to.keep <- columns.to.keep[columns.to.keep != "Category"]
    columns.to.keep <- c("Id", columns.to.keep)
  }

  data <- data[,columns.to.keep]
  data <- cbind(data, as.data.frame(PdDistrict.mat))

  return(data)
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

get.model <- function(train)
{
  require(nnet)
  m <- multinom(Category ~ DayOfWeek + X + Y + WoY + DoM + Month + Year + Hour
                + PdDistrictNORTHERN + PdDistrictPARK + PdDistrictINGLESIDE
                + PdDistrictBAYVIEW + PdDistrictRICHMOND + PdDistrictCENTRAL
                + PdDistrictTARAVAL + PdDistrictTENDERLOIN + PdDistrictMISSION
                + PdDistrictSOUTHERN,
                data = train,
                entropy = TRUE)

  return(m)
}

get.preds <- function(model, input.data)
{
  require(nnet)
  predict(model, newdata = input.data, type = "probs")
}

# run one fold of cross-validation
cv.fold <- function(fold, train)
{
  logdebug("in cv.fold")
  train.indices <- fold$train
  test.indices <- fold$test

  cv.train <- train[train.indices,]
  cv.test <- train[-train.indices,]

  # only include categories in the test set that appear in the train set
  categories <- as.character(unique(cv.train$Category))
  cv.test.rows <- which(cv.test$Category %in% categories)
  cv.test <- cv.test[cv.test.rows,]

  m <- bayes.loc.model(cv.train, 3, 3)

  pred <- get.bayes.preds(m, cv.test)

  mll <- mult.log.loss(pred, cv.test)

  return(mll)
}

# run cross-validation over a small sample of training data
run.sample <- function(iter, train)
{
  loginfo(paste("Now on sample", iter))

  train <- incl.sample(train)
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

# generate a submission file
# train: the data to train on
# test: the target data to be classified
# subm.name: the name of this submission (a short name to describe it)
generate.subm <- function(train, test, subm.name)
{

  init.subm.dir(subm.name)
  addHandler(writeToFile,
             file = get.log.file(subm.name),
             level = 'INFO')

  loginfo(paste("Current HEAD ref:", get.head.ref()))
  loginfo(paste("Current commit:", get.current.commit()))

  train.sample <- incl.sample(train)
  save.sample(train.sample, subm.name)
  m <- bayes.loc.model(train.sample, 3, 3)
  loginfo("The call used to train the model:")
  loginfo(m$call)

  preds <- as.data.frame(get.bayes.preds(m, test))

  id.df <- data.frame(Id = test[,"Id"])

  output.df <- cbind(id.df, preds)

  write.csv(output.df, get.subm.file(subm.name), row.names = FALSE)
  zip.subm.file(subm.name)
}

# takes the name of a submission as an argument, and returns the corresponding
# directory path
get.subm.dir.name <- function(subm.name)
{
  proj.root <- getwd()
  subm.dir <- paste(proj.root, "submissions", subm.name, sep = "/")
  return(subm.dir)
}

# make a new directory for a given submission
make.subm.dir <- function(subm.name)
{
  new.dir <- get.subm.dir.name(subm.name)
  dir.create(new.dir, recursive = TRUE)
}

# initialize the directory for a given submission
init.subm.dir <- function(subm.name)
{
  subm.dir <- get.subm.dir.name(subm.name)
  if (dir.exists(subm.dir)) {
    unlink(subm.dir, recursive = TRUE)
  }

  make.subm.dir(subm.name)
}

# takes the name of a submission as an argument, and returns the log file for
# that submission
get.log.file <- function(subm.name)
{
  subm.dir <- get.subm.dir.name(subm.name)
  log.file <- paste(subm.dir, "log.txt", sep = "/")
  return(log.file)
}

# takes the name of a submission as an argument, and returns the name of the file
# containing the list of rows comprising the sample of training data used to 
# train the model
get.sample.file <- function(subm.name)
{
  subm.dir <- get.subm.dir.name(subm.name)
  sample.file <- paste(subm.dir, "sample.txt", sep = "/")
  return(sample.file)
}

# based on the submission name, return the absolute path of the submission file
get.subm.file <- function(subm.name)
{
  subm.dir <- get.subm.dir.name(subm.name)
  subm.file <- paste(subm.dir, "submission.csv", sep = "/")
  return(subm.file)
}

# zip the submission file for submission to kaggle
zip.subm.file <- function(subm.name)
{
  subm.dir <- get.subm.dir.name(subm.name)
  subm.file <- paste(subm.dir, "submission.csv", sep = "/")
  zip.file <- paste(subm.dir, "submission.zip", sep = "/")
  zip(zip.file, subm.file)
}

# store the rows that make up a sample of a given data set
# data.sample: the sample whose row indices we want to record
# subm.name: the name of the submission that this sample is being used for
save.sample <- function(data.sample, subm.name)
{
  sample.file <- get.sample.file(subm.name)
  sample.indices <- as.numeric(rownames(data.sample))
  write(sample.indices, file = sample.file, sep = "\n")
}

# get the current ref that the HEAD branch is pointing to in git
get.head.ref <- function()
{
  proj.root <- getwd()
  head.file <- paste(proj.root, ".git", "HEAD", sep = "/")
  head.ref.str <- readChar(head.file, file.info(head.file)$size)
  head.ref <- unlist(strsplit(readChar(".git/HEAD", file.info(".git/HEAD")$size), " "))[2]
  head.ref <- gsub("\n", "", head.ref)

  return(head.ref)
}

# get the current commit that HEAD is pointing to in git
get.current.commit <- function()
{
  proj.root <- getwd()
  ref.file <- paste(proj.root, ".git", get.head.ref(), sep = "/")
  commit.hash <- readChar(ref.file, file.info(ref.file)$size)
  commit.hash <- gsub("\n", "", commit.hash)

  return(commit.hash)
}

### Data-munging functions

# convert "day of week" values ("Sunday", "Monday", etc.) to numbers 1-7
day.to.num <- function(day)
{
  day.numbers <- c(Sunday = 1,
                   Monday = 2,
                   Tuesday = 3,
                   Wednesday = 4,
                   Thursday = 5,
                   Friday = 6,
                   Saturday = 7)

  output <- day.numbers[day]

  return(output)
}
