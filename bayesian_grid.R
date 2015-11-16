# min and max values of x and y
# this is over both train and test sets
# currently hard-coded, maybe later I'll write a more intelligent way to do this
min.x <- -122.5136
max.x <- -122.3649
min.y <- 37.70788
max.y <- 37.82062

# bucket coordinates to a certain degree of precision (granularity).
# used in making a discretized coordinate grid for feeding into a
# bayesian classifier
bucket.coords <- function(v, min.v, max.v, num.buckets = 10)
{

  # translate coordinates so they start at 0
  # round to the nearest 1/10000 because otherwise rounding errors mess things up
  v <- round(v - min.v, digits = 4)

  # determine how big each bucket should be
  incr <- (max.v - min.v) / num.buckets

  # assign each element to its bucket. Bucket will be an integer, 0 to (num.buckets - 1)
  buckets <- v %/% incr

  # buckets should be 1-indexed because R is 1-indexed
  buckets <- buckets + 1

  # items having the max value in x will have been placed in the (num.buckets + 1)th bucket
  # put them in the (num.bucket)th bucket instead
  buckets[which(buckets > num.buckets)] <- num.buckets

  return(buckets)
}

# create initial grid of all zeroes
zeroes.grid <- function(num.rows, num.columns)
{
  output <- matrix(data = rep.int(0, num.rows * num.columns), num.rows, num.columns)
  return(output)
}

# returns a discretized grid, where each grid square contains a count of the number
# of pairs of coordinates that were within that square
get.loc.matrix <- function(x, y, num.x.buckets = 10, num.y.buckets = num.x.buckets)
{
  x.buckets <- bucket.coords(x, min.x, max.x, num.x.buckets)
  y.buckets <- bucket.coords(y, min.y, max.y, num.y.buckets)
  grid <- zeroes.grid(num.y.buckets, num.x.buckets)
  for (i in 1:length(x.buckets)) {
    grid[y.buckets[i], x.buckets[i]] <- grid[y.buckets[i], x.buckets[i]] + 1
  }

  grid
}

# re-allocate a little bit of probability mass to the places in the grid where probability = 0
# just so that there is no place in the grid where probability = 0
smooth.grid <- function(grid, reserve = 0.0001)
{
  reduction.factor <- 1 - reserve
  zero.indices <- which(grid == 0)
  grid <- grid / sum(grid) # make this into a probability distribution
  grid <- grid * reduction.factor
  addition <- reserve / length(zero.indices)
  grid[zero.indices] <- addition

  return(grid)
}

# calculate the probability of seeing these coordinates if we're sampling from the
# distribution represented by this grid
grid.prob <- function(x, y, grid)
{
  x.bucket <- bucket.coords(x, min.x, max.x, ncol(grid))
  y.bucket <- bucket.coords(y, min.y, max.y, nrow(grid))
  return(grid[x.bucket, y.bucket])
}

get.category.grid <- function(category, data, num.x.buckets = 10, num.y.buckets = num.x.buckets)
{
  category.rows <- which(data$Category == category)
  data <- data[category.rows,]
  grid <- get.loc.matrix(data$X, data$Y, num.x.buckets, num.y.buckets)
  grid <- smooth.grid(grid)

  grid
}

get.category.grids <- function(data, num.x.buckets = 10, num.y.buckets = num.x.buckets)
{
  categories <- as.character(unique(data$Category))
  names(categories) <- categories # hack to retain category names as indices into list

  lapply(categories, get.category.grid, data = data, num.x.buckets = num.x.buckets,
         num.y.buckets = num.y.buckets)
}

category.raw.cts <- function(data)
{
  categories <- as.character(data$Category)
  prior <- table(categories)
  return(prior)
}

# get the whole bayesian location model
bayes.loc.model <- function(data)
{
  grids <- get.category.grids(data)
  prior <- smooth.grid(category.raw.cts(data))

  return(list(grids = grids, prior = prior))
}

# get a prediction for a single row using bayesian location model
bayes.loc.pred <- function(m, x, y)
{
  x.bucket <- bucket.coords(x, min.x, max.x, num.buckets = 10)
  y.bucket <- bucket.coords(y, min.y, max.y, num.buckets = 10)

  grids <- m$grids
  prior <- m$prior
  categories <- names(prior)

  get.prob <- function(category, prior, grids) {
    prior.prob <- prior[category]
    grid <- grids[[category]]
    cond.prob <- grid[y.bucket, x.bucket]

    return(prior.prob * cond.prob)
  }

  probs <- sapply(categories, get.prob, prior, grids)

  return(probs)
}