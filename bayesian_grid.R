# bucket coordinates to a certain degree of precision (granularity).
# used in making a discretized coordinate grid for feeding into a
# bayesian classifier
bucket.coords <- function(x, num.buckets = 10)
{

  # translate coordinates so they start at 0
  x <- x - min(x)

  # determine how big each bucket should be
  x.incr <- max(x) / num.buckets

  # assign each element to its bucket
  buckets <- x %/% x.incr

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
  x.buckets <- bucket.coords(x, num.x.buckets)
  y.buckets <- bucket.coords(y, num.y.buckets)
  grid <- zeroes.grid(num.y.buckets, num.x.buckets)
  for (i in 1:length(x.buckets)) {
    grid[y.buckets[i], x.buckets[i]] <- grid[y.buckets[i], x.buckets[i]] + 1
  }

  grid
}