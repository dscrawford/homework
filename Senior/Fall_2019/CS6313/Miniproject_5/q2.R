require(boot)
rm(list = ls())
options(digits=5)

in_confidence_interval = function(mean, CI) {
  return(mean >= CI[1] && mean <= CI[2])
}

coverage_probability = function(lambda, CI) {
  return(sum(replicate(100, in_confidence_interval(mean(rexp(10, lambda)), CI))) / 100)
}

coverage_probability_percentile = function(X, indices, lambda) {
  D  = sort(X[indices])
  CI = c(D[alpha / 2 * length(D) + 1], D[(1 - alpha / 2) * length(D) + 1])
  return(coverage_probability(lambda, CI))
}

coverage_probability_large_sample = function(X, indices, lambda) {
  D  = X[indices]
  CI = mean(D) + c(-1, 1) * qnorm(0.975) * sd(D) / sqrt(length(D))
  return(coverage_probability(lambda, CI))
}

D = expand.grid(c(5, 10, 30, 100),c(0.01, 0.1, 1, 10))
alpha = 0.05
b     = 999

r = apply(D, 1, FUN=function(x) boot(rexp(x[1], x[2]), coverage_probability_percentile, 
                                     lambda = as.numeric(x[2]), R=999))

t = apply(D, 1, FUN=function(x) boot(rexp(x[1], x[2]), coverage_probability_large_sample, 
                                     lambda = as.numeric(x[2]), R=999)$t0)

x = D[1,]