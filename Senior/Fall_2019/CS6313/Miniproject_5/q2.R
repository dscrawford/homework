require(boot)
rm(list = ls())
options(digits=5)

in_confidence_interval = function(mean, CI) {
  return(mean >= CI[1] & mean <= CI[2])
}

capture_percentage = function(B, t) {
  CI = percentile_confidence_interval(B$t)
  return(sum(in_confidence_interval(t, CI)) / length(B$t))
}

generate_sample = function(D, par) {
  return(rexp(par[1], par[2]))
}

percentile_confidence_interval = function(D) {
  est = sort(D)
  n = length(est)
  alpha = 0.05
  return(c(est[alpha / 2 * n + 1], est[(1 - alpha/2) * n + 1]))
}

run_test = function(n, lambda, MC) {
  tMean = 1 / lambda
  B = replicate(5000, boot(data = rexp(n, lambda), 
                           statistic = mean, R=b, sim = "parametric", 
                           ran.gen = generate_sample, mle = c(n, lambda)))
  
}

df = expand.grid(n=c(5, 10, 30, 100), lambda=c(0.01, 0.1, 1, 10))
alpha = 0.05
b     = 999
MC = replicate(999, mean(rexp(100, 1/100)))

results = apply(df, 1, function(x) run_test(as.numeric(x[1]), as.numeric(x[2])))

n = 5
lambda = 0.01
B = replicate(5000, boot(data = rexp(n, lambda), 
                         statistic = mean, R=b, sim = "parametric", 
                         ran.gen = generate_sample, mle = c(n, lambda)))

#apply(df, 1, FUN=function(x) boot(unlist(df$sample),
#                                  mean, lambda = as.numeric(df[2]), R=b))

# What to do:
# Generate 5000 means
# Check from each generated mean if it is captured by the bootstrap interval

