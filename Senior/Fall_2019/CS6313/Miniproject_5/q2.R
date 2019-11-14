require(boot)
rm(list = ls())
options(digits=5)
set.seed(1234)

in_confidence_interval = function(mean, CI) {
  return(mean >= CI[1] & mean <= CI[2])
}

generate_sample = function(D, lambda) {
  n = length(D)
  return(rexp(n = n, rate = lambda))
}

# Calculates bootstrap percentile and large sample confidence interval.
Confidence_Intervals = function(n, lambda) {
  b  = 999
  D  = rexp(n, lambda)
  x  = mean(D)
  s  = sd(D)
  perc = boot.ci(boot(data = D, statistic = mean, R=b, sim = "parametric", ran.gen = generate_sample, mle = 1 / x, ), type = "perc")$perc[c(4,5)]
  norm = x + c(-1, 1) * qnorm(0.975) * s / sqrt(n)
  return(c(norm, perc))
}

MC_Coverage_Probability = function(n, lambda) {
  B = t(replicate(5000, Confidence_Intervals(n, lambda)))
  norm = sum(apply(B, 1, function(x) in_confidence_interval(1 / lambda, x[c(1,2)]))) / 5000
  perc = sum(apply(B, 1, function(x) in_confidence_interval(1 / lambda, x[c(3,4)]))) / 5000
  return(c(norm, perc))
}

df = expand.grid(n=c(5, 10, 30, 100), lambda=c(0.01, 0.1, 1, 10))

results = apply(df, 1, function(x) MC_Coverage_Probability(as.numeric(x[1]), as.numeric(x[2])))
norm = results[1,]
perc = results[2,]
print(data.frame(df, normal_coverage_prob = norm, percentile_coverage_prob = perc))
print(summary(norm - perc))