# Clear all global values before each run and set number of sig figs.
rm(list = ls())
options(digits=5)

gpa = read.csv('gpa.csv')
set.seed(1234)

# Generate indices from a dataset X using uniform function
bootstrap_sample = function(X, n) {
  return(floor(runif(n, 1, length(X) + 1)))
}

# Get a bootstrap sample for two paired sets X and Y,
# then return sample correlation between the two.
bootstrap_estimate = function(X, Y, n) {
  indices = bootstrap_sample(X, n)
  return(cor(X[indices], Y[indices]))
}

# Scatter plot
plot(gpa$gpa, gpa$act, xlab='GPA', ylab = 'ACT',
     main = "Scatterplot of ACT vs GPA")

# Point estimate of desired rho
rho = cor(gpa$gpa, gpa$act)

# Generated a sorted list of estimates of rho through bootstrapping.
b = 1000
n = 120
bootstrap_rhos = sort(
    replicate(b, bootstrap_estimate(gpa$gpa, gpa$act, n))
  )

# Calculate se, bias, and Confidence interval from bootstrap samples.
bias = 1/b * sum(bootstrap_rhos - rho)
se = sqrt(1/(b-1) * sum((bootstrap_rhos - mean(bootstrap_rhos))^2))
alpha = 1 - 0.95
CI = c(bootstrap_rhos[(b + 1) * alpha / 2],
       bootstrap_rhos[(b + 1) * (1  - alpha / 2)])

cat("Bootstrap results:",
    "\n Point estimate                       : ", rho,
    "\n Bootstrap estimate of bias           : ", bias,
    "\n Bootstrap estimate of standard error : ", se,
    "\n Percentile Confidence Interval       : [", CI[1], "," ,CI[2], "]\n")