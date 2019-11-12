par(mar=c(5,5,4,1)+.1)
rm(list = ls())
options(digits = 10)

#Values of x to be used for estimation
X = c(21.72, 14.65, 50.42, 28.78, 11.23)
#Get MLE for theta for value of X
theta.est = length(X) / sum(log(X))
print(theta.est)

#Negative log likelihood function of f_x
neg.log.f_x = function(par, dat) {
  return( -1 * (length(dat) * log(par) - sum( (par + 1) * log(dat))))
}

#Use negative because optim minimizes, we want to maximum
#Get numerical estimates from optim()
numerical = optim(par = 0.001, neg.log.f_x, dat = X, method="L-BFGS-B", lower=0.001, hessian=TRUE)
print(numerical)

#To estimate confidence interval, assume theta.numerical.est is normal and theta is an unbiased estimator.
confidence = 0.95
alpha = (1 - 0.95)
standard.error = sqrt(diag(solve(numerical$hessian)))
estimator = numerical$par

#Get confidence interval from formula theta_est +/- Z_(alpha/2) * standard_error
confidence.interval = c(estimator - qnorm(1 - alpha/2) * standard.error, estimator + qnorm(1 - alpha/2) * standard.error)
print(confidence.interval)
