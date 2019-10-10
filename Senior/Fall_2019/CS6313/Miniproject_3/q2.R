par(mar=c(5,5,4,1)+.1)
rm(list = ls())
options(digits = 10)

#Values of x to be used for estimation
X = c(21.72, 14.65, 50.42, 28.78, 11.23)
#Get MLE for theta for value of X
theta.est = 1 / log(X)
print(theta.est)

log.f_x = function(par, dat) {
  return(1)
}
#negative f_x function because optim by default minimizes, so flip to maximize.
neg.f_x = function(par, dat) {
  return( sum( -1 * par / (dat[dat>=1]^(par + 1)) ) )
}

neg.log.f_x = function(par, dat) {
  return(sum(-1 * (log(par) - (par +1) * log(dat))))
}


#Get numerical estimates from optim()
numerical = optim(par = 0.1, neg.log.f_x, dat = X, method="L-BFGS-B", lower=0, hessian=TRUE)
numerical
print(numerical)

#To estimate confidence interval, assume theta.numerical.est is normal and theta is an unbiased estimator.
confidence = 0.95
alpha = (1 - 0.95)
#
standard.error = mean(sqrt(diag(solve(numerical$hessian))))
#If theta estimator is normal, then estimator is mean of sample
estimator = mean(numerical$par)

#Get confidence interval from formula theta_est +/- Z_(alpha/2) * standard_error
confidence.interval = c(estimator - qnorm(1 - alpha/2) * standard.error, estimator + qnorm(1 - alpha/2) * standard.error)
print(confidence.interval)