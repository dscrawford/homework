#Values of x to be used for estimation
X = c(21.72, 14.65, 50.42, 28.78, 11.23)

#Get MLE for theta for value of X
theta.est = 1 / log(X)
print(theta.est)

#negative f_x function because optim by default minimizes, so flip to maximize.
neg.f_x = function(x, theta) {
  if (x >= 1)
    return(-1 * theta / (x^(theta + 1)))
  return(0)
}

#Get numerical estimates from optim()
theta.numerical.est = unlist(lapply(X, function(x) {optim(par = 0, f_x, x = x, method = "Brent", lower=-10, upper=10)$par[1]}))
print(theta.numerical.est)

#To estimate confidence interval, assume theta.numerical.est is normal and theta is an unbiased estimator.
confidence = 0.95
alpha = (1 - 0.95)
standard.error = sd(theta.numerical.est)
#If theta estimator is normal, then estimator is mean of sample
estimator = mean(theta.numerical.est)

#Get confidence interval from formula theta_est +/- Z_(alpha/2) * standard_error
confidence.interval = c(estimator - qnorm(1 - alpha/2) * standard.error, estimator + qnorm(1 - alpha/2) * standard.error)
print(confidence.interval)