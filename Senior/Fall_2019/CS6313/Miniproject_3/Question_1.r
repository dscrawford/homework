#Method of moments uses 2*sample average as its estimate.
MSE_from_MME = function(theta, n) {
  return(theta^2 / (3 * n))
}

Theta_for_MME = function(D) {
  return(2 * mean(D))
}

MME_Get_MSE_old = function(N, n, theta) {
  theta_est = mean(replicate(N, Theta_for_MME(runif(n, 0, theta))))
  return(MSE_from_MME(theta_est, n))
}

MSE_from_MLE = function(theta, n) {
  return((2 * theta^2) / ((n + 2) * (n + 1)))
}

Theta_for_MLE = function(D) {
  return(max(D))
}

MLE_Get_MSE_old = function(N, n, theta) {
  theta_est = mean(replicate(N, Theta_for_MLE(runif(n, 0, theta))))
  return(MSE_from_MLE(theta_est, n))
}

MSE = function(thetas, theta_target) {
  return(var(thetas) + (mean(thetas) - theta_target)^2)
}

#Method of moments uses 2*sample average as its estimate.
Theta_for_MME = function(D) {
  return(2 * mean(D))
}

Theta_for_MLE = function(D) {
  return(max(D))
}

MLE_Get_MSE = function(N, n, theta) {
  thetas = replicate(N, Theta_for_MLE(runif(n, 0, theta)))
  return(MSE(thetas, theta))
}

MME_Get_MSE = function(N, n, theta) {
  thetas = replicate(N, Theta
                     _for_MME(runif(n, 0, theta)))
  return(MSE(thetas,theta))
}

MSE = function(thetas, theta_target) {
  return(var(thetas) + (mean(thetas) - theta_target)^2)
}
