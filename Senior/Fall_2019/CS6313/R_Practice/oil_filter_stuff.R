gen_X_of = function(U, V) {
  if (U < 1/5) {
    return(-1/5 * log(V,))
  }
  return(-1/20 * log(V, exp(1)))
}

nsim = function(epsilon, alpha) {
  return(
    ceiling( 1/4 * (qnorm(1 - (alpha/2))/epsilon)^2)
  )
}

pt.in.square = function() {
  return(runif(2, -1, 1))
}

is.pt.in.circle = function(pt) {
  return(sum(pt^2) <= 1)
}

pi.estimate = function(N) {
  x = replicate(N, is.pt.in.circle(pt.in.square()))
  return(4 * mean(x))
}