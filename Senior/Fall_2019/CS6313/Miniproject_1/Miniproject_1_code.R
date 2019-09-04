#PDF from exercise 4.11
density.function <- function(t) {
  if (t[1] <= 0)
    return(c(0))
  return(c(0.2 * exp(-0.1 * t) - 0.2 * exp(-0.2 * t)))
}
#Generate block lifetimes x_A and x_B, choose longest living one to be T. Do this n times.
t <- replicate(10000, max(c(rexp(1,1/10),rexp(1,1/10))))
#Histogram of lifetimes of satellites with curve to fit the data.
hist(t,breaks=20, probability = TRUE)
x <- t
curve(expr = sapply(x, density.function), add = TRUE)
#Show sample average (should be close to 15)
mean(t)
#Show percentage of satellites that last longer than 15 years
gt_15_years = length(t[t>15]) / length (t)
gt_15_years
#Difference between estimated results using Monte Carlo and integrals.
abs(gt_15_years - 0.3964)
