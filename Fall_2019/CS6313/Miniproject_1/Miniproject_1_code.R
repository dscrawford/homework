density.function <- function(t) {
  if (t[1] <= 0)
    return(c(0))
  return(c(0.2 * exp(-0.1 * t) - 0.2 * exp(-0.2 * t)))
}
t <- replicate(100000, max(c(rexp(1,1/10),rexp(1,1/10))))
hist(t,breaks=20, probability = TRUE)
x <- t
curve(expr = sapply(x, density.function), add = TRUE)
mean(t)
gt_15_years = length(t[t>15]) / length (t)
gt_15_years
abs(gt_15_years - 0.3964)
