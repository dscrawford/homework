DE_tot = c(0)
find_errors = function() {
  Last3 = c(28,22,18)
  DE = sum(Last3)
  Ti = 0; X = min(Last3)
  while (X > 0) {
    lambda = min(Last3)
    U = runif(1); X = 0;
    while (U >= exp(-lambda)) {
      U = U * runif(1); X = X + 1;
    }
    Ti = Ti + 1; DE = DE + X
    Last3 = c(Last3[2:3], X)
  }
  # DE_tot = c(DE_tot, DE)
  return(Ti - 1)
}
A = replicate(1000, find_errors())