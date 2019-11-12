# Clear all global values before each run, set plot properties, and set number of sig figs.
par(mar=c(5,5,4,1)+.1)
rm(list = ls())
options(digits = 10)

# Initial values and functions
n = c(1,2,3,5,10,30)
theta = c(1,5,50,100)
getunif = function(x){ # Get 1000 runif values from (0,x)
  return(runif(1000,0,x))
}

theta.mle = c();
theta.mme = c();
mse.mle = c();
mse.mme = c();

# Use monte carlo method to generate population and sample from that population
#population = lapply(theta, getunif)
for(max in theta){
  population = runif(1000,0,max)
  for(sampleSize in n){r =
    # cat("for n=" , sampleSize, "and population in (0 ,",max,")", "\n")
    population.sample = sample(population,sampleSize)
    # Find MLE estimator
    population.sample = sort(population.sample)
    theta.mle = append(theta.mle, population.sample[length(population.sample)])
    # cat("theta.mle", theta.mle[length(theta.mle)], "\n")
    # Find MME estimator
    theta.mme = append(theta.mme, mean(population.sample)*2)
    # cat("theta.mme", theta.mme[length(theta.mme)], "\n")
    
    # Error for MLE
    mse.mle = append(mse.mle, (2*theta.mle[length(theta.mle)]^2/((sampleSize+2)*(sampleSize+1))))
    # cat("mse.mle", mse.mle[length(mse.mle)], "\n")
    # Error for MME
    mse.mme = append(mse.mme , (theta.mme[length(theta.mme)]^2/(3*sampleSize)))
    # cat("mse.mme", mse.mme[length(mse.mme)], "\n")
  }
}

# Create a 6x4 matrix with each colum representing n[i] and each row representing theta[i]
theta.mle.matrix = matrix(theta.mle,ncol = 6, nrow = 4, byrow = TRUE)
theta.mme.matrix = matrix(theta.mme,ncol = 6, nrow = 4, byrow = TRUE)
mse.mle.matrix = matrix(mse.mle,ncol = 6, nrow = 4, byrow = TRUE)
mse.mme.matrix = matrix(mse.mme,ncol = 6, nrow = 4, byrow = TRUE)

# Plot the lines of each theta with respect to n
matplot(t(theta.mle.matrix), type = c("b"),pch=1,col = 1:4, xaxt = "n", 
        xlab = "sample size n", ylab = expression(hat(theta)[MLE]))
axis(1, at=1:6, labels=n)


matplot(t(theta.mme.matrix), type=c("b"),pch=1,col = 1:4, xaxt = "n", 
        xlab = "sample size n", ylab = expression(hat(theta)[MME]))
axis(1, at=1:6, labels=n)

# Plot the errors of each theta with respect to n
matplot(t(mse.mle.matrix), type = c("b"),pch=1,col = 1:4, xaxt = "n", 
        xlab = "sample size n", ylab = expression(MSE(hat(theta)[MLE])))
axis(1, at=1:6, labels=n)


matplot(t(mse.mme.matrix), type = c("b"),pch=1,col = 1:4, xaxt = "n", 
        xlab = "sample size n", ylab = expression(MSE(hat(theta)[MME])))
axis(1, at=1:6, labels=n)


