# Clear all global values before each run and set number of sig figs.
rm(list = ls())
options(digits=5)

set.seed(1234)
volt = read.csv('VOLTAGE.csv')

remoteV = volt$voltage[volt$location == 1]
localV  = volt$voltage[volt$location == 0]
qqnorm(localV, main = "Local location voltages")
qqline(localV)
hist(localV, breaks=10, xlab = 'voltage', main = 'Local voltage histogram')

qqnorm(remoteV, main = "Remote location voltages")
qqline(remoteV)
hist(remoteV, breaks=10, xlab = 'voltage', main = 'Remote voltage histogram')

x1bar = mean(remoteV)
x2bar = mean(localV)
s1    = sd(remoteV)
s2    = sd(localV)
n     = length(remoteV)
m     = length(localV)
diff  = x1bar - x2bar
se = sqrt(s1^2 / n + s2^2 / m)
alpha = 0.05
CI = diff + c(-1, 1) * qnorm(1 - alpha / 2) * se

cat("Confidence Interval: [", CI[1], ",", CI[2], "]")