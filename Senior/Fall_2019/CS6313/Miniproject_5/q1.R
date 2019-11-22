library(ggplot2)
library(distributions3)
library(DAAG)
MyData = read.csv(file="bodytemp-heartrate.csv", header=TRUE, sep=",")

male = MyData[MyData$gender == 1,]
female = MyData[MyData$gender == 2,]

# For Body Temperature
# See if data is normal
qqnorm(male$body_temperature, main = "Male Body Temperature Readings")
qqline(male$body_temperature)
qqnorm(female$body_temperature, main = "Female Body Temperature Readings")
qqline(female$body_temperature)

# create side by side boxplot
# make a data frame in long format for plotting
test_results <- data.frame(
  Temperature = c(male$body_temperature, female$body_temperature),
  Gender = c(
    rep("Male", length(male$body_temperature)),
    rep("Female", length(female$body_temperature))
  )
)

ggplot(test_results, aes(x = Gender, y = Temperature, color = Gender), main = "Distribution of Male and Female") +
  geom_boxplot(outlier.colour = "black", outlier.shape = 17) +
  geom_jitter() +
  scale_color_brewer(type = "qual", palette = 2) +
  theme_minimal() +
  theme(legend.position = "none") 
# Female not normal but data size is large

# With H_0 = 0, calculate our Z-stat
z_stat = (mean(male$body_temperature) - mean(female$body_temperature) - 0) / 
  sqrt(var(male$body_temperature)/length(male$body_temperature) + 
         var(female$body_temperature)/length(female$body_temperature))

# Calculate 2 sided p value
Z <- Normal(0, 1)  # make a standard normal r.v.
1 - cdf(Z, abs(z_stat)) + cdf(Z, -abs(z_stat))
2 * cdf(Z, -abs(z_stat))



# For Heart Rate
# See if data is normal
qqnorm(male$heart_rate, main = "Male Heart Rate Readings")
qqline(male$heart_rate)
qqnorm(female$heart_rate, main = "Female Heart Rate Readings")
qqline(female$heart_rate)

# create side by side boxplot
# make a data frame in long format for plotting
test_results <- data.frame(
  Heart_Rate = c(male$heart_rate, female$heart_rate),
  Gender = c(
    rep("Male", length(male$heart_rate)),
    rep("Female", length(female$heart_rate))
  )
)

ggplot(test_results, aes(x = Gender, y = Heart_Rate, color = Gender), main = "Distribution of Male and Female") +
  geom_boxplot(outlier.colour = "black", outlier.shape = 17) +
  geom_jitter() +
  scale_color_brewer(type = "qual", palette = 2) +
  theme_minimal() +
  theme(legend.position = "none") 
# Female not normal but data size is large

# With H_0 = 0, calculate our Z-stat
z_stat = (mean(male$heart_rate) - mean(female$heart_rate) - 0) / 
  sqrt(var(male$heart_rate)/length(male$heart_rate) + 
         var(female$heart_rate)/length(female$heart_rate))

# Calculate 2 sided p value
Z <- Normal(0, 1)  # make a standard normal r.v.
1 - cdf(Z, abs(z_stat)) + cdf(Z, -abs(z_stat))
2 * cdf(Z, -abs(z_stat))


# To test if heart rate and body temperature are correlated

# create side by side boxplot
boxplot(MyData$body_temperature, main="Body Temperature", sub=paste("Outlier rows: ", boxplot.stats(cars$speed)$out))  # box plot for 'speed'
boxplot(MyData$heart_rate, main="Heart Rate", sub=paste("Outlier rows: ", boxplot.stats(cars$dist)$out))  # box plot for 'distance'

##  FOR MALES ##
# Plot scatter plot with regression line
plot(male$body_temperature ~ male$heart_rate, data = data.frame(male),
     main = "Scatter plot for Male Body Temperature vs. Heart Rate")
abline(lm(male$body_temperature ~ male$heart_rate, data = data.frame(male)))
cor(male$body_temperature, male$heart_rate)
# t-stat, p-value and Linear Regression Diagnostics
linearMod <- lm(body_temperature ~ heart_rate, data=male)  
summary(linearMod)
modelSummary <- summary(linearMod)  # capture model summary as an object
modelCoeffs <- modelSummary$coefficients  # model coefficients
beta.estimate <- modelCoeffs["heart_rate", "Estimate"]  # get beta estimate for speed
std.error <- modelCoeffs["heart_rate", "Std. Error"]  # get std.error for speed
t_value <- beta.estimate/std.error  # calc t statistic
p_value <- 2*pt(-abs(t_value), df=nrow(male)-ncol(male))  # calc p Value
f_statistic <- linearMod$fstatistic[1]  # fstatistic
f <- summary(linearMod)$fstatistic  # parameters for model p-value calc
model_p <- pf(f[1], f[2], f[3], lower=FALSE)
# AIC and BIC
AIC(linearMod)
BIC(linearMod) 
#k-fold cross validation
test_results <- data.frame(
  body_temperature = male$body_temperature,
  heart_rate = male$heart_rate
)
cvResults <- suppressWarnings(CVlm(data = test_results, form.lm=body_temperature ~ heart_rate, m=5, dots=FALSE, seed=29, legend.pos="topleft",  printit=FALSE, main="Body temperature ~ Heart rate"));  # performs the CV
attr(cvResults, 'ms') 

##  FOR FEMALES ##
plot(female$body_temperature ~ female$heart_rate, data = data.frame(female),
     main = "Scatter plot for Female Body Temperature vs. Heart Rate")
abline(lm(female$body_temperature~ female$heart_rate, data = data.frame(female)))
cor(female$body_temperature, female$heart_rate)
# t-stat, p-value and Linear Regression Diagnostics
linearMod <- lm(body_temperature ~ heart_rate, data=female)  
summary(linearMod)
modelSummary <- summary(linearMod)  # capture model summary as an object
modelCoeffs <- modelSummary$coefficients  # model coefficients
beta.estimate <- modelCoeffs["heart_rate", "Estimate"]  # get beta estimate for speed
std.error <- modelCoeffs["heart_rate", "Std. Error"]  # get std.error for speed
t_value <- beta.estimate/std.error  # calc t statistic
p_value <- 2*pt(-abs(t_value), df=nrow(female)-ncol(female))  # calc p Value
f_statistic <- linearMod$fstatistic[1]  # fstatistic
f <- summary(linearMod)$fstatistic  # parameters for model p-value calc
model_p <- pf(f[1], f[2], f[3], lower=FALSE)
# AIC and BIC
AIC(linearMod)
BIC(linearMod) 
#k-fold cross validation
test_results <- data.frame(
  body_temperature = female$body_temperature,
  heart_rate = female$heart_rate
)
cvResults <- suppressWarnings(CVlm(data = test_results, form.lm=body_temperature ~ heart_rate, m=5, dots=FALSE, seed=29, legend.pos="topleft",  printit=FALSE, main="Body temperature ~ Heart rate"));  # performs the CV
attr(cvResults, 'ms') 

##  FOR ALL  ##
plot(MyData$body_temperature ~ MyData$heart_rate, data = data.frame(MyData),
     main = "Scatter plot people's Body Temperature vs. Heart Rate")
abline(lm(MyData$body_temperature ~ MyData$heart_rate, data = data.frame(MyData)))
cor(MyData$body_temperature, MyData$heart_rate)
# t-stat, p-value and Linear Regression Diagnostics
linearMod <- lm(body_temperature ~ heart_rate, data=MyData)  
summary(linearMod)
modelSummary <- summary(linearMod)  # capture model summary as an object
modelCoeffs <- modelSummary$coefficients  # model coefficients
beta.estimate <- modelCoeffs["heart_rate", "Estimate"]  # get beta estimate for speed
std.error <- modelCoeffs["heart_rate", "Std. Error"]  # get std.error for speed
t_value <- beta.estimate/std.error  # calc t statistic
p_value <- 2*pt(-abs(t_value), df=nrow(MyData)-ncol(MyData))  # calc p Value
f_statistic <- linearMod$fstatistic[1]  # fstatistic
f <- summary(linearMod)$fstatistic  # parameters for model p-value calc
model_p <- pf(f[1], f[2], f[3], lower=FALSE)
# AIC and BIC
AIC(linearMod)
BIC(linearMod)  
#k-fold cross validation
test_results <- data.frame(
  body_temperature = MyData$body_temperature,
  heart_rate = MyData$heart_rate
)
cvResults <- suppressWarnings(CVlm(data = test_results, form.lm=body_temperature ~ heart_rate, m=5, dots=FALSE, seed=29, legend.pos="topleft",  printit=FALSE, main="Body temperature ~ Heart rate"));  # performs the CV
attr(cvResults, 'ms') 