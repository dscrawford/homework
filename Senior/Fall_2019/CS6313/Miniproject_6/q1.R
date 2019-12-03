rm(list = ls())

pCancer = read.csv('prostate_cancer.csv')
attach(pCancer)

# Below graphs show that psa can be made linear with a log transformation.
plot(psa)
plot(log(psa))


y = log(psa)

# No linear relationship here
plot(cancervol, y)
# Log transformation more linear
plot(log(cancervol), y)
fit1 = lm(y ~ log(cancervol))
abline(fit1)

# Same here, log transformation improves linearity
plot(weight, y, col = "blue", )
plot(log(weight), y, col = "red")
fit2 = lm(y ~ log(weight))
abline(fit2)

# Age by itself indicates higher log psa levels
plot(age, y)
fit3 = lm(y ~ age)
abline(fit3)

# No linear relationship here, line is nearly flat but abundance of 0 results may contribute to that.
plot(benpros, y)
fit4 = lm(y ~ benpros)
abline(fit4)

# vesinv is a qualatative variable, but it is binary, so it can be used as it is.
plot(vesinv, y)
fit5 = lm(y ~ vesinv)
abline(fit5)
# The binary classification shows a linear relationship at first glance.

# Need to generate dummy variables for each category in gleason
dummies = lapply(unique(gleason), function(x) as.numeric(gleason == x))
gleason6 = unlist(dummies[1])
gleason7 = unlist(dummies[2])
gleason8 = unlist(dummies[3])

# Category 6 has a negative linear correlation it seems, it is relatively flat though.
plot(gleason6, y)
fit6 = lm(y ~ gleason6)
abline(fit6)


# Category 7 does not seem to correlatve linearly at all.
plot(gleason7, y)
fit7 = lm(y ~ gleason7)
abline(fit7)


# Category 8 seems to have a much stronger correlation than other categories.
plot(gleason8, y)
fit8 = lm(y ~ gleason8)
abline(fit8)

# capspen  does not seem to be linear, but may still be a useful feature.
plot(capspen, y)
fit9 = lm(y ~ capspen)
abline(fit9)

ffit1 = lm(y ~ log(cancervol) + log(weight) + age + vesinv + gleason6 + gleason8)
summary(ffit1)

# According to our summary, our parameters age, gleason6 and capspen don't seem to give
# signficant signficant results. We will also train a model without these parameters.

ffit2 = lm(y ~ log(cancervol) + log(weight) + vesinv + gleason8)
summary(ffit2)

# Despite the values not providing signifcant results, it seems to have worsened the performance of our model
anova(ffit1, ffit2)

# It is possible that capspen is not a useful feature, so we will train another line to fit without it.
ffit3 = lm(y ~ log(cancervol) + log(weight) + age + vesinv + gleason6 + gleason8)
summary(ffit3)

anova(ffit1, ffit3)
