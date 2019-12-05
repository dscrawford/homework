# Clear all global values before each run, and set seed
rm(list = ls())
library(olsrr)
set.seed(1234)

MyData = read.csv(file="prostate_cancer.csv", header=TRUE, sep=",")
str(MyData)
attach(MyData)
headers = dput(names(MyData))

# Visualize the predictors
hist(psa)
hist(cancervol)
hist(weight)
hist(age)
hist(benpros)
table(vesinv)
hist(capspen)
table(gleason)

# Factor
vesinv = factor(vesinv)
gleason = factor(gleason)

# Testing without transformation
cat("Testing", paste(headers[-2:-1], collapse = ", "), "\n")

fit.all = lm(psa ~ cancervol + weight + age + benpros + factor(vesinv) + capspen + factor(gleason), data=MyData)
test.all = ols_step_all_possible(fit.all)
plot(test.all)

ols_step_best_subset(fit.all)

test.for = ols_step_forward_p(fit.all)

test.back = ols_step_backward_p(fit.all)

test.opt = lm(psa ~ cancervol + age + benpros + vesinv + gleason)
summary(test.opt)

# Transform the dataset
psa.log = log1p(psa)
cancervol.log = log1p(cancervol)
weight.log = log1p(weight)
benpros.log = log1p(benpros)
capspen.log = log1p(capspen)
hist(psa.log)
hist(cancervol.log)
hist(weight.log)
hist(age)
hist(benpros.log)
table(vesinv)
hist(capspen.log)
table(gleason)

# Testing transformed dataset
fit.log.all = lm(psa.log ~ cancervol.log + weight.log + age + benpros.log + vesinv + capspen.log + gleason, data=MyData)
test.log.all = ols_step_all_possible(fit.log.all)
plot(test.log.all)

ols_step_best_subset(fit.log.all)

test.log.opt = lm(psa.log ~ cancervol.log + weight.log + vesinv + gleason)
summary(test.log.opt)