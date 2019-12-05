year = seq(3, 13)
profit = c(0,0,1,0,1,1,1,1,0,1,1)
investment = c(17,23,31,29,33,39,39,40,41,44,47)

y = investment
z = factor(profit)
x = year

# 11.5.a
fit = lm(y ~ x + z)

summary(fit)

# 11.5.b
sum(fit$coefficients * c(1, 1, 15))

# 11.5.c
sum(fit$coefficients * c(1, 0, 15))

# 11.5.d
anova(fit)

