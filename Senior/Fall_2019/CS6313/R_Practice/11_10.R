year = seq(1790, 2010, by = 10)
pop  = c(3.9,5.3,7.2,9.6,12.9,17.1,23.2,31.4,38.6,50.2,63.0,76.2,
         92.2,106.0,123.2,132.2,151.3,179.3,203.3,226.5,248.7,281.4,
         308.7)
x = year - 1800
y = pop

fit = lm(y ~ x)

s = summary(fit)

s$r.squared

anova(fit)

sum(fit$coefficients * c(1, 2015 - 1800))

sum(fit$coefficients * c(1, 2020 - 1800))


# 11.11

z = (year - 1800)^2

fit2 = lm(y ~ x + z)

sum(fit2$coefficients * c(1, 2015 - 1800, (2015 - 1800)^2))

sum(fit2$coefficients * c(1, 2020 - 1800, (2020 - 1800)^2))

summary(fit2)$r.squared

anova(fit2)
