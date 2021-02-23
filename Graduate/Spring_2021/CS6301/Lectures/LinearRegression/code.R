require(ISLR)
require(MASS)
data()
View(Boston)
Boston
nrow(Boston)
ncol(Boston)
?Boston
cor(Boston$crim, Boston$medv)
cor(Boston$dis, Boston$medv)
require(corrplot)
install.packages(c('corrplot'))
require(corrplot)
M <- cor(Boston)
M
corrplot(M, method = "circle")
corrplot(M, method = "number")
?Boston
row.names(corMat)
corMat <- as.data.frame(corrplot(M, method="number"))
row.names(corMat)[abs(corMat$medv) > 0.50]


lm.fit = lm(medv~lstat, data=Boston)
summary(lm.fit)

# t value for intercept
34.55384 / 0.56263

# t value for lstat
-0.95005 / 0.03873

# Degrees of freedom
nrow(Boston) - (2)

# Model which is not much better than previous model, given tax variable
lm.fit = lm(medv~lstat + tax, data=Boston)
summary(lm.fit)

lm.fit = lm(medv~lstat + tax + crim, data=Boston)
summary(lm.fit)

lm.fit.1=lm(medv~lstat + tax, data=Boston)
summary(lm.fit.1)

lm.fit.total=lm(medv~.-age, data=Boston)
summary(lm.fit.total)

plot(Boston$lstat, Boston$medv)
abline(lm.fit.total, lwd=3, col="red")


lm.fit=lm(medv~lstat,data=Boston)
lm.fit2=lm(medv~rm+I(lstat^2), data=Boston)

lm.fit5=lm(medv~poly(lstat,5), data=Boston)

summary(lm.fit5)
