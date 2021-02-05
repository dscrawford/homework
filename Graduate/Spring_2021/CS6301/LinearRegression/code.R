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
