# Create a matrix from list [1,2,3,4,5,6] with 2 rows (entered column-wide)
x1 <- matrix(1:6, nrow=2)
# Same thing, but row wise
x2 <- matrix(1:6, nrow=2, byrow=T)
# Basic functions for matrices
nrow(x)
ncol(x)


## data frame
emp.data <- data.frame(
  emp_id = c(1:6),
  emp_name = c("Rick", "Dan", "Michelee", "Ryan", "Gary", "Dan"),
#  stringsAsFactors = FALSE <-- NO LONGER NEEDED
)


# reading table from web address
survey <- read.table("http://www.andrew.cmu.edu/user/achoulde/94842/data/survey_data.csv",
                     header=TRUE, sep=",")

x <- as.data.frame(survey)
class(x)

u <- as.matrix(survey)
class(u)

head(survey)

summary(survey)

# Structure of survey
str(survey)

p <- survey$Program
str(p)
# distinct(p) # DNE
unique(p)

colnames(mtcars)

# How to extract columns from a data frame
x <- mtcars$mpg
typeof(x)
mtcars["mpg"]
mtcars$mpg
mtcars[,1]

# Create data frame with first column =x, second column = y
x <- c(1:5)
y <- c(6:10)

## vectors -> shorter one repeats if you combine two vectors

df = data.frame(first = x, second = y)
df

## quick tools
summary(mtcars$mpg)
hist(mtcars$mpg)

df1 = data.frame(CustomerId = c(1:6), 
                 Product = c(rep("Oven", 3), rep("Television", 3)))
df2 = data.frame(CustomerId = c(4:9),
                 Product = c(rep("Television", 3), rep("Air conditioner", 3)))

# Column bind to cbind
# Different than join, this is perform the operation column wise.
cbind(df1, df2)
# Also have row bind
rbind(df1, df2)

merge(df1, df2)

USArrests["Assault"]
USArrests[,c("UrbanPop")]

# Exclude a column by name
USArrests[,-which(names(USArrests) == "Murder")]

# exact match
v <- c("Houston", "Dallas", "Austin")

grep("dal*", v, ignore.case = T)

# For loops (slower than apply functions, not parrellelized)
for (i in 1:length(v)) {
  print(v[i])
}

i = 1
while (i <= length(v)) {
  print(v[i])
  i = i + 1
}

# List (list of any type of elements)
l <- list(a = 1:10, b = seq(2,20, 2), c = matrix(1:10, nrow = 5))
# same here
l$a
l[1]

# Access b's 2nd element
l[[2]][2]


v <- 1:10
## square each element of the vector
sapply(v,function(x) x^2)

u <- sapply(v, function(x) x^2)
z <- sapply(v, function(x) x%%2)

# List vector operations
l <- list(a = 1:10, b = seq(2,20, 2))
# lapply takes a list and outputs a list
l1 <- lapply(l, function(x) x^2)
# sapply applies to a list and concatenates into a vector
l2 <- sapply(l, function(x) x^2)

# Apply marigin wise (1 = rowwise, 2= columnwise)
m <- matrix(1:10, nrow = 5)
apply(m, function(x) x^2, MARGIN = 2)

# transpose
t(m)

# Select first 5 rows, first 2 columns
u[1:5, 1:2]

us <- USArrests
# sort the data by murder (order returns order indices!)
us[order(us$Murder),]


# Make your own user-defined apply functions
addOne <- function(x) {
  x + 1
}

addTwoNumbers <- function(x, y) {
  
}
sapply(v, addOne)

l <- list(a = 1:10, b = 11:20, c = c(1))
lapply(l, mean)
lapply(l, var)

complex_list <- list(a = 1:10, b = matrix(1:10, nrow = 5))
is.matrix(complex_list[[2]])
is.vector(complex_list[[1]])

# function that handles vector and matrix depending on their type

complex_function <- function(x, f) {
  if (is.vector(x)) {
    f(x)
  }
  else if (is.matrix(x)) {
    apply(x, f, MARGIN=1)
  }
}

lapply(complex_list, function(x) complex_function(x, mean))


plot(1:10, type="l", col="red", lwd=3)
abline(v=5, lty=2)
plot(1:10, type="p", col="red", lwd=3)
abline(v=5, lty=2)
plot(1:10, type="s", col="red", lwd=3)
abline(v=5, lty=2)
plot(1:10, type="o", col="red", lwd=3)
abline(v=5, lty=2)

# UTD Students have a mean GPA = 3.0 and SD = 0.6
sample <- rnorm(1000000, 3.0, 0.6)
hist(sample, breaks=100)

plot(x=rnorm(500), y=rnorm(500), xlab="x", ylab="y", main="Bi-variate Norm. Distr.")

plot(iris[,-5], col=iris[,5], main="Iris Plot", legend = TRUE)
