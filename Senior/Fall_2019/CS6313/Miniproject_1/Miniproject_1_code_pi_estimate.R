# For draw.circle
library(plotrix)
size <- 10000000
# Function returns a boolean based on if a point is in a circle at (0.5, 0.5) with 0.5 radius
inside_circle <- function(x) {
  return(sqrt( (x[1] - 0.5)^2 + (x[2] - 0.5)^2 ) < 0.5)
}
# Target prediction
p = (pi * 0.5^2)
p
# Uniformally generate points on a plane between 0 and 1
points <- cbind(runif(size), runif(size))
# Check which points are in the circle
points.in.circle <- apply(points, 1, inside_circle)
# Plot points, rectangle and circle
plot(points, asp=1, xlab = "x", ylab="y")
rect(0,0,1,1)
draw.circle(0.5,0.5,0.5, border = "red")
# Get amount of points that were in circle compared to total amount of points
p_pred = length(points.in.circle[points.in.circle == TRUE]) / length(points.in.circle)
# p_pred is approximately the area of the circle. Use it to solve for pi = area / r^2
p_pred / 0.5^2
