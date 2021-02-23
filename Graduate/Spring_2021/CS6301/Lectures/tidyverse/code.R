install.packages("tidyverse")
library(tidyverse)
?tibble
tibble(x = 1:5, y = 1, z = x ^ 2 + y)

t <- read_csv(readr_example("mtcars.csv"))

type(t)
class(t)

as_tibble(t)

t[, cyl>2]

t[t$cyl == 4, ]
filter(t, cyl == 4)
t[,"cyl">2]

data(nycflights13)

nycflights13::flights

nrow(flights)

flights[, "days" == 1 & "month" == 1]

flights[(flights[,"month"] == 1 & flights[,"day"]==1)]

arrange(flights, -day)
arrange(flights, -dep_delay)
arrange(flights, dep_delay)