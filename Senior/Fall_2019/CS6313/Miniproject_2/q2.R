# Clear all global values before each run and set number of sig figs.
rm(list = ls())
options(digits = 5)

# summarise()
library(dplyr)

# Open motorcycle.csv
accidents = read.csv(file="./motorcycle.csv", header = TRUE, sep = ",")

# Plot and display summary stats
outliers = boxplot(accidents$Fatal.Motorcycle.Accidents,
        horizontal = TRUE,
        xlab = "Fatal Accidents",
        ylab = "Counties",
        main = "Number of Fatal Accidents in Counties")$out

accidents %>%
  summarise(min = min(Fatal.Motorcycle.Accidents),
            max = max(Fatal.Motorcycle.Accidents),
            mean = mean(Fatal.Motorcycle.Accidents),
            median = median(Fatal.Motorcycle.Accidents),
            range = diff(range(Fatal.Motorcycle.Accidents)),
            sd = sd(Fatal.Motorcycle.Accidents),
            FirstQuartile = quantile(Fatal.Motorcycle.Accidents, .25),
            ThirdQuartile = quantile(Fatal.Motorcycle.Accidents, .75))

# Print the counties with min, max number of accidents and outliers
min = which(min(accidents$Fatal.Motorcycle.Accidents) == accidents$Fatal.Motorcycle.Accidents)
print(accidents$County[min])
max = which(max(accidents$Fatal.Motorcycle.Accidents) == accidents$Fatal.Motorcycle.Accidents)
print(accidents$County[max])
outliers.location = which(outliers == accidents$Fatal.Motorcycle.Accidents)
print(accidents$County[outliers.location])
