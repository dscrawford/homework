# Clear all global values before each run and set number of sig figs.
rm(list = ls())
options(digits = 10)

# summarise()
library(dplyr)

# Open roadrace.csv
racers = read.csv(file="./roadrace.csv", header = TRUE, sep = ",")

# Split Maine racers and other racers and plot bar graph
racers.MaineSplit = split(racers, racers$Maine)
racers.MaineSplit.occurences = rev(as.vector(table(racers$Maine)))
barplot(
  racers.MaineSplit.occurences,
  names.arg = c("Maine", "Away"),
  xlab = "Origin",
  ylab = "Number of People",
  main = "Racer's Origin")

# Summary of people who are in Maine and Away
summary(racers.MaineSplit.occurences)


# Create histograms for Maine and Away racers and find their 5 point summary
hist(racers.MaineSplit$Maine$Time..minutes.,
     ylim = c(0,1800),
     xlim = c(25,160),
     xlab = "Time in Minutes",
     main = "Maine Racer Time distribution")
hist(racers.MaineSplit$Away$Time..minutes.,
     ylim = c(0,500),
     xlim = c(25,160),
     xlab = "Time in Minutes",
     main = "Away Racer Time distribution")

# Summarize min, max, mean, sd, first and third quartile from each group from Maine attribute by Time in minutes
racers %>% group_by(Maine) %>%
  summarise(min = min(Time..minutes.),
            max = max(Time..minutes.),
            mean = mean(Time..minutes.),
            median = median(Time..minutes.),
            range = diff(range(Time..minutes.)),
            sd = sd(Time..minutes.),
            FirstQuartile = quantile(Time..minutes., .25),
            ThirdQuartile = quantile(Time..minutes., .75))


# Create boxplots for Maine and Away
boxplot(racers.MaineSplit$Maine$Time..minutes.,
        racers.MaineSplit$Away$Time..minutes.,
        main = "Time in minutes of racers",
        horizontal = TRUE,
        xlab = "Minutes",
        names = c('Maine', 'Away'))

# Convert age attribute to integers for easier use.
racers$Age = as.integer(racers$Age)

# Create male and female boxplots for runner's age
racers.SexSplit = split(racers, racers$Sex)
boxplot(racers.SexSplit$M$Age,
        racers.SexSplit$F$Age,
        horizontal = TRUE,
        names = c("Male", "Female"),
        xlab = "Age",
        main = "Racer Age Base on Sex")


# Summarize min, max, mean, sd, first and third quartile from each group of Sex's Age
racers %>% group_by(Sex,) %>%
  summarise(min = min(Age),
            max = max(Age),
            mean = mean(Age),
            median = median(Age),
            range = diff(range(Age)),
            sd = sd(Age),
            FirstQuartile = quantile(Age, .25),
            ThirdQuartile = quantile(Age, .75))
