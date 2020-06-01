
# Hannes Windisch
# HarvardX: PH125.9x Data Science: Capstone
# 1.6.2020

###################################
# Project MovieLens 
# Prediction of Movie Rating System
###################################

# This R Script calculates the RMSE for movie ratings.
# All other analysis can be found in th Rmd and the pdf file.

# This is the RMSE function used for evaluating the quality of the method
RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
    }

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###########################
# Exploratory data analysis
###########################

head(edx) %>%
  print.data.frame()

edx %>%
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))

edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25) +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating distribution")

####################
# Model 0 - Raw mean
####################

mu <- mean(edx$rating)
mu

# Calculate RMSE on validation set
rmse_mu <- RMSE(validation$rating, mu)
rmse_mu

# Save results in tibble
rmse_results <- tibble(method = "raw_mean", RMSE = rmse_mu)

#########################
# Model 1 - Movie effects
#########################

m1 <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Calculate bi
pred_1 <- mu + validation %>%
  left_join(m1, by = 'movieId') %>%
  .$b_i

# Calculate RMSE on validation set
rmse_1 <- RMSE(pred_1, validation$rating)
rmse_1

# Save results in tibble
rmse_results <- bind_rows(rmse_results, tibble(method = "movie effects", RMSE = rmse_1))

########################
# Movie and User effects
########################

# Calculate bu
m2 <- edx %>%
  left_join(m1, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

pred_2 <- validation %>%
  left_join(m1, by = 'movieId') %>%
  left_join(m2, by = 'userId') %>%
  mutate(pred_2 = mu + b_i + b_u) %>%
  .$pred_2

# Calculate RMSE on validation set
rmse_2 <- RMSE(pred_2, validation$rating)
rmse_2

# Save results in tibble
rmse_results <- bind_rows(rmse_results, tibble(method = "movie & user effects", RMSE = rmse_2))

#########################################
# Regularized movie and user effect model
#########################################

# Evaluate optimal lambda on training set
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l) {
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  pred <- edx %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(pred, edx$rating))
})

# get lambda of min. RMSE
lambdas[which.min(rmses)]

# Calculate regularized parameters bi und bu for optimal RMSE using training set
l <- 0.5
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))
b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

# Calculate prediction on validation set
pred <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate RMSE on validation set
rmse_3 <- RMSE(pred, validation$rating)
rmse_3

# Save results in tibble
rmse_results <- bind_rows(rmse_results, tibble(method = "Regularized movie and user effect model", RMSE = rmse_3))

# RMSE overview per method
rmse_results %>% knitr::kable()
