#install.packages('earth')
library(earth)
library(dplyr)
set.seed(1)

# Load the data
scenario <- "housing"
n <- 10000
num_trial <- 100


file_name <- paste('data_real/',scenario,'.csv',sep="")
data <- read.csv(file_name)


data <- data %>% mutate(
    avg_occupancy = population / households,
    log_median_house_value = log(median_house_value)
  )



lower_avg_occupancy <- quantile(data$avg_occupancy, 0.025, na.rm = TRUE)
upper_avg_occupancy <- quantile(data$avg_occupancy, 0.975, na.rm = TRUE)
lower_median_income <- quantile(data$median_income, 0.025, na.rm = TRUE)
upper_median_income <- quantile(data$median_income, 0.975, na.rm = TRUE)


data <- data %>% filter(
    avg_occupancy >= lower_avg_occupancy & avg_occupancy <= upper_avg_occupancy,
    median_income >= lower_median_income & median_income <= upper_median_income
  )

data_final <- data %>% select(avg_occupancy, median_income, log_median_house_value)
print(dim(data_final)) # 18662 by 3




formula1 <- as.formula('log_median_house_value ~ avg_occupancy + median_income')
formula2 <- as.formula('res1_squared ~ avg_occupancy + median_income')
formula3 <- as.formula('log_median_house_value^2 ~ avg_occupancy + median_income') # y is squared

Prop_M5_95_holder <- numeric(num_trial)
Prop_M6_95_holder <- numeric(num_trial)
Prop_M5_90_holder <- numeric(num_trial)
Prop_M6_90_holder <- numeric(num_trial)


for(trial_ind in 1:num_trial){
  
  df_shuffled <- data_final[sample(nrow(data_final)), ]
  
  train_set <- df_shuffled[1:n, ]
  test_set <- df_shuffled[(n+1):nrow(df_shuffled), ]
  
  # Method 5
  earth.mod1 <- earth(formula1, data = train_set) 
  prediction1 <- predict(earth.mod1, train_set[,1:2]) 
  res1_squared <- (train_set$log_median_house_value - prediction1)^2
  earth.mod2 <- earth(formula2, data = train_set[,1:2]) # USE res1_squared
  
  # Test set
  test_prediction1 <- predict(earth.mod1, test_set[,1:2]) # col 1 = avg_occupancy, col 2 = median_income
  test_prediction2 <- predict(earth.mod2, test_set[,1:2]) # col 1 = avg_occupancy, col 2 = median_income
  test_prediction2[test_prediction2 <= 0] <- 0
  
  # 95% interval
  lower_bound <- test_prediction1 - 1.96 * sqrt(test_prediction2)
  upper_bound <- test_prediction1 + 1.96 * sqrt(test_prediction2)
  
  is_in_interval <- (test_set[,3] >= lower_bound) & (test_set[,3] <= upper_bound) # col 3 = log_median_house_value
  Prop_M5_95_holder[trial_ind] <- mean(is_in_interval)
  
  # 90% interval
  lower_bound <- test_prediction1 - 1.65 * sqrt(test_prediction2)
  upper_bound <- test_prediction1 + 1.65 * sqrt(test_prediction2)
  
  is_in_interval <- (test_set[,3] >= lower_bound) & (test_set[,3] <= upper_bound) # col 3 = log_median_house_value
  Prop_M5_90_holder[trial_ind] <- mean(is_in_interval)
  
  
  # Method 6
  earth.mod3 <- earth(formula3, data = train_set) # USE squared y
  
  # Test set
  test_prediction3 <- predict(earth.mod3, test_set[,1:2])
  test_g_direct <- test_prediction3 - test_prediction1^2
  test_g_direct[test_g_direct <= 0] <- 0
  
  # 95% interval
  lower_bound <- test_prediction1 - 1.96 * sqrt(test_g_direct)
  upper_bound <- test_prediction1 + 1.96 * sqrt(test_g_direct)
  
  is_in_interval <- (test_set[,3] >= lower_bound) & (test_set[,3] <= upper_bound) # col 3 = log_median_house_value
  Prop_M6_95_holder[trial_ind] <- mean(is_in_interval)
  
  # 90% interval
  lower_bound <- test_prediction1 - 1.65 * sqrt(test_g_direct)
  upper_bound <- test_prediction1 + 1.65 * sqrt(test_g_direct)
  
  is_in_interval <- (test_set[,3] >= lower_bound) & (test_set[,3] <= upper_bound) # col 3 = log_median_house_value
  Prop_M6_90_holder[trial_ind] <- mean(is_in_interval)
}





################
# PRINT & SAVE #
################


mean(Prop_M5_95_holder)
sd(Prop_M5_95_holder)

mean(Prop_M5_90_holder)
sd(Prop_M5_90_holder)

mean(Prop_M6_95_holder)
sd(Prop_M6_95_holder)

mean(Prop_M6_90_holder)
sd(Prop_M6_90_holder)


write.csv(Prop_M5_95_holder, paste(scenario,'_M5_95_n',n,'_holder.csv',sep=""))
write.csv(Prop_M5_90_holder, paste(scenario,'_M5_90_n',n,'_holder.csv',sep=""))
write.csv(Prop_M6_95_holder, paste(scenario,'_M6_95_n',n,'_holder.csv',sep=""))
write.csv(Prop_M6_90_holder, paste(scenario,'_M6_90_n',n,'_holder.csv',sep=""))

