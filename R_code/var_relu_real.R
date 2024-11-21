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
    log_median_house_value = log(median_house_value),
    log_population = log(population)
)


data <- data %>% 
  select(avg_occupancy, median_income, log_population, log_median_house_value) %>%
  mutate(
    avg_occupancy = (avg_occupancy - min(avg_occupancy)) / (max(avg_occupancy) - min(avg_occupancy)),
    median_income = (median_income - min(median_income)) / (max(median_income) - min(median_income)),
    log_population = (log_population - min(log_population)) / (max(log_population) - min(log_population))
  )




data_final <- data %>% select(avg_occupancy, median_income, log_population, log_median_house_value)
print(dim(data_final)) 


formula1 <- as.formula('log_median_house_value ~ log_population + avg_occupancy + median_income')
formula2 <- as.formula('res1_squared ~ log_population + avg_occupancy + median_income')
formula3 <- as.formula('log_median_house_value^2 ~ log_population + avg_occupancy + median_income') # y is squared

Prop_M5_95_holder <- numeric(num_trial)
Prop_M6_95_holder <- numeric(num_trial)
Prop_M5_90_holder <- numeric(num_trial)
Prop_M6_90_holder <- numeric(num_trial)



for(trial_ind in 1:num_trial){
  
  df_shuffled <- data_final[sample(nrow(data_final)), ] # shuffling
  
  train_set <- df_shuffled[1:n, ]
  test_set <- df_shuffled[(n+1):nrow(df_shuffled), ]
  
  # Method 5
  earth.mod1 <- earth(formula1, data = train_set) 
  prediction1 <- predict(earth.mod1, train_set[,1:3]) 
  res1_squared <- (train_set$log_median_house_value - prediction1)^2
  earth.mod2 <- earth(formula2, data = train_set[,1:3]) # USE res1_squared
  
  # Empirical quantile
  g_train <- predict(earth.mod2, train_set[,1:3])
  ri <- (train_set$log_median_house_value[g_train > 0] - prediction1[g_train > 0])/sqrt(g_train[g_train > 0])
  
  # Test set
  test_prediction1 <- predict(earth.mod1, test_set[,1:3]) # col 1,2,3
  test_prediction2 <- predict(earth.mod2, test_set[,1:3]) # col 1,2,3
  test_prediction2[test_prediction2 <= 0] <- 0
  
  # 95% interval
  lower_q <- quantile(ri, 0.025) # -1.96
  upper_q <- quantile(ri, 0.975) # 1.96
  
  lower_bound <- test_prediction1 + lower_q * sqrt(test_prediction2)
  upper_bound <- test_prediction1 + upper_q * sqrt(test_prediction2)
  
  is_in_interval <- (test_set[,4] >= lower_bound) & (test_set[,4] <= upper_bound) # col 4 = log_median_house_value
  Prop_M5_95_holder[trial_ind] <- mean(is_in_interval)
  
  
  
  # 90% interval
  lower_q <- quantile(ri, 0.05) # -1.65
  upper_q <- quantile(ri, 0.95) # 1.65
  
  lower_bound <- test_prediction1 + lower_q * sqrt(test_prediction2)
  upper_bound <- test_prediction1 + upper_q * sqrt(test_prediction2)
  
  is_in_interval <- (test_set[,4] >= lower_bound) & (test_set[,4] <= upper_bound) # col 4 = log_median_house_value
  Prop_M5_90_holder[trial_ind] <- mean(is_in_interval)
  
  
  # Method 6
  earth.mod3 <- earth(formula3, data = train_set) # USE squared y
  
  
  # Empirical quantile
  h_train <- predict(earth.mod3, train_set[,1:3])
  g_train <- h_train - prediction1^2
  ri <- (train_set$log_median_house_value[g_train > 0] - prediction1[g_train > 0])/sqrt(g_train[g_train > 0])
  
  
  
  # Test set
  test_prediction3 <- predict(earth.mod3, test_set[,1:3])
  test_g_direct <- test_prediction3 - test_prediction1^2
  test_g_direct[test_g_direct <= 0] <- 0
  
  # 95% interval
  lower_q <- quantile(ri, 0.025) # -1.96
  upper_q <- quantile(ri, 0.975) # 1.96
  lower_bound <- test_prediction1 + lower_q * sqrt(test_g_direct)
  upper_bound <- test_prediction1 + upper_q * sqrt(test_g_direct)
  
  is_in_interval <- (test_set[,4] >= lower_bound) & (test_set[,4] <= upper_bound) # col 4 = log_median_house_value
  Prop_M6_95_holder[trial_ind] <- mean(is_in_interval)
  
  # 90% interval
  lower_q <- quantile(ri, 0.05) # -1.65
  upper_q <- quantile(ri, 0.95) # 1.65
  lower_bound <- test_prediction1 + lower_q * sqrt(test_g_direct)
  upper_bound <- test_prediction1 + upper_q * sqrt(test_g_direct)
  
  is_in_interval <- (test_set[,4] >= lower_bound) & (test_set[,4] <= upper_bound) # col 4 = log_median_house_value
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

