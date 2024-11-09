#install.packages('earth')
library(earth)

# Load the data
scenario <- "s1"
n <- 2000


file_name <- paste('data/',scenario,'_data_n',n,'.csv',sep="")
combined_data <- read.csv(file_name)
colnames(combined_data) # trial,f_value,g_value,y_value,x1,...,xp   
last_ind <- dim(combined_data)[2]
num_trials <- 100
half_n <- n/2

if(scenario %in% c('s1','s2','s3')){
  formula1 <- as.formula('y_value ~ x1+x2')
  formula2 <- as.formula('res1_squared ~ x1+x2')
  formula3 <- as.formula('y_value^2 ~ x1+x2')
}else{
  formula1 <- as.formula('y_value ~ x1+x2+x3+x4+x5')
  formula2 <- as.formula('res1_squared ~ x1+x2+x3+x4+x5')
  formula3 <- as.formula('y_value^2 ~ x1+x2+x3+x4+x5')
}

MSE_g_M5_holder <- numeric(num_trials)
MSE_g_M6_holder <- numeric(num_trials)

for(trial_ind in 1:num_trials){
  
  trial_data <- subset(combined_data, trial == trial_ind)
  trial_data_first <- trial_data[1:half_n,] # FIRST HALF
  trial_data_second <- trial_data[(half_n+1):n,] # SECOND HALF
  rm(trial_data)
  
  # Method 5
  earth.mod1 <- earth(formula1, data = trial_data_first) # USE FIRST
  prediction1 <- predict(earth.mod1, trial_data_second[,5:last_ind]) # col5 is x1 # USE SECOND
  res1_squared <- (trial_data_second$y_value - prediction1)^2 # USE SECOND
  
  earth.mod2 <- earth(formula2, data = trial_data_second[,5:last_ind]) # USE res1_squared # USE SECOND
  prediction2 <- predict(earth.mod2, trial_data_second[,5:last_ind]) # col5 is x1 # USE SECOND
  MSE_g_M5_holder[trial_ind] <- mean( (prediction2 - trial_data_second$g_value)^2 ) # USE SECOND
  
  # Method 6
  earth.mod3 <- earth(formula3, data = trial_data_second) # USE y_squared # USE SECOND
  prediction3 <- predict(earth.mod3, trial_data_second[,5:last_ind]) # col5 is x1 # USE SECOND
  
  g_direct <- prediction3 - prediction1^2
  MSE_g_M6_holder[trial_ind] <- mean( (g_direct - trial_data_second$g_value)^2 ) # USE SECOND
  
}



################
# PRINT & SAVE #
################

print(scenario)
print(file_name)

mean(MSE_g_M5_holder)
sd(MSE_g_M5_holder)

mean(MSE_g_M6_holder)
sd(MSE_g_M6_holder)


write.csv(MSE_g_M5_holder, paste('SPLIT_',scenario,'_M5_n',n,'_holder.csv',sep=""))
write.csv(MSE_g_M6_holder, paste('SPLIT_',scenario,'_M6_n',n,'_holder.csv',sep=""))

