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
  
  # Method 5
  earth.mod1 <- earth(formula1, data = trial_data) 
  prediction1 <- predict(earth.mod1, trial_data[,5:last_ind]) # col5 is x1
  res1_squared <- (trial_data$y_value - prediction1)^2
  
  earth.mod2 <- earth(formula2, data = trial_data[,5:last_ind]) # USE res1_squared
  prediction2 <- predict(earth.mod2, trial_data[,5:last_ind]) # col5 is x1
  MSE_g_M5_holder[trial_ind] <- mean( (prediction2 - trial_data$g_value)^2 )
  
  # Method 6
  earth.mod3 <- earth(formula3, data = trial_data) # USE y_squared
  prediction3 <- predict(earth.mod3, trial_data[,5:last_ind]) # col5 is x1
  
  g_direct <- prediction3 - prediction1^2
  MSE_g_M6_holder[trial_ind] <- mean( (g_direct - trial_data$g_value)^2 )
  
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


write.csv(MSE_g_M5_holder, paste(scenario,'_M5_n',n,'_holder.csv',sep=""))
write.csv(MSE_g_M6_holder, paste(scenario,'_M6_n',n,'_holder.csv',sep=""))

