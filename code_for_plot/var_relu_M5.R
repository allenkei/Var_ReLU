#install.packages('earth')
library(earth)

# Load the data
scenario <- "s3"
n <- 6000

file_name <- paste('data/',scenario,'_data_n',n,'.csv',sep="")
combined_data <- read.csv(file_name)
colnames(combined_data) # trial,f_value,g_value,y_value,x1,...,xp   
last_ind <- dim(combined_data)[2]



formula1 <- as.formula('y_value ~ x1+x2')
formula2 <- as.formula('res1_squared ~ x1+x2')


trial_data <- subset(combined_data, trial == 1) # use the first one
rm(combined_data)

# Method 5
earth.mod1 <- earth(formula1, data = trial_data) 
prediction1 <- predict(earth.mod1, trial_data[,5:last_ind]) # col5 is x1
res1_squared <- (trial_data$y_value - prediction1)^2

earth.mod2 <- earth(formula2, data = trial_data[,5:last_ind]) # USE res1_squared
prediction2 <- predict(earth.mod2, trial_data[,5:last_ind]) # col5 is x1


write.csv(prediction2, paste(scenario,'_M5_ghat.csv',sep=""))

  


