library(ggplot2)
library(patchwork)
library(cowplot)

s1_data <- read.csv('data/s1_data_n6000.csv')
s2_data <- read.csv('data/s2_data_n6000.csv')
s3_data <- read.csv('data/s3_data_n6000.csv')
s1_data <- s1_data[1:6000,]
s2_data <- s2_data[1:6000,]
s3_data <- s3_data[1:6000,]


s1_M1_prediction1 <- c(read.csv("result_plot/s1_M1_ghat.csv")[,1])
s1_M3_prediction1 <- c(read.csv("result_plot/s1_M3_ghat.csv")[,1])
s1_M5_prediction1 <- c(read.csv("result_plot/s1_M5_ghat.csv")[,2]) # col 1 is index
s1_data$M1 <- s1_M1_prediction1; rm(s1_M1_prediction1)
s1_data$M3 <- s1_M3_prediction1; rm(s1_M3_prediction1)
s1_data$M5 <- s1_M5_prediction1; rm(s1_M5_prediction1)

s2_M1_prediction1 <- c(read.csv("result_plot/s2_M1_ghat.csv")[,1])
s2_M3_prediction1 <- c(read.csv("result_plot/s2_M3_ghat.csv")[,1])
s2_M5_prediction1 <- c(read.csv("result_plot/s2_M5_ghat.csv")[,2]) # col 1 is index
s2_data$M1 <- s2_M1_prediction1; rm(s2_M1_prediction1)
s2_data$M3 <- s2_M3_prediction1; rm(s2_M3_prediction1)
s2_data$M5 <- s2_M5_prediction1; rm(s2_M5_prediction1)

s3_M1_prediction1 <- c(read.csv("result_plot/s3_M1_ghat.csv")[,1])
s3_M3_prediction1 <- c(read.csv("result_plot/s3_M3_ghat.csv")[,1])
s3_M5_prediction1 <- c(read.csv("result_plot/s3_M5_ghat.csv")[,2]) # col 1 is index
s3_data$M1 <- s3_M1_prediction1; rm(s3_M1_prediction1)
s3_data$M3 <- s3_M3_prediction1; rm(s3_M3_prediction1)
s3_data$M5 <- s3_M5_prediction1; rm(s3_M5_prediction1)



##############
# Scenario 1 #
##############

#s1_global_min <- min(s1_data$g_value, s1_data$M1, s1_data$M3, s1_data$M5)
#s1_global_max <- max(s1_data$g_value, s1_data$M1, s1_data$M3, s1_data$M5)

#color_scale1 <- scale_color_gradientn(
#  colors = c( 'deepskyblue2', 'white', 'darkorange2'),
#  limits = c(s1_global_min, s1_global_max),
#  name = "y"
#)


s1_data <- s1_data %>%
  mutate(
    g_value_modified = ifelse(g_value > 1, 1.01, g_value),
    M1_modified = ifelse(M1 > 1, 1.01, M1),
    M3_modified = ifelse(M3 > 1, 1.01, M3),
    M5_modified = ifelse(M5 > 1, 1.01, M5)
  )




custom_color_scale1 <- scale_color_gradientn(
  colors = c('deepskyblue1', 'deepskyblue', 'white', 'darkorange', 'darkorange1','lightgrey'),
  limits = c(0,1.01),
  values = c(0, 0.25, 0.5, 0.75, 1), 
  breaks = c(0, 0.25, 0.5, 0.75, 1),
  name = ""
)


p11 <- ggplot(s1_data, aes(x = x1, y = x2, color = g_value_modified)) +
  geom_point() +
  labs(x = "x1", y = "x2") +
  theme_minimal() +
  custom_color_scale1 + # color_scale1 +
  ggtitle("g(x)") +
  theme(legend.position = "left")

p12 <- ggplot(s1_data, aes(x = x1, y = x2, color = M1_modified)) +
  geom_point() +
  labs(x = "x1", y = "x2") +
  theme_minimal() +
  custom_color_scale1 + # color_scale1 +
  ggtitle("ReLU NN") +
  theme(legend.position = "none") 

p13 <- ggplot(s1_data, aes(x = x1, y = x2, color = M3_modified)) +
  geom_point() +
  labs(x = "x1", y = "x2") +
  theme_minimal() +
  custom_color_scale1 + # color_scale1 +
  ggtitle("RF") +
  theme(legend.position = "none") 

p14 <- ggplot(s1_data, aes(x = x1, y = x2, color = M5_modified)) +
  geom_point() +
  labs(x = "x1", y = "x2") +
  theme_minimal() +
  custom_color_scale1 + # color_scale1 +
  ggtitle("MARS") +
  theme(legend.position = "none") 


##############
# Scenario 2 #
##############
#s2_global_min <- min(s2_data$g_value, s2_data$M1, s2_data$M3, s2_data$M5)
#s2_global_max <- max(s2_data$g_value, s2_data$M1, s2_data$M3, s2_data$M5)


#color_scale2 <- scale_color_gradientn(
#  colors = c( 'deepskyblue2', 'deepskyblue1', 'deepskyblue', 'white', 'darkorange', 'darkorange1', 'darkorange2'),
#  limits = c(s2_global_min, s2_global_max),
#  name = "y"
#)


s2_data <- s2_data %>%
  mutate(
    g_value_modified = ifelse(g_value > 1, 1.01, g_value),
    M1_modified = ifelse(M1 > 1, 1.01, M1),
    M3_modified = ifelse(M3 > 1, 1.01, M3),
    M5_modified = ifelse(M5 > 1, 1.01, M5)
  )




custom_color_scale2 <- scale_color_gradientn(
  colors = c('deepskyblue1', 'deepskyblue', 'white', 'darkorange', 'darkorange1','lightgrey'),
  limits = c(0,1.01),
  values = c(0, 0.25, 0.5, 0.75, 1), 
  breaks = c(0, 0.25, 0.5, 0.75, 1),
  name = ""
)


p21 <- ggplot(s2_data, aes(x = x1, y = x2, color = g_value_modified)) +
  geom_point() +
  labs(x = "x1", y = "x2") +
  theme_minimal() +
  custom_color_scale2 + # color_scale2 +
  ggtitle("g(x)") +
  theme(legend.position = "left")

p22 <- ggplot(s2_data, aes(x = x1, y = x2, color = M1_modified)) +
  geom_point() +
  labs(x = "x1", y = "x2") +
  theme_minimal() +
  custom_color_scale2 + # color_scale2 +
  ggtitle("ReLU NN") +
  theme(legend.position = "none")

p23 <- ggplot(s2_data, aes(x = x1, y = x2, color = M3_modified)) +
  geom_point() +
  labs(x = "x1", y = "x2") +
  theme_minimal() +
  custom_color_scale2 + # color_scale2 +
  ggtitle("RF") +
  theme(legend.position = "none")

p24 <- ggplot(s2_data, aes(x = x1, y = x2, color = M5_modified)) +
  geom_point() +
  labs(x = "x1", y = "x2") +
  theme_minimal() +
  custom_color_scale2 + # color_scale2 +
  ggtitle("MARS") +
  theme(legend.position = "none")



##############
# Scenario 3 #
##############
#s3_global_min <- min(s3_data$g_value, s3_data$M1, s3_data$M3, s3_data$M5)
#s3_global_max <- max(s3_data$g_value, s3_data$M1, s3_data$M3, s3_data$M5)


#color_scale3 <- scale_color_gradientn(
#  colors = c( 'deepskyblue2', 'deepskyblue1', 'deepskyblue', 'white', 'darkorange', 'darkorange1', 'darkorange2'),
#  limits = c(s3_global_min, s3_global_max),
#  name = "y"
#)




s3_data <- s3_data %>%
  mutate(
    g_value_modified = ifelse(g_value > 1, 1.01, g_value),
    M1_modified = ifelse(M1 > 1, 1.01, M1),
    M3_modified = ifelse(M3 > 1, 1.01, M3),
    M5_modified = ifelse(M5 > 1, 1.01, M5)
  )




custom_color_scale3 <- scale_color_gradientn(
  colors = c('deepskyblue1', 'deepskyblue', 'white', 'darkorange', 'darkorange1','lightgrey'),
  limits = c(0, 1.01),
  values = c(0, 0.25, 0.5, 0.75, 1), 
  breaks = c(0, 0.25, 0.5, 0.75, 1),
  name = ""
)


p31 <- ggplot(s3_data, aes(x = x1, y = x2, color = g_value_modified)) +
  geom_point() +
  labs(x = "x1", y = "x2") +
  theme_minimal() +
  custom_color_scale3 + # color_scale3 +
  ggtitle("g(x)") +
  theme(legend.position = "left")

p32 <- ggplot(s3_data, aes(x = x1, y = x2, color = M1_modified)) +
  geom_point() +
  labs(x = "x1", y = "x2") +
  theme_minimal() +
  custom_color_scale3 + # color_scale3 +
  ggtitle("ReLU NN") +
  theme(legend.position = "none")

p33 <- ggplot(s3_data, aes(x = x1, y = x2, color = M3_modified)) +
  geom_point() +
  labs(x = "x1", y = "x2") +
  theme_minimal() +
  custom_color_scale3 + # color_scale3 +
  ggtitle("RF") +
  theme(legend.position = "none")

p34 <- ggplot(s3_data, aes(x = x1, y = x2, color = M5_modified)) +
  geom_point() +
  labs(x = "x1", y = "x2") +
  theme_minimal() +
  custom_color_scale3 + # color_scale3 +
  ggtitle("MARS") +
  theme(legend.position = "none")


combined_plot <- (p11 | p12 | p13 | p14) /
  (p21 | p22 | p23 | p24) /
  (p31 | p32 | p33 | p34) 

print(combined_plot)







