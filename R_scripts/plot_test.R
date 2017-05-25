library(rhdf5)
library(ggplot2)  
library(RColorBrewer)  
library(reshape2)  
rute <- "/Users/cristobal/Documents/Tesis/Codigo/Neural_LSTM/Checkpoint/8_time_steps/19/2_layer/4_units/1_fold/weights/final_weights.hdf5"
data <- h5ls(rute)
dense <- h5read(rute,"/dense_1/dense_1")
lstm <- h5read(rute,"/lstm_1/lstm_1")
lstm2 <- h5read(rute,"/lstm_2/lstm_2")

recurrent_matrix <- lstm$`recurrent_kernel:0`

print(ggplot(lstm$`kernel:0`, aes(x = Var1, y = Var2, fill = value)) + 
        ggtitle(titulo_lstm) + geom_tile() + sc)

values <- c(recurrent_matrix)
## Scale your values to range between 0 and 1
rr <- range(values)
svals <- (values-rr[1])/diff(rr)
# [1] 0.2752527 0.0000000 0.9149839 0.3680242 1.0000000 0.2660587

## Play around with ends of the color range
f <- colorRamp(c("green", "blue","red"))
colors <- rgb(f(svals)/255)

## Check that it works
image(seq_along(svals), 1, as.matrix(seq_along(svals)), col=colors,
      axes=FALSE, xlab="", ylab="")







