library(png)

path_dir <- "/Users/cristobal/Documents/Tesis/Codigo/Neural_LSTM/Checkpoint/8_time_steps/"
cases <- c("19","55","91")
layers <- c("1_layer","2_layer")
cells <- c("1_units","2_units","4_units","8_units","16_units","24_units","32_units","40_units","48_units")
folds <- c("1_fold","2_fold")

path_view <- paste(path_dir,cases[1],"/",layers[1],"/",cells[1],"/",folds[1],"/heat_images/",sep = "")
kernel_view <- paste("19 _ 1_units _kernel _capa1 _iteration_ 1 .png")
recurrent_kernel_view <- paste("19 _ 1_units _recurrent_kernel _capa1 _iteration_ 1 .png")

img_view <- paste(path_view,kernel_view,sep = "")


png(filename = "19 _ 1_units _recurrent_kernel _capa1 _iteration_ 1 .png", width=10, height=10)
plot(xlim=c(0,255),ylim=c(0,255))

png("19 _ 1_units _recurrent_kernel _capa1 _iteration_ 1 .png", 490, 350)
plot(x, y, pch=19, col=rgb(0.5, 0.5, 0.5, 0.5), cex=1.5)
abline(lm(y ~ x))
dev.off()