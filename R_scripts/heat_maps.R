library(rhdf5)
library(ggplot2)  
library(RColorBrewer)  
library(reshape2)  

#Examples para los paths
#path_weights <- "/Users/cristobal/Documents/Tesis/Codigo/Neural_LSTM/Checkpoint/19/1_layer/1_units/1_fold/weights/"
#path_save <- "/Users/cristobal/Documents/Tesis/Codigo/Neural_LSTM/Checkpoint/19/1_layer/1_units/1_fold/heat_images/"

check_max_min <- function(min_value,max_value){
  if(max_value > min_value*-1){
    return(max_value)
  }
  else{
    return(min_value*-1)
  }
}

check_range_1layer <- function(path_weights,
                        path_save,
                        case,
                        units){
  min_kernel <- c()
  max_kernel <- c()
  min_recurrent_kernel <- c()
  max_recurrent_kernel <- c()
  min_dense <- c()
  max_dense <- c()
  
  file_list <- list.files(path_weights)
  for (file in file_list){
    rute <- paste(path_weights,file,sep = "")
    data <- h5ls(rute)
    dense <- h5read(rute,"/dense_1/dense_1")
    lstm <- h5read(rute,"/lstm_1/lstm_1")
    
    #Ranges kernels
    rr_kernel <- range(lstm$`kernel:0`)
    
    min_kernel <- c(min_kernel, rr_kernel[1])
    max_kernel <- c(max_kernel, rr_kernel[2])
    
    #Ranges recurrents_kernels
    rr_recurrent_kernel <- range(lstm$`recurrent_kernel:0`)
    
    min_recurrent_kernel <- c(min_recurrent_kernel, rr_recurrent_kernel[1])
    max_recurrent_kernel <- c(max_recurrent_kernel, rr_recurrent_kernel[2])
    
    #Dense
    rr_dense <- range(dense$`kernel:0`)
    
    min_dense <- c(min_dense, rr_dense[1])
    max_dense <- c(max_dense, rr_dense[2])
    
  }
  
  simetric_range_kernel <-  check_max_min(min(min_kernel),max(max_kernel))
  simetric_range_recurrent_kernel <-  check_max_min(min(min_recurrent_kernel),max(max_recurrent_kernel))
  simetric_range_dense <-  check_max_min(min(min_dense),max(max_dense))
  
  ans <- list()
  ans[["min_kernel"]] <- simetric_range_kernel*-1
  ans[["max_kernel"]] <- simetric_range_kernel
  ans[["min_recurrent_kernel"]] <- simetric_range_recurrent_kernel*-1
  ans[["max_recurrent_kernel"]] <- simetric_range_recurrent_kernel
  ans[["min_dense"]] <- simetric_range_dense*-1
  ans[["max_dense"]] <- simetric_range_dense
  ans
}

check_range_2layer <- function(path_weights,
                       path_save,
                       case,
                       units){
  min_kernel <- c()
  max_kernel <- c()
  min_kernel2 <- c()
  max_kernel2 <- c()
  min_recurrent_kernel <- c()
  max_recurrent_kernel <- c()
  min_recurrent_kernel2 <- c()
  max_recurrent_kernel2 <- c()
  min_dense <- c()
  max_dense <- c()
  
  file_list <- list.files(path_weights)
  for (file in file_list){
    rute <- paste(path_weights,file,sep = "")
    data <- h5ls(rute)
    dense <- h5read(rute,"/dense_1/dense_1")
    lstm <- h5read(rute,"/lstm_1/lstm_1")
    lstm2 <- h5read(rute,"/lstm_2/lstm_2")
    
    #Ranges kernels
    rr_kernel <- range(lstm$`kernel:0`)
    rr_kernel2 <- range(lstm2$`kernel:0`)
    
    min_kernel <- c(min_kernel, rr_kernel[1])
    max_kernel <- c(max_kernel, rr_kernel[2])
    
    min_kernel2 <- c(min_kernel2, rr_kernel2[1])
    max_kernel2 <- c(max_kernel2, rr_kernel2[2])
    
    #Ranges recurrents_kernels
    rr_recurrent_kernel <- range(lstm$`recurrent_kernel:0`)
    rr_recurrent_kernel2 <- range(lstm2$`recurrent_kernel:0`)
    
    min_recurrent_kernel <- c(min_recurrent_kernel, rr_recurrent_kernel[1])
    max_recurrent_kernel <- c(max_recurrent_kernel, rr_recurrent_kernel[2])
    
    min_recurrent_kernel2 <- c(min_recurrent_kernel2, rr_recurrent_kernel2[1])
    max_recurrent_kernel2 <- c(max_recurrent_kernel2, rr_recurrent_kernel2[2])
    
    #Dense
    rr_dense <- range(dense$`kernel:0`)
    
    min_dense <- c(min_dense, rr_dense[1])
    max_dense <- c(max_dense, rr_dense[2])
    
  }
  
  simetric_range_kernel <-  check_max_min(min(min_kernel),max(max_kernel))
  simetric_range_recurrent_kernel <-  check_max_min(min(min_recurrent_kernel),max(max_recurrent_kernel))
  simetric_range_kernel2 <-  check_max_min(min(min_kernel2),max(max_kernel2))
  simetric_range_recurrent_kernel2 <-  check_max_min(min(min_recurrent_kernel2),max(max_recurrent_kernel2))
  simetric_range_dense <-  check_max_min(min(min_dense),max(max_dense))
  
  ans <- list()
  ans[["min_kernel"]] <- simetric_range_kernel*-1
  ans[["max_kernel"]] <- simetric_range_kernel
  ans[["min_kernel2"]] <- simetric_range_kernel2*-1
  ans[["max_kernel2"]] <- simetric_range_kernel2
  ans[["min_recurrent_kernel"]] <- simetric_range_recurrent_kernel*-1
  ans[["max_recurrent_kernel"]] <- simetric_range_recurrent_kernel
  ans[["min_recurrent_kernel2"]] <- simetric_range_recurrent_kernel2*-1
  ans[["max_recurrent_kernel2"]] <- simetric_range_recurrent_kernel2
  ans[["min_dense"]] <- simetric_range_dense*-1
  ans[["max_dense"]] <- simetric_range_dense
  ans
}



##Funcion para guardar los heat maps de las matrices del modelo cuando tiene 2 capas
save_heat_maps_1layer <- function(path_weights,
                           path_save,
                           case,
                           units,
                           ranges){
  file_list <- list.files(path_weights)
  period = 1
  i = period
  
  for (file in file_list){
    rute <- paste(path_weights,file,sep = "")
    data <- h5ls(rute)
    dense <- h5read(rute,"/dense_1/dense_1")
    lstm <- h5read(rute,"/lstm_1/lstm_1")
    
    dense_melted <- melt(dense$`kernel:0`,varnames = c("Output","Units"), value.name = "weights")
    lstm_kernel_melted <- melt(lstm$`kernel:0`,varnames = c("Gates","Units"), value.name = "weights")
    lstm_recurrent_kernel_melted <- melt(lstm$`recurrent_kernel:0`,varnames = c("Gates","Units"), value.name = "weights")
    
    hm.palette <- colorRampPalette(rev(brewer.pal(11, 'Spectral')), space='Lab')
    
    #Titulos
    titulo_lstm <- paste(case,"_",units,"_kernel","_layer1","_iteration_",i,sep="")
    titulo_lstm_recurrent <- paste(case,"_",units,"_recurrent_kernel","_layer1","_iteration_",i,sep="")
    titulo_dense <- paste(case,"_",units,"_dense","_iteration_",i, sep="")
    
    #Files
    file_lstm = paste(titulo_lstm,'.png',sep="")
    file_lstm_recurrent = paste(titulo_lstm_recurrent,'.png',sep="")
    file_dense = paste(titulo_dense,'.png',sep="")
    
    #Primera capa
    (ggplot(lstm_kernel_melted, aes(x = Gates, y = Units, fill = weights)) + ggtitle(titulo_lstm) + scale_fill_gradientn(limits = c(ranges$min_kernel,ranges$max_kernel), colours = c("black", "white", "green"),breaks = c(ranges$min_kernel,0,ranges$max_kernel),labels=format(c(ranges$min_kernel,0,ranges$max_kernel))) + geom_tile())
    ggsave(file = file_lstm,path=path_save)
    (ggplot(lstm_recurrent_kernel_melted, aes(x = Gates, y = Units, fill = weights)) + ggtitle(titulo_lstm_recurrent) + scale_fill_gradientn(limits = c(ranges$min_recurrent_kernel,ranges$max_recurrent_kernel), colours = c("black", "white", "green"),breaks = c(ranges$min_recurrent_kernel,0,ranges$max_recurrent_kernel),labels=format(c(ranges$min_recurrent_kernel,0,ranges$max_recurrent_kernel))) + geom_tile())
    ggsave(file = file_lstm_recurrent,path=path_save)
    (ggplot(dense_melted, aes(x = Output, y = Units, fill = weights)) + ggtitle(titulo_dense)+ scale_fill_gradientn(limits = c(ranges$min_dense,ranges$max_dense), colours = c("black", "white", "green"),breaks = c(ranges$min_dense,0,ranges$max_dense),labels=format(c(ranges$min_dense,0,ranges$max_dense))) + geom_tile())
    ggsave(file = file_dense,path=path_save)
    
    i=i+period
    
  }
}

##Funcion para guardar los heat maps de las matrices del modelo cuando tiene 2 capas
save_heat_maps_2layer <- function(path_weights,
                                  path_save,
                                  case,
                                  units,
                                  ranges){
  file_list <- list.files(path_weights)
  period = 1
  i = period
  
  for (file in file_list){
    rute <- paste(path_weights,file,sep = "")
    data <- h5ls(rute)
    dense <- h5read(rute,"/dense_1/dense_1")
    lstm <- h5read(rute,"/lstm_1/lstm_1")
    lstm2 <- h5read(rute,"/lstm_2/lstm_2")
    
    dense_melted <- melt(dense$`kernel:0`,varnames = c("Output","Units"), value.name = "weights")
    lstm_kernel_melted <- melt(lstm$`kernel:0`,varnames = c("Gates","Units"), value.name = "weights")
    lstm_recurrent_kernel_melted <- melt(lstm$`recurrent_kernel:0`,varnames = c("Gates","Units"), value.name = "weights")
    
    lstm_kernel_melted_2 <- melt(lstm2$`kernel:0`,varnames = c("Gates","Units"), value.name = "weights")
    lstm_recurrent_kernel_melted_2 <- melt(lstm2$`recurrent_kernel:0`,varnames = c("Gates","Units"), value.name = "weights")
    
    hm.palette <- colorRampPalette(rev(brewer.pal(11, 'Spectral')), space='Lab')
    
    #Primera capa titulos
    titulo_lstm <- paste(case,"_",units,"_kernel","_layer1","_iteration_",i,space="")
    titulo_lstm_recurrent <- paste(case,"_",units,"_recurrent_kernel","_layer1","_iteration_",i,space="")
    
    file_lstm = paste(titulo_lstm,'.png',sep="")
    file_lstm_recurrent = paste(titulo_lstm_recurrent,'.png',sep="")
    
    #Segunda capa titulos
    titulo_lstm_2 <- paste(case,"_",units,"_kernel","_layer2","_iteration_",i,space="")
    titulo_lstm_recurrent_2 <- paste(case,"_",units,"_recurrent_kernel","_layer2","_iteration_",i,space="")
    titulo_dense <- paste(case,"_",units,"_dense","_iteration_",i,sep="")
    
    file_lstm_2 = paste(titulo_lstm_2,'.png',sep="")
    file_lstm_recurrent_2 = paste(titulo_lstm_recurrent_2,'.png',sep="")
    file_dense = paste(titulo_dense,'.png',sep="")
  
    #Primera capa
    (ggplot(lstm_kernel_melted, aes(x = Gates, y = Units, fill = weights)) + ggtitle(titulo_lstm) + scale_fill_gradientn(limits = c(ranges$min_kernel,ranges$max_kernel),colours = c("black", "white", "green"),breaks = c(ranges$min_kernel,0,ranges$max_kernel),labels=format(c(ranges$min_kernel,0,ranges$max_kernel))) + geom_tile())
    ggsave(file = file_lstm,path=path_save)
    (ggplot(lstm_recurrent_kernel_melted, aes(x = Gates, y = Units, fill = weights)) + ggtitle(titulo_lstm_recurrent) + scale_fill_gradientn(limits = c(ranges$min_recurrent_kernel,ranges$max_recurrent_kernel), colours = c("black", "white", "green"),breaks = c(ranges$min_recurrent_kernel,0,ranges$max_recurrent_kernel),labels=format(c(ranges$min_recurrent_kernel,0,ranges$max_recurrent_kernel))) + geom_tile())
    ggsave(file = file_lstm_recurrent,path=path_save)
    print("llegue")
    #Segunda capa
    (ggplot(lstm_kernel_melted_2, aes(x = Gates, y = Units,fill=weights)) + ggtitle(titulo_lstm_2) +  scale_fill_gradientn(limits = c(ranges$min_kernel2,ranges$max_kernel2), colours = c("black", "white", "green"),breaks = c(ranges$min_kernel2,0,ranges$max_kernel2),labels=format(c(ranges$min_kernel2,0,ranges$max_kernel2))) + geom_tile())
    ggsave(file = file_lstm_2,path=path_save)
    (ggplot(lstm_recurrent_kernel_melted_2, aes(x = Gates, y = Units, fill = weights)) + ggtitle(titulo_lstm_recurrent_2)+ scale_fill_gradientn(limits = c(ranges$min_recurrent_kernel2,ranges$max_recurrent_kernel2), colours = c("black", "white", "green"),breaks = c(ranges$min_recurrent_kernel2,0,ranges$max_recurrent_kernel2),labels=format(c(ranges$min_recurrent_kernel2,0,ranges$max_recurrent_kernel2))) + geom_tile())
    ggsave(file = file_lstm_recurrent_2,path=path_save)
    (ggplot(dense_melted, aes(x = Output, y = Units, fill = weights)) + ggtitle(titulo_dense) +scale_fill_gradientn(limits = c(ranges$min_dense,ranges$max_dense), colours = c("black", "white", "green"),breaks = c(ranges$min_dense,0,ranges$max_dense),labels=format(c(ranges$min_dense,0,ranges$max_dense))) + geom_tile())
    ggsave(file = file_dense,path=path_save)
    
    i=i+period
    
  }
}

##Path weights example /Users/cristobal/Documents/Tesis/Codigo/Neural_LSTM/Checkpoint/19/1_layer/1_units/1_fold/weights

path_dir <- "/Users/cristobal/Documents/Tesis/Codigo/Neural_LSTM/Checkpoint/Stateless/25_time_steps/"
#cases <- c("11","55","91")
#layers <- c("1_layer","2_layer")
#cells <- c("1_units","2_units","4_units","8_units","16_units","24_units","32_units","40_units","48_units")
#folds <- c("1_fold","2_fold")

cases <- c("11","99")
layers <- c("1_layer","2_layer")
cells <- c("2_units","4_units","8_units")
folds <- c("1_fold","2_fold")
ranges_1layer <- list()
ranges_2layer <- list()
for (case in cases){
  for (layer in layers){
    for(cell in cells){
      for(fold in folds){
        path_weights <- ""
        path_save <- ""
        path_weights <- paste(path_dir,case,"/",layer,"/",cell,"/",fold,"/weights/",sep = "")
        path_save <- paste(path_dir,case,"/",layer,"/",cell,"/",fold,"/heat_images/",sep = "")
        if(layer == "1_layer"){
          ranges_1layer <- check_range_1layer(path_weights = path_weights,path_save = path_save,case = case,units = cell)
        }
        if(layer == "2_layer"){
          ranges_2layer <- check_range_2layer(path_weights = path_weights,path_save = path_save,case = case,units = cell)
        }
      }
    }
  }
}

##Se guarda los ploteos
for (case in cases){
  for (layer in layers){
    for(cell in cells){
      for(fold in folds){
        path_weights <- ""
        path_save <- ""
        path_weights <- paste(path_dir,case,"/",layer,"/",cell,"/",fold,"/weights/",sep = "")
        path_save <- paste(path_dir,case,"/",layer,"/",cell,"/",fold,"/heat_images/",sep = "")
        print(path_weights)
        print(path_save)
        if(layer == "1_layer"){
          save_heat_maps_1layer(path_weights = path_weights,path_save = path_save,case = case,units = cell,ranges = ranges_1layer)
        }
        if(layer == "2_layer"){
          save_heat_maps_2layer(path_weights = path_weights,path_save = path_save,case = case,units = cell,ranges = ranges_2layer)
        }
      }
    }
  }
}
