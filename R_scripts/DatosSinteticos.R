library(ARI)
library(carSignal)
library(Rounding)
library(signal)
library(pracma)
library(MASS)


#' Read data
#' 
#' @param path
#' @param file
#' @param sep 
#' @param columns
#' 
#' @return data
read_data <- function(path,
                      file,
                      sep="\t",
                      columns = c("time", "cbfvs","abp","cbfvr"))
{
  path_file <- paste(path,file,sep="")
  ari_19 = read.table(path_file, 
                      sep=sep, 
                      header = FALSE,
                      col.names=columns, 
                      strip.white=TRUE)
}

read_normal_data <- function(path,
                      file,
                      sep="\t",
                      columns = c("time", "cbfv","abp"))
{
  path_file <- paste(path,file,sep="")
  ari_19 = read.table(path_file, 
                      sep=sep, 
                      header = FALSE,
                      col.names=columns, 
                      strip.white=TRUE)
}

#' Ploting function
#' 
#' @param simulated_data
#' @param templates_parameters
#' @param normalised_real_abp
#' @param specific_ari
#' 
#' @return teorical_response
simulated_data <- function(simulated_data,
                           time_instants,
                           templates_parameters,
                           normalised_real_abp,
                           specific_ari = FALSE,
                           index_ari = 1)
{
  step <- round(length(simulated_data$cbfvs)/length(templates_parameters$ARI))
  aux <- 1
  teorical_response <- c(1:length(simulated_data$cbfvs))
  if(specific_ari == FALSE){
    for(i in seq(length(templates_parameters$ARI))){
      index <- i
      if((i*step) > length(simulated_data$cbfvs)){
        #print(templates_parameters$ARI[i])
        limit <- (length(simulated_data$cbfvs)-aux)
        aux_cbfv <- get.theoretical.CBFV.response.simulated(T = templates_parameters$T[i],
                                                  D = templates_parameters$D[i],
                                                  K = templates_parameters$K[i],
                                                  ABP.normalised = normalised_real_abp[aux:(aux+limit)],
                                                  time.instants = time_instants[aux:(aux+limit)],
                                                  stabilisation.time = 10)
        #print(aux_cbfv)
        teorical_response[aux:(aux+limit)] <- aux_cbfv$CBFV.theoretical.response[1:limit]
      }
      else{
        #print(templates_parameters$ARI[i])
        aux_cbfv <- get.theoretical.CBFV.response.simulated(T = templates_parameters$T[i],
                                                  D = templates_parameters$D[i],
                                                  K = templates_parameters$K[i],
                                                  ABP.normalised = normalised_real_abp[aux:(aux+step)-1],
                                                  time.instants = time_instants[aux:(aux+step)-1],
                                                  stabilisation.time = 10)
        #print(aux_cbfv)
        teorical_response[aux:(aux+step)-1] <- aux_cbfv$CBFV.theoretical.response[1:step]
        #print(teorical_response[aux:(aux+step)-1])
        #print(aux_cbfv$CBFV.theoretical.response[1:step])
      }
      aux <- aux + step
    }
  }
  else{
    print("Se esta usando indice fijo")
    teorical_response <- get.theoretical.CBFV.response(T = templates_parameters$T[index_ari],
                                              D = templates_parameters$D[index_ari],
                                              K = templates_parameters$K[index_ari],
                                              ABP.normalised = normalised_real_abp,
                                              time.instants = time_instants,
                                              stabilisation.time = 6)
    teorical_response <- teorical_response$CBFV.theoretical.response
  }
  teorical_response
}

#' Simulando respuesta teorica de flujo
#' 
#' @param specific_ari
#' @param ari_index
#' @param start
#' @param end
#' @param time.instants
#' @param ABP.normalised
#' @param sampling.time
#' @param stabilisation.time
#' @param templates_parameters
#' @param interpolation_method: puede ser "linear", "constant", "nearest", "spline", "cubic"
#' 
#' 
get.theoretical.CBFV.response.simulated <- function(
  specific_ari,
  ari_index,
  start,
  end,
  time.instants,
  ABP.normalised,
  sampling.time = min(round(diff(time.instants), 3)),
  stabilisation.time = 1,
  templates_parameters,
  interpolation_method = "spline"
)
{
  print("sampling.time simulacion")
  print(sampling.time)
  # Initialises the answer
  ans <- list()
  if(specific_ari){
    parameters_lists <- parameters_lists(len_array = length(ABP.normalised),
                                         templates_parameters = templates_parameters,
                                         specific_ari = specific_ari,
                                         ari_index = ari_index)
    
    T <- parameters_lists[["T"]]
    D <- parameters_lists[["D"]]
    K <- parameters_lists[["K"]]
    
  }
  else{
    
    interpolation_points <- seq(start,end,length.out = length(ABP.normalised))
    #print(interpolation_points)
    T <-pracma::interp1(templates_parameters[["ARI"]], 
                        templates_parameters[["T"]], 
                        interpolation_points, interpolation_method)
    D <-pracma::interp1(templates_parameters[["ARI"]], 
                        templates_parameters[["D"]], 
                        interpolation_points, interpolation_method)
    K <-pracma::interp1(templates_parameters[["ARI"]], 
                        templates_parameters[["K"]], 
                        interpolation_points, interpolation_method)
    
  }
  #print(T)
  #print(D)
  #print(K)
  
  ans[["T"]] <- T
  ans[["D"]] <- D
  ans[["K"]] <- K
  ans[["time.instants"]] <- time.instants
  ans[["sampling.time"]] <- sampling.time
  ans[["ABP.normalised"]] <- ABP.normalised
  
 # print(ans)
  
  frequency <- 1 / ans[["sampling.time"]]
  nsamples <- length(ans[["time.instants"]])
  nsamples.stabilisation <-
    round(stabilisation.time / ans[["sampling.time"]])
  
  P <- c(
    rep(ans[["ABP.normalised"]][1], nsamples.stabilisation),
    ans[["ABP.normalised"]],
    rep(ans[["ABP.normalised"]][nsamples], nsamples.stabilisation)
  )
  
  ans[["T"]] <- c(
    rep(ans[["T"]][1], nsamples.stabilisation),
    ans[["T"]],
    rep(ans[["T"]][nsamples], nsamples.stabilisation)
  )
  
  ans[["D"]] <- c(
    rep(ans[["D"]][1], nsamples.stabilisation),
    ans[["D"]],
    rep(ans[["D"]][nsamples], nsamples.stabilisation)
  )
  
  ans[["K"]] <- c(
    rep(ans[["K"]][1], nsamples.stabilisation),
    ans[["K"]],
    rep(ans[["K"]][nsamples], nsamples.stabilisation)
  )
  
  # Gets dP
  dP <- P - 1
  
  # Applies Tiecks' equations to obtain the CBFV signal
  X1 <- vector(mode = "numeric", length = length(P))
  X2 <- vector(mode = "numeric", length = length(P))
  CBFV <- vector(mode = "numeric", length = length(P))
  
  X1[1] <- 0.0
  X2[1] <- 0.0
  CBFV[1] <- 1.0
  for(t in 2:length(P))
  {
    divisor <- frequency * ans[["T"]][t]
    X1[t] <- X1[t-1] + (dP[t] - X2[t-1]) / divisor
    X2[t] <- X2[t-1] + (X1[t] - 2 * ans[["D"]][t] * X2[t-1]) / divisor
    CBFV[t] <- 1 + dP[t] - ans[["K"]][t] * X2[t]
  }
  
  if(nsamples.stabilisation > 0)
    CBFV <- CBFV[-(1:nsamples.stabilisation)]
  CBFV <- CBFV[1:nsamples]
  ans[["CBFV.theoretical.response"]] <- CBFV
  #invisible(ans)
  teorical_response <- ans[["CBFV.theoretical.response"]]
  teorical_response
}

#' Ploting function
#' @param time_instants
#' @param cbfv_curve
#' @param title
#' 
#' @return Plot of data
ploting_simulated_data <- function(time_instants, 
                                   cbfv_curve, 
                                   title = "CBFV")
{
  plot(time_instants,cbfv_curve,"l",main = title)
}


#' Se crea un arreglo del tamanio len_array con los parametros T, D y K 
#' repetidos len_array/length(templates_parameter$ARI)
#' 
#' @param len_array: largo del arreglo
#' 
parameters_lists <- function(len_array,
                           templates_parameters,
                           specific_ari = FALSE,
                           ari_index = 0)
{
  step <- round(len_array/length(templates_parameters$ARI))
  aux <- 1
  
  T_list <- c(1:len_array)
  D_list <- c(1:len_array)
  K_list <- c(1:len_array)
  
  ans <- list()
  ans[["T"]] <- T_list
  ans[["D"]] <- D_list
  ans[["K"]] <- K_list
  
  for(i in seq(length(templates_parameters$ARI))){
    if(specific_ari){i=ari_index}
    if((i*step) > len_array){
      limit <- (len_array-aux)
      ans[["T"]][aux:(aux+limit)] <- rep(templates_parameters$T[i],limit)
      ans[["D"]][aux:(aux+limit)] <- rep(templates_parameters$D[i],limit)
      ans[["K"]][aux:(aux+limit)] <- rep(templates_parameters$K[i],limit)
    }
    else{

      ans[["T"]][aux:(aux+step)] <- rep(templates_parameters$T[i],step)
      ans[["D"]][aux:(aux+step)] <- rep(templates_parameters$D[i],step)
      ans[["K"]][aux:(aux+step)] <- rep(templates_parameters$K[i],step)
      
    }
    aux <- aux + step
  }
  
  ans
}


#' Normalization signal
#' 
#' @param signal : signal
#' 
#' @return Normalize signal
normalize_signal <- function(signal_real, min, max){
  ((signal_real - min) / (max - min))
}

inverse_normalize <- function(signal_normalize, min, max){
  (min + signal_normalize*(max - min))
}


#' Simulate_ari
#' 
#' Ejecucion del programa
simulate_ari <- function(path = "/Users/cristobal/Documents/Tesis/Codigo/DataGeneration/",
                         file = "191.txt",
                         file_output = "",
                         start = 0, 
                         end = 9, 
                         specific_ari = FALSE, 
                         ari_index = 1,
                         interpolation_method = "spline",
                         statabilisation_time = 1)
{
  
  #Se leen los datos
  simulated_data <- read_data(path=path,file = file)
  
  #Se obtienen los 9 parametros del ARI. De 0 a 9. Tambien se pueden obtener interpolados
  #templates_parameters <- get.AT.decimal.templates.parameters()
  templates_parameters <- get.AT.templates.parameters()
  
  #Se normaliza la senial de presion
  normalised_real_abp <- scale(simulated_data$abp)

  #Se utiliza esta funcion solamente para obtener la frecuencia de muestreo. Se resta 0.2 para eliminar el ultimo elemento
  time_instants <- get.normalised.ABP.stimulus(sampling.time = 0.2,
                                               time.until.release = 0,
                                               time.after.release = (length(simulated_data$cbfvs)*(0.2))-0.2)
  time_instants <- time_instants$time.instants
  
  #Se obtiene la respuesta teorica de flujo a la senial real de presion
  teorical_response <- get.theoretical.CBFV.response.simulated(start = start,
                                                                    end = end,
                                                                    specific_ari = specific_ari,
                                                                    ari_index = ari_index,
                                                                    ABP.normalised = normalised_real_abp,
                                                               time.instants = time_instants,
                                                               stabilisation.time = statabilisation_time,
                                                               templates_parameters = templates_parameters,
                                                               interpolation_method = interpolation_method)

  #se utiliza los valores del flujo real para transformar los datos simulados normalizados a una escala real
  #cbfv.transform <- scale(simulated_data$cbfvr)
  #cbfv.transform * attr(cbfv.transform, 'scaled:scale') + attr(cbfv.transform, 'scaled:center')

  #se plotean los graficos de flujo real, simulados
  #Presion del sujeto
  ploting_simulated_data(time_instants = time_instants,
                         cbfv_curve = scale(simulated_data$abp),
                         title = "ABP")
  #Simulado Rooney
  ploting_simulated_data(time_instants = time_instants,
                         cbfv_curve = scale(simulated_data$cbfvs),
                         title = "Simulado Rooney")
  
  #Simulado creado
  #teorical_response <- teorical_response * attr(cbfv.transform, 'scaled:scale') + attr(cbfv.transform, 'scaled:center')
  ploting_simulated_data(time_instants = time_instants,
                         cbfv_curve = scale(teorical_response),
                         title = paste("CBFV simulado ",toString(start)," to ",toString(end),sep=""))

  teorical_response_transform <- inverse_normalize(normalize_signal(teorical_response,min(teorical_response),max(teorical_response)), 
                                                   min(simulated_data$cbfvs), 
                                                   max(simulated_data$cbfvs))
  
  
  #Ploteos juntos
  plot(time_instants,scale(simulated_data$cbfvs),type="s",col="red", ylab = "CBFV", main = "Simulated ARI", sub = "Red->Rooney: Green->Cristobal")
  lines(time_instants,scale(teorical_response),col="green", ylab = "CBFV")
  
  plot(time_instants,(simulated_data$abp),type="s",col="red", ylab = "CBFV", main = "Simulated ARI", sub = "Red->Rooney: Green->Cristobal: Blue->CBFV real", ylim = range(40:125))
  lines(time_instants,(teorical_response_transform),col="green", ylab = "CBFV")
  #lines(time_instants,(simulated_data$cbfvr),col="blue", ylab = "CBFV")
  
  plot(time_instants,normalize_signal(simulated_data$cbfvs,min(simulated_data$cbfvs),max(simulated_data$cbfvs)),type="s",col="red", ylab = "CBFV", main = "Simulated ARI", sub = "Red->Rooney: Green->Cristobal")
  lines(time_instants,normalize_signal(teorical_response_transform,min(simulated_data$cbfvs),max(simulated_data$cbfvs)),col="green", ylab = "CBFV")
  
  
  #write file with data simulada
  
  data <- matrix(data = c(time_instants,teorical_response_transform,simulated_data$abp), nrow = length(time_instants),ncol = 3,dimnames = NULL)
  write.matrix(data, 
               file = paste("/Users/cristobal/Documents/Tesis/Codigo/DataGeneration/GeneratedData/",file_output,sep=""), 
               sep = "\t")
  
  print(cor(teorical_response,simulated_data$cbfvs,method = "pearson"))  
  print(cor(teorical_response,simulated_data$cbfvs,method = "spearman"))  
}


#' Simulate_ari
#' 
#' Ejecucion del programa
simulate_ari_original <- function(path = "",
                                  file = "",
                                  file_output = "",
                                  start = 1, 
                                  end = 9, 
                                  specific_ari = FALSE, 
                                  ari_index = 1,
                                  interpolation_method = "spline",
                                  statabilisation_time = 1)
{
  
  #Se leen los datos#time_frecuency, cbfv, abp
  subject_data <- read_normal_data(path=path,file = file)
  sampling_time <- subject_data$time[2] - subject_data$time[1]
  print("sampling_time")
  print(sampling_time)
  
  #Se obtienen los 9 parametros del ARI. De 0 a 9. Tambien se pueden obtener interpolados
  #templates_parameters <- get.AT.decimal.templates.parameters()
  templates_parameters <- get.AT.templates.parameters()
  
  #Se normaliza la senial de presion
  normalised_real_abp <- scale(subject_data$abp)
  
  #Se utiliza esta funcion solamente para obtener la frecuencia de muestreo. Se resta 0.2 para eliminar el ultimo elemento
  time_instants <- get.normalised.ABP.stimulus(sampling.time = sampling_time,
                                               time.until.release = 0,
                                               time.after.release = (length(subject_data$cbfv)*(sampling_time))-sampling_time)
  time_instants <- time_instants$time.instants
  
  #Se obtiene la respuesta teorica de flujo a la senial real de presion
  teorical_response <- get.theoretical.CBFV.response.simulated(start = start,
                                                               end = end,
                                                               specific_ari = specific_ari,
                                                               ari_index = ari_index,
                                                               ABP.normalised = normalised_real_abp,
                                                               time.instants = time_instants,
                                                               stabilisation.time = statabilisation_time,
                                                               templates_parameters = templates_parameters,
                                                               interpolation_method = interpolation_method)
  
  #se utiliza los valores del flujo real para transformar los datos simulados normalizados a una escala real
  #cbfv.transform <- scale(simulated_data$cbfvr)
  #cbfv.transform * attr(cbfv.transform, 'scaled:scale') + attr(cbfv.transform, 'scaled:center')

  
  #Simulado creado
  #teorical_response <- teorical_response * attr(cbfv.transform, 'scaled:scale') + attr(cbfv.transform, 'scaled:center')
  
  #check si teorical_response esta normalizada
 
  #teorical_response_transform <- inverse_normalize(teorical_response, 
  #                                                 min(subject_data$cbfv), 
  #                                                 max(subject_data$cbfv))
  
  teorical_response_transform <- inverse_normalize(normalize_signal(teorical_response,min(teorical_response),max(teorical_response)), 
                                                   min(subject_data$cbfv), 
                                                   max(subject_data$cbfv))
  
  
  #write file with data simulada
  data <- matrix(data = c(time_instants,teorical_response_transform,subject_data$abp), nrow = length(time_instants),ncol = 3,dimnames = NULL)
  write.matrix(data, file = file_output, sep = "\t")
  
}


#' ari_index= este indice esta defasado. Cuando es 1 corresponde a ARI 0. Cuando es 10 corresponde a ARI 9
#' path es donde se encuentran los datos reales de los sujetos
#' path_save es el directorio raiz de donde se guardaran los datos simulados
#' 
#' 
process_subjects <- function(start = 1, end = 9,specific_ari = TRUE,ari_index = 1){
  path = "/Users/cristobal/Documents/Tesis/Codigo/Neural_LSTM/Subjects_data"
  path_save = "/Users/cristobal/Documents/Tesis/Codigo/Neural_LSTM/Subjects_simulated/"
  ##TO DO
  dirs = list.dirs(path)
  for (dir in dirs){
    files = list.files(dir)
    for (file in files){
      aux_file = unlist(strsplit(file,"_"))
      for (element in aux_file){
        if (element == "train.txt"){
          print("trabajando archivo train")
          print(file)
          names_aux = unlist(strsplit(file,"_"))
          print(names_aux[1])
          path = paste(dir,"/",sep = "")
          print("path")
          print(path)
          dir_save = paste(path_save,names_aux[1],"/",sep = "")
          if (dir.exists(dir_save) == FALSE){
            dir.create(dir_save)
          }
          if(specific_ari == FALSE){
            file_output = paste(dir_save,1,start,end,".txt",sep = "")
          }
          if(specific_ari == TRUE){
            file_output = paste(dir_save,1,(ari_index-1),(ari_index-1),".txt",sep = "")
          }
          
          print("file_output")
          print(file_output)
          simulate_ari_original(path = path,
                                file = file,
                                file_output = file_output,
                                start = start, 
                                end = end, 
                                specific_ari = specific_ari, 
                                ari_index = ari_index,
                                interpolation_method = "spline",
                                statabilisation_time = 1)
          
        }
        else if (element == "test.txt"){
          print("trabajando archivo test")
          print(file)
          names_aux = unlist(strsplit(file,"_"))
          print(names_aux[1])
          path = paste(dir,"/",sep = "")
          print("path")
          print(path)
          dir_save = paste(path_save,names_aux[1],"/",sep = "")
          if (dir.exists(dir_save) == FALSE){
            dir.create(dir_save)
          }
          if(specific_ari == FALSE){
            file_output = paste(dir_save,2,start,end,".txt",sep = "")
          }
          if(specific_ari == TRUE){
            file_output = paste(dir_save,2,(ari_index-1),(ari_index-1),".txt",sep = "")
          }
          print("file_output")
          print(file_output)
          simulate_ari_original(path = path,
                                file = file,
                                file_output = file_output,
                                start = start, 
                                end = end, 
                                specific_ari = specific_ari, 
                                ari_index = ari_index,
                                interpolation_method = "spline",
                                statabilisation_time = 1)
        }
        else{
          print("do nothing")
        }
      }
    }
  }
}

