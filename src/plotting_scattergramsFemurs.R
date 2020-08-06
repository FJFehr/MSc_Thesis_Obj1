# This script plots and saves scattergram output
# Fabio Fehr
# 2 July 2020

library(tidyverse)
setwd("/media/fabio/Storage/UCT/Thesis/Coding/MSc_Thesis_Obj1/src")
saving_directory = "../results/"

# NOTE: The signs of the AE components might need to be reversed. *-1

# Shape parameters PCA
bPCA <- read.csv("../results/femur_PCA_ShapeParamaters_b.csv",header = F) %>%
  as.data.frame()

# # Shape parameters Linear AutoEncoder
# bLinearAE <-  read.csv("faust_AE_ShapeParamaters_b.csv",header = F) %>%
#   as.data.frame() %>% 
#   mutate(class = as.factor(rep(seq(0,9),10)))
# # Were reversed so getting correct orientation
# # bLinearAE$V2 <- bLinearAE$V2 *-1
# 
# # Shape parameters Non-linear AutoEncoder
# bNonlinearAE <- read.csv("faust_nonlinear_AE_ShapeParamaters_b.csv",header = F)%>%
#   as.data.frame() %>% 
#   mutate(class = as.factor(rep(seq(0,9),10)))
# bNonlinearAE$V2 <- bNonlinearAE$V2 *-1

# Eigen values from PCA
eigen <- read.csv("../results/femur_PCA_Eigen.csv",header = F) %>%
  as.data.frame()

#####################################################################################################################
### Function for plotting ###########################################################################################
#####################################################################################################################

# plotting a scattergram coloured by class
plot_scattergram <- function(b,eigenValues,dim1=1,dim2=2,title,name,saveBoolean=T){
  
  vector_data <- data.frame(x1 = c(sqrt(eigenValues[dim1,1])*-3,0),
                            x2 = c(sqrt(eigenValues[dim1,1])*3,0),
                            y1 = c(0,sqrt(eigenValues[dim2,1])*-3),
                            y2 = c(0,sqrt(eigenValues[dim2,1])*3))
  
  ggplot()+
    geom_point(data =b, aes(x=b[,dim1], y = b[,dim2]))+
    geom_segment(aes(x = x1, y = y1, xend = x2, yend = y2),
                 data = vector_data,arrow = arrow(ends = "both",
                                                  type = "closed",
                                                  length = unit(0.01, "npc")))+
    
    labs(title = title,
         x = bquote(b[.(dim1)]),
         y = bquote(b[.(dim2)]),
         parse = T)+
    
    geom_text(aes(x = sqrt(eigenValues[dim1,1])*-3,y = 0, label=deparse(bquote(-3*sqrt(lambda[.(dim1)])))),
              parse = TRUE,
              vjust=-0.5)+
    geom_text(aes(sqrt(eigenValues[dim1,1])*3,0, label=deparse(bquote(+3*sqrt(lambda[.(dim1)])))),
              parse = TRUE,
              vjust=-0.5)+
    geom_text(aes(0, sqrt(eigenValues[dim2,1])*-3, label=deparse(bquote(-3*sqrt(lambda[.(dim2)])))),
              parse = TRUE,
              hjust=-0.2)+
    geom_text(aes(0, sqrt(eigenValues[dim2,1])*3, label=deparse(bquote(+3*sqrt(lambda[.(dim2)])))),
              parse = TRUE,
              hjust=-0.2)+
    geom_text(aes(0, 0, label=(paste(expression(bar(x))))),
              parse = TRUE,
              hjust=-0.5,vjust = -0.5)+
    
    theme_light()
  
  if(saveBoolean){
    ggsave(name, width = 10, height = 7)
  }
}

# Compare the PCA and linearAE
plot_compare_scattergram <- function(b1,b2,label1,label2,eigenValues,dim1=1,dim2=2,title,name,saveBoolean=T){
  
  vector_data <- data.frame(x1 = c(sqrt(eigenValues[dim1,1])*-3,0),
                            x2 = c(sqrt(eigenValues[dim1,1])*3,0),
                            y1 = c(0,sqrt(eigenValues[dim2,1])*-3),
                            y2 = c(0,sqrt(eigenValues[dim2,1])*3))
  
  ggplot()+
    geom_point(data =b1, aes(x=b1[,dim1], y = b1[,dim2],colour = label1),alpha = 0.5)+
    geom_point(data =b2, aes(x=b2[,dim1], y = b2[,dim2],colour = label2),alpha = 0.5)+
    geom_segment(aes(x = x1, y = y1, xend = x2, yend = y2),
                 data = vector_data,arrow = arrow(ends = "both",
                                                  type = "closed",
                                                  length = unit(0.01, "npc")))+
    
    labs(title = title,
         x = bquote(b[.(dim1)]),
         y = bquote(b[.(dim2)]),
         parse = T)+
    
    geom_text(aes(x = sqrt(eigenValues[dim1,1])*-3,y = 0, label=deparse(bquote(-3*sqrt(lambda[.(dim1)])))),
              parse = TRUE,
              vjust=-0.5)+
    geom_text(aes(sqrt(eigenValues[dim1,1])*3,0, label=deparse(bquote(+3*sqrt(lambda[.(dim1)])))),
              parse = TRUE,
              vjust=-0.5)+
    geom_text(aes(0, sqrt(eigenValues[dim2,1])*-3, label=deparse(bquote(-3*sqrt(lambda[.(dim2)])))),
              parse = TRUE,
              hjust=-0.2)+
    geom_text(aes(0, sqrt(eigenValues[dim2,1])*3, label=deparse(bquote(+3*sqrt(lambda[.(dim2)])))),
              parse = TRUE,
              hjust=-0.2)+
    geom_text(aes(0, 0, label=(paste(expression(bar(x))))),
              parse = TRUE,
              hjust=-0.5,vjust = -0.5)+
    
    scale_color_manual(values=c("dodgerblue","grey20"))+
    labs(colour = NULL)+
    theme_light()
  
  if(saveBoolean){
    ggsave(name, width = 10, height = 7)
  }
}

# compares 2 scattergrams with smooth curves fitted to them
plot_compare_scattergram2 <- function(b1,b2,label1,label2,eigenValues,dim1=1,dim2=2,title,name,saveBoolean=T){
  
  vector_data <- data.frame(x1 = c(sqrt(eigenValues[dim1,1])*-3,0),
                            x2 = c(sqrt(eigenValues[dim1,1])*3,0),
                            y1 = c(0,sqrt(eigenValues[dim2,1])*-3),
                            y2 = c(0,sqrt(eigenValues[dim2,1])*3))
  
  ggplot()+
    geom_point(data =b1, aes(x=b1[,dim1], y = b1[,dim2],colour = label1),alpha = 0.5)+
    geom_point(data =b2, aes(x=b2[,dim1], y = b2[,dim2],colour = label2),alpha = 0.5)+
    geom_smooth(data =b1,aes(x=b1[,dim1], y = b1[,dim2],colour = label1))+
    geom_smooth(data =b2, aes(x=b2[,dim1], y = b2[,dim2],colour = label2))+
    geom_segment(aes(x = x1, y = y1, xend = x2, yend = y2),
                 data = vector_data,arrow = arrow(ends = "both",
                                                  type = "closed",
                                                  length = unit(0.01, "npc")))+
    
    labs(title = title,
         x = bquote(b[.(dim1)]),
         y = bquote(b[.(dim2)]),
         parse = T)+
    
    geom_text(aes(x = sqrt(eigenValues[dim1,1])*-3,y = 0, label=deparse(bquote(-3*sqrt(lambda[.(dim1)])))),
              parse = TRUE,
              vjust=-0.5)+
    geom_text(aes(sqrt(eigenValues[dim1,1])*3,0, label=deparse(bquote(+3*sqrt(lambda[.(dim1)])))),
              parse = TRUE,
              vjust=-0.5)+
    geom_text(aes(0, sqrt(eigenValues[dim2,1])*-3, label=deparse(bquote(-3*sqrt(lambda[.(dim2)])))),
              parse = TRUE,
              hjust=-0.2)+
    geom_text(aes(0, sqrt(eigenValues[dim2,1])*3, label=deparse(bquote(+3*sqrt(lambda[.(dim2)])))),
              parse = TRUE,
              hjust=-0.2)+
    geom_text(aes(0, 0, label=(paste(expression(bar(x))))),
              parse = TRUE,
              hjust=-0.5,vjust = -0.5)+
    
    scale_color_manual(values=c("dodgerblue","goldenrod3"))+
    labs(colour = NULL)+
    theme_light()
  
  if(saveBoolean){
    ggsave(name, width = 10, height = 7)
  }
}

#####################################################################################################################
### Call plotting functions #########################################################################################
#####################################################################################################################


# Plot 2 shape parameters for the PCA FAUST dataset 
plot_scattergram(b = bPCA,
                 eigenValues = eigen,
                 dim1 = 1,
                 dim2 = 2,
                 title = "Femur PCA scattergram",
                 name = paste0(saving_directory,"femur_PCA_Scattergram.png"), # change to .pdf if you want a pdf
                 saveBoolean = T)

# Compare PCA and linear AE shape parameters for the FAUST dataset 
plot_compare_scattergram(b1 = bPCA,
                         b2 = bLinearAE,
                         label1 = "PCA",
                         label2 = "Linear AE",
                         eigenValues = eigen,
                         dim1 = 1,
                         dim2 = 2,
                         title = "FAUST PCA vs Linear AE scattergram",
                         name = paste0(saving_directory,"faust_PCAvslinearAE_Scattergram.png"), # change to .pdf if you want a pdf
                         saveBoolean = T)



# Compare Linear and Non-linear AE shape parameters for the FAUST dataset 
plot_compare_scattergram2(b1 = bLinearAE,
                          b2 = bNonlinearAE,
                          label1 = "Linear AE",
                          label2 = "Non-linear AE",
                          eigenValues = eigen,
                          dim1 = 1,
                          dim2 = 2,
                          title = "FAUST Linear vs Non-Linear AE scattergram",
                          name = paste0(saving_directory,"faust_LinearAEvsNonLinearAE_Scattergram.png"), # change to .pdf if you want a pdf
                          saveBoolean = T)
