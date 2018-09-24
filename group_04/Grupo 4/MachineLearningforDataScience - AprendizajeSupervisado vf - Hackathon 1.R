rm(list=ls())

#########################################################################
### -- Machine Learning for Data Science -- ## 
#########################################################################
### Hackaton 1 ## 

#########################################################################

# ######### 1) LIBRERIAS A UTILIZAR ################# 


install.packages("sqldf")
install.packages("ggvis")
install.packages("party")
install.packages("Boruta")
install.packages("pROC")
install.packages("randomForest")
install.packages("e1071")
install.packages("caret")
install.packages("glmnet")
install.packages("mboost")
install.packages("adabag")
install.packages("xgboost")
install.packages("ROCR")
install.packages("C50")
install.packages("mlr")
install.packages("lattice")
install.packages("gmodels")
install.packages("gplots")
install.packages("DMwR")
install.packages("UBL")
install.packages("rminer")
install.packages("polycor")
install.packages("class")
install.packages("neuralnet")
install.packages("reticulate")

library(sqldf)
library(ggvis)
library(party)
library(Boruta)
library(pROC)
library(randomForest)
library(e1071)
library(caret)
library(glmnet)
library(mboost)
library(adabag)
library(xgboost)
library(ROCR)
library(C50)
library(mlr)
library(lattice)
library(gmodels)
library(gplots)
library(DMwR)
library(UBL)
library(rminer)
library(polycor)
library(class)
library(neuralnet)
library(reticulate)


######### 2) EXTRAYENDO LA DATA ################# 

getwd()
setwd("C:/Users/Yaneth/Desktop/Data/hackaton 1")

train2<-read.csv("train.csv",na.strings = c(""," ",NA)) # leer la data de entrenamiento
test<-read.csv("test.csv",na.strings = c(""," ",NA))  # leer la data de Validacion 

table(train2$riesgo)

names(train) # visualizar los nombres de la data
head(train)  # visualizar los 6 primeros registros
str(train)   # ver la estructura de la data

######### 3) EXPLORACION DE LA DATA ################# 

# tablas resumen
summary(train) # tabla comun de obtener
summarizeColumns(train) # tabla mas completa

resumen=data.frame(summarizeColumns(train))

## Graficos para variables cuantitativas

# histogramas y Cajas

#Veamos la variable Edad
hist(train$edad, breaks = 100, main = "Edad del cliente",xlab = "Edad",col="blue")

table(train$meses)
#Veamos la variable Meses
hist(train$meses, breaks = 100, main = "Número de días de atraso máximo durante los 12 meses antes de la aprobación del crédito",xlab = "Retraso",col="red")

#Veamos la variable  antiguedad
table(train$max_ant)
hist(train$max_ant, breaks = 100, main = "Máxima antigüedad con Tarjeta de crédito en el Sistema Financiero (en meses).",xlab = "Antiguedad",col="red")

#Veamos la variable Ingreso
hist(train$ingreso, breaks = 100, main = "Ingreso mensual",xlab = "Ingreso",col="blue")

#Veamos la variable Score
hist(train$score, breaks = 100, main = "Score con el que fue aprobada la Tarjeta de crédito en el Banco",xlab = "Score",col="blue")


#Veamos los Outliers
#Se visualizan valores atipicos

#Veamos la variable max_ant
bwplot(train$max_ant, layout = c(1, 1),main = "Màxima antiguedad",xlab = "Antiguedad", col="blue")

#Veamos la variable ingreso
bwplot(train$ingreso, layout = c(1, 1),main = "Ingreso",xlab = "Ingreso", col="blue")


####### Tablas para variables cualitativas

#Veamos la variable tipo de vivienda
CrossTable(train$tipo_vivienda,prop.t=FALSE,prop.r=TRUE,prop.c=FALSE,prop.chisq=FALSE)

#Veamos la variable zona  --- ver por target
CrossTable(train$zona,prop.t=FALSE,prop.r=TRUE,prop.c=FALSE,prop.chisq=FALSE)

dim(train)
table(train$zona, train$riesgo)
#Veamos la variable nivel
CrossTable(train$nivel,prop.t=FALSE,prop.r=TRUE,prop.c=FALSE,prop.chisq=FALSE)

# Graficos de cualitativos
Tabla1=table(train$tipo_vivienda)
Tabla2=table(train$zona)
Tabla3=table(train$nivel)
Tabla4=table(train$riesgo)

par(mfrow=c(1,2))
balloonplot(t(Tabla1), main ="Tabla de Contingencia Tipo de Vivienda",xlab ="Tipo Vivienda", label = FALSE, show.margins = FALSE)
balloonplot(t(Tabla2), main ="Tabla de Contingencia Zona",xlab ="Zona", label = FALSE, show.margins = FALSE)
balloonplot(t(Tabla3), main ="Tabla de Contingencia Nivel",xlab ="Nivel", label = FALSE, show.margins = FALSE)
balloonplot(t(Tabla4), main ="Tabla de Contingencia Riesgo",xlab ="RIESGO", label = FALSE, show.margins = FALSE)


# comentarios de la data

# 1. LoanAmount tiene (614 - 592) 22 valores perdidos.
# 2. Loan_Amount_Term tiene (614 - 600) 14 valores perdidos.
# 3. Credit_History tiene (614 - 564) 50 valores perdidos.
# 4. Nosotros podemos tambi?n observar que cerca del 84% de los solicitantes al pr?stamo 
# tienen un historial crediticio. ?C?mo? La media del campo Credit_History es 0.84 
# (Recordemos, Credit_History tiene o toma el valor 1 para aquellos que tienen 
#   historial crediticio y 0 en caso contrario).
# 5. La variable ApplicantIncome parece estar en l?nea con las espectativas al 
# igual que CoapplicantIncome.

######### 4) IMPUTACION DE LA DATA ################# 

# revisar valores perdidos

perdidos=data.frame(resumen$name,resumen$na,resumen$type); colnames(perdidos)=c("name","na","type")

perdidos <- resumen %>% select(name, na, type)
perdidos

#----------------------------------------------------------
##  RECODIFICACIONES : TRAIN

######### recodificando EDAD  - numerico

train$meses_cod=1/train$meses
hist(train$meses_cod, breaks = 100, main = "Edad reprocesado",xlab = "Edad",col="blue")


######### recodificando MAXIMA ANTIGUEDAD

train$max_ant_cod=ifelse(train$max_ant==0,0,
                         ifelse(train$max_ant >=1,1,
                                train$max_ant))

train$max_ant_cod=as.factor(train$max_ant_cod)
train$max_ant_cod


######### recodificando EDAD  - ingreso

train$ingreso_cod=sqrt(sqrt(train$ingreso))
hist(train$ingreso_cod, breaks = 100, main = "Ingreso reprocesado",xlab = "Ingreso",col="blue")

######### recodificando TIPO DE VIVIENDA
train$tipo_vivienda_cod=ifelse(train$tipo_vivienda==1,1,
                       ifelse(train$tipo_vivienda==3,1,
                              ifelse(train$tipo_vivienda==2,2,
                                     ifelse(train$tipo_vivienda==4,3,
                                            train$tipo_vivienda))))
train$tipo_vivienda_cod=as.factor(train$tipo_vivienda_cod)
train$tipo_vivienda_cod

######### recodificando NIVEL
train$nivel_cod=ifelse(train$nivel==1,1,
                               ifelse(train$nivel==3,1,
                                      ifelse(train$nivel==2,2,
                                             ifelse(train$nivel==4,3,
                                                    ifelse(train$nivel==5,4,
                                                           train$nivel)))))
train$nivel_cod=as.factor(train$nivel_cod)
train$nivel_cod

####### Target : riesto como factor

train$riesgo=as.factor(train$riesgo)
train$riesgo

####### zona como factor

train$zona=as.factor(train$zona)
train$zona


# ---------------------------------------------------

# recodificando Dependents --- También del TEST

######### recodificando EDAD  - numerico

test$meses_cod=1/test$meses
hist(test$meses_cod, breaks = 100, main = "Edad reprocesado",xlab = "Edad",col="blue")


######### recodificando MAXIMA ANTIGUEDAD

test$max_ant_cod=ifelse(test$max_ant==0,0,
                        ifelse(test$max_ant >=1,1,
                               test$max_ant))

test$max_ant_cod=as.factor(test$max_ant_cod)
test$max_ant_cod


######### recodificando EDAD  - ingreso

test$ingreso_cod=sqrt(sqrt(test$ingreso))
hist(test$ingreso_cod, breaks = 100, main = "Ingreso reprocesado",xlab = "Ingreso",col="blue")

######### recodificando TIPO DE VIVIENDA
test$tipo_vivienda_cod=ifelse(test$tipo_vivienda==1,1,
                              ifelse(test$tipo_vivienda==3,1,
                                     ifelse(test$tipo_vivienda==2,2,
                                            ifelse(test$tipo_vivienda==4,3,
                                                   test$tipo_vivienda))))
test$tipo_vivienda_cod=as.factor(test$tipo_vivienda_cod)
test$tipo_vivienda_cod

######### recodificando NIVEL
test$nivel_cod=ifelse(test$nivel==1,1,
                      ifelse(test$nivel==3,1,
                             ifelse(test$nivel==2,2,
                                    ifelse(test$nivel==4,3,
                                           ifelse(test$nivel==5,4,
                                                  test$nivel)))))
test$nivel_cod=as.factor(test$nivel_cod)
test$nivel_cod


####### zona como factor

test$zona=as.factor(test$zona)
test$zona


#---------------------------------------------------------------------------------------------------

### SELECCIONAMOS LAS VARIABLES A UTILIZAR PARA TRAIN Y TEST


train_selec <- train %>% select(edad, meses_cod, max_ant_cod, ingreso_cod, score, tipo_vivienda_cod, zona, nivel_cod, riesgo)
head(train_selec)
test_selec <- test %>% select(edad, meses_cod, max_ant_cod, ingreso_cod, score, tipo_vivienda_cod, zona, nivel_cod)
head(test_selec)


#-------------------------------------------------------

# partcionando la data en numericos y factores

numericos <- sapply(train_selec, is.numeric) # variables cuantitativas
factores <- sapply(train_selec, is.factor)  # variables cualitativas

train_numericos <-  train_selec[ , numericos]
train_factores <- train_selec[ , factores]

head(train_numericos)
head(train_factores)

# APLICAR LA FUNCION LAPPLY PARA DISTINTAS COLUMNAS CONVERTIR A FORMATO NUMERICO
#n1=min(dim(train_factores))
#train_factores[2:(n1-1)] <- lapply(train_factores[2:(n1-1)], as.numeric)
#train_factores[2:(n1-1)] <- lapply(train_factores[2:(n1-1)], as.factor)

numericos <- sapply(test_selec, is.numeric) # variables cuantitativas
factores <- sapply(test_selec, is.factor)  # variables cualitativas

test_numericos <-  test_selec[ , numericos]
test_factores <- test_selec[ , factores]

head(test_numericos)
head(test_factores)

# APLICAR LA FUNCION LAPPLY PARA DISTINTAS COLUMNAS CONVERTIR A FORMATO NUMERICO
#n1=min(dim(test_factores))
#test_factores[2:(n1)] <- lapply(test_factores[2:(n1)], as.numeric)
#test_factores[2:(n1)] <- lapply(test_factores[2:(n1)], as.factor)

# Para train y test

train=cbind(train_numericos,train_factores[,])
test=cbind(test_numericos,test_factores[,])

head(train)
head(test)

##----------------------------------------------------------------------
## Imputacion Parametrica

#Podemos imputar los valores perdidos por la media o la moda

# data train
train_parametrica <- impute(train, classes = list(factor = imputeMode(), 
                                    integer = imputeMode(),
                                    numeric = imputeMean()),
              dummy.classes = c("integer","factor"), dummy.type = "numeric")
train_parametrica=train_parametrica$data[,1:min(dim(train))]
head(train_parametrica)

# data test
test_parametrica  <- impute(test, classes = list(factor = imputeMode(), 
                                    integer = imputeMode(),
                                    numeric = imputeMean()), 
               dummy.classes = c("integer","factor"), dummy.type = "numeric")
test_parametrica=test_parametrica$data[,1:min(dim(test))]
head(test_parametrica)

summary(train_parametrica)
summarizeColumns(train_parametrica)
summarizeColumns(test_parametrica)

str(train_parametrica)
str(test_parametrica)

# ------------------------------- no se utilizó los de arriba - imputación no paramétrica

#--------------------------------  CONTINUANOS

######### 6) BALANCEO DE LOS DATOS Y SELECCION DE DRIVERS #################

## Particionando la Data

set.seed(1234)
sample <- createDataPartition(train_parametrica$riesgo, p = .70,list = FALSE,times = 1)
dim(train_parametrica)
dim(sample)

data.train <- train_parametrica[ sample,]
data.prueba <- train_parametrica[-sample,]

dim(data.train)
dim(data.prueba)

#---------------------------------------------------------------------
# Balanceo de los datos

# Balanceo mediante SMOTE

table(data.train$riesgo)

data_smoote <- SMOTE(riesgo ~ .,data.train  , perc.over = 100, perc.under=200)
table(data_smoote$riesgo)

# Nota SMOTE:
# Vamos a crear observaciones positivas adicionales usando SMOTE.
# Establecimos perc.over = 100 duplicar la cantidad de casos positivos 
# y configuramos perc.under=200 para mantener la mitad de lo que se cre? 
# como casos negativos.



#-----------------------------------------------------
######## seleccion de variables más importantes

##  Mediante ML

# Utilizando Boruta

str(data.train)

pdf("seleccion de variables.pdf")
Boruta(riesgo~.,data=data.train,doTrace=2)->Bor.hvo;
plot(Bor.hvo,las=3);
Bor.hvo$finalDecision

# Utilizando RF

set.seed(1234)
rand <- randomForest( riesgo ~ ., data = data.train,   # Datos a entrenar 
                      ntree=100,           # N?mero de ?rboles
                      mtry = 3,            # Cantidad de variables
                      importance = TRUE,   # Determina la importancia de las variables
                      replace=T)           # muestras con reemplazo


varImpPlot(rand)

# Utilizando Naive Bayes

naive <- fit(riesgo~., data=data.train, model="naiveBayes")
naive.imp <- Importance(naive, data=data.train)
impor.naive=data.frame(naive.imp$imp); rownames(impor.naive)=colnames(data.train)
barplot(naive.imp$imp,horiz = FALSE,names.arg = colnames(data.train),las=2)
impor.naive

dev.off()



### ---------------- revisar estar correlaciones

# matriz de correlaciones no parametricas completas
correlaciones=hetcor(data.train, use = "pairwise.complete.obs")
correlaciones
correlaciones=correlaciones$correlations

# guardamos las correlaciones
#write.csv(correlaciones,"correlaciones.csv")


# Grafico del comportamiento de las variables
data.train %>% ggvis(~ingreso_cod, ~riesgo, fill = ~ingreso_cod) %>% layer_points()



#####################################################################################################
######### 7) MODELADO DE LA DATA #################

# data de entrenamiento 

data.train.1 <- data.train %>% select(edad, meses_cod, max_ant_cod, ingreso_cod, score, zona, nivel_cod, riesgo)
head(data.train.1)

# data de validacion
data.test.1 <- data.prueba %>% select(edad, meses_cod, max_ant_cod, ingreso_cod, score, zona, nivel_cod, riesgo)
head(data.test.1)


m=min(dim(data.train.1))


# modelo 1.- Logistico

modelo1=glm(riesgo~.,data=data.train.1,family = binomial(link = "logit"))
summary(modelo1)

proba1=predict(modelo1, newdata=data.test.1,type="response")

AUC1 <- roc(data.test.1$riesgo, proba1)

## calcular el AUC
auc_modelo1=AUC1$auc

## calcular el GINI
gini1 <- 2*(AUC1$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo1,data.test.1,type="response")
PRED=ifelse(PRED<=0.5,0,1)
PRED=as.factor(PRED)

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity1=as.numeric(tabla$byClass[1])

# Precision
Accuracy1=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error1=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo1
gini1
Accuracy1
error1
Sensitivity1

# modelo 2.- KNN

# Utilizar libreria para ML, tratamiento de la data . pIDE ESTANDARIZACION
install.packages("ML")
library(ML)

n=ncol(data.train.1)
data.train.2=data.train.1
data.train.2[,1:(n-1)]=lapply(data.train.2[,1:(n-1)],as.character)
data.train.2[,1:(n-1)]=lapply(data.train.2[,1:(n-1)],as.numeric)
data.train.2[,1:(n-1)]=lapply(data.train.2[,1:(n-1)],scale)

data.test.2=data.test.1
data.test.2[,1:(n-1)]=lapply(data.test.2[,1:(n-1)],as.character)
data.test.2[,1:(n-1)]=lapply(data.test.2[,1:(n-1)],as.numeric)
data.test.2[,2:(n-1)]=lapply(data.test.2[,2:(n-1)],scale)

#create a task
trainTask <- makeClassifTask(data = data.train.2,target = "riesgo")
testTask <- makeClassifTask(data = data.test.2, target = "riesgo")

trainTask <- makeClassifTask(data = data.train.2,target = "riesgo", positive = "1")

# Modelado KNN

set.seed(1234)
knn <- makeLearner("classif.knn",prob = TRUE,k = 10) # REQUIERO 10 VECINOS

qmodel <- train(knn, trainTask)
qpredict <- predict(qmodel, testTask)
length(qpredict$data$truth)

response=as.numeric(qpredict$data$response[1:1350])
response=ifelse(response==2,1,0)
proba2=response

# curva ROC
AUC2 <- roc(data.test.2$riesgo, proba2) 
auc_modelo2=AUC2$auc

# Gini
gini2 <- 2*(AUC2$auc) -1

# Calcular los valores predichos
PRED <-response

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity2=as.numeric(tabla$byClass[1])

# Precision
Accuracy2=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error2=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo2
gini2
Accuracy2
error2
Sensitivity2

# modelo 3.- Naive Bayes ******************************************

modelo3=naiveBayes(riesgo~.,data = data.train.1)

##probabilidades
proba3<-predict(modelo3, newdata=data.test.1,type="raw")
proba3=proba3[,2]

# curva ROC
AUC3 <- roc(data.test.1$riesgo, proba3) 
auc_modelo3=AUC3$auc

# Gini
gini3 <- 2*(AUC3$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo3,data.test.1,type="class")

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity3=as.numeric(tabla$byClass[1])

# Precision
Accuracy3=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error3=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo3
gini3
Accuracy3
error3
Sensitivity3

# modelo 4.- Arbol CHAID *****************************************

modelo4<-ctree(riesgo~.,data = data.train.1, 
               controls=ctree_control(mincriterion=0.95))

##probabilidades
proba4=sapply(predict(modelo4, newdata=data.test.1,type="prob"),'[[',2)

# curva ROC	
AUC4 <- roc(data.test.1$riesgo, proba4) 
auc_modelo4=AUC4$auc

# Gini
gini4 <- 2*(AUC4$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo4, newdata=data.test.1,type="response")

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity4=as.numeric(tabla$byClass[1])

# Precision
Accuracy4=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error4=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo4
gini4
Accuracy4
error4
Sensitivity4

# modelo 5.- Arbol CART    *********************************************

arbol.completo <- rpart(riesgo~.,data = data.train.1,method="class",cp=0, minbucket=0)
xerr <- arbol.completo$cptable[,"xerror"] ## error de la validacion cruzada
minxerr <- which.min(xerr)
mincp <- arbol.completo$cptable[minxerr, "CP"]

modelo5 <- prune(arbol.completo,cp=mincp)

##probabilidades
proba5=predict(modelo5, newdata=data.test.1,type="prob")[,2]

# curva ROC
AUC5 <- roc(data.test.1$riesgo, proba5) 
auc_modelo5=AUC5$auc

# Gini
gini5 <- 2*(AUC5$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo5, newdata=data.test.1,type="class")

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity5=as.numeric(tabla$byClass[1])

# Precision
Accuracy5=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error5=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo5
gini5
Accuracy5
error5
Sensitivity5

# modelo 6.- Arbol c5.0    *****************************************************************

modelo6 <- C5.0(riesgo~.,data = data.train.1,trials = 55,rules= TRUE,tree=FALSE,winnow=FALSE)

##probabilidades
proba6=predict(modelo6, newdata=data.test.1,type="prob")[,2]

# curva ROC
AUC6 <- roc(data.test.1$riesgo, proba6) 
auc_modelo6=AUC6$auc

# Gini
gini6 <- 2*(AUC6$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo6, newdata=data.test.1,type="class")

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity6=as.numeric(tabla$byClass[1])

# Precision
Accuracy6=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error6=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo6
gini6
Accuracy6
error6
Sensitivity6

# modelo 7.- SVM Radial

modelo7=svm(riesgo~.,data = data.train.1,kernel="radial",costo=100,gamma=1,probability = TRUE, method="C-classification")

##probabilidades
proba7<-predict(modelo7, newdata=data.test.1,decision.values = TRUE, probability = TRUE) 
proba7=attributes(proba7)$probabilities[,2]

# curva ROC
AUC7 <- roc(data.test.1$riesgo, proba7) 
auc_modelo7=AUC7$auc

# Gini
gini7 <- 2*(AUC7$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo7,data.test.1,type="class")

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity7=as.numeric(tabla$byClass[1])

# Precision
Accuracy7=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error7=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo7
gini7
Accuracy7
error7
Sensitivity7

# modelo 8.- SVM Linear

modelo8=svm(riesgo~.,data = data.train.1,kernel="linear",costo=100,probability = TRUE, method="C-classification")

##probabilidades
proba8<-predict(modelo8, newdata=data.test.1,decision.values = TRUE, probability = TRUE) 
proba8=attributes(proba8)$probabilities[,2]

# curva ROC
AUC8 <- roc(data.test.1$riesgo, proba8) 
auc_modelo8=AUC8$auc

# Gini
gini8 <- 2*(AUC8$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo8,data.test.1,type="class")

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity8=as.numeric(tabla$byClass[1])

# Precision
Accuracy8=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error8=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo8
gini8
Accuracy8
error8
Sensitivity8

# modelo 9.- SVM sigmoid

modelo9=svm(riesgo~.,data = data.train.1,kernel="sigmoid",costo=100,probability = TRUE, method="C-classification")

##probabilidades
proba9<-predict(modelo9, newdata=data.test.1,decision.values = TRUE, probability = TRUE) 
proba9=attributes(proba9)$probabilities[,2]

# curva ROC
AUC9 <- roc(data.test.1$riesgo, proba9) 
auc_modelo9=AUC9$auc

# Gini
gini9 <- 2*(AUC9$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo9,data.test.1,type="class")

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity9=as.numeric(tabla$byClass[1])

# Precision
Accuracy9=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error9=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo9
gini9
Accuracy9
error9
Sensitivity9

# modelo 10.- Random Forest

set.seed(1234)
modelo10 <- randomForest( riesgo~.,data = data.train.1,   # Datos a entrenar 
                         ntree=100,           # N?mero de ?rboles
                         mtry = 1,            # Cantidad de variables
                         importance = TRUE,   # Determina la importancia de las variables
                         replace=T) 

##probabilidades
proba10<-predict(modelo10, newdata=data.test.1,type="prob")
proba10=proba10[,2]

# curva ROC
AUC10 <- roc(data.test.1$riesgo, proba10) 
auc_modelo10=AUC10$auc

# Gini
gini10 <- 2*(AUC10$auc) -1

# Calcular los valores predichos
PRED <-predict(modelo10,data.test.1,type="class")

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity10=as.numeric(tabla$byClass[1])

# Precision
Accuracy10=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error10=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo10
gini10
Accuracy10
error10
Sensitivity10

# modelo 11.- Redes Neuronales

set.seed(1234)
neuralnet.learner <- makeLearner("classif.neuralnet",predict.type = "prob",hidden=c(10,15),
                                 act.fct = "logistic",algorithm = "rprop+",threshold = 0.01,stepmax = 2e+05)

qmodel <- train(neuralnet.learner, trainTask)
qpredict <- predict(qmodel, testTask)

##probabilidades
proba11=qpredict$data$prob.1

# curva ROC
AUC11 <- roc(data.test.1$riesgo, proba11) 
auc_modelo11=AUC11$auc

# Gini
gini11 <- 2*(AUC11$auc) -1

# Calcular los valores predichos
PRED <-qpredict$data$response

length(PRED) # TAbla de testeo de train

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity11=as.numeric(tabla$byClass[1])

# Precision
Accuracy11=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error11=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo11
gini11
Accuracy11
error11
Sensitivity11


# modelo 12.- Boosting

set.seed(1234)
modelo12<-boosting(riesgo~.,data = data.train.1)

##probabilidades
proba<-predict(modelo12,data.test.1)
proba12=(proba)$prob[,2]

# curva ROC
AUC12 <- roc(data.test.1$riesgo, proba12) 
auc_modelo12=AUC12$auc

# Gini
gini12 <- 2*(AUC12$auc) -1

# Calcular los valores predichos
PRED <-proba$class

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity12=as.numeric(tabla$byClass[1])

# Precision
Accuracy12=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error12=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo12
gini12
Accuracy12
error12
Sensitivity12

#rm(data_smoote,data.test.2,correlaciones,data.train.2,predictores.train,train,train_no_parametrica,train_no_parametrica2,x)

# modelo 13.- XGboost

set.seed(1234)
getParamSet("classif.xgboost")

xg_set <- makeLearner("classif.xgboost",objective = "binary:logistic", predict.type = "prob",
                      max_depth=3,eta=0.5,nthread=2,nrounds=4)

qmodel <- train(xg_set, trainTask)
qpredict <- predict(qmodel, testTask)

##probabilidades
proba13=qpredict$data$prob.1

# curva ROC
AUC13 <- roc(data.test.1$riesgo, proba13) 
auc_modelo13=AUC13$auc

# Gini
gini13 <- 2*(AUC13$auc) -1

# Calcular los valores predichos
PRED <-qpredict$data$response

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity13=as.numeric(tabla$byClass[1])

# Precision
Accuracy13=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error13=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo13
gini13
Accuracy13
error13
Sensitivity13


#------------------------------------------------------------------------------------------------------
##################### ENSAMBLANDO MIS MEJORES MODELOS

# tabla target


# modelo YANETH - - Ensamble de Modelos (CART, CHAID, C5.0)

##probabilidades

ensamble=data.frame(proba1,proba2,proba6, proba7,proba8,proba10, proba12,proba13);colnames(ensamble)=c("LOGISTICO","KNN","C50","SVM RADIAL", "SVM LINEAL", "RANDOM FOREST", "BOOSTING", "XGBOOST")
ensamble$ensamble=apply(ensamble, 1, mean)
ensamble$response=ifelse(ensamble$ensamble<=0.5,0,1)

proba_FINAL=ensamble$ensamble

# curva ROC
AUC_FINAL <- roc(data.test.1$riesgo, proba_FINAL) 
auc_modeloFINAL=AUC_FINAL$auc

# Gini
gini_FINAL <- 2*(AUC_FINAL$auc) -1

# Calcular los valores predichos
PRED <-ensamble$response

# Calcular la matriz de confusi?n
tabla=confusionMatrix(PRED,data.test.1$riesgo,positive = "1")

# sensibilidad
Sensitivity_FINAL=as.numeric(tabla$byClass[1])

# Precision
Accuracy_FINAL=tabla$overall[1]

# Calcular el error de mala clasificaci?n
error_FINAL=mean(PRED!=data.test.1$riesgo)

# indicadores
auc_modelo_FINAL
gini_FINAL
Accuracy_FINAL
error_FINAL
Sensitivity_FINAL


# FIN ----------------------------------------------


##--------------------------------- TABLA DE RESULTADOS FINAL

## --Tabla De Resultados ####

AUC=rbind(auc_modelo1,
          auc_modelo2,
          auc_modelo6,
          auc_modelo7,
          auc_modelo8,
          auc_modelo10,
          auc_modelo12,
          auc_modelo13,
          auc_modeloFINAL)

GINI=rbind(gini1,
           gini2,
           gini6,
           gini7,
           gini8,
           gini10,
           gini12,
           gini13,
           gini_FINAL)

Accuracy=rbind(Accuracy1,
               Accuracy2,
               Accuracy6,
               Accuracy7,
               Accuracy8,
               Accuracy10,
               Accuracy12,
               Accuracy13,
               Accuracy_FINAL)

ERROR= rbind(error1,
             error2,
             error6,
             error7,
             error8,
             error10,
             error12,
             error13,
             error_FINAL)

SENSIBILIDAD=rbind(Sensitivity1,
                   Sensitivity2,
                   Sensitivity6,
                   Sensitivity7,
                   Sensitivity8,
                   Sensitivity10,
                   Sensitivity12,
                   Sensitivity13,
                   Sensitivity_FINAL)

resultado=data.frame(AUC,GINI,Accuracy,ERROR,SENSIBILIDAD)
rownames(resultado)=c('Logistico',
                      'KNN',
                      'Arbol_c50',
                      'SVM_Radial',
                      'SVM_Linear',
                      'Random_Forest',
                      'Boosting',
                      'XGboosting',
                      'Ensamblado Total (8 modelos)'
)
resultado=round(resultado,2)
resultado

## Resultado Ordenado #####

# ordenamos por el Indicador que deseamos, quiza Accuracy en forma decreciente

Resultado_ordenado <- resultado[order(-Accuracy),] 
Resultado_ordenado

#######---------------------------------------------------
## enviando resultados mejores modelos

#7: SVM RADIAL

length(test)
head(test)
dim(test)

test_final <- test %>% select(edad, meses_cod, max_ant_cod, ingreso_cod, score,  zona, nivel_cod)

proba7_f<-predict(modelo7, newdata=test_final,decision.values = TRUE, probability = TRUE) 
proba7_f=attributes(proba7_f)$probabilities[,2]

PRED_7 <-predict(modelo7,test_final,type="class")

write.csv(PRED_7, "SVM RADIAL RIESGO.CSV")

#13: RANDOM FOREST

proba10_F<-predict(modelo10, newdata=test_final,type="prob")
proba10_F=proba10_F[,2]

# Calcular los valores predichos
PRED_10F <-predict(modelo10,test_final,type="class")

write.csv(PRED_10F, "RANDOM FOREST RIESGO.CSV")



#1 LOGISTICO ----------- FINAL

proba1_F=predict(modelo1, newdata=test_final,type="response")

hist(proba1)

# Calcular los valores predichos
PRED_1s <-predict(modelo1,test_final,type="response")

hist(PRED_1f)
PRED_1f=ifelse(PRED_1s>=0.87,1,0)
PRED_1f=as.factor(PRED_1f)

table(PRED_1f)

write.csv(PRED_1f, "LOGISTICO RIESGO _2.CSV")

