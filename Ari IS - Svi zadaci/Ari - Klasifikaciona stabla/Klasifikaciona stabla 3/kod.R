

data <- read.csv("wines.csv", stringsAsFactors = F)

str(data)

# izbacio sam ovu vrednost jer nam redni broj naravno nece biti potreban
data$X <- NULL

tempSub <- subset(data, (data$country == "Argentina" 
                      | data$country == "France" | data$country == "Italy"))

apply(tempSub, MARGIN = 2, FUN = function(x) sum(is.na(x)))
apply(tempSub, MARGIN = 2, FUN = function(x) sum(x == "-", na.rm = T))
apply(tempSub, MARGIN = 2, FUN = function(x) sum(x == "", na.rm = T))
apply(tempSub, MARGIN = 2, FUN = function(x) sum(x == " ", na.rm = T))

# Designation ima 1065 praznih stringova, region ima 9
# nijedna varijabla nema '-'
# Price ima 39 NA vrednosti i te redove moramo ukloniti
# jer se od varijable Price kreira izlazna varijabla Price_Category

# izbacujemo sve observacije sa nepoznatom varijablom Price
data <- tempSub[complete.cases(tempSub[,5]),]

length(unique(data$designation))
# posto designation ima 1859 razlicitih vrednosti, a ukupan broj observacija
# je 3388 nema poente da je pretvorimo u faktor zato cemo je ukloniti
data$designation <- NULL

length(unique(data$region))
# ima 458 sto je mnogo za faktor varijablu, pa i region takodje brisemo
data$region <- NULL

data$description <- NULL # description nema uticaja na dalje analze jer ne utice na cenu

# Country ima samo 3 razlicite vrednosti, jer smo od toga rekli da nam se sastoji subset
# pa cemo je pretvoriti u faktor
data$country <- factor(data$country)
levels(data$country)

length(unique(data$title))
# Title takodje ima previse razlicitih vrednosti, pa i njega uklanjamo
data$title <- NULL

length(unique(data$province))
# 23 nivoa, mozemo da ostavimo
data$province <- as.factor(data$province)

str(data)

length(unique(data$variety))
# variety takodje ima previse razlicitih vrednosti, pa i njega uklanjamo
data$variety <- NULL

length(unique(data$winery))
# winery takodje ima previse razlicitih vrednosti, pa i njega uklanjamo
data$winery <- NULL

str(data)

percentil30 <- quantile(data$price, 0.3)

data$price_category <- ifelse(data$price <= percentil30, 
                                 yes = "cheap", no = "not_cheap")

# price nam sada vise nije potrebna jer smo napravili izlaznu
# price category
data$price <- NULL
data$price_category <- factor(data$price_category)

str(data)

# zavrsili smo sa sredjivanjem podataka


# pravimo trening i test setove
library(caret)
set.seed(1010)
indexes <- createDataPartition(data$price_category, p = 0.8, list = FALSE)
train.data <- data[indexes, ] 
test.data <- data[-indexes, ] 

# sada kreiramo klasifikaciono stablo

library(rpart)
tree1 <- rpart(price_category ~ ., 
               data = train.data,
               method = "class")
tree1

# kao sto vidimo, uzeo je samo MovingTime kao najdominantniji prediktor

library(rpart.plot)
rpart.plot(tree1, extra = 104) # extra 104 pokazuje brojke na odredjen nacin

# sledece sto radimo je pravimo predikciju
tree1.pred <- predict(tree1, newdata = test.data, type = "class")

# sada pravimo matricu konfuzije

# na glavnoj dijagonali matrice konfuzije nam se nalazi broj tacnih
# predikcija, a na sporednoj broj pogresnih predikcija 
tree1.cm <- table(true = test.data$price_category, predicted = tree1.pred)
tree1.cm

# vidimo da ce nam metrike biti solidne, a da 
# je precision i recall skoro isti


# Dobili smo matricu konfuzije
#             predicted
# true        cheap not_cheap
# cheap       128        75
# not_cheap    74       400

# napisemo funkciju za evaluaciju i odradimo je na cm
# OVO TAKODJE SAMI UCITE DA PISETE! 
# ISPOD JE PRIMER KAD JE YES POZITIVNA KLASA
# KAD JE NO ONDA OBRNEMO INDEKSE, STAVIO SAM ISPOD
getEvaluationMetrics <- function(cm) {
  # levo je kad je YES pozitivna
  # desno je kad je NO pozitivna
  TP <- cm[1,1] 
  TN <- cm[2,2] 
  FP <- cm[2,1] 
  FN <- cm[1,2] 
  
  accuracy <- sum(diag(cm)) / sum(cm) # tacno predvidjene / sve
  precision <- TP / (TP + FP)      # tacno predvidjenje pozitivne / sve predvidjene pozitivne (prva kolona ili druga u zavisnosti od pozitivne klase)
  recall <- TP / (TP + FN)         # tacno predvidjenje pozitivne / prvi ili drugi red u zavisnosti od pozitivne klase
  F1 <- (2 * precision * recall) / (precision + recall)
  
  c(Accuracy = accuracy, 
    Precision = precision, 
    Recall = recall, 
    F1 = F1)
  
}

eval.tree1 <- getEvaluationMetrics(tree1.cm) 
eval.tree1
# accuracy = procenat tacnih predikcija, ovde smo od ukupnog broja observacija
# u test setu, sto je 677, tacno predvideli 528, pa nam je tacnost visoka

# precision = udeo onih koje smo predvideli da su pozitivne koje su stvarno pozitivne
# ovde smo od 202 koje smo predvideli da su pozitivne, tacno predvideli 128, odnosno
# da su jeftina vina, a za 74 smo pogresili i rekli da su jeftina, a zapravo nisu
# pa nam je precision 0.634

# recall = udeo observacija koje su stvarno pozitivne koje smo predvideli da su pozitivne
# ovde od ukupno 203 stvarno jeftina vina, 128 predvideli tacno, a za 75 smo rekli
# da nisu jeftina, a zapravo jesu, pa nam je recall 0.63

# F1 = sluzi za evaluaciju modela kada su precision i recall u balansu, 
# govori koliko je dobar model, u nasem slucaju je 0.632, pa cemo pokusati
# da napravimo bolji primenom krosvalidacije

##################################
##################################

# poslednji deo cross validacija: kucaj <folds> u cheatsheetu
library(e1071)
library(caret)

# radimo 10-fold crossvalidation
numFolds <- trainControl(method = "cv", number = 10) 

# gledamo koja je cp vrednost se pokazala najbolje za nas model
cpGrid <- expand.grid(.cp = seq(from = 0.001, to = 0.05, by = 0.001)) 

set.seed(1010)
crossvalidation <- train(x = train.data[,-4],
                         y = train.data$price_category,
                         method = "rpart", 
                         trControl = numFolds, # numFolds sto smo dobili iznad
                         tuneGrid = cpGrid) # cpGrid sto smo dobili iznad

crossvalidation

# dobili smo da je najbolji cp = 0.013, to cemo iskoristiti za nase novo drvo
# pa uporediti vrednosti
plot(crossvalidation)

# direktno uzimamo cp iz krosvalidacije
cpValue <- crossvalidation$bestTune$cp

# prune nam smanjuje nase drvo i pravi jednostavniji model
# poenta je da napravimo sto jednostavnije drvo sa sto
# boljim evaluacionim metrikama
# prune prima kao parametre staro drvo i novi cp
# a cp koji smo dobili krosvalidacijom je 0.05
# posle toga samo  napravimo novu predikciju za nase novo stablo
# napravimo novu matricu konfuzije, izracunamo metrike
# i uporedjujemo sa vrednostima prethodnog ili prethodnih stabala
# u ovom slucaju necemo raditi prune, jer je nase drvo vec najjednostavnije
# moguce, zato cemo napraviti novo samo sa drugom vrednoscu complexity parametra
# tree2 <- prune(tree1, cp = cpValue)
tree2 <- rpart(price_category ~ ., 
               data = train.data,
               method = "class", 
               control = rpart.control(cp = cpValue))

# pravimo predickije
tree2.pred <- predict(tree2, newdata = test.data, type = "class")

# pravimo konfuzionu matricu za drugi model
tree2.cm <- table(true = test.data$price_category, predicted = tree2.pred) # OVO NEMA U CHEATSHEETU
tree2.cm

# dobili smo isto, tako da ce nam vrednosti metrika biti totalno iste

eval.tree2 <- getEvaluationMetrics(tree2.cm)

eval.tree1
eval.tree2

# sa sledecom linijom koda ispisujemo i uporedjujemo vrednosti na lep nacin
data.frame(rbind(eval.tree1, eval.tree2), row.names = c("prvi","drugi"))

# kao sto vidimo, nase metrike se nisu promenile
# tako da je nas prvi model zapravo bio savrsen
# da su drugacije metrike samo biste ih prokomentarisali
# i rekli koji model je na kraju bolji



