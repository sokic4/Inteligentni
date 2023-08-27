
data <- read.csv("cars.csv", stringsAsFactors = F)

str(data)
summary(data)
# vidimo da year ima 2 NA vrednosti, a za character varijable cemo proveriti
# da li mozemo da ih pretvorimo u faktor/numericku ili izbacimo

# proveravamo za price da li ima neke nedostajuce vrednosti
# jer ako ima, onda moramo te redove da uklonimo jer koristimo price
# za kreiranje izlazne varijable
sum(is.na(data$price))
sum(data$price == "-")
sum(data$price == "")
sum(data$price == " ")

# trazimo vrednost 85og percentila
percentil85 <- quantile(data$price, 0.85)
percentil85

# vrednost yes imaju sve observacije s price varijablom iznad 85og percentila
# ostale dobijaju vrednost no
data$High_Price <- ifelse(data$price > percentil85, yes = "yes", no = "no")

# izbacujemo price jer smo nju koristili da bismo kreirali novu izlaznu varijablu
# a High_Price pretvaramo u factor
data$price <- NULL
data$High_Price <- as.factor(data$High_Price)

# sredjujemo podatke
str(data)
# X je redni broj i nece uticati na izlaznu varijablu, njega cemo izbaciti
# marku i model cemo pretvoriti u faktor jer je logicno da ce one uticati na cenu
# generation_name cemo izbaciti jer imamo informaciju o godini proizvodnje
# godina proizvodnje, kilometraza, zapremina motora, tip goriva
# ce takodje uticati na cenu, njih pretvaramo u kategoricke varijable
# grad i provincija nam nisu potrebne jer ne uticu na cenu automobila kao ostale varijable
data$X <- NULL
data$generation_name <- NULL
data$city <- NULL
data$province <- NULL

# Naive Bayes radi samo sa faktorskim varijablama i numerickim koje imaju normalnu raspodelu
# sad cemo proveriti da li numericke imaju normalnu raspodelu, jer ako nemaju
# onda moramo da ih diskretizujemo da bismo ih koristili za dalje analize

# pre nego sto to uradimo cemo da izbacimo sve NA vrednosti
apply(data, MARGIN = 2, FUN = function(x) sum(is.na(x)))
apply(data, MARGIN = 2, FUN = function(x) sum(x == "-", na.rm = T))
apply(data, MARGIN = 2, FUN = function(x) sum(x == "", na.rm = T))
apply(data, MARGIN = 2, FUN = function(x) sum(x == " ", na.rm = T))
# zapremina motora ima jednu '-' i jednu NA vrednost
# godina ima 2 NA vrednosti


# zapreminu motora i godinu menjamo srednjom vrednoscu ili medijanom u zavisnosti 
# od toga da li imaju normalnu raspodelu ili ne

data$vol_engine[data$vol_engine == "-" | is.na(data$vol_engine)] <- NA
data$vol_engine <- as.numeric(data$vol_engine)

length(unique(data$mark))
length(unique(data$model))
length(unique(data$vol_engine))
length(unique(data$fuel))

data$mark <- as.factor(data$mark)
data$model <- as.factor(data$model)
data$fuel <- as.factor(data$fuel)

levels(data$mark)
levels(data$model)
levels(data$vol_engine)
levels(data$fuel)

# sredjujemo vol_engine
shapiro.test(head(data$vol_engine, n = 5000))
# nema normalnu raspodelu, pa menjamo medijanom

medianVol <- median(data$vol_engine, na.rm = TRUE)
data$vol_engine[is.na(data$vol_engine)] <- medianVol

# sad sredjujemo godinu
shapiro.test(head(data$year, n = 5000))
# nema normalnu raspodelu, pa menjamo medijanom

medianYear <- median(data$year, na.rm = TRUE)
data$year[is.na(data$year)] <- medianYear

shapiro.test(head(data$mileage, n = 5000))
shapiro.test(head(data$vol_engine, n = 5000))
# ni godina ni kilometraza ni kubikaza nemaju normalnu raspodelu, pa moramo da ih diskretizujemo
# funkcija discretize radi sa numerickim varijablama, pa moramo int da pretvorimo u num

str(data)
# year i vol_engine su vec numericke, samo mileage pretvaramo u numeric jer
# discretize funkcija prima numericki dataframe

data$mileage <- as.numeric(data$mileage)

library(bnlearn)
discretized <- discretize(data[,c(3,4,5)],
                          method = "quantile",
                          breaks = c(5,5,3))
summary(discretized)

newData <- as.data.frame(cbind(discretized, data[,c(1,2,6,7)]))
# spojimo dataframe sa diskretizovanim varijablama
# sa varijablama iz originalnog dataframea koje imaju normalnu raspodelu ili su faktorske
# u newData

str(newData)
# dodajemo izlaznu varijablu u newData
newData$High_Price <- data$High_Price

# sad su nam ostale sve faktor varijable i mozemo da nastavimo dalje

# train i test
library(caret)
set.seed(1010)
indexes <- createDataPartition(newData$High_Price, p= 0.80, list = F)
train.set <- newData[indexes,]
test.set <- newData[-indexes,]

# sad pravimo model
# install.packages("e1071")
library(e1071)
nb1 <- naiveBayes(High_Price ~ ., data = train.set)
nb1
# A-priori probability pokazuje prave verovatnoce
# u skoro 15% slucajeva ce biti visoka cena, a u 85% nece

nb1.pred <- predict(nb1, newdata = test.set, type = "class")
nb1.pred
# ako umesto type = "class" (ovako dobijamo klase kao predikcije)
# stavimo "raw", onda dobijamo konkretne verovatnoce za cheap i not_cheap
# model ce naravno da izabere vecu verovatnocu kao resenje
# mi cemo kasnije moci da podesavamo threshold, odnosno
# birati odredjenu klasu samo ako je verovatnoca veca od vrednosti
# koju smo mi zadali

nb1.cm <- table(true = test.set$High_Price, predicted = nb1.pred)
nb1.cm
# po matrici konfuzije vidimo da smo izuzetno veliki broj vrednosti pogodili
# pa ce nam sve metrike biti visoke
# na glavnoj dijagonali nam se nalaze tacno predvidjene vrednosti
# i mi smo od ukupno 1999 observacija tacno predvideli cak 1876
# 92 smo pogresno predvideli da ce biti visoka cena, a zapravo nece, 
# pa ce nam precision biti veci, ali nesto manji od recalla jer smo 
# samo 31 predvideli da ce biti niska cena umesto visoka

# kreiramo funkciju evaluacija
getEvaluationMetrics <- function(cm) {
  
  TP <- cm[2,2] # true positive
  TN <- cm[1,1] # true negative
  FP <- cm[1,2] # false positive
  FN <- cm[2,1] # false negative
  
  accuracy <- sum(diag(cm)) / sum(cm)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1 <- (2 * precision * recall) / (precision + recall)
  
  c(Accuracy = accuracy, 
    Precision = precision, 
    Recall = recall, 
    F1 = F1)
}

nb1.eval <- getEvaluationMetrics(nb1.cm)
nb1.eval

# accuracy = procenat tacnih predikcija, ovde smo od ukupnog broja observacija
# u test setu, sto je 1999, tacno predvideli 1876, pa nam je tacnost visoka, odnosno 92.79%

# precision = udeo onih koje smo predvideli da su pozitivne koje su stvarno pozitivne
# ovde smo od 358 automobila za koja smo rekli da imaju visoku cenu
# tacno predvideli 266, 
# a za 92 smo rekli da imaju visoku cenu 
# umesto da nemaju, pa nam je precision 74.3%

# recall = udeo observacija koje su stvarno pozitivne koje smo predvideli da su pozitivne
# od ukupno 297 automobila sa visokom cenom smo tacno predvideli 266, a 31 smo pogresili i
# rekli da nemaju visoku cenu, zato nam je i recall 89.56%

# F1 = sluzi za evaluaciju modela kada su precision i recall u balansu, 
# govori koliko je dobar model, ovde nam je vrednost F1 statistike 81.22%,
# sto znaci da je nas model solidan

# sad trazimo threshold preko ROC krive
# to je optimalna verovatnoca za specificity i sensitivity
# onda se pravi nova predikcija za ROC krivu, ali TYPE = RAW
# da bismo videli tacne verovatnoce za svaku klasu
nb2.pred.prob <- predict(nb1, newdata = test.set, type = "raw")
nb2.pred.prob

# kreiranje ROC krive, kucaj PROC u cheatsheetu

# install.packages("pROC")
library(pROC)
nb2.roc <- roc(response = as.integer(test.set$High_Price),
               predictor = nb2.pred.prob[,2],
               levels = c(1,2))

# response je izlazna varijabla, ali vrednost treba da bude integer vrednost
# za predictor vrednost dajemo verovatnocu pozitivne klase, odnosno prva kolona
# levels je uredjen da prvo ide negativna, pa pozitivna klasa

# sensitivity odgovara recallu (TPR - true positive rate)
# u odnosu na sve automobile koji imaju visoku cenu, koji je udeo onih koje smo 
# mi predvideli da imaju visoku cenu

# specificity je isto samo se odnosi na negativnu klasu (FPR - false positive rate)
# u odnosu na sve automobile koji imaju nisku cenu, koji je udeo onih koje smo
# mi predvideli da imaju nisku cenu

nb2.roc$auc
# sto je AUC - area under the curve veca, to se klasifikator smatra boljim,
# 0.9753 je u nasem slucaju (znaci da je izuzetno dobar model, sve preko 0.9 se smatraju bas dobrim modelima)
# sada pravimo plot da bismo videli threshold

plot.roc(nb2.roc, print.thres = TRUE, print.thres.best.method = "youden")
# print.thres je TRUE da bi se ispisao najbolji threshold
# youden method bira threshold gde je suma specificity i sensitivity maximalna
# treshold je 0.156, specificity je 0.901, a sensitivity je 0.956

# coords u CHEATSHEET-u, da nadjemo koordinate naseg thresholda
nb2.coords <- coords(nb2.roc, 
                     ret = c("accuracy", "spec", "sens", "thr"), 
                     x = "local maximas", transpose = FALSE)
nb2.coords

# vraca koordinate za nasu nb2.roc 
# ret znaci sta da nam vrati od parametara (ima ih mnogo, ove 4 su najbitnije)
# local maximas znaci da vrati sve lokalne maximume (tacke na roc krivi)
# transpose uvek stavljamo FALSE zbog prikaza rezultata nb2.coords
# (da ne bude transponovano)

# ovo radimo da bismo izabrali najbolji threshold da maximiziramo 
# specificity i sensitivity (ovo se trazi u zadatku, a mogu da vam daju i 
# da nadjete threshold samo za najveci specificity npr.)

# na slici vidimo da nam je  treshold je 0.156, 
# specificity je 0.901, a sensitivity je 0.956 
# to nam je nb2.coords[8,4] i mozemo to da koristimo prema slici
# medjutim mozemo i sami da biramo u zavinosti od zadatka
# ako trazi da specificity bude sto veci ili sensitivity sto veci
# ovde trazimo threshold gde je njihova suma najveca, zato uzimamo 8,4
prob.threshold <- nb2.coords[7,4]
prob.threshold

# sad radimo predikciju sa novim thresholdom
nb2.pred <- ifelse(test = nb2.pred.prob[,2] >= prob.threshold, 
                   yes = "yes", no = "no")

# stavili smo nb2.pred.prob[,2] jer nam je pozitivna klasa u drugoj koloni

# pretvaramo u faktor da ne bude vektor karaktera
nb2.pred <- as.factor(nb2.pred)

nb2.cm <- table(true = test.set$High_Price, predicted = nb2.pred)
nb1.cm
nb2.cm

nb2.eval <- getEvaluationMetrics(nb2.cm)
nb1.eval
nb2.eval

data.frame(rbind(nb1.eval, nb2.eval), row.names = c("one", "two"))

# accuracy i precision su nam gori nego u prethodnom modelu, pogotovo precision
# dakle ovde smo za 169 automobila rekli da imaju visoku cenu, a zapravo nemaju
# a u prethodnom modelu smo isto pogresili za 92 automobila

# recall se povecao, pogresili smo samo za 13 automobila da nemaju skupu cenu, 
# a u proslom modelu 31, ali nam je F1 statistika slabija
# jer nam se znacajno smanjio precision u odnosu na prethodni model
# dakle nas novi model ima samo veci recall, ostale metrike su gore
# pa zakljucujemo da je prvi model sa default thresholdom 0.5
# bolji od drugog sa novim thresholdom 0.156









