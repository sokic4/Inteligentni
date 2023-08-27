# RADI SA FAKTOR VARIJABLAMA!
# RADI I SA NUMERICKIM AKO IMAJU NORMALNU RASPODELU, 
# A AKO NEMAJU, MORAJU DA SE DISKRETIZUJU!

# POSTOJE 2 VRSTE DISKRETIZACIJE:
# 1. DISKRETIZACIJA U INTERVALE JEDNAKE DUZINE (intervalna diskretizacija)
# 2. DISKRETIZACIJA U INTERVALE ISTE FREKVENCIJE (kvantilna diskretizacija)
# kvantilna obicno daje bolje rezultate, pa cemo nju ovde koristiti

data <- read.csv("wines.csv", stringsAsFactors = FALSE)
str(data)
# ovaj dataframe sam morao sam da popravim jer nemamo konkretan CSV fajl
# R iz nekog razloga dodaje kolonu X koja oznacava redni broj observacije
# pokusao sam da izbrisem, ali neuspesno
# zbog toga radim sledecu liniju koda, da izbacim to X odmah, pa normalno radimo zadatak
data$X <- NULL

str(data)

dataSub <- subset(data, (country == "France" | country == "Argentina" | country == "Italy"))
summary(dataSub)

apply(dataSub, MARGIN = 2, FUN = function(x) sum(is.na(x)))
apply(dataSub, MARGIN = 2, FUN = function(x) sum(x == "-", na.rm = T))
apply(dataSub, MARGIN = 2, FUN = function(x) sum(x == "", na.rm = T))
apply(dataSub, MARGIN = 2, FUN = function(x) sum(x == " ", na.rm = T))

# Designation ima 1065 praznih stringova, region ima 9
# nijedna varijabla nema '-'
# Price ima 39 NA vrednosti i te redove moramo ukloniti
# jer se od varijable Price kreira izlazna varijabla Price_Category

# izbacujemo sve observacije sa nepoznatom varijablom Price
dataSub <- dataSub[complete.cases(dataSub[,5]),]
# moze i ovako dataSub <- dataSub[!is.na(dataSub$price),]

length(unique(dataSub$designation))
# posto designation ima 1859 razlicitih vrednosti, a ukupan broj observacija
# je 3388 nema poente da je pretvorimo u faktor zato cemo je ukloniti
dataSub$designation <- NULL

length(unique(dataSub$region))
# ima 458 sto je mnogo za faktor varijablu, pa i region takodje brisemo
dataSub$region <- NULL

dataSub$description <- NULL # description nema uticaja na dalje analze jer ne utice na cenu

# Country ima samo 3 razlicite vrednosti, jer smo od toga rekli da nam se sastoji subset
# pa cemo je pretvoriti u faktor
dataSub$country <- factor(dataSub$country)
levels(dataSub$country)

length(unique(dataSub$title))
# Title takodje ima previse razlicitih vrednosti, pa i njega uklanjamo
dataSub$title <- NULL

length(unique(dataSub$province))
# 23 nivoa, mozemo da ostavimo
dataSub$province <- as.factor(dataSub$province)

str(dataSub)

length(unique(dataSub$variety))
# variety takodje ima previse razlicitih vrednosti, pa i njega uklanjamo
dataSub$variety <- NULL

length(unique(dataSub$winery))
# winery takodje ima previse razlicitih vrednosti, pa i njega uklanjamo
dataSub$winery <- NULL

str(dataSub)

str(dataSub)
prviKvartil <- quantile(dataSub$price, 0.25)
prviKvartil
dataSub$price_category <- ifelse(dataSub$price <= prviKvartil, 
                                 yes = "cheap", no = "not_cheap")

# price nam sada vise nije potrebna jer smo napravili izlaznu
# price category
dataSub$price <- NULL
dataSub$price_category <- factor(dataSub$price_category)

str(dataSub)

# zavrsili smo sa sredjivanjem podataka, sada radimo diskretizaciju numerickih varijabli
# ako nemaju normalnu raspodelu,
# pravimo train i test setove i pravimo model

shapiro.test(dataSub$points)
# Nema normalnu raspodelu, pa radimo diskretizaciju !

# vidimo koliko vinarija upada u neki opseg poena
# ovo nije toliko bitno za ispit
library(ggplot2)
ggplot(dataSub, aes(x = points)) + geom_histogram()

dataSub$points <- as.numeric(dataSub$points)
points <- dataSub$points
points.df <- as.data.frame(points)

# pretvorili smo points u numeric data frame 
# jer funckija discretized (ova ispod)
# prima to kao parametar

# install.packages("bnlearn")
library(bnlearn)
discretized <- discretize(data = points.df,
                          method = "quantile",
                          breaks = c(5))
# diskretizujemo sve koje nemaju normalnu raspodelu
# npr. da je bilo vise kolona onda bi bilo
# discretized <- discretize(dataSub[,c(2,3,6,7,9)],
#                           method = "quantile",
#                           breaks = c(5,2,5,2,5)) 
# stavljamo da podeli u 5 intervala sve kolone, to je neki rule of thumb,
# ako javlja gresku onda promenimo broj intervala
# u novijim verzijama Ra ne bi trebalo da javi gresku
# pa sami moramo da vidimo i izmenimo broj intervala da bi bio
# sto pravilniji broj observacija u svakom intervalu
# neka varijabla nekad ima jako nepravilnu raspodelu, odnosno
# jako mali opseg vrednosti da bi napravio 5 intervala
# zato moramo da smanjimo broj intervala

# u ovom konkretnom primeru sam ostavio 5 intervala, jer kad stavimo
# drugi broj onda daje previse drugaciji broj observacija u invervalima

summary(discretized)

newData <- as.data.frame(cbind(discretized, dataSub[,c(1,3,4)]))
# spojimo dataframe sa diskretizovanim varijablama
# sa varijablama iz originalnog dataframea koje imaju normalnu raspodelu ili su faktorske
# u newData

str(newData)

# sad su nam ostale sve faktor varijable i mozemo da nastavimo dalje

# train i test
library(caret)
set.seed(1010)
indexes <- createDataPartition(newData$price_category, p= 0.80, list = F)
train.set <- newData[indexes,]
test.set <- newData[-indexes,]


# install.packages("e1071")
library(e1071)
nb1 <- naiveBayes(price_category ~ ., data = train.set)
nb1
# A-priori probability pokazuje prave verovatnoce
# u 26% slucajeva ce vino biti jeftino, a u 74% slucajeva nece biti jeftino

nb1.pred <- predict(nb1, newdata = test.set, type = "class")
nb1.pred
# ako umesto type = "class" (ovako dobijamo klase kao predikcije)
# stavimo "raw", onda dobijamo konkretne verovatnoce za cheap i not_cheap
# model ce naravno da izabere vecu verovatnocu kao resenje
# mi cemo kasnije podesiti threshold, odnosno
# birati odredjenu klasu samo ako je verovatnoca veca od vrednosti
# koju smo mi zadali

nb1.cm <- table(true = test.set$price_category, predicted = nb1.pred)
nb1.cm

# pozitivna klasa je cheap, receno u zadatku

#                       predicted
#       true       cheap        not_cheap
#       cheap        98        80
#       not_cheap    49       450

getEvaluationMetrics <- function(cm) {
  
  TP <- cm[1,1] # true positive
  TN <- cm[2,2] # true negative
  FP <- cm[2,1] # false positive
  FN <- cm[1,2] # false negative
  
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
# u test setu, sto je 677, tacno predvideli 548, pa nam je tacnost solidna, odnosno 80.9%

# precision = udeo onih koje smo predvideli da su pozitivne koje su stvarno pozitivne
# ovde smo od 147 vina za koja smo rekli da su jeftina
# pogodili 98 da jesu, a za 49 smo rekli da su jeftina
# umesto da nisu, pa nam je precision nesto nizi, odnosno 66.7%

# recall = udeo observacija koje su stvarno pozitivne koje smo predvideli da su pozitivne
# od ukupno 178 vina koja su jeftina smo tacno predvideli 98, a 80 smo pogresili i
# rekli da nisu jeftina, zato nam je i recall 55.05%

# F1 = sluzi za evaluaciju modela kada su precision i recall u balansu, 
# govori koliko je dobar model, ovde nam je vrednost F1 statistike 60.3%,
# sto je lose jer su nam niski i precision i recall, morali bismo da poboljsamo model

# sad trazimo threshold preko ROC krive
# to je optimalna verovatnoca za specificity i sensitivity
# onda se pravi nova predikcija za ROC krivu, ali TYPE = RAW
# da bismo videli tacne verovatnoce za svaku klasu

nb2.pred.prob <- predict(nb1, newdata = test.set, type = "raw")
nb2.pred.prob

# kreiranje ROC krive, kucaj PROC u cheatsheetu


# install.packages("pROC")
library(pROC)
nb2.roc <- roc(response = as.integer(test.set$price_category),
               predictor = nb2.pred.prob[,1],
               levels = c(2,1))
plot.roc(nb2.roc)

# response je izlazna varijabla, ali vrednost treba da bude integer vrednost (NE PISE U CHEATSHEETU !!!!)
# za predictor vrednost dajemo verovatnocu pozitivne klase, odnosno prva kolona
# levels (NE PISE U CHEATSHEETU !!!!!) je uredjen da prvo ide negativna, pa pozitivna klasa
# zato smo mi stavili c(2,1) jer smo prvo zadali poziciju
# negativne klase (not_cheap), pa pozitivnu pozitivne (cheap)

# sensitivity odgovara recallu (TPR - true positive rate)
# u odnosu na sve prodavnice koje imaju jeftino vino (koje su pozitivne),
# koji je udeo onih koje smo mi predvideli da imaju jeftino vino (da jesu pozitivne)

# specificity je isto samo se odnosi na negativnu klasu (FPR - false positive rate)
# u odnosu na sve prodavnice koje imaju skupo vino (koje su negativne),
# koji je udeo onih koje smo mi predvideli da imaju skupo vino (da jesu negativne)

# da je not_cheap pozitivna klasa 
# onda ide predictor = nb2.pred.prob[,2],levels = c(1,2)

nb2.roc$auc
# sto je AUC - area under the curve veca, to se klasifikator smatra boljim
# ako je 1 onda moze perfektno da razlikuje koja je pozitivna, a koja negativna klasa
# a ako je na primer 0.7 onda ima 70% sanse da razlikuje koja je pozitivna, a koja negativna
# a ako je 0.5 onda ne moze da razlikuje, to je najgora situacija
# 0.8374 odnosno 83.74% je u nasem slucaju (znaci da je dobra, sve preko 0.9 se smatraju bas dobrim modelima)
# sada pravimo plot da bismo videli threshold

plot.roc(nb2.roc, print.thres = TRUE, print.thres.best.method = "youden")
# print.thres je TRUE da bi se ispisao najbolji threshold
# youden method bira threshold gde je suma specificity i sensitivity maximalna
# treshold je 0.242, specificity je 0.756, a sensitivity je 0.764

# coords u CHEATSHEET-u, da nadjemo koordinate naseg thresholda
nb2.coords <- coords(nb2.roc, 
                     ret = c("accuracy", "spec", "sens", "thr"), 
                     x = "local maximas")
nb2.coords

# vraca koordinate za nasu nb2.roc 
# ret (return) znaci sta da nam vrati od parametara (ima ih vise, ove 4 su najbitnije)
# local maximas znaci da vrati sve lokalne maximume (tacke na roc krivi)
# transpose uvek stavljamo FALSE zbog prikaza rezultata nb2.coords
# (da ne bude transponovano)

# ovo radimo da bismo izabrali najbolji threshold da maximiziramo
# specificity i sensitivity (ovo se trazi u zadatku, a mogu da vam daju i 
# da nadjete threshold samo za najveci specificity npr.)

# prvi model ima dobar accuracy, ali nizi precision i recall, ako zelimo da povecamo recall
# potrebno je da nadjemo vrednost koja ima visoki specificity i 
# accuracy prilicno visok
# ovo je potrebno izabrati kako zelimo da pobosljamo model

# na slici vidimo da nam je  treshold je 0.242, 
# specificity je 0.756, a sensitivity je 0.764 
# to nam je nb2.coords[23,4] i mozemo to da koristimo te rezultate
# sto smo dobili sa plota
# medjutim mozemo i sami da biramo u zavinosti od zadatka
# ako trazi da specificity bude sto veci ili sensitivity sto veci
# ovde trazimo threshold gde je njihova suma najveca, zato uzimamo 23,4
prob.threshold <- nb2.coords[23,4]
prob.threshold

# sad radimo predikciju sa novim thresholdom
nb2.pred <- ifelse(test = nb2.pred.prob[,1] >= prob.threshold, 
                   yes = "cheap", no = "not_cheap")

# stavili smo nb2.pred.prob[,1] jer nam je pozitivna klasa u prvoj koloni
# u yes se upisuje vrednost nase pozitivne klase, u ovom slucaju to je cheap
# da je bila pozitivna klasa not_cheap onda bi bilo ovako
# nb2.pred <- ifelse(test = nb2.pred.prob[,2] >= prob.threshold, yes = "not_cheap", no = "cheap")

# pretvaramo u faktor da ne bude vektor karaktera
nb2.pred <- as.factor(nb2.pred)

nb2.cm <- table(true = test.set$price_category, predicted = nb2.pred)
nb1.cm
nb2.cm

nb2.eval <- getEvaluationMetrics(nb2.cm)
nb1.eval
nb2.eval

data.frame(rbind(nb1.eval, nb2.eval), row.names = c("one", "two"))

# accuracy i precision su nam gori nego u prethodnom modelu
# recall se znacajno povecao, pa je to zato povuklo i F1 statistiku da 
# se poveca
# dakle nas novi model ima slabiji accuracy i precision
# a bolji recall i f1 statistiku
