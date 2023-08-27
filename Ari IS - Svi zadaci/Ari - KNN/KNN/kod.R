
##################################################################
# RADIO SAM BEZ DAYOFWEEK
# ZA DOMACI UBACITE I DAYOFWEEK I VIDITE KAKO SE PROMENIO NAS MODEL
# NA KRAJU BI TREBALO OVO DA DOBIJETE AKO STE LEPO ODRADILI SA ISTIM SEEDOM
# Accuracy Precision    Recall        F1 
# 0.6750000 0.7142857 0.6818182 0.6976744
##################################################################

# SVE VARIJABLE ZA KNN MORAJU DA BUDU NUMERICKE, A IZLAZNA VARIJABLA FAKTORSKA !!!
# NAKON SREDJIVANJA DATASETA MORAMO DA STANDARDIZUJEMO NUMERICKE VARIJABLE !
# AKO JE p > 0.05 onda je normalna raspodela, ako je p < 0.05 onda nije !

data <- read.csv("travel-times.csv", stringsAsFactors = F)

str(data)
summary(data)

# sve varijable moraju da budu numericke i
# moramo da ih standardizujemo, odnosno dovedemo u slican 
# opseg vrednosti da bismo izvrsili KNN algoritam
# neke faktorske cemo nekad pretvarati u numericke, a neke cemo totalno izbaciti
# varijable Date, StartTime, DayOfWeek
# nisu numericke, niti su nam potrebne za model
# kao u prethodnom zadatku stabla, tako da cemo ih izbaciti

length(unique(data$Date))
length(unique(data$StartTime))
length(unique(data$DayOfWeek))
length(unique(data$GoingTo))
# vidimo da Date i StartTime imaju previse razlicitih vrednosti
# pa cemo njih 100% izbaciti
# a sto se tice DayOfWeek, to cemo sad izbaciti, a za domaci cete sami ubaciti
# i videti kako se menja vas model
data$Date <- NULL
data$StartTime <- NULL
data$DayOfWeek <- NULL


apply(data, MARGIN = 2, FUN = function(x) sum(is.na(x)))
apply(data, MARGIN = 2, FUN = function(x) sum(x == "-", na.rm = T))
apply(data, MARGIN = 2, FUN = function(x) sum(x == "", na.rm = T))
apply(data, MARGIN = 2, FUN = function(x) sum(x == " ", na.rm = T))

# 2 '-' ima FuelEconomy
# Prazne stringove ima GoingTo, FuelEconomy i Comments. 
# Comments ima previse nedostajucih vrednosti i treba ukloniti kolonu
# ali cemo to uraditi kasnije jer nam treba za izlaznu varijablu

# opet radimo isto kao prosli put za GoingTo
# NA vrednosti pretvaramo u Work jer njega ima vise
# ali ovaj put moramo faktor da pretvorimo u numeric
# jer nam je to potrebno za izvrsavanje KNN algoritma
data$GoingTo[data$GoingTo == ""] <- NA
table(data$GoingTo)
data$GoingTo[is.na(data$GoingTo)] <- "Work"
data$GoingTo <- as.numeric(as.factor(data$GoingTo))

# pretvaramo nedostajuce vrednosti za FuelEconomy u NA i, posto je to character
# varijabla trenutno, moramo da je pretvorimo u numericku zbog KNN algoritma
# mozemo i samo da kazemo as.numeric(data$FuelEconomy) i on ce sam da pretvori
# sve vrednosti u NA
data$FuelEconomy[data$FuelEconomy == "" | data$FuelEconomy == "-"] <- NA
class(data$FuelEconomy)
data$FuelEconomy <- as.numeric(data$FuelEconomy)

# proveravamo kakvu raspodelu ima FuelEconomy da bismo videli s kojom
# vrednoscu da zamenimo NA vrednosti
shapiro.test(data$FuelEconomy)
# nema normlanu raspodelu, pa NA menjamo medijanom

medijanaFuelEco <- median(data$FuelEconomy, na.rm = TRUE)
data$FuelEconomy[is.na(data$FuelEconomy)] <- medijanaFuelEco

str(data)
# u zadatku nam kaze da pravimo novu varijablu Take407All koja uzima
# vrednost YES ako je vrednost Congestion407 manja od vrednosti na 
# 60-tom percentilu, a NO ako je veca
# ako koristimo funkciju summary(data), 
# dobicemo min, max, 1. kvartil, medijanu, sredinu, 3. kvartil
# ali ne znamo koji je 60ti percentil, za to koristimo funkciju
# quantile koje ima u CHEATSHEETu
percentil60 <- quantile(data$Congestion407, 0.6)
percentil60
data$Take407All <- ifelse(data$Congestion407 < percentil60 
                          & data$Comments == "", yes = "Yes", no = "No")
# sad mozemo da izbacimo Congestion407 i Comments jer smo ih iskoristili
# za kreiranje izlazne varijable i nisu nam vise potrebne
data$Congestion407 <- NULL
data$Comments <- NULL
str(data)
# Take407 je character varijabla, pretvaramo je u factor, s njom ne radimo
# ona je izlazna, prediktor varijable moraju da budu numericke
data$Take407All <- as.factor(data$Take407All)

# sada su nam sve varijable numericke, jedino sto nam preostaje
# je da ih standardizujemo, dovedemo u slican opseg vrednosti kako
# bismo izvrsili nas KNN algoritam

########################################
########################################
# Pre nego sto krenemo sa autlajerima, prvo da vidite sta je to
# i kako smo dosli do toga, za primer cu uzeti jednu varijablu:
# boxplot(data$MaxSpeed) - crta grafik sa 5 tacaka: min, 1. kvartil, medijana, 3. kvartil, max
# boxplot.stats(data$MaxSpeed) - prikazuje statistiku varijable, dobili smo:


# $stats      - ovo je upravo ovih 5 tacaka iznad, respektivno
# [1] 119.0 124.9 127.4 129.8 137.1

# $n          - ovo je broj observacija
# [1] 205
#
# $conf       - confidence interval, interval poverenja za vrednost medijane
# [1] 126.8593 127.9407

# $out        - ovo su konkretni autlajeri, ima ih 6
# [1] 137.8 112.2 114.4 140.9 138.0 137.7
# length(boxplot.stats(data$MaxSpeed)$out) - vraca koliko autlajera imamo

# Sada kad smo prosli ovu funkciju da razumete sta je, iskoristicemo 
# funkciju apply kako bi je izvrsili na sve kolone i vidimo
# da li ima autlajera
########################################
########################################

# kad izvrsimo sledecu funkciju, pise nam za svaku varijablu
# koliko ima outlajera
apply(X = data[,2:8], 2, FUN = function(x) length(boxplot.stats(x)$out))
# posto ima outlajera onda obavezno moramo da uradimo standardizaciju
# na koji nacin zavisi da li varijabla ima ili nema normalnu raspodelu
apply(X = data[,2:8], 2, FUN = function(x) shapiro.test(x))
# nijedna nema normalnu raspodelu

# sve p vrednosti su manje od 0.05, pa je center = median(x) scale = IQR(x), x nam je ta kolona
# ako je neka veca od 0.05, odnosno ima normalnu raspodelu onda pisemo center = TRUE scale = TRUE
# center = TRUE znaci da uzima MEAN, a scale = TRUE da deli sa standardnom devijacijom
# IQR = Interquartile range vam je razlika izmedju 3. i 1. kvartila 
# naravno izbacite kolone koje ne trebaju u odredjenoj funkciji
# data.std je u pocetku matrica, zato mi moramo da ga pretvorimo u dataframe
# da bismo nad njim testirali algoritam
data.std <- apply(X = data[,2:8], 2, FUN = function(x) scale(x, center = median(x), scale = IQR(x)))
data.std <- as.data.frame(data.std)
# sad dodamo Take407All jer je ona ta poslednja, factor varijabla
data.std$Take407All <- as.factor(data$Take407All)
# ako zelimo da specificiramo kojim redosledom idu leveli
# onda mozemo da koristimo factor umesto as.factor, npr.
# factor(data$Take407All, levels = ("No", "Yes"))
# onda ce ici prvo No, pa onda Yes, a sa as.factor sam bira

# dodamo faktorsku kao integer, nije potrebno da je skaliramo jer je binarna
data.std$GoingTo <- as.integer(data$GoingTo)
# promenimo redosled da nam poslednja bude izlazna
data.std <- data.std[,c(9,1:8)]

# ?scale
# PRIMER KAD IMA I NEKIH SA NORMALNOM RASPODELOM, SAMO DODAMO KOLONE NA SLEDECI NACIN:
# data.std$Kolona <- as.vector(scale(data$Kolona, center = TRUE, scale = TRUE)) | scale(x, center = mean(x), scale = sd(x))
# as.vector jer nam inicijalno vraca matricu, a mi ne zelimo matricu u koloni

str(data.std)
summary(data.std)

# sada, kao sto vidite u data.std dataframe-u, imamo standardizovane
# vrednosti i mozemo da idemo dalje



# kad smo zavrsili sa standardizacijom, pravimo train i test setove
library(caret)
set.seed(1010)
indexes <- createDataPartition(data.std$Take407All, p = 0.8, list = FALSE)
train.data <- data.std[indexes, ]
test.data <- data.std[-indexes, ]

# krosvalidacija za 10 iteracija
# nacin za pronalazenje sto boljih vrednosti za parametar
# koji nam je potreban
# za krosvalidaciju uvek e1071 !!!
library(e1071)
library(caret)
numFolds <- trainControl(method = "cv", number = 10) 
# za sledecu funkciju smo izabrali neparne brojeve od 3 do 25
# jer nam je K koliko najblizih vrednosti gledamo
# i mora da bude neparan da bi jedna klasa bila dominantnija
# u odnosu na drugu
kGrid = expand.grid(.k = seq(from = 3, to = 25, by = 2))
# OBAVEZNO SET.SEED !!!
# funkcija ispod je kao rpart kod klasifikacionih stabala
# samo za method pisemo knn, za trControl numFolds koji smo dobili iznad
# i za tuneGrid kGrid koji smo dobili iznad

set.seed(1010)

# KUCAS "train(" U CHEATSHEETU
knn.cv <- train(x = train.data[,-9],
                y = train.data$Take407All,
                method = "knn",
                trControl = numFolds,
                tuneGrid = kGrid)

knn.cv
plot(knn.cv)
# dobili smo da je najbolji value za k = 19, to mozemo uzeti linjom koda ispod

best_k <- knn.cv$bestTune$k

# KUCAS KNN U CHEATSHEETU
library(class)
knn.pred <- knn(train = train.data[,-9], # training data without the output (class) variable
                test = test.data[,-9], # test data without the output (class) variable
                cl = train.data$Take407All, # output (class) variable is specified here
                k = best_k)

# Yes nam je pozitivna klasa
getEvaluationMetrics <- function(cm){
  
  TP <- cm[2,2]
  TN <- cm[1,1]
  FP <- cm[1,2]
  FN <- cm[2,1]
  
  accuracy <- sum(diag(cm)) / sum(cm)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1 <- (2*precision*recall) / (precision + recall)
  
  c(Accuracy = accuracy, 
    Precision = precision, 
    Recall = recall, 
    F1 = F1)
  
}

knn.cm <- table(true = test.data$Take407All, predicted = knn.pred)
knn.cm
# vidimo da ce nam model imati slabije metrike
# jer je tacno predvideo samo 26/40 observacija
# vidimo da nam su nam vrednosti na sporednoj dijagonali iste
# tako da zakljucujemo da ce nam precision, recall i F1 statistika
# biti iste vrednosti

knn.eval <- getEvaluationMetrics(knn.cm)
knn.eval

# accuracy = procenat tacnih predikcija, ovde smo od ukupnog broja observacija
# u test setu, sto je 40, tacno predvideli 26, pa nam je tacnost niza, odnosno 0.65

# precision = udeo onih koje smo predvideli da su pozitivne koje su stvarno pozitivne
# ovde smo od 22 koje smo rekli da su pozitivne, pogodili 15, odnosno da treba ici autoputem,
# a za 7 smo rekli da treba, a zapravo ne treba, pa nam je precision 0.681

# recall = udeo observacija koje su stvarno pozitivne koje smo predvideli da su pozitivne
# ista nam je vrednost kao za precision jer su iste vrednosti za FP i FN
# dakle od 22 observacije gde treba ici autoputem, rekli smo da treba ici za 15 sto je tacno,
# a za 7 smo rekli da ne treba, a u stvari treba

# F1 = sluzi za evaluaciju modela kada su precision i recall u balansu, 
# govori koliko je dobar model, ovde nam je vrednost F1 statistike 0.681,
# isto kao precision i recall jer su oni potpuno jednaki

# ZA DOMACI UBACITE I DAYOFWEEK I VIDITE KAKO SE PROMENIO NAS MODEL
# NA KRAJU BI TREBALO OVO DA DOBIJETE AKO STE LEPO ODRADILI SA ISTIM SEEDOM
# Accuracy Precision    Recall        F1 
# 0.6750000 0.7142857 0.6818182 0.6976744 