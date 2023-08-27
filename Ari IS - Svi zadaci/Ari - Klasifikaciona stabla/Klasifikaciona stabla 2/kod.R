

data <- read.csv("apps.csv", stringsAsFactors = F)

str(data)
summary(data)
# u Ratingu imamo 1474 NA vrednosti, ali to cemo resiti u 4. koraku

treciKvartil <- quantile(data$Reviews, 0.75, na.rm = T)
treciKvartil
# treci kvartil nam je 54775.5 i za sve observacije sa brojem recenzija
# veceg od navedenog cemo dati vrednost High_Reviews Yes, a obrnuto No

data$High_Reviews <- ifelse(data$Reviews > treciKvartil, yes = "Yes", no = "No")

# pretvaramo varijablu price u numeric da bismo uradili 2. korak
# takodje moramo da sredimo NA vrednosti prvo jer u starijim verzijama
# R-a ovo nece odmah raditi, a oni su u zadatku naveli drugaciji redosled
sum(is.na(data$Price))
sum(data$Price == "-", na.rm = T)
sum(data$Price == "", na.rm = T)
sum(data$Price == " ", na.rm = T)

data$Price[data$Price == "-"] <- NA
data$Price <- as.numeric(data$Price)

# sledeci shapiro test nam baca gresku sample size must be between 3 and 5000, 
# pa moramo da ogranicimo broj redova na nesto izmedju 3 i 5000 
# i onda da vidimo raspodelu
shapiro.test(data$Price)

shapiro.test(head(data$Price, n = 5000))
# nema normalnu raspodelu, menjamo medijanom

medianPrice <- median(data$Price, na.rm = T)
data$Price[is.na(data$Price)] <- medianPrice


# ostavljamo sve observacije koje imaju cenu manju ili jednaku 350
dataSub <- subset(data, data$Price <= 350)

str(dataSub)

# sada gledamo koje varijable su nam potrebne za model, a koje ne
# App nam nije potrebno za model jer se svako ime aplikacije razlikuje i nece uticati
# na nas rezultat
# Kategorija igrice nam takodje nije potrebna
# Reviews izbacujemo zbog nove varijable High_Reviews
# Size, odnosno velicina aplikacije nema nikakve veze sa kolicinom recenzija
# Price takodje izbacujemo jer je velika vecina aplikacija besplatna, pa nece uticati
# na nas model
# Genres, Last.Updated, Current.Ver i Android.Ver takodje nemaju nikakve veze sa 
# brojem recenzija za neku aplikaciju

dataSub$App <- NULL
dataSub$Category <- NULL
dataSub$Reviews <- NULL
dataSub$Size <- NULL
dataSub$Price <- NULL
dataSub$Genres <- NULL
dataSub$Last.Updated <- NULL
dataSub$Current.Ver <- NULL
dataSub$Android.Ver <- NULL

# proveravamo nedostajuce vrednosti
apply(dataSub, MARGIN = 2, FUN = function(x) sum(is.na(x)))
apply(dataSub, MARGIN = 2, FUN = function(x) sum(x == "-", na.rm = T))
apply(dataSub, MARGIN = 2, FUN = function(x) sum(x == "", na.rm = T))
apply(dataSub, MARGIN = 2, FUN = function(x) sum(x == " ", na.rm = T))
# ovo cemo proveriti opet kad izbacimo NA vrednosti iz Ratinga, da bismo videli
# da nema ostalih tipova nedostajucih vrednosti

# Varijabla Rating ima 1472 NA vrednosti, njih cemo zameniti srednjom vrednoscu
# ili medijanom u zavisnosti od toga kakvu raspodelu varijabla ima

shapiro.test(dataSub$Rating)
# ovo nam baca gresku sample size must be between 3 and 5000, pa moramo da
# ogranicimo broj redova na nesto izmedju 3 i 5000 i onda da vidimo raspodelu

shapiro.test(head(dataSub$Rating, n=5000))
# vidimo da nemamo normalnu raspodelu jer je p-value manje od 0.05
# pa NA vrednosti menjamo medijanom!

mediana <- median(dataSub$Rating, na.rm = T)
mediana

dataSub$Rating[is.na(dataSub$Rating)] <- mediana

str(dataSub)
# vidimo da sve character varijable mozemo da pretvorimo u factor

dataSub$Type <- as.factor(dataSub$Type)
dataSub$Content.Rating <- as.factor(dataSub$Content.Rating)
dataSub$High_Reviews <- as.factor(dataSub$High_Reviews)

levels(dataSub$Type)
levels(dataSub$Content.Rating)
levels(dataSub$High_Reviews)
table(dataSub$Type)

# zavrsili smo sa sredjivanjem podataka

# pravimo trening i test setove
library(caret)
set.seed(1010)
indexes <- createDataPartition(dataSub$High_Reviews, p = 0.8, list = FALSE)
train.data <- dataSub[indexes, ] # svi oni koji se nalaze u tih 80%
test.data <- dataSub[-indexes, ] # svi oni koji se NE nalazed u tih 80%, ostalih 20%


# trazimo najbolju vrednost za CP preko krosvalidacije od sa 10 iteracija

library(e1071)
library(caret)
numFolds = trainControl(method = "cv", number = 10) 
cpGrid = expand.grid(.cp = seq(from = 0.001, to = 0.05, by = 0.001)) 

set.seed(1010)
crossvalidation <- train(x = train.data[,-5],
                         y = train.data$High_Reviews,
                         method = "rpart", 
                         trControl = numFolds, # numFolds sto smo dobili iznad
                         tuneGrid = cpGrid) # cpGrid sto smo dobili iznad

crossvalidation


plot(crossvalidation)

# dobili smo da nam je najbolja vrednost za CP 0.005, tu nam je najveci accuracy,
# pa cemo tu vrednost koristiti da napravimo novo drvo
cpValue <- crossvalidation$bestTune$cp


tree1 <- rpart(High_Reviews ~ ., 
               data = train.data,
               method = "class", 
               control = rpart.control(cp = cpValue))

library(rpart.plot)
rpart.plot(tree1, extra = 104)

# pravimo predikcije sa novim modelom
tree1.pred <- predict(tree1, newdata = test.data, type = "class")

# matrica konfuzije za nov model
tree1.cm <- table(true = test.data$High_Reviews, predicted = tree1.pred)
tree1.cm
# vidimo da ce model da nam bude izuzetno dobar i imati visoke metrike

# Yes nam je pozitivna klasa
getEvaluationMetrics <- function(cm) {
  TP <- cm[2,2]
  TN <- cm[1,1]
  FP <- cm[1,2]
  FN <- cm[2,1]
  
  accuracy = sum(diag(cm))/sum(cm) # tacno predvidjene / sve
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
# u test setu, sto je 2164, tacno predvideli 2018, pa nam je tacnost visoka, 0.932

# precision = udeo onih koje smo predvideli da su pozitivne koje su stvarno pozitivne
# ovde smo od 526 observacija za koje smo rekli da imaju visok broj recenzija
# za 461 predvideli tacno
# a za 65 smo rekli da imaju, a zapravo nemaju, pa nam
# je precision 0.876

# recall = udeo observacija koje su stvarno pozitivne koje smo predvideli da su pozitivne
# ovde smo od ukupno 542 observacije sa visokim brojem recenzija 461 predvideli tacno,
# a za 81 smo rekli da nemaju visok broj recenzija, a zapravo imaju, pa nam 
# je recall nesto manji od precisiona, odnosno 0.85

# F1 = sluzi za evaluaciju modela kada su precision i recall u balansu, 
# govori koliko je dobar model, u nasem slucaju je 0.863, pa mozemo da 
# zakljucimo da jeste dobar



