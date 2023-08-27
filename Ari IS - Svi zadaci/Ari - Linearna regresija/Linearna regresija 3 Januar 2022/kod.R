

data <- read.csv("medals.csv", stringsAsFactors = F)

str(data)

# proveravamo koje nedostajuce vrednosti ima
sum(is.na(data$Team.NOC))
sum(data$Team.NOC == "", na.rm = T)
sum(data$Team.NOC == "-", na.rm = T)
sum(data$Team.NOC == " ", na.rm = T)

# pravimo podskup
data <- subset(data, data$Team.NOC != "-")

str(data)
length(unique(data$Team.NOC))
# naziv drzave nam nije potreban za linearnu regresiju, a Silver cemo pretvoriti
# u numericku varijablu, ali prvo da pogledamo koje su nedostajuce vrednosti

data$Team.NOC <- NULL

apply(data, MARGIN = 2, FUN = function(x) sum(is.na(x)))
apply(data, MARGIN = 2, FUN = function(x) sum(x == "", na.rm = T))
apply(data, MARGIN = 2, FUN = function(x) sum(x == "-", na.rm = T))
apply(data, MARGIN = 2, FUN = function(x) sum(x == " ", na.rm = T))

# Gold je izlazna varijabla i ima dve nedostajuce vrednosti
# posto je izlazna, te dve observacije moramo da uklonimo iz dataseta
# Silver ima jedan "-", pa cemo to zameniti adekvatnijom vrednoscu

data <- data[complete.cases(data[,1]),]
# izbacili smo dve nedostajuce vrednosti iz Gold

# sredjujemo silver, gledamo da li ima normalnu raspodelu ili nema
# pa u odnosu nedostajucu vrednost menjamo srednjom vrednoscu ili medijanom
# ali prvo moramo sve te "-" da pretvorimo u NA vrednosti da bismo
# varijablu pretvorili u numeric jer je trenutno character

data$Silver[data$Silver == "-"] <- NA
data$Silver <- as.numeric(data$Silver)

shapiro.test(data$Silver)
# nema normalnu raspodelu, menjamo medijanom

medianSilver <- median(data$Silver, na.rm = T)

data$Silver[is.na(data$Silver)] <- medianSilver

# zavrsili smo sredjivanje podataka, sada pravimo korelacionu matricu

matrica <- cor(data)

library(corrplot)
corrplot(matrica, method = "number", type = "upper", diag = F)
# vidimo da su sve 3 varijable visoko korelisane
# napravicemo model sa sve 3, pa cemo videti kakve rezultate nam daje

library(caret)
set.seed(1010)
indexes <- createDataPartition(data$Gold, p = 0.8, list = FALSE)
train.data <- data[indexes, ]
test.data <- data[-indexes, ]

lm1 <- lm(Gold ~ ., data = train.data)
summary(lm1)

# za svako povecanje Silver povecava nam se Gold za 0.657 zlatnih medalja
# za svako povecanje Bronze povecava nam se Gold za 0.253 zlatnih medalja
# izgled linearne krive y = -0.09714 + 0.0657x + 0.253y
# max zvezdica je 3, sto veci broj zvezdica, to je znacajnija predikcija
# ovde vidimo da su nam obe varijable znacajne
# intercept znaci kolika bi vrednost bila da su svi prediktori 0 (nije uvek realno)
# residual predstavlja razliku izmedju stvarnih i predvidjenih vrednosti
# i ovde iznosi 1.671
# r-squared znaci da nas model objasnjava 92.2% varijabilieteta zavisne promenljive Gold
# f-statistika je 413.8, a p-value < 0.05, dakle postoji zavisnost izmedju 
# zavisne promenljive i prediktora i ima smisla razmatrati ovaj nas model

# pravimo 4 plota
graphics.off()
par(mfrow = c(1,1))
par(mfrow = c(2,2))
plot(lm1)

# Prva slika govori koliko je prepostavka o linearnosti zadovoljenja, 
# predikcija se moze smatrati merodavnom jer je crvena linija blizu 
# toga da bude ravna, odnosno tackice su blizu toga da budu jednako rasporedjene
# Residuals su reziduali (razlika izmedju stvarnih i predvidjenih vrednosti)
# a fitted values predvidjene vrednosti 
# ovde se tezi da reziduali budu 0, tako da sto je crvena linija
# bliza 0, to je nas model bolji
# u nasem slucaju pretpostavka linearnosti nije zadovoljavajuca

# druga slika govori o tome da li su reziduali normalno rasporedjeni
# U ovom slucaju mozemo reci da jesu, jer prate isprekidanu liniju

# treca slika proverava da li rezidulali imaju jednake 
# varijanse (homoskedasticnost), ukoliko imamo horizontalnu liniju
# bilo gde na plotu, znaci da imaju
# kod nas nije horizontalna, tako da se razlikuju

# cetvrta da li ima observacija sa veoma velikim/malim vrednostima 
# tj. ekstemnim vrednostima, Kukova distanca nam je preko isprekidanih crvenih linija
# ako je neka observacija preko te linije, znaci da imamo ekstremne vrednosti
# ovde imamo observacije koje su problematicne, odnosno van Kukove distance

# mozemo da poboljsamo model tako sto proverimo multikolinearnost

library(car)
sqrt(vif(lm1))
# moracemo jednu varijablu da izbacimo, obe daju rezultat preko 2,
# a posto su iste vrednosti, napravicemo jedan model sa Silver
# a drugi sa Bronze i uporediti rezultate

lm2 <- lm(Gold ~ Bronze, data = train.data)
summary(lm2)

lm3 <- lm(Gold ~ Silver, data = train.data)
summary(lm3)

# vidimo da nam model sa Silver daje bolje rezulate i da je znacajnija
# manji je residual standard error u odnosu na modelom sa bronze
# odnosno na lm2 je 2.5, a na lm3 je 1.797
# takodje model opisuje veci procenat varijabiliteta zavisne promenljive Gold
# 82.3% je sa Bronze kao prediktorom, a 90.86% sa Silver
# malo je veci i Std. error
# tako da je model lm3 u svakom aspektu od lm2

graphics.off()
par(mfrow = c(1,1))
par(mfrow = c(2,2))
plot(lm3)

# plotovi nam se nisu nesto poboljsali

# prva slika, odnosno pretpostavka o linearnosti jos uvek nije zadovoljena
# druga slika, sad nam residuali nismo normalno rasporedjeni
# treca slika, residuali jos uvek nemaju jednake varijanse
# cetvrta slika, jos uvek imamo ekstremne vrednosti koje su problematicne


lm3.pred <- predict(lm3, newdata = test.data)
head(lm3.pred)
head(test.data$Gold)

# mozete da da koristite ovaj ggplot, ali nije neophodno
test.data$Gold_pred <- lm3.pred

library(ggplot2)
ggplot(test.data) +
  geom_density(aes(x = Gold, color = 'actual')) +
  geom_density(aes(x = Gold_pred, color = 'predicted'))



# NISU REKLI U ZADATKU ZA RSS I TSS, ALI SAM JA IPAK URADIO

RSS <- sum((lm3.pred - test.data$Gold)^2)

# razlika izmedju vrednosti u testu i srednje vrednosti u trainu,
# nju sumiramo i kvadriramo da bismo dobili TSS
TSS <- sum((mean(train.data$Gold) - test.data$Gold)^2)

# ovo je formula za RSQUARED
rsquared <- 1 - RSS / TSS
rsquared
# ukupan objasnjeni varijabilitet je 58.68%


summary(lm3)
# uporedjujemo sa rsquared nad trainom i nad testom i vidimo 
# da je veca na trainu
# ukupan objasnjeni varijabilitet je 58.68%, a na trainu je 90.86%

# RMSE = Root Mean Squared Error, koliku gresku pravimo s predikcijama
# RSS / broj observacija u test setu
RMSE <- sqrt(RSS/nrow(test.data))
RMSE

# pravimo gresku 3.9626 zlatnih medalja za izlaznu varijablu Gold

# OVO NEMA U CHEATSHEETU !
mean(test.data$Gold) # mean, srednja vrednost nam je 2.941 poena
RMSE/mean(test.data$Gold)
# greska iznosi 134.72% od srednje vrednosti poena


# sad cemo isto da proverimo za lm1 i videti koji nam je model bolji

lm1.pred <- predict(lm1, newdata = test.data)
head(lm1.pred)
head(test.data$Gold)

# mozete da da koristite ovaj ggplot, ali nije neophodno
test.data$Gold_pred <- lm1.pred

library(ggplot2)
ggplot(test.data) +
  geom_density(aes(x = Gold, color = 'actual')) +
  geom_density(aes(x = Gold_pred, color = 'predicted'))



# NISU REKLI U ZADATKU ZA RSS I TSS, ALI SAM JA IPAK URADIO

RSS <- sum((lm1.pred - test.data$Gold)^2)

# razlika izmedju vrednosti u testu i srednje vrednosti u trainu,
# nju sumiramo i kvadriramo da bismo dobili TSS
TSS <- sum((mean(train.data$Gold) - test.data$Gold)^2)

# ovo je formula za RSQUARED
rsquared <- 1 - RSS / TSS
rsquared
# ukupan objasnjeni varijabilitet je 64.09%


summary(lm1)
# uporedjujemo sa rsquared nad trainom i nad testom i vidimo 
# da je veca na trainu
# ukupan objasnjeni varijabilitet je 64.09%, a na trainu je 92.2%

# RMSE = Root Mean Squared Error, koliku gresku pravimo s predikcijama
# RSS / broj observacija u test setu
RMSE <- sqrt(RSS/nrow(test.data))
RMSE

# pravimo gresku 3.694 zlatnih medalja za izlaznu varijablu Gold

# OVO NEMA U CHEATSHEETU !
mean(test.data$Gold) # mean, srednja vrednost nam je 2.941 poena
RMSE/mean(test.data$Gold)
# greska iznosi 125.6% od srednje vrednosti poena

# model lm1 kada iskoristimo sve izlazne varijable nam je bolji 
# od lm3 kada izbacimo Bronze i iskoristimo samo Silver



