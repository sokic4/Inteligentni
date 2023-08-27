
library("ISLR")
?Credit

data <- Credit

summary(data)
str(data)

dataSub <- subset(data, data$Student == "No")
# sad izbacujemo varijablu student jer znamo da svi nisu studenti
dataSub$Student <- NULL
# ID nam nije potreban
dataSub$ID <- NULL



# sve varijable moraju da budu numericke za linearnu regresiju
# pa sve faktor koje imamo pretvaramo u numeric!
str(dataSub)
dataSub$Gender <- as.numeric(dataSub$Gender)
dataSub$Married <- as.numeric(dataSub$Married)
dataSub$Ethnicity <- as.numeric(dataSub$Ethnicity)

# sad gledamo korelacije
library(corrplot)
matrica <- cor(dataSub)
matrica[,10] # izabrali smo 10. kolonu/varijablu Balance
corrplot(matrica, method = "number", type = "upper", diag = FALSE) 
# kao sto vidimo, samo varijable Income, Limit i Rating imaju koeficijent
# korelacije veci od 0.4, odnosno 0.49, 0.9 i 0.9 respektivno
# pa cemo samo njih koristiti za nas model, kao sto je receno u zadatku


# pravimo trening i test setove
library(caret)
set.seed(1010)
indexes <- createDataPartition(dataSub$Balance, p = 0.8, list = FALSE)
train.data <- dataSub[indexes, ]
test.data <- dataSub[-indexes, ]

# pravimo model
# sad za lm uzimamo samo ove koje imaju koeficijent veci od 0.4
lm1 <- lm(Balance ~ Income + Limit + Rating, data = train.data)
summary(lm1)
# za svako povecanje Income za jednu jedinicu (1000 dolara u nasem slucaju) 
# smanjuje nam se Balance za 7.85$
# za svako povecanje Limit za jednu jedinicu povecava nam se Balance za 0.123$
# za svako povecanje Rating za jednu jedinicu povecava nam se Balance za 2.13$
# vidimo da su nam sve varijable koje smo izabrali znacajne
# residual predstavlja razliku izmedju predvidjenih i stvarnih vrednosti
# i ovde iznosi 101.8
# r-squared, nas model opisuje 94.72% varijabilieteta zavisne promenljive
# f-statistika je 1697, a p-value < 0.05, dakle postoji zavisnost izmedju 
# zavisne promenljive i prediktora i ima smisla razmatrati ovaj nas model

# proveravamo multikolinearnost
library(car)
vif(lm1)
sort(sqrt(vif(lm1)))

# varijable koje imaju sqrt(vif(lm)) veci od 2 su problematicne
# postoji velika multikolinearnost izmedju nekih nasih varijabli
# izbacicemo prvo Rating, pa cemo da vidimo sta ce se promeniti

lm2 <- lm(Balance ~ Income + Rating, data = train.data)
sqrt(vif(lm2))
# sada je sve u redu

summary(lm2)

# vidimo da su nam Income i Limit znacajni prediktori
# za svako povecanje Income za jednu jedinicu (1000 dolara u nasem slucaju) 
# smanjuje nam se balance za 7.74$
# za svako povecanje Rating jednu jedinicu, povecava se Balance za 3.93$
# residual predstavlja razliku izmedju predvidjenih i stvarnih vrednosti
# i ovde iznosi 104
# r-squared, nas model opisuje 94.46% varijabilieteta zavisne promenljive
# f-statistika je 2430, a p-value < 0.05, dakle postoji zavisnost izmedju 
# ovih varijabli

# pravimo 4 plota
graphics.off()
par(mfrow = c(1,1)) # da imamo samo 1 red i 1 kolonu za grafove
par(mfrow = c(2,2)) # da imamo 2 reda i 2 kolone za grafove
plot(lm2)


# Prva slika govori koliko je prepostavka o linearnosti zadovoljenja, 
# predikcija se moze smatrati merodavnom jer je crvena linija blizu 
# toga da bude ravna, odnosno tackice su blizu toga da budu jednako rasporedjene
# Residuals su reziduali (razlika izmedju stvarnih i predvidjenih vrednosti)
# a fitted values predvidjene vrednosti
# ovde se tezi da reziduali budu 0, tako da sto je crvena linija
# bliza 0, to je nas model bolji
# u nasem slucaju je pretpostavka linearnosti nije zadovoljena

# druga slika govori o tome da li su reziduali normalno rasporedjeni
# U ovom slucaju su veoma blizu isprekidanoj liniji, mozemo reci da su blizu
# toga da budu normalno rasporedjeni

# treca slika proverava da li rezidulali imaju jednake 
# varijanse (homoskedasticnost), ukoliko imamo horizontalnu liniju
# bilo gde na plotu, znaci da imaju, kod nas nije horizontalna i mozemo
# reci da nemaju

# cetvrta da li ima observacija sa veoma velikim/malim vrednostima 
# tj. ekstemnim vrednostima, Kukova distanca nam je preko isprekidanih crvenih linija
# ako je neka observacija preko te linije, znaci da imamo ekstremne vrednosti
# u nasem slucaju se ne vidi Kukova distanca, odnosno nemamo ekstremne vrednosti
# koje ce nam praviti problem

# nas model ne ispunjava idealne uslove, ali je prihvatljiv


lm2.pred <- predict(lm2, newdata = test.data)
head(lm2.pred)

test.data$Balance_pred <- lm2.pred

library(ggplot2)
ggplot(test.data) +
  geom_density(aes(x = Balance, color = 'actual')) +
  geom_density(aes(x = Balance_pred, color = 'predicted'))

# residual je razlika izmedju stvarne i predvidjene vrednosti,
# nju sumiramo i kvadriramo da bismo dobili RSS:
RSS <- sum((lm2.pred - test.data$Balance)^2)

# razlika izmedju vrednosti u testu i srednje vrednosti u trainu,
# nju sumiramo i kvadriramo da bismo dobili TSS
TSS <- sum((mean(train.data$Balance) - test.data$Balance)^2)

# ovo je formula za RSQUARED
rsquared <- 1 - RSS / TSS
rsquared
# ukupan objasnjeni varijabilitet je 94.58%

summary(lm2)
# uporedjujemo sa rsquared nad trainom i nad testom i vidimo 
# da je veca na testu
# ukupan objasnjeni varijabilitet je 94.58%, a na trainu je 94.46%

# RMSE = Root Mean Squared Error, koliku gresku pravimo s predikcijama
RMSE <- sqrt(RSS/nrow(test.data))
RMSE

# pravimo gresku 101.32$ za izlaznu varijablu Balance

mean(test.data$Balance) # mean, srednja vrednost nam je 488.11$
RMSE/mean(test.data$Balance)
# greska iznosi 20.75% od srednje vrednosti poena, sto je malo veca greska

















