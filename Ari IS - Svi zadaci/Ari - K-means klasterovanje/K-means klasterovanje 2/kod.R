

dataSet <- read.csv("world-happiness-report-2017.csv", stringsAsFactors = F)
str(dataSet)
summary(dataSet)

# varijable koje cemo koristiti u modelu su sve osim Country 
# jer nam je to character varijabla

dataSet <- subset(dataSet, (dataSet$Economy.GDP.per.Capita > 0))

all(complete.cases(dataSet)) 
# vraca TRUE ako nijedna nema NA vrednosti, a FALSE ako barem 1 ima neku NA vrednost
which(complete.cases(dataSet) == F) 
# vraca koji redovi imaju NA vrednosti, ovde kaze da NA vrednosti imaju
# redovi 87, 139 i 147, to cemo srediti kasnije

apply(dataSet, MARGIN = 2, FUN = function(x) sum(is.na(x)))
apply(dataSet, MARGIN = 2, FUN = function(x) sum(x == "-", na.rm = T))
apply(dataSet, MARGIN = 2, FUN = function(x) sum(x == "", na.rm = T))
apply(dataSet, MARGIN = 2, FUN = function(x) sum(x == " ", na.rm = T))

# NA vrednosti imaju Health.Life.Expectancy i Generosity, to cemo srediti
# gledamo za Health.Life.Expectancy
shapiro.test(dataSet$Health.Life.Expectancy)
# nema normalnu raspodelu, menjamo medijanom
median <- median(dataSet$Health.Life.Expectancy, na.rm = T)
dataSet$Health.Life.Expectancy[is.na(dataSet$Health.Life.Expectancy)] <- median

# gledamo za Generosity
shapiro.test(dataSet$Generosity)
# nema normalnu raspodelu, menjamo medijanom
median <- median(dataSet$Generosity, na.rm = T)
dataSet$Generosity[is.na(dataSet$Generosity)] <- median

# zavrsili smo sredjivanje

# proveravamo outliere
apply(dataSet[,2:12], 2, FUN = function(x) length(boxplot.stats(x)$out))
# Family i Trust.Government.Corruption imaju 4 i 2 outliera

boxplot(dataSet$Family, xlab = 'Family')
boxplot(dataSet$Trust.Government.Corruption, xlab = 'Trust.Government.Corruption')
# Family ima samo outliere sa ekstremno niskim vrednostima
# Trust.Government.Corruption ima 1 outlier sa ekstremno visokom vrednoscu

# sredjujemo preko funkcije Winsorize()
library(DescTools)
?Winsorize
Family_w <- Winsorize(dataSet$Family, probs = c(0.05, 1))
dataSet$Family <- Family_w
# proveravamo da li smo izbacili outliere
boxplot(dataSet$Family, xlab = 'Family') 
# jesmo

# Dystopia.Residual ima i preniske i previsoke vrednosti, pa cemo koristiti
# i donji i gornji percentil ovaj put
Trust.Government.Corruption_w <- 
        Winsorize(dataSet$Trust.Government.Corruption, probs = c(0, 0.95))
dataSet$Trust.Government.Corruption <- Trust.Government.Corruption_w
boxplot(dataSet$Trust.Government.Corruption, xlab = 'Trust.Government.Corruption')
# izbacili smo i ovde

# zavrseno sredjivanje outliera


# normalizacija i model i elbow
normalize_var <- function( x ) {
  if ( sum(x, na.rm = T) == 0 ) x
  else ((x - min(x, na.rm = T))/(max(x, na.rm = T) - min(x, na.rm = T)))
}

# normalizujemo numericke kolone
data.norm <- as.data.frame(apply(dataSet[,2:12], 2, normalize_var))
summary(data.norm)

# nase varijable ne smeju da budu visoko korelisane jer ce to uticati
# negativno na nas model, tako da proveravamo multikolinearnost
data_cor <- cor(data.norm)

library(corrplot)
corrplot.mixed(data_cor)
# vidimo da su nam Happiness.Rank, Happiness.Score,
# Whisker.high, Whisker.low, Economy.GDP.per.Capita
# Family, Health.Life.Expectancy, Freedom
# visoko korelisani, pa cemo izbacivati jednu po jednu i videti sta se desava

data.norm$Happiness.Rank <- NULL
data.norm$Happiness.Score <- NULL
data.norm$Whisker.high <- NULL
data.norm$Whisker.low <- NULL
data.norm$Health.Life.Expectancy <- NULL

data_cor <- cor(data.norm)
corrplot.mixed(data_cor)


# pravimo eval.metrics i k 2:8
eval.metrics <- data.frame()

# kmeans u cheatsheetu, direktno je ugradjena funkcija u R, ne vadimo iz paketa
# moramo da navedemo koji numericki dataframe koristi, koliko klastera pravi,
# max iteracija (da ne bi isao u nedogled) i koliko puta pozivamo algoritam
# jer inicijalno moze da nam da losije pozicije klastera, pa moramo da ga izvrsimo
# vise puta da bismo nasli sto bolje
for(k in 2:8){
  set.seed(1010) # jer nasumicno biramo klastere
  km <- kmeans(data.norm, centers = k, iter.max = 20, nstart = 1000)
  eval.metrics <- rbind(eval.metrics, 
                        c(k, km$tot.withinss, km$betweenss/km$totss))
  # na postojeci sadrzaj dataframea, dodaj tot.withinss i racio betweenss / totss
}

colnames(eval.metrics) <- c("clusters", "tot.withinss", "ratio")
eval.metrics

# crtamo krivu i gledamo gde je najveci prelom
library(ggplot2)
ggplot(data = eval.metrics, mapping = aes(x = 2:8, y = tot.withinss)) + 
                                geom_line() + 
                                geom_point()

source("Utility.R")
diff_df <- apply(eval.metrics[,2:3], 2, compute.difference)
diff_df

# da bismo imali kolonu za k da znamo na sta se odnose ove vrednosti
diff_df <- cbind(k = 2:8, diff_df)
diff_df

# u prvoj koloni smo dobili NA jer radimo razliku izmedju dve susedne vrednosti
# 2. kolona predstavlja smanjenje tot.withinss kad je k+1
# 3. kolona je povecanje racia za k+1

# po ovom diff_df vidimo da je 3 najbolji broj klastera
# jer ima najvecu razliku u odnosu na prethodni i za tot.withinss i za ratio

sample.3k <- kmeans(x = data.norm, centers = 3, iter.max = 20, nstart = 1000)
sample.3k
# ovo pokazuje:

# 3 klastera od 55, 71, 28 observacija

# srednje vrednosti klastera po varijablama

# clustering vector: koja observacija pripada kom klasteru
# suma kvadrata odstupanja observacija od centra klastera (za svaki pise pojedinacno), 
# sto manja to bolja jer je bliza centru klastera naravno
# vidimo da 2. klaster najvise odstupa, sto je i logicno jer on ima
# najvise observacija

# withinss => suma kvadrata odstupanja observacija od centra njihovog klastera
# between_SS => suma kvadrata odstupanja centara klastera od globalnog centra 
# total_SS => suma kvadrata odstupanja svake observacije od globalnog centra
# sto je ratio veci, to je bolji, u nasem slucaju je 46%
# ako je veci ne znaci da cemo moci da interpretiramo rezultate
# najbitnije je da krajnji rezultati budu razumljivi

# pozivamo summary.stats iz Utility.R 
# prvi parametar je dataset, drugi raspored po klasterima, a treci je broj klastera
sum.stats <- summary.stats(data.norm, sample.3k$cluster, 3)
sum.stats

# mean pokazuje srednju vrednost, a SD nam je standardna devijacija,
# odnosno odstupanje (disperzije) od centra
# freq je broj zemalja, u prvom klasteru je 55, u drugom 71, broj u trecem je 28 zemalja,

# imamo mali disbalans sto se tice klastera u pitanju njihove velicine

# trazimo ono sto je specificno za svaki od ovih klastera

# disperzije od centra su nam svuda slicne

# slicni komentari kao za prvi zadatak, pisite slobodno i gde odstupaju

# treci klaster imaju sve vece vrednosti, pa mozemo reci da su drzave viseg statusa
# drugi klaster srednje vrednosti, pa mozemo reci da su drzave srednjeg statusa
# prvi klaster nize vrednosti, pa mozemo reci da su drzave nizeg statusa













