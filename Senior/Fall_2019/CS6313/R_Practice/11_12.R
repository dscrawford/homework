size = c(6,7,7,8,10,10,15);
tables = c(4,20,20,10,10,2,1);
os = c(1,1,1,1,2,2,2);
os = factor(os);
req = c(40,55,50,41,17,26,16);

fit1 = lm(req ~ size + tables)
summary(fit1);

fit2 = lm(req ~ size + tables + os)
summary(fit2);

