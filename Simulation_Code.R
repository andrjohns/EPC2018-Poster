library(lme4)
library(rstanarm)
library(data.table)
library(parallel)
library(pbapply)

#Initialise cluster object
cl <- makeCluster(parallel::detectCores())

#Load required packages and create data-generating function in each cluster
clusterEvalQ(cl,{
    library(lme4)
    library(rstanarm)
    library(data.table)
    library(parallel)

form<-as.formula(c("~env+(1|ID)"))
N.ind<-10
N.obs<-5
simdat <-data.frame(ID=factor(rep(1:N.ind,each=N.obs)),env=runif(N.ind*N.obs,0,1))
beta<-c(7,2)
names(beta)<-c("(Intercept)","env")
theta <- 5
names(theta) <- "ID.(Intercept)"
sigma <- 1

sim.data <- function(...){
response<-simulate(form,newdata=simdat,family=gaussian,
                   newparams=list(theta = theta,beta = beta, sigma = sigma))
simdat$resp<-as.vector(response[,1])

return(simdat)
}
})

#Create function to run REML and Bayes analysis and extract relevant estimates
sim.err <- function(...) {
  tmpdata = sim.data()
  fit <- lmer(resp ~ env + (1|ID) ,data=tmpdata)
  bfit <- stan_lmer(resp ~ env + (1|ID),data=tmpdata)
  return(
      data.frame("int_ml" = fixef(fit)[1],
                 "int_se_ml" = coef(summary(fit))[,"Std. Error"][1],
                 "int_bys" = fixef(bfit)[1],
                 "int_se_bys" = bfit$stan_summary[1,2],
                 "b1_ml" = fixef(fit)[2],
                 "b1_se_ml" = coef(summary(fit))[,"Std. Error"][2],
                 "b1_se_bys" = bfit$stan_summary[2,2],
                 "b1_bys" = fixef(bfit)[2],
                 row.names = NULL)
  )
}

#Run analyses and extract estimates
n <- rbindlist(pblapply(1:1000,sim.err,cl=cl))


#Plot data
ggplot(data=n,aes(x=int_se_ml)) +
    geom_histogram(fill="orange",colour="black",bins=40) +
    geom_label(x=2.7,
               y=75,
               aes(label=paste0("Mean: ",round(mean(n$int_se_ml),digits=3),"\n",
                                "95% CI: [",round(quantile(n$int_se_ml,0.025),digits=3),", ",
                                round(quantile(n$int_se_ml,0.975),digits=3),"]")),
               label.padding = unit(1, "lines"),hjust=0) +
    geom_vline(xintercept = c(mean(n$int_se_ml),quantile(n$int_se_ml,0.025),quantile(n$int_se_ml,0.975)),linetype=c("solid","dashed","dashed")) +
    labs(y="Count",title="Intercept Estimate Standard Errors \n 1000 Montecarlo Simulations with REML") +
    scale_x_continuous(breaks=seq(0,3.5,0.5),name="Standard Error of the Estimate - Intercept") + ylim(0,90) + 
    theme(
        text = element_text(family="sans"),
        axis.title = element_text(size=rel(2)),
        axis.text = element_text(size=rel(1.5)),
        panel.background = element_rect(fill="gray90",colour="gray90",size=0.5,linetype="solid"),
        panel.grid.major = element_line(colour="white",size=0.5,linetype="solid"),
        panel.grid.minor = element_line(colour="white",size=0.25,linetype="solid"),
        plot.title = element_text(size=rel(2.5),hjust=0.5),
        axis.line = element_line(linetype = "solid")
    )


ggplot(data=n,aes(x=int_se_bys)) +
    geom_histogram(fill="blue",colour="black",bins=40) +
    geom_label(x=0.1,
               y=125,
               aes(label=paste0("Mean: ",round(mean(n$int_se_bys),digits=3),"\n",
                                "95% CI: [",round(quantile(n$int_se_bys,0.025),digits=3),", ",
                                round(quantile(n$int_se_bys,0.975),digits=3),"]")),
               label.padding = unit(1, "lines"),hjust=0) +
    geom_vline(xintercept = c(mean(n$int_se_bys),quantile(n$int_se_bys,0.025),quantile(n$int_se_bys,0.975)),linetype=c("solid","dashed","dashed")) +
    labs(y="Count",title="Intercept Estimate Standard Errors \n 1000 Montecarlo Simulations with Bayes") +
    scale_x_continuous(breaks=seq(0,0.20,0.05),limits=c(0,0.16),name="Standard Error of the Estimate - Intercept") + ylim(0,150) +
    theme(
        text = element_text(family="sans"),
        axis.title = element_text(size=rel(2)),
        axis.text = element_text(size=rel(1.5)),
        panel.background = element_rect(fill="gray90",colour="gray90",size=0.5,linetype="solid"),
        panel.grid.major = element_line(colour="white",size=0.5,linetype="solid"),
        panel.grid.minor = element_line(colour="white",size=0.25,linetype="solid"),
        plot.title = element_text(size=rel(2.5),hjust=0.5),
        axis.line = element_line(linetype = "solid")
    )


ggplot(data=n) + geom_histogram(aes(x=int_ml),bins=40,colour="black",fill="orange",alpha=0.4) + geom_histogram(aes(x=int_bys),bins=40,colour="black",fill="blue",alpha=0.4) +
    theme(
        text = element_text(family="sans"),
        axis.title = element_text(size=rel(1.5)),
        panel.background = element_rect(fill="gray90",colour="gray90",size=0.5,linetype="solid"),
        panel.grid.major = element_line(colour="white",size=0.5,linetype="solid"),
        panel.grid.minor = element_line(colour="white",size=0.25,linetype="solid"),
        plot.title = element_text(size=rel(2.25),hjust=0.5),
        axis.line = element_line(linetype = "solid")
    )