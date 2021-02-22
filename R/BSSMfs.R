
#' Fit the Bayesian subject-specific bi-level feature selection model described in Baer et al. (2021)
#'
#' This function implements a Markov chain Monte Carlo (MCMC) Gibbs sampler to fit
#' a Bayesian specific-specific bi-level feature selection model;
#' this model is developed for the data scenario where we have time-invariant feature data and
#' irregularly-spaced longitudinal outcome data;
#' this model specifies time-varying
#' feature parameters, and allows for the incorporation of group structure into
#' the feature selection process.loadmathjax
#' \
#'
#' @param Y_transpose A \mjeqn{J \times n}{}
#' matrix of longitudinal outcome data,
#' where \mjeqn{J}{J}
#'  is the number of occasions,
#' and \mjeqn{n}{n} is the number of subjects.
#' This model assumes multivariate normality of these outcome data.
#'
#' @param X An \mjeqn{n \times p}{}
#' matrix of time-invariant feature data, where
#' \mjeqn{p}{p} is the number of features.
#'
#'
#' @param Z A list consisting of
#' \mjeqn{n \underset{J\times q}{\mathbf{Z}_i}}{}
#' matrices, where each
#' \mjeqn{\underset{J\times q}{\mathbf{Z}_i}}{}
#' is the matrix
#' of (time-varying) subject-specific feature data
#' corresponding to the subject-specific/random effect parameters,
#' \mjeqn{\underset{q\times 1}{\mathop{{{\mathbf{b}}_{i}}}}\,}{}
#' for
#' \mjeqn{i = 1, \cdots, n}{}.
#'
#' @param p_k A
#' \mjeqn{K \times 1}{}
#' matrix of the feature group sizes,
#' \mjeqn{p_k}{},
#' where
#' \mjeqn{K}{K}
#' is the number of feature groups.
#' Note that
#' \mjeqn{\sum\limits_{k=1}^{K}{{{p}_{k}}}=p}{}.
#'
#'
#' @param nsim The total number of MCMC
#' iterations to be performed.
#'
#' @param burn The number of MCMC
#' iterations to be discarded as part of the burn-in period.
#'
#' @param thin The thinning interval applied to the MCMC
#' sample.
#'
#' @param mcmc_div The interval at which the MCMC
#' iteration number will be reported; the
#' default value is
#' \code{floor((nsim-burn)/4)}.
#'
#' @param covariate_select A vector denoting which of the
#' \mjeqn{p}{p} features the user wishes to remove from
#' the feature selection process; note that setting
#' \code{covariate_select<-c(NA)} removes no features from the feature
#' selection process; \code{covariate_select<-c(NA)} is the default setting.
#'
#'
#' @param d The degrees of freedom hyper-prior for the inverse-Wishart-distributed
#' covariance matrix,
#' \mjeqn{\underset{J\times J}{\mathbf{\Sigma}}}{};
#' note that
#' \mjeqn{\underset{J\times J}{\mathbf{\Sigma}}}{}
#' represents the prior
#' covariance for each
#' of the
#'\mjeqn{l = 1, \cdots, p}{}
#' time-varying feature parameters
#' within the matrix
#' \mjeqn{\underset{p\times J}{\mathbf{B}}}{};
#' we recommend setting \code{d} to the smallest possible
#' value, which is
#' \mjeqn{J}{J};
#' the default value is \mjeqn{J}{J}.
#'
#' @param Q The
#' \mjeqn{J\times J}{}
#' scale matrix hyper-prior for the inverse-Wishart-distributed
#' covariance matrix,
#' \mjeqn{\underset{J\times J}{\mathbf{\Sigma}}}{}.
#' Note that
#' \code{Q}
#' must be positive-definite.
#'
#' @param nu_0 The degrees of freedom hyper-prior for the inverse-Wishart-distributed
#' covariance matrix,
#' \mjeqn{\underset{q\times q}{\mathbf{G}}}{};
#' note that
#' \mjeqn{\underset{q\times q}{\mathbf{G}}}{}
#' represents the prior
#' covariance for each
#' of the
#' vectors of subject-specific (random effect) parameters,
#' \mjeqn{\underset{q\times 1}{\mathbf{b}_i}}{}.
#' We recommend setting \code{nu_0} to the smallest possible
#' value, which is
#' \mjeqn{q}{q};
#' the default value is \mjeqn{q}{q}.
#'
#' @param C_0 The
#' \mjeqn{q\times q}{}
#' scale matrix hyper-prior for the inverse-Wishart-distributed
#' covariance matrix,
#' \mjeqn{\underset{q\times q}{\mathbf{G}}}{}.
#' Note that
#' \code{C_0}
#' must be positive-definite.
#'
#' @param alpha The shape hyper-prior for
#' the inverse-gamma-distributed
#' \mjeqn{\sigma^2}{};
#' note that \mjeqn{\sigma^2}{}
#' represents the prior
#' variance for
#' each vector of longitudinal outcome data,
#' of the longitudinal outcome data,
#' \mjeqn{\underset{J\times 1}{\mathbf{y}_i}}{}
#' for
#' \mjeqn{i = 1, \cdots, n}{}.
#' That is,
#' \mjeqn{\operatorname{cov}\left( \underset{J\times 1}{\mathbf{y}_i} \right) =
#' {{\sigma }^{2}}\underset{J\times J}{\mathop{{{\mathbf{I}}_{J}}}}\,}{}
#' \mjeqn{}{}.
#' The default value is \mjeqn{10^-3}{}.
#'
#' @param gamma The scale hyper-prior for
#' the inverse-gamma-distributed
#' \mjeqn{\sigma^2}{}.
#' The default value is \mjeqn{10^-3}{}.
#'
#' @param o The shape hyper-prior for
#' the inverse-gamma-distributed
#' \mjeqn{s^2}{};
#' note that \mjeqn{s^2}{}
#' represents the prior
#' variance
#' of the
#' spike-and-slab feature selection parameter,
#' \mjeqn{\tau_{l}^2}{}.
#' The default value is \mjeqn{10^-3}{}.
#'
#' @param u The scale hyper-prior for
#' the inverse-gamma-distributed
#' \mjeqn{s^2}{}.
#' The default value is \mjeqn{10^-3}{}.
#'
#' @param a_1 The first shape hyper-prior for
#' the Beta-distributed
#' \mjeqn{{{\theta }_{{\tilde{\beta }}}}}{};
#' note that
#' \mjeqn{{{\theta }_{{\tilde{\beta }}}}}{}
#' represents the prior
#' probability
#' of the
#' group-level feature selection indicator variable,
#' \mjeqn{\pi_{0k}}{},
#' equaling
#' \mjeqn{1}{}.
#' The default value is \mjeqn{1}{1}.
#'
#' @param b_1 The second shape hyper-prior for
#' the Beta-distributed
#' \mjeqn{{{\theta }_{{\tilde{\beta }}}}}{}.
#' The default value is \mjeqn{1}{1}.
#'
#' @param g The first shape hyper-prior for
#' the Beta-distributed
#' \mjeqn{{{\theta }_{{{\tau }^{2}}}}}{};
#' note that
#' \mjeqn{{{\theta }_{{{\tau }^{2}}}}}{}
#' represents the prior
#' probability
#' of the
#' individual feature selection indicator variable,
#' \mjeqn{\pi_{0l}}{},
#' equaling
#' \mjeqn{1}{}.
#' The default value is \mjeqn{1}{1}.
#'
#' @param h The second shape hyper-prior for
#' the Beta-distributed
#' \mjeqn{{{\theta }_{{{\tau }^{2}}}}}{}.
#' The default value is \mjeqn{1}{}.
#'
#' @param raw_MCMC_output If set to \code{TRUE}, the output for \code{BSSMfs} will include a data frame
#' containing un-summarized MCMC output.
#' The default value is \code{FALSE}.
#'
#' @return If \code{raw_MCMC_output=TRUE}, \code{BSSMfs} returns a list with two data frames named
#' \code{"MCMC_output"} and \code{"MCMC_summary"};
#' \code{"MCMC_output"} is a data frame with the un-summarized MCMC output, while
#' \code{"MCMC_summary"} is a data frame with the summarized MCMC output.
#' If \code{raw_MCMC_output=FALSE}, \code{BSSMfs} returns a list with only one data frame named
#' \code{"MCMC_summary"}
#'
#' The \code{"MCMC_output"} data frame containing the un-summarized MCMC output has the following columns:
#' \itemize{
#'  \item{"feature"}{} A number denoting the \mjeqn{l=1,\cdots,p}{} feature parameters.
#'
#'  \item{"occasion"}{} A character denoting the \mjeqn{j=1,\cdots,J}{} occasions; each occasion
#'  is denoted by the character \code{"tj"}.
#'
#'  \item{"feature_group"}{} A number denoting the \mjeqn{k=1,\cdots,K}{} feature groups.
#'
#'  \item{"subject"}{} A number denoting the \mjeqn{i=1,\cdots,n}{} subjects.
#'
#'  \item{"parameter"}{} A character denoting the model parameter; the model parameters include:
#'
#'  \itemize{
#'
#'  \item{"b"}{} the subject-specific parameters,
#'  \mjeqn{\underset{q\times 1}{\mathop{{{\mathbf{b}}_{i}}}}\,}{}.
#'
#'  \item{"beta"}{} The feature parameters,
#'  \mjeqn{\underset{p\times J}{\mathop{\mathbf{B}}}\,}{}.
#'
#'  \item{"deviance"}{} The deviance of our model;
#'  \mjeqn{
#'  deviance=nJ\log \left( 2\pi {{\sigma }^{2}} \right)+
#'  \frac{1}{{{\sigma }^{2}}}
#'  {{\left(vec\left( \underset{J\times n}{\mathop{{{\mathbf{Y}}^{T}}}}\, \right) -
#'   \underset{nJ\times Jp}{\mathop{\mathbf{X}}}\,
#'  vec\left( \underset{p\times J}{\mathop{\mathbf{B}}}\, \right) -
#'  \underset{nJ\times nq}{\mathop{\mathbf{Z}}}\,
#'  \underset{nq\times 1}{\mathop{\mathbf{b}}}\, \right)}^{T}}
#'  \left(vec\left( \underset{J\times n}{\mathop{{{\mathbf{Y}}^{T}}}}\, \right)-
#'   \underset{nJ\times Jp}{\mathop{\mathbf{X}}}\,
#'  vec\left( \underset{p\times J}{\mathop{\mathbf{B}}}\, \right)-
#'  \underset{nJ\times nq}{\mathop{\mathbf{Z}}}\,
#'  \underset{nq\times 1}{\mathop{\mathbf{b}}}\, \right)+C
#'  }{},
#'  where
#'  \mjeqn{C}{}
#'  is a constant that cancels out upon comparing models.
#'  The deviance can be used to calculate the deviance information criteria
#'  (DIC).
#'
#'  \item{"G"}{} The subject-specific parameter covariance matrix,
#'  \mjeqn{\underset{q\times q}{\mathop{\mathbf{G}}}\,}{}.
#'
#'  \item{"MSE"}{} The mean squared error with respect to predicting the longitudinal
#'  outcome data,
#'  \mjeqn{\underset{J\times 1}{\mathbf{y}_i}}{}.
#'  Note that
#'  \mjeqn{
#'  \left( \underset{J\times J}{\mathop{{{\mathbf{I}}_{J}}}}\,
#'  \otimes \underset{1\times p}{\mathop{\mathbf{x}_{i}^{T}}}\, \right)
#'  vec\left( \underset{p\times J}{\mathop{\mathbf{B}}}\, \right)+
#'  \underset{J\times q}{\mathop{{{\mathbf{Z}}_{i}}}}\,
#'  \underset{q\times 1}{\mathop{{{\mathbf{b}}_{i}}}}\,}{}
#'  is used to predict
#'  \mjeqn{\underset{J\times 1}{\mathbf{y}_i}}{}.
#'
#'  \item{"pi_0_k"}{} The group-level feature selection parameters,
#'  \mjeqn{{{\pi}_{0k}}}{}.
#'
#'  \item{"pi_0_l"}{} The individual feature selection parameters,
#'  \mjeqn{{{\pi}_{0l}}}{}.
#'
#'  \item{"s_2"}{} The variance parameter of
#'  \mjeqn{\tau _{l}^{2}}{},
#'  \mjeqn{s^{2}}{}.
#'
#'  \item{"Sigma"}{} The feature parameter covariance matrix,
#'  \mjeqn{\underset{J\times J}{\mathop{\mathbf{\Sigma }}}\,}{}.
#'
#'  \item{"sigma2"}{} The variance parameter of
#'  \mjeqn{\underset{J\times 1}{\mathbf{y}_i}}{},
#'  \mjeqn{\sigma^{2}}{}.
#'
#'  \item{"tau_2_l"}{} The spike-and-slab parameters,
#'  \mjeqn{\tau _{l}^{2}}{}.
#'
#'  \item{"theta_beta"}{} The probability parameter for group-level feature selection,
#'  \mjeqn{{{\theta }_{{\tilde{\beta }}}}}{}.
#'
#'  \item{"theta_tau_2"}{} The probability parameter for individual feature selection,
#'  \mjeqn{{{\theta }_{\tau^{2}}}}{}.
#'
#'  \item{"y"}{} The predicted values of
#'  \mjeqn{\underset{J\times 1}{\mathbf{y}_i}}{}.
#'
#'  }
#'
#'
#'  \item{"row_occasion"}{} A character denoting the \mjeqn{J^2}{} elements of
#'  the feature parameter covariance matrix,
#'  \mjeqn{\underset{J\times J}{\mathop{\mathbf{\Sigma }}}\,}{};
#'  each element
#'  is denoted by the character \code{"tj"}.
#'
#'  \item{"q"}{} A number denoting the \mjeqn{q}{} subject-specific/random effect parameters,
#'  \mjeqn{\underset{q\times 1}{\mathop{{{\mathbf{b}}_{i}}}}\,}{}.
#'
#'  \item{"row_q"}{} A number denoting the \mjeqn{q^2}{} elements of the
#'  subject-specific parameter covariance matrix,
#'  \mjeqn{\underset{q\times q}{\mathop{\mathbf{G}}}\,}{}.
#'
#'  \item{"comp_time"}{} The elapsed computation time for all \code{"nsim"}
#'  MCMC iterations to be completed.
#'
#'  \item{"mcmc_iter"}{} The MCMC iteration number.
#'
#'  \item{"value"}{} The numeric value of model parameter at each MCMC iteration.
#'
#' }
#'
#' The \code{"MCMC_summary"} data frame containg the summarized MCMC output has the following columns:
#' \itemize{
#'
#'  \item{"feature"}{} A number denoting the \mjeqn{l=1,\cdots,p}{} feature parameters.
#'
#'  \item{"occasion"}{} A character denoting the \mjeqn{j=1,\cdots,J}{} occasions; each occasion
#'  is denoted by the character string \code{"tj"}.
#'
#'  \item{"feature_group"}{} A number denoting the \mjeqn{k=1,\cdots,K}{} feature groups.
#'
#'  \item{"subject"}{} A number denoting the \mjeqn{i=1,\cdots,n}{} subjects.
#'
#'  \item{"parameter"}{} A character denoting the model parameter; the model parameters are the
#'  same here as for the \code{"MCMC_output"} data frame.
#'
#'  \item{"row_occasion"}{} A character denoting the \mjeqn{J^2}{} elements of
#'  the feature parameter covariance matrix,
#'  \mjeqn{\underset{J\times J}{\mathop{\mathbf{\Sigma }}}\,}{};
#'  each element
#'  is denoted by the character \code{"tj"}.
#'
#'  \item{"q"}{} A number denoting the \mjeqn{q}{} subject-specific/random effect parameters.
#'
#'  \item{"row_q"}{} A number denoting the \mjeqn{q^2}{} elements of the
#'  subject-specific parameter covariance matrix,
#'  \mjeqn{\underset{q\times q}{\mathop{\mathbf{G}}}\,}{}.
#'
#'  \item{"MCMC_mean"}{} The numeric posterior mean for each model parameter.
#'
#'  \item{"MCMC_median"}{} The numeric posterior median for each model parameter.
#'
#'  \item{"MCMC_sd"}{} The numeric posterior standard deviation for each model parameter.
#'
#'  \item{"MCMC_LCL"}{} The numeric posterior \mjeqn{2.5^{th}}{} percentile for each model parameter;
#'  used to construct the \mjeqn{95}{} percent credible interval (CI) for each model parameter.
#'
#'  \item{"MCMC_UCL"}{} The numeric MCMC \mjeqn{97.5^{th}}{} percentile for each model parameter;
#'  used to construct the \mjeqn{95}{} percent CI for each model parameter.
#'
#'  \item{"Geweke_diag"}{} The numeric Geweke MCMC diagnostic for each model parameter; calculated using
#'  the \code{coda} R package.
#'
#'  \item{"lag_5"}{} The numeric lag 5 autocorrelation for each model parameter; calculated using
#'  the \code{coda} R package.
#'
#'  \item{"lag_10"}{} The numeric lag 10 autocorrelation for each model parameter; calculated using
#'  the \code{coda} R package.
#'
#'  \item{"ESS"}{} The numeric MCMC effective sample size for each model parameter; calculated using
#'  the \code{coda} R package.
#'
#'  \item{"comp_time"}{} The elapsed computation time for all \code{"nsim"}
#'  MCMC iterations to be completed.
#'
#'
#'
#' }
#'
#' @examples
#'
#' set.seed(123)
#'
#' #Define data settings:
#'
#' n<-100
#' p<-16
#' K<-4
#' J<-3
#' p_k<-rep(p/K,K)
#' q<-2
#'
#' #----------------------------------------------------------------------------------#
#'
#' #Generate time-invariant feature data:
#'
#' X<-matrix(rnorm(n*p),nrow=n,ncol=p)
#'
#' #----------------------------------------------------------------------------------#
#'
#' #Define true B:
#'
#' Beta_k<-list()
#'
#' for(k in 1:K){
#'
#'  Beta_k[[k]]<-matrix(rnorm(J*p_k[k]),nrow=p_k[k],ncol=J)
#'
#' }
#'
#' #Introduce sparsity:
#'
#' #Group-level feature sparsity:
#'
#' Beta_k[[1]]<-0*Beta_k[[1]]
#'
#' #Individual feature sparsity:
#'
#' Beta_k[[2]][1,]<-0*Beta_k[[2]][1,]
#'
#' #Collapse:
#'
#' Beta<-dplyr::bind_rows(lapply(Beta_k,as.data.frame))%>%as.matrix
#'
#' #vec(Beta):
#'
#' beta<-matrix(Beta,nrow=J*p,ncol=1)
#'
#' #----------------------------------------------------------------------------------#
#'
#' #Define subject-specific feature data (time-varying):
#'
#' Z<-list()
#'
#' for(i in 1:n){
#'
#'   Z[[i]]<-matrix(0,nrow=J,ncol=q)
#'
#'   #For random intercept:
#'
#'   Z[[i]][,1]<-1
#'
#'   #For random slope:
#'
#'   Z[[i]][,2]<-sort(rexp(n=J,rate=1/20))
#'
#'   #Make first occasion 0:
#'
#'   Z[[i]][,2][1]<-0
#'
#' }
#'
#' #----------------------------------------------------------------------------------#
#'
#' #Define sigma2:
#'
#' sigma2<-1/20
#'
#' sigma2_I_J<-sigma2*diag(J)
#'
#' #Define G:
#'
#' G<-matrix(c(1/2,-10^-3,
#'             -10^-3,10^-4),
#'             q,q,byrow = T)
#'
#' #----------------------------------------------------------------------------------#
#'
#' #Generate outcome data:
#'
#' Y_transpose<-matrix(0,nrow=J,ncol=n)
#'
#' for (i in 1:n){
#'
#'   #This is X_i (J x Jp):
#'
#'   X_kron_i<-diag(J)%x%t(X[i,])
#'
#'   #Mean structure:
#'
#'   SS_MS_i<-X_kron_i%*%beta
#'
#'   #Covariance structure:
#'
#'   cov_y_i<-Z[[i]]%*%G%*%t(Z[[i]])+sigma2_I_J
#'
#'   #Outcome data:
#'
#'   Y_transpose[,i]<-MASS::mvrnorm(n=1,
#'                      mu=SS_MS_i,
#'                      Sigma=cov_y_i)
#'
#' }
#'
#' #----------------------------------------------------------------------------------#
#'
#' #Fit model to simulated data:
#'
#' I_J<-diag(J)
#'
#' I_q<-diag(q)
#'
#' model_results<-
#'
#' BSSMfs(
#'
#'     Y_transpose,
#'     X,
#'     Z,
#'     p_k,
#'
#'     nsim=2000,
#'     burn=1000,
#'     thin=1,
#'
#'     Q=10^-3*I_J,
#'
#'     C_0=10^-3*I_q
#'  )
#'
#' #Inspect feature parameter estimates:
#'
#' model_results$MCMC_summary%>%dplyr::filter(parameter=='beta')
#'
#' #Inspect MSE of longitudinal outcome data:
#'
#' model_results$MCMC_summary%>%dplyr::filter(parameter=='MSE')
#'
#' #Calculate and inspect DIC, using the posterior variance of the
#' #deviance to calculate the effective number of model parameters:
#'
#' model_results$MCMC_summary%>%
#' dplyr::filter(parameter=='deviance')%>%
#' dplyr::mutate(DIC=MCMC_mean+.5*(MCMC_sd^2))
#'
#' @references


#----------------------------------------------------------------------------------#

#' @import dplyr
#' @import MCMCpack
#' @import MASS
#' @import truncnorm
#' @import coda
#' @import tidyr

#' @export BSSMfs

BSSMfs<-function(
  Y_transpose,
  X,
  Z,
  p_k,
  #-------#

  nsim,
  burn,
  thin,

  mcmc_div=floor((nsim-burn)/4),

  #-------#

  covariate_select=c(NA),

  #-------#

  d=nrow(Y_transpose),
  Q,

  nu_0=ncol(Z[[1]]),
  C_0,

  alpha=10^-3,
  gamma=10^-3,

  o=10^-3,
  u=10^-3,

  a_1=1,
  b_1=1,

  g=1,
  h=1,

  #-------#

  raw_MCMC_output=F

){

  #----------------------------------------------------------------------------------#


  #Organize observed/simulated data in order to fit MCMC algorithm:####

  #Computational time:

  start_comp_time<-Sys.time()

  #----------------------------------------------------------------------------------#

  #Define J and n:

  J<-nrow(Y_transpose)

  n<-ncol(Y_transpose)

  #Create vec(Y^T) (Jn x 1):

  vec_Y_transpose<-matrix(Y_transpose,nrow=(J*n),ncol=1)


  #----------------------------------------------------------------------------------#

  #Define K:

  K<-length(p_k)

  #----------------------------------------------------------------------------------#

  #X (n x p):

  #Define p:

  p<-ncol(X)

  X_df<-X%>%as.data.frame

  rownames(X_df)<-1:n

  #feature names:

  feature_labels<-paste('p',1:p,sep='_')

  names(X_df)<-feature_labels

  #----------------------------------------------------------------------------------#

  #X_kron (nJ*Jp):

  X_kron_mat_i<-list()

  for (i in 1:n){

    #X_i is J x Jp

    X_kron_mat_i[[i]]<-diag(J)%x%(X_df[i,]%>%as.matrix)

  }


  X_kron_mat<-lapply(X_kron_mat_i,as.data.frame)

  #Collapse into a single df:

  #X (nJ*Jp):

  X_kron_mat<-dplyr::bind_rows(X_kron_mat)%>%as.matrix


  #----------------------------------------------------------------------------------#

  #For group level model only:

  #Creating x_ik (1 x p_k) and x_i(-k) (1 x p_(-k))

  X_dim_df<-data.frame(n=n,p_k=p_k,k=1:K)

  names(X_dim_df)<-c('n','p_k','k')

  X_dim_df<-X_dim_df%>%
    dplyr::mutate(cum_p_k=cumsum(p_k),
                  lag_cum_p_k=lag(cum_p_k)+1)

  X_dim_df$lag_cum_p_k[1]<-1



  X_partition_k<-list()

  X_partition_NOT_k<-list()


  print(('Creating X_ik'))

  for(i in 1:n){

    X_partition_k[[i]]<-list()

    X_partition_NOT_k[[i]]<-list()


    for(k in 1:K){

      k_select<-X_dim_df[k,]


      p_select<-k_select$lag_cum_p_k:k_select$cum_p_k

      p_NOT_select<-which(((1:p)%in%p_select)==F)


      #X_ik (J X Jp_k)

      X_partition_k[[i]][[k]]<-(diag(J))%x%as.matrix(X_df[i,p_select])

      #X_i_NOT_k (J x Jp_NOT_k)

      X_partition_NOT_k[[i]][[k]]<-(diag(J))%x%as.matrix(X_df[i,p_NOT_select])


    }#end k loop

    names(X_partition_k[[i]])<-paste('k',1:K,sep='_')

    names(X_partition_NOT_k[[i]])<-paste('k',1:K,sep='_')

  }#end i loop

  names(X_partition_k)<-paste('n',1:n,sep='_')

  names(X_partition_NOT_k)<-paste('n',1:n,sep='_')


  #----------------------------------------------------------------------------------#

  #For group level model only:

  #Creating X_k (nJ x Jp_k)
  #and X_-k (nJ x Jp_-k)

  X_partition_k_list<-list()

  X_partition_NOT_k_list<-list()

  print(('Creating X_k'))

  for(k in 1:K){

    X_partition_k_list[[k]]<-lapply(X_partition_k,"[", k)

    X_partition_NOT_k_list[[k]]<-lapply(X_partition_NOT_k,"[", k)

    #----------------------------------------------------------#


    X_partition_k_list[[k]]<-lapply(X_partition_k_list[[k]],as.data.frame)

    X_partition_NOT_k_list[[k]]<-lapply(X_partition_NOT_k_list[[k]],as.data.frame)

    #----------------------------------------------------------#

    #Collapse:

    #X_k (nJ x Jp_k)

    X_partition_k_list[[k]]<-dplyr::bind_rows(X_partition_k_list[[k]])

    X_partition_k_list[[k]]<-as.matrix(X_partition_k_list[[k]])


    #X_NOT_k (nJ x Jp_NOT_k)

    X_partition_NOT_k_list[[k]]<-dplyr::bind_rows(X_partition_NOT_k_list[[k]])

    X_partition_NOT_k_list[[k]]<-as.matrix(X_partition_NOT_k_list[[k]])

  }

  names(X_partition_k_list)<-paste('k',1:K,sep='_')

  names(X_partition_NOT_k_list)<-paste('k',1:K,sep='_')

  #----------------------------------------------------------------------------------#

  #For just bi-level feature selection:

  X_dim_df_bi_level<-data.frame(
    k=rep(1:K,as.vector(p_k)),
    p=1:p)


  #Creating x_il (1 x 1) and x_i(-l) (1 x (p-1))

  X_partition_l<-list()

  X_partition_NOT_l<-list()

  print(('Creating X_il'))

  for(i in 1:n){

    X_partition_l[[i]]<-list()

    X_partition_NOT_l[[i]]<-list()

    for(l in 1:p){

      p_select<-X_dim_df_bi_level[l,]$p

      p_NOT_select<-which(((1:p)%in%p_select)==F)

      #X_il (J X J)

      X_partition_l[[i]][[l]]<-(diag(J))%x%as.matrix(X_df[i,p_select])

      #X_i_NOT_k (J x Jp_NOT_k)

      X_partition_NOT_l[[i]][[l]]<-(diag(J))%x%as.matrix(X_df[i,p_NOT_select])

    }#end l loop

    names(X_partition_l[[i]])<-paste('l',1:p,sep='_')

    names(X_partition_NOT_l[[i]])<-paste('l',1:p,sep='_')

  }#end i loop

  names(X_partition_l)<-paste('n',1:n,sep='_')

  names(X_partition_NOT_l)<-paste('n',1:n,sep='_')


  #----------------------------------------------------------------------------------#

  #For the bi-level selection model only:

  #Creating X_l (nJ x J)
  #and X_-l (nJ x J(p-1))

  X_partition_l_list<-list()

  X_partition_NOT_l_list<-list()

  print(('Creating X_l'))

  for(l in 1:p){

    X_partition_l_list[[l]]<-lapply(X_partition_l,"[",l)

    X_partition_NOT_l_list[[l]]<-lapply(X_partition_NOT_l,"[",l)

    #----------------------------------------------------------#

    X_partition_l_list[[l]]<-lapply(X_partition_l_list[[l]],as.data.frame)

    X_partition_NOT_l_list[[l]]<-lapply(X_partition_NOT_l_list[[l]],as.data.frame)

    #----------------------------------------------------------#

    #Collapse:

    #X_l (nJ x J)

    X_partition_l_list[[l]]<-dplyr::bind_rows(X_partition_l_list[[l]])

    X_partition_l_list[[l]]<-as.matrix(X_partition_l_list[[l]])

    #X_NOT_l (nJ x J(p-1))

    X_partition_NOT_l_list[[l]]<-dplyr::bind_rows(X_partition_NOT_l_list[[l]])

    X_partition_NOT_l_list[[l]]<-as.matrix(X_partition_NOT_l_list[[l]])

  }#end l loop

  names(X_partition_l_list)<-paste('l',1:p,sep='_')

  names(X_partition_NOT_l_list)<-paste('l',1:p,sep='_')


  #----------------------------------------------------------------------------------#



  #----------------------------------------------------------------------------------#

  #Z (nJ X nq):

  #Z_i (J x q)

  print('creating Z')

  #Number of REs:

  q<-ncol(Z[[1]])

  Z_mat<-matrix(0,nrow=n*J,ncol=n*q)

  Z_dim_df<-data.frame(i=1:n,J=J,q=q)

  Z_dim_df<-Z_dim_df%>%
    dplyr::mutate(cum_J=cumsum(J),
                  cum_q=cumsum(q),
                  lag_cum_J=lag(cum_J)+1,
                  lag_cum_q=lag(cum_q)+1)

  Z_dim_df$lag_cum_J[1]<-1

  Z_dim_df$lag_cum_q[1]<-1

  for(i in 1:n){

    #Z (nJ x nq)

    Z_mat[Z_dim_df$lag_cum_J[i]:Z_dim_df$cum_J[i],
          Z_dim_df$lag_cum_q[i]:Z_dim_df$cum_q[i]]<-Z[[i]]

  }


  #----------------------------------------------------------------------------------#


  #################################

  #Start of MCMC function:####

  #################################


  MV_LMM_MCMC_fun<-function(nsim, burn, thin){


    #----------------------------------------------------------------------------------#

    #Initial values for stochastic model parameters:####

    #----------------------------------------------------------------------------------#

    print('Creating data objects re: initial MCMC values')

    #####Beta and (p x J) and beta (pJ x 1):#####

    Beta_dim_df<-data.frame(p_k=p_k,k=1:K)

    names(Beta_dim_df)<-c('p_k','k')

    Beta_dim_df<-Beta_dim_df%>%
      dplyr::mutate(cum_p_k=cumsum(p_k),
                    lag_cum_p_k=lag(cum_p_k)+1)

    Beta_dim_df$lag_cum_p_k[1]<-1

    Beta_dim_df<-Beta_dim_df%>%
      dplyr::mutate(total_p_NOT_k=p-p_k)

    #----------------------------------------------------------------------------------#


    #Want to partition beta (pJ x 1) into beta_k (p_k*J x 1) and beta_(-k) (p_(-k)*J x 1)

    beta<-matrix(rep(0,p*J),nrow=p*J,ncol=1)

    Beta<-matrix(beta,nrow=p,ncol=J)


    Beta_partition_k<-list()

    Beta_partition_NOT_k<-list()


    beta_partition_k<-list()

    beta_partition_NOT_k<-list()


    for(k in 1:K){

      k_select_Beta<-Beta_dim_df[k,]

      p_select<-k_select_Beta$lag_cum_p_k:k_select_Beta$cum_p_k

      p_NOT_select<-which(((1:(p))%in%p_select)==F)


      #B_k (p_k x J)

      Beta_partition_k[[k]]<-matrix((Beta[p_select,]),nrow=p_k[k],ncol=J)

      #B_-k (p_-k x J)

      Beta_partition_NOT_k[[k]]<-matrix((Beta[p_NOT_select,]),nrow=p-p_k[k],ncol=J)


      #For vec(Beta):

      #beta_k (p_kJ x 1)

      beta_partition_k[[k]]<-
        matrix(Beta_partition_k[[k]],
               nrow=k_select_Beta$p_k*J,ncol=1)

      #beta_-k (p_-kJ x 1)

      beta_partition_NOT_k[[k]]<-
        matrix(Beta_partition_NOT_k[[k]],
               nrow=k_select_Beta$total_p_NOT_k*J,ncol=1)


    }


    names(Beta_partition_k)<-paste('k',1:K,sep='_')

    names(Beta_partition_NOT_k)<-paste('k',1:K,sep='_')


    names(beta_partition_k)<-paste('k',1:K,sep='_')

    names(beta_partition_NOT_k)<-paste('k',1:K,sep='_')


    #----------------------------------------------------------------------------------#

    #Beta_l and Beta_NOT_l:####

    #For just bi-level feature selection:

    Beta_dim_df_bi_level<-data.frame(
      p=1:p)

    #Want to partition Beta into Beta_l^T (1 x J) and
    #Beta_(-l)^T ((p-1) x J)

    Beta_partition_l<-list()

    Beta_partition_NOT_l<-list()


    beta_partition_l<-list()

    beta_partition_NOT_l<-list()


    for(l in 1:p){

      p_select<-Beta_dim_df_bi_level$p[l]

      p_NOT_select<-which(((1:p)%in%p_select)==F)

      #B_l^T (1 X J)

      Beta_partition_l[[l]]<-
        (Beta[p_select,])%>%matrix(.,nrow=1,ncol=J)

      #B_-l^T ((p-1) X J)

      Beta_partition_NOT_l[[l]]<-
        (Beta[p_NOT_select,])%>%matrix(.,nrow=p-1,ncol=J)


      #beta_l (J X 1)

      beta_partition_l[[l]]<-
        matrix(Beta_partition_l[[l]],nrow=J,ncol=1)

      #beta_-l^T ((p-1)J x 1)

      beta_partition_NOT_l[[l]]<-
        matrix(Beta_partition_NOT_l[[l]],nrow=J*(p-1),ncol=1)

    }#end l loop

    names(Beta_partition_l)<-paste('l',1:p,sep='_')

    names(Beta_partition_NOT_l)<-paste('l',1:p,sep='_')


    names(beta_partition_l)<-paste('l',1:p,sep='_')

    names(beta_partition_NOT_l)<-paste('l',1:p,sep='_')



    #----------------------------------------------------------------------------------#


    #N.b. Beta_tilde is only for the bi-level selection model:

    #Beta_tilde (p x J) and beta_tilde (pJ x 1):####

    beta_tilde<-matrix(rep(0,p*J),nrow=p*J,ncol=1)

    Beta_tilde<-matrix(beta_tilde,nrow=p,ncol=J)


    Beta_tilde_partition_k<-list()

    Beta_tilde_partition_NOT_k<-list()


    beta_tilde_partition_k<-list()

    beta_tilde_partition_NOT_k<-list()


    for(k in 1:K){

      k_select_Beta<-Beta_dim_df[k,]

      p_select<-k_select_Beta$lag_cum_p_k:k_select_Beta$cum_p_k

      p_NOT_select<-which(((1:(p))%in%p_select)==F)


      #B_tilde_k (p_k x J)

      Beta_tilde_partition_k[[k]]<-matrix((Beta_tilde[p_select,]),nrow=p_k[k],ncol=J)

      #B_tilde_-k (p_-k x J)

      Beta_tilde_partition_NOT_k[[k]]<-matrix((Beta_tilde[p_NOT_select,]),nrow=p-p_k[k],ncol=J)


      #For vec(Beta_tilde):

      #beta_tilde_k (p_kJ x 1)

      beta_tilde_partition_k[[k]]<-
        matrix(Beta_tilde_partition_k[[k]],
               nrow=k_select_Beta$p_k*J,ncol=1)

      #beta_tilde_-k (p_-kJ x 1)

      beta_tilde_partition_NOT_k[[k]]<-
        matrix(Beta_tilde_partition_NOT_k[[k]],
               nrow=k_select_Beta$total_p_NOT_k*J,ncol=1)


    }


    names(Beta_tilde_partition_k)<-paste('k',1:K,sep='_')

    names(Beta_tilde_partition_NOT_k)<-paste('k',1:K,sep='_')


    names(beta_tilde_partition_k)<-paste('k',1:K,sep='_')

    names(beta_tilde_partition_NOT_k)<-paste('k',1:K,sep='_')



    #Beta_tilde_l and Beta_tilde_NOT_l:####

    #For just bi-level feature selection:

    #Want to partition Beta into Beta_l^T (1 x J) and
    #Beta_(-l)^T ((p-1) x J)

    Beta_tilde_partition_l<-list()

    Beta_tilde_partition_NOT_l<-list()


    beta_tilde_partition_l<-list()

    beta_tilde_partition_NOT_l<-list()


    for(l in 1:p){

      p_select<-Beta_dim_df_bi_level$p[l]

      p_NOT_select<-which(((1:p)%in%p_select)==F)

      #B_tilde_l^T (1 X J)

      Beta_tilde_partition_l[[l]]<-
        (Beta_tilde[p_select,])%>%matrix(.,nrow=1,ncol=J)

      #B_tilde_-l^T ((p-1) X J)

      Beta_tilde_partition_NOT_l[[l]]<-
        (Beta_tilde[p_NOT_select,])%>%matrix(.,nrow=p-1,ncol=J)


      #beta_tilde_l (J X 1)

      beta_tilde_partition_l[[l]]<-
        matrix(Beta_tilde_partition_l[[l]],nrow=J,ncol=1)

      #beta_tilde_-l^T ((p-1)J x 1)

      beta_tilde_partition_NOT_l[[l]]<-
        matrix(Beta_tilde_partition_NOT_l[[l]],nrow=J*(p-1),ncol=1)

    }#end l loop

    names(Beta_tilde_partition_l)<-paste('l',1:p,sep='_')

    names(Beta_tilde_partition_NOT_l)<-paste('l',1:p,sep='_')


    names(beta_tilde_partition_l)<-paste('l',1:p,sep='_')

    names(beta_tilde_partition_NOT_l)<-paste('l',1:p,sep='_')

    #----------------------------------------------------------------------------------#


    #----------------------------------------------------------------------------------#

    #pi_0k:####

    pi_0<-matrix(rep(1,K),nrow=K,ncol=1)


    #----------------------------------------------------------------------------------#

    #pi_0_l_k:####

    pi_0_l<-matrix(rep(1,p),nrow=p,ncol=1)

    #----------------------------------------------------------------------------------#

    #theta_beta:####

    theta_beta<-1

    #----------------------------------------------------------------------------------#

    #theta_tau_2:####

    theta_tau_2<-1

    #----------------------------------------------------------------------------------#

    #b (nq x 1):####

    b_list<-list()

    for(i in 1:n){

      b_list[[i]]<-matrix(0,
                          nrow=q,ncol=1)

    }

    b<-dplyr::bind_rows(lapply(b_list,as.data.frame))%>%as.matrix

    #----------------------------------------------------------------------------------#

    #sigma_2:####

    sigma_2<-1

    #----------------------------------------------------------------------------------#

    #s2####

    s_2<-1

    #----------------------------------------------------------------------------------#

    #Sigma (J x J):####

    Sigma<-1*diag(J)

    #----------------------------------------------------------------------------------#


    #tau_2_l (p x 1):####

    tau_2_l<-matrix(rep(1,p),nrow=p,ncol=1)

    #----------------------------------------------------------------------------------#

    #G (q x q)####

    G<-1*diag(q)

    #----------------------------------------------------------------------------------#

    #----------------------------------------------------------------------------------#

    #MCMC Info.:####


    # Number of MCMC Iterations

    nsim<-nsim

    # Thinning interval

    thin<-thin

    # Burnin

    burn<-burn

    #Total MCMC iterations:

    total_sim<-(nsim-burn)/thin

    #---------------------------------------------------------------------------------#


    #MCMC storage data objects:#####

    print('MCMC data storage objects')

    #For labeling MCMC storage objects:

    time_labels<-paste('t',1:J,sep='')

    #---------------------------------------------------------------------------------#


    #theta_beta:####

    theta_beta_mcmc_df<-data.frame(value=0,
                                   feature_group_size=NA,
                                   feature_group=NA,
                                   subject=NA,
                                   parameter='theta_beta',
                                   row_occasion=NA,
                                   occasion=NA,
                                   q=NA,
                                   row_q=NA,
                                   feature=NA)

    theta_beta_mcmc_df<-lapply(1:total_sim, function(x) theta_beta_mcmc_df)%>%
      dplyr::bind_rows()

    theta_beta_mcmc_df$mcmc_iter=1:total_sim


    #---------------------------------------------------------------------------------#

    #theta_tau_2:#####

    theta_tau_2_mcmc_df<-data.frame(value=0,
                                    feature_group_size=NA,
                                    feature_group=NA,
                                    occasion=NA,
                                    subject=NA,
                                    parameter='theta_tau_2',
                                    row_occasion=NA,
                                    q=NA,
                                    row_q=NA,
                                    feature=NA)

    theta_tau_2_mcmc_df<-
      lapply(1:total_sim, function(x) theta_tau_2_mcmc_df)%>%
      dplyr::bind_rows()

    theta_tau_2_mcmc_df$mcmc_iter=1:total_sim

    #---------------------------------------------------------------------------------#

    #sigma_2:#####

    sigma_2_mcmc_df<-data.frame(value=0,
                                feature_group_size=NA,
                                feature_group=NA,
                                subject=NA,
                                parameter='sigma2',
                                row_occasion=NA,
                                occasion=NA,
                                q=NA,
                                row_q=NA,
                                feature=NA)

    sigma_2_mcmc_df<-lapply(1:total_sim, function(x) sigma_2_mcmc_df)%>%
      dplyr::bind_rows()

    sigma_2_mcmc_df$mcmc_iter=1:total_sim


    #---------------------------------------------------------------------------------#

    #s_2:#####

    s_2_mcmc_df<-data.frame(value=0,
                            feature_group_size=NA,
                            feature_group=NA,
                            subject=NA,
                            parameter='s_2',
                            row_occasion=NA,
                            occasion=NA,
                            q=NA,
                            row_q=NA,
                            feature=NA)

    s_2_mcmc_df<-lapply(1:total_sim, function(x) s_2_mcmc_df)%>%
      dplyr::bind_rows()

    s_2_mcmc_df$mcmc_iter=1:total_sim



    #---------------------------------------------------------------------------------#

    #Sigma (J X J):####

    Sigma_mcmc_df<-as.data.frame(matrix(rep(0,J^2),nrow=J,ncol=J))

    names(Sigma_mcmc_df)<-time_labels

    Sigma_mcmc_df$row_occasion<-time_labels

    Sigma_mcmc_df<-Sigma_mcmc_df%>%
      tidyr::gather(occasion,value,-row_occasion)%>%
      dplyr::mutate(feature_group_size=NA,
                    feature_group=NA,
                    subject=NA,
                    parameter='Sigma',
                    q=NA,
                    row_q=NA,
                    feature=NA)


    Sigma_mcmc_list<-lapply(1:total_sim, function(x) Sigma_mcmc_df)


    #---------------------------------------------------------------------------------#

    #G (q X q):####

    G_mcmc_df<-as.data.frame(matrix(rep(0,q^2),nrow=q,ncol=q))

    names(G_mcmc_df)<-as.character(1:q)

    G_mcmc_df$row_q<-1:q

    G_mcmc_df<-G_mcmc_df%>%
      tidyr::gather(q,value,-row_q)%>%
      dplyr::mutate(feature_group_size=NA,
                    feature_group=NA,
                    subject=NA,
                    parameter='G',
                    feature=NA,
                    occasion=NA,
                    row_occasion=NA)


    G_mcmc_list<-lapply(1:total_sim, function(x) G_mcmc_df)





    #---------------------------------------------------------------------------------#

    #Y_transpose (J X n):####

    Y_transpose_mcmc_df<-as.data.frame(matrix(rep(0,J*n),nrow=J,ncol=n))

    names(Y_transpose_mcmc_df)<-as.character(1:n)

    Y_transpose_mcmc_df$occasion<-time_labels

    Y_transpose_mcmc_df<-Y_transpose_mcmc_df%>%
      tidyr::gather(subject,value,-occasion)%>%
      dplyr::mutate(feature_group_size=NA,
                    feature_group=NA,
                    parameter='y',
                    feature=NA,
                    q=NA,
                    row_q=NA,
                    row_occasion=NA)


    Y_transpose_mcmc_list<-lapply(1:total_sim, function(x) Y_transpose_mcmc_df)


    #---------------------------------------------------------------------------------#

    #pi_0_k:####

    pi_0_k_mcmc_df<-data.frame(value=rep(0,K),
                               feature_group_size=as.vector(p_k),
                               feature_group=1:K,
                               occasion=NA,
                               subject=NA,
                               parameter='pi_0_k',
                               row_occasion=NA,
                               q=NA,
                               row_q=NA,
                               feature=NA)


    pi_0_k_mcmc_list<-lapply(1:total_sim, function(x) pi_0_k_mcmc_df)



    #---------------------------------------------------------------------------------#

    #pi_0_l_k:####

    pi_0_l_k_mcmc_list<-list()

    for(k in 1:K){

      k_select<-Beta_dim_df[k,]

      p_select<-k_select$lag_cum_p_k:k_select$cum_p_k


      pi_0_l_k_mcmc_list[[k]]<-matrix(0,nrow=p_k[k],ncol=1)

      pi_0_l_k_mcmc_list[[k]]<-as.data.frame(pi_0_l_k_mcmc_list[[k]])

      pi_0_l_k_mcmc_list[[k]]$feature_group_size<-p_k[k]

      pi_0_l_k_mcmc_list[[k]]$feature<-p_select

      names(pi_0_l_k_mcmc_list[[k]])[1]<-'value'

      pi_0_l_k_mcmc_list[[k]]<-pi_0_l_k_mcmc_list[[k]]%>%
        dplyr::mutate(
          feature_group=k,
          subject=NA,
          occasion=NA,
          subject=NA,
          parameter='pi_0_l_k',
          row_occasion=NA,
          q=NA,
          row_q=NA,
          parameter='pi_0_l_k')

    }


    #Collapse:

    pi_0_l_k_mcmc_df<-dplyr::bind_rows(pi_0_l_k_mcmc_list)


    pi_0_l_k_mcmc_list_combo<-lapply(1:total_sim, function(x) pi_0_l_k_mcmc_df)


    #---------------------------------------------------------------------------------#

    #b_i:####

    b_mcmc_df<-as.data.frame(matrix(0,nrow=q,ncol=n))

    names(b_mcmc_df)<-as.character(1:n)

    b_mcmc_df<-b_mcmc_df%>%
      dplyr::mutate(q=1:q)%>%
      tidyr::gather(subject,value,-q)%>%
      dplyr::mutate(
        feature_group_size=NA,
        feature_group=NA,
        parameter='b',
        row_occasion=NA,
        occasion=NA,
        row_q=NA,
        feature=NA)

    b_mcmc_list<-lapply(1:total_sim, function(x) b_mcmc_df)


    #---------------------------------------------------------------------------------#

    #Beta:####

    Beta_mcmc_list<-list()

    for(k in 1:K){

      Beta_mcmc_list[[k]]<-matrix(0,nrow=Beta_dim_df$p_k[k],ncol=J)

      Beta_mcmc_list[[k]]<-as.data.frame(Beta_mcmc_list[[k]])

      names(Beta_mcmc_list[[k]])<-time_labels

      Beta_mcmc_list[[k]]$feature_group_size<-Beta_dim_df$p_k[k]

      Beta_mcmc_list[[k]]<-Beta_mcmc_list[[k]]%>%
        tidyr::gather(occasion,value,-feature_group_size)


    }


    p_k_index<-list()

    for(k in 1:K){

      p_k_index[[k]]<-X_dim_df$lag_cum_p_k[k]:X_dim_df$cum_p_k[k]

      p_k_index[[k]]<-as.character(p_k_index[[k]])

    }

    for(k in 1:K){

      Beta_mcmc_list[[k]]$feature=as.numeric(p_k_index[[k]])

      Beta_mcmc_list[[k]]<-Beta_mcmc_list[[k]]%>%
        dplyr::mutate(
          feature_group=k,
          subject=NA,
          parameter='beta',
          row_occasion=NA,
          q=NA,
          row_q=NA)

    }


    #Collapse:

    beta_mcmc_df<-dplyr::bind_rows(Beta_mcmc_list)


    beta_mcmc_list_combo<-lapply(1:total_sim, function(x) beta_mcmc_df)


    #---------------------------------------------------------------------------------#

    #tau_2_l_k:####

    tau_2_l_k_mcmc_list<-list()

    for(k in 1:K){

      k_select<-Beta_dim_df[k,]

      p_select<-k_select$lag_cum_p_k:k_select$cum_p_k


      tau_2_l_k_mcmc_list[[k]]<-matrix(0,nrow=p_k[k],ncol=1)

      tau_2_l_k_mcmc_list[[k]]<-as.data.frame(tau_2_l_k_mcmc_list[[k]])

      tau_2_l_k_mcmc_list[[k]]$feature_group_size<-p_k[k]

      tau_2_l_k_mcmc_list[[k]]$feature<-p_select

      names(tau_2_l_k_mcmc_list[[k]])[1]<-'value'

      tau_2_l_k_mcmc_list[[k]]<-tau_2_l_k_mcmc_list[[k]]%>%
        dplyr::mutate(
          feature_group=k,
          subject=NA,
          occasion=NA,
          subject=NA,
          parameter='tau_2_l_k',
          row_occasion=NA,
          q=NA,
          row_q=NA)

    }


    #Collapse:

    tau_2_l_k_mcmc_df<-dplyr::bind_rows(tau_2_l_k_mcmc_list)


    tau_2_l_k_mcmc_list_combo<-lapply(1:total_sim, function(x) tau_2_l_k_mcmc_df)


    #---------------------------------------------------------------------------------#

    #MSE:####

    MSE_mcmc_df<-data.frame(value=0,
                            feature_group_size=NA,
                            feature_group=NA,
                            subject=NA,
                            parameter='MSE',
                            row_occasion=NA,
                            occasion=NA,
                            q=NA,
                            row_q=NA,
                            feature=NA)

    MSE_mcmc_df<-lapply(1:total_sim, function(x) MSE_mcmc_df)%>%
      dplyr::bind_rows()

    MSE_mcmc_df$mcmc_iter=1:total_sim

    #----------------------------------------------------------------------------------#

    #Deviance:####

    deviance_mcmc_df<-
      data.frame(value=0,
                 feature_group_size=NA,
                 feature_group=NA,
                 subject=NA,
                 parameter='deviance',
                 row_occasion=NA,
                 occasion=NA,
                 q=NA,
                 row_q=NA,
                 feature=NA)

    deviance_mcmc_df<-lapply(1:total_sim, function(x) deviance_mcmc_df)%>%bind_rows()

    deviance_mcmc_df$mcmc_iter=1:total_sim

    #----------------------------------------------------------------------------------#

    #######################################################

    #Start Gibbs sampler:####

    #######################################################

    tmp<-proc.time() # Store current time

    print('MCMC Sampling')

    #I_J:

    I_J<-diag(J)

    for (mcmc in 1:nsim){

      #Progress bar:

      if(mcmc %% mcmc_div == 0){
        cat(mcmc)
        cat("..")
      }

      #----------------------------------------------------------------------------------#

      #Initialize this counter for indexing l=1,...,p

      index_l<-1

      #----------------------------------------------------------------------------------#

      #These for loops pertain to model parameters indexed by l_k:


      #Start k loop:

      for(k in 1:K){

        #Start l loop for Gibbs sampler:

        for(l in 1:p_k[k]){

          #Calculate z_l_k (Jn x 1):

          z_l_k<-
            vec_Y_transpose-
            (X_partition_NOT_l_list[[index_l]]%*%
               beta_partition_NOT_l[[index_l]]+
               Z_mat%*%b)

          #----------------------------------------------------------------------------------#

          #Calculate m_l_k (nJ x1):

          m_l_k<-X_partition_l_list[[index_l]]%*%
            beta_tilde_partition_l[[index_l]]

          #----------------------------------------------------------------------------------#

          #Calculate sigma_l_k_inverse (1x1)

          sigma_l_k_inverse<-s_2/(sigma_2+(s_2*t(m_l_k)%*%m_l_k))

          #----------------------------------------------------------------------------------#

          #Update tau_2_l_k:####

          tau_2_l[index_l]<-


            (pi_0_l[index_l]*
               truncnorm::rtruncnorm(n=1,
                                     a=0,b=Inf,
                                     mean=sigma_l_k_inverse*t(m_l_k)%*%z_l_k,
                                     sd=sqrt(sigma_2*sigma_l_k_inverse)
               )
            )+

            (1-pi_0_l[index_l])*matrix(0,nrow=1,ncol=1)


          #----------------------------------------------------------------------------------#

          #Update pi_0_l_k:####

          if(k%in%covariate_select){

            pi_0_l[index_l]<-1

          }

          #Calculate f_l_k

          f_l_k<-t(z_l_k)%*%m_l_k*sigma_l_k_inverse*t(m_l_k)%*%z_l_k

          exp_term_l_k<-exp(f_l_k/(2*sigma_2))

          if(exp_term_l_k==Inf){

            exp_term_l_k<-
              10^20

          }

          #Calculate numerator of pi_0_l_k posterior prob. parameter:

          pi_0_l_k_prob_num<-

            theta_tau_2*

            2*sqrt(sigma_2/s_2)*

            exp_term_l_k*

            sqrt(sigma_l_k_inverse)*

            pnorm(
              q=sqrt(f_l_k/sigma_2),

              mean=0,

              sd=1
            )

          if(pi_0_l_k_prob_num==Inf){

            pi_0_l_k_prob_num<-
              10^20

          }


          pi_0_l[index_l]<-
            rbinom(
              n=1,
              size=1,
              prob=pi_0_l_k_prob_num/((1-theta_tau_2)+pi_0_l_k_prob_num)
            )





          #----------------------------------------------------------------------------------#

          #----------------------------------------------------------------------------------#

          #Update counter:

          index_l<-index_l+1

        }#end of l loop for Gibbs sampler

      }#end of k loop for Gibbs sampler



      #----------------------------------------------------------------------------------#

      #----------------------------------------------------------------------------------#


      #This for loop pertains to model parameters indexed by k:

      #----------------------------------------------------------------------------------#

      #Initialize this counter for indexing p_k and p_-k

      index_k_start<-1

      #----------------------------------------------------------------------------------#

      for(k in 1:K){

        #Update index for p_k and p_-k:

        index_k_end <- index_k_start + p_k[k] - 1


        #Update V_k (p_k x p_k) via tau_2_l (p x 1):####

        if(p_k[k]>1){

          V_k<-diag(c(tau_2_l[index_k_start:index_k_end]))

        }else{

          V_k<-matrix(tau_2_l[index_k_start:index_k_end],
                      nrow=1,
                      ncol=1)

        }


        #Calculate Omega_k (Jp_k x Jp_k):

        Omega_k<-
          (
            (I_J%x%V_k)%*%
              t(X_partition_k_list[[k]])%*%X_partition_k_list[[k]]%*%
              (I_J%x%V_k)
          )+
          (sigma_2*solve((Sigma%x%diag(p_k[k]))))


        #Calculate zeta_k (Jn x 1):

        zeta_k<-
          vec_Y_transpose-
          (X_partition_NOT_k_list[[k]]%*%beta_partition_NOT_k[[k]]+
             Z_mat%*%b)



        #----------------------------------------------------------------------------------#

        #Update pi_0k (K x 1):####

        if(k%in%covariate_select){

          pi_0[k]<-1

        }


        #Define f_k (1x1)

        f_k<-
          t(zeta_k)%*%X_partition_k_list[[k]]%*%
          (I_J%x%V_k)%*%
          solve(Omega_k)%*%
          (I_J%x%V_k)%*%
          t(X_partition_k_list[[k]])%*%zeta_k


        exp_term_k<-exp((1/(2*sigma_2))*
                          f_k)


        if(exp_term_k==Inf){

          exp_term_k<-
            10^20

        }



        pi_0_k_prob_num<-

          theta_beta*(sigma_2)^(0.5*J*p_k[k])*

          (det(Sigma%x%diag(p_k[k])))^(-1/2)*

          exp_term_k*

          (det(solve(Omega_k)))^(1/2)


        if(pi_0_k_prob_num==Inf){

          pi_0_k_prob_num<-
            10^20

        }



        pi_0[k]<-
          rbinom(
            n=1,
            size=1,
            prob=pi_0_k_prob_num/((1-theta_beta)+pi_0_k_prob_num)
          )





        #----------------------------------------------------------------------------------#

        #Update beta_tilde_k (p_k*J x 1):####

        beta_tilde_partition_k[[k]]<-
          (
            (pi_0[k]*MASS::mvrnorm(n=1,
                                   mu=solve(Omega_k)%*%
                                     (I_J%x%V_k)%*%
                                     t(X_partition_k_list[[k]])%*%zeta_k,
                                   Sigma=sigma_2*solve(Omega_k))
            )
            %>%matrix(.,nrow=p_k[k]*J,ncol=1)
          ) +
          (
            (1-pi_0[k])*matrix(0,nrow=(p_k[k]*J))
          )


        #Calculate Beta_tilde_k (p_k x J) via updated beta_tilde_k (p_k*J x 1):

        Beta_tilde_partition_k[[k]]<-
          beta_tilde_partition_k[[k]]%>%matrix(.,nrow=p_k[k],ncol=J,byrow=F)


        #Then, calculate Beta_k (p_k x J) via V_k for bi-level selection model:

        Beta_partition_k[[k]]<-
          V_k%*%Beta_tilde_partition_k[[k]]

        #Update beta_k (p_kJ x 1) for output:

        beta_partition_k[[k]]<-
          matrix(Beta_partition_k[[k]],
                 nrow=p_k[k]*J,ncol=1)

        #----------------------------------------------------------------------------------#

        #----------------------------------------------------------------------------------#

        #Update counter:

        index_k_start = index_k_end + 1

      }#end of k loop for Gibbs sampler

      #----------------------------------------------------------------------------------#

      #----------------------------------------------------------------------------------#

      #N.b. we need to update the partitions (l and k) of Beta and
      #Beta_tilde:

      #Create updated Beta (p x J) via Beta_partition_k:####

      Beta<-dplyr::bind_rows(lapply(Beta_partition_k,as.data.frame))%>%
        as.matrix

      #Create updated beta (pJ x 1) via Beta:

      beta<-
        matrix(Beta,
               nrow=p*J,ncol=1)


      #Update Beta_tilde (p x J):####

      Beta_tilde<-dplyr::bind_rows(lapply(Beta_tilde_partition_k,as.data.frame))%>%
        as.matrix

      #----------------------------------------------------------------------------------#


      #----------------------------------------------------------------------------------#

      #Partition beta


      for(k in 1:K){

        k_select_Beta<-Beta_dim_df[k,]

        p_select<-k_select_Beta$lag_cum_p_k:k_select_Beta$cum_p_k

        p_NOT_select<-which(((1:(p))%in%p_select)==F)

        #B_-k (p_-k x J)

        Beta_partition_NOT_k[[k]]<-matrix((Beta[p_NOT_select,]),nrow=p-p_k[k],ncol=J)

        #For vec(Beta):

        #beta_-k (p_-kJ x 1)

        beta_partition_NOT_k[[k]]<-
          matrix(Beta_partition_NOT_k[[k]],
                 nrow=k_select_Beta$total_p_NOT_k*J,ncol=1)

      }


      #Beta_l and Beta_NOT_l:

      for(l in 1:p){

        p_select<-Beta_dim_df_bi_level$p[l]

        p_NOT_select<-which(((1:p)%in%p_select)==F)

        #B_-l^T ((p-1) X J)

        Beta_partition_NOT_l[[l]]<-
          (Beta[p_NOT_select,])%>%
          matrix(.,nrow=p-1,ncol=J)

        #beta_-l^T ((p-1)J x 1)

        beta_partition_NOT_l[[l]]<-
          matrix(Beta_partition_NOT_l[[l]],nrow=J*(p-1),ncol=1)

      }#end l loop


      #----------------------------------------------------------------------------------#

      #Beta_tilde (p x J) and beta_tilde (pJ x 1):

      #Want to partition Beta into Beta_l^T (1 x J) and
      #Beta_(-l)^T ((p-1) x J)

      for(l in 1:p){

        p_select<-Beta_dim_df_bi_level$p[l]

        #B_tilde_l^T (1 X J)

        Beta_tilde_partition_l[[l]]<-
          (Beta_tilde[p_select,])%>%matrix(.,nrow=1,ncol=J)

        #beta_tilde_l (J X 1)

        beta_tilde_partition_l[[l]]<-
          matrix(Beta_tilde_partition_l[[l]],nrow=J,ncol=1)


      }#end l loop

      #----------------------------------------------------------------------------------#

      #----------------------------------------------------------------------------------#

      #Update b (nq x 1):####


      #Create variable for easier expression of full conditional posterior of b:

      Z_t_Z_term<-solve(t(Z_mat)%*%Z_mat+sigma_2*solve(diag(n)%x%G))

      b<-
        MASS::mvrnorm(
          n=1,
          mu=Z_t_Z_term%*%t(Z_mat)%*%(vec_Y_transpose-X_kron_mat%*%beta),
          Sigma=sigma_2*Z_t_Z_term
        )%>%
        as.matrix

      #Separate (i.e. vec) by subject for full conditional posterior of G:

      b_mat<-matrix(b,nrow=q,ncol=n,byrow=F)


      #----------------------------------------------------------------------------------#

      #Update G (q x q):####

      #N.b. this is equal to sum(b_list[[i]]%*%t(b_list[[i]]))!

      sum_b_mat_outer_product<-b_mat%*%t(b_mat)

      G<-MCMCpack::riwish(v=n+nu_0,
                          S=sum_b_mat_outer_product+C_0)

      #----------------------------------------------------------------------------------#

      #Update theta_beta (1 x 1):####

      theta_beta<-rbeta(n=1,
                        shape1=sum(pi_0)+a_1,
                        shape2=b_1+K-sum(pi_0))


      #----------------------------------------------------------------------------------#


      #Update theta_tau_2 (1 x 1):####

      theta_tau_2<-rbeta(n=1,
                         shape1=(pi_0_l%>%unlist%>%sum)+g,
                         shape2=h+(p)-(pi_0_l%>%unlist%>%sum))


      #----------------------------------------------------------------------------------#

      #Define residual term:

      z<-vec_Y_transpose-(X_kron_mat%*%beta+Z_mat%*%b)

      #----------------------------------------------------------------------------------#

      #Update sigma_2 (1 x 1):####

      sigma_2<-MCMCpack::rinvgamma(n=1,
                                   shape=(.5*(n*J))+alpha,
                                   scale=(.5*(t(z)%*%z))+gamma)

      #----------------------------------------------------------------------------------#


      #Update Sigma (J x J):####

      #Identify for which partitions of Beta_tilde pi_0_k =1

      p_k_selected<-p_k[(pi_0==1)]

      Sigma<-MCMCpack::riwish(v=d+sum(p_k_selected),
                              S=(t(Beta_tilde)%*%Beta_tilde)+Q
      )

      #----------------------------------------------------------------------------------#


      #Update s_2 (1 x 1):####

      s_2<-MCMCpack::rinvgamma(n=1,
                               shape=(pi_0_l%>%unlist%>%sum/2)+o,
                               scale=(((tau_2_l%>%unlist)^2%>%sum)/2)+u
      )


      # if(is.nan(s_2)==T|s_2==Inf){
      #
      #   s_2<-MCMCpack::rinvgamma(n=1,
      #                            shape=o,
      #                            scale=u
      #   )
      #
      # }

      #----------------------------------------------------------------------------------#

      #----------------------------------------------------------------------------------#

      #Generate predicted values of Y (n x J):####

      #Subject-specific mean structure:

      SS_MS<-(X_kron_mat%*%beta)+((Z_mat%*%b%>%as.matrix))

      vec_Y_transpose_predicted<-
        SS_MS


      #Calculate MSE:####

      # MSE<-sum((vec_Y_transpose-vec_Y_transpose_predicted)^2)/(n*J)

      MSE<-sum((vec_Y_transpose-SS_MS)^2)/(n*J)


      #----------------------------------------------------------------------------------#

      #Calculate deviance:####

      model_deviance<-
        (n*J*log(2*pi*sigma_2))+
        (t(z)%*%z*(1/sigma_2))

      #----------------------------------------------------------------------------------#

      #----------------------------------------------------------------------------------#

      # Store MCMC samples: #####

      if (mcmc>burn && mcmc%%thin==0) {

        #----------------------------------------------------------------------------------#

        #theta_beta:

        theta_beta_mcmc_df$value[(mcmc-burn)/thin]<-theta_beta


        #----------------------------------------------------------------------------------#


        #theta_tau_2:

        theta_tau_2_mcmc_df$value[(mcmc-burn)/thin]<-theta_tau_2

        #----------------------------------------------------------------------------------#


        #sigma_2:

        sigma_2_mcmc_df$value[(mcmc-burn)/thin]<-sigma_2


        #----------------------------------------------------------------------------------#

        #MSE:

        MSE_mcmc_df$value[(mcmc-burn)/thin]<-MSE

        #----------------------------------------------------------------------------------#

        #Deviance:

        deviance_mcmc_df$value[(mcmc-burn)/thin]<-model_deviance

        #----------------------------------------------------------------------------------#

        #s_2:

        s_2_mcmc_df$value[(mcmc-burn)/thin]<-s_2

        #----------------------------------------------------------------------------------#


        #Sigma:

        Sigma_mcmc_list[[(mcmc-burn)/thin]]$value<-matrix(Sigma,nrow=J*J,ncol=1)

        Sigma_mcmc_list[[(mcmc-burn)/thin]]$mcmc_iter<-((mcmc-burn)/thin)



        #----------------------------------------------------------------------------------#

        #G:

        G_mcmc_list[[(mcmc-burn)/thin]]$value<-matrix(G,nrow=q*q,ncol=1)

        G_mcmc_list[[(mcmc-burn)/thin]]$mcmc_iter<-((mcmc-burn)/thin)



        #----------------------------------------------------------------------------------#

        #Y_pred:

        Y_transpose_mcmc_list[[(mcmc-burn)/thin]]$value<-vec_Y_transpose_predicted

        Y_transpose_mcmc_list[[(mcmc-burn)/thin]]$mcmc_iter<-((mcmc-burn)/thin)

        #----------------------------------------------------------------------------------#


        #----------------------------------------------------------------------------------#

        #tau_2_l:

        tau_2_l_k_mcmc_list_combo[[(mcmc-burn)/thin]]$value<-as.vector(tau_2_l)

        tau_2_l_k_mcmc_list_combo[[(mcmc-burn)/thin]]$mcmc_iter<-((mcmc-burn)/thin)


        #----------------------------------------------------------------------------------#

        #pi_0k:

        pi_0_k_mcmc_list[[(mcmc-burn)/thin]]$value<-as.vector(pi_0)

        pi_0_k_mcmc_list[[(mcmc-burn)/thin]]$mcmc_iter<-((mcmc-burn)/thin)


        #----------------------------------------------------------------------------------#

        #pi_0_l_k:

        pi_0_l_k_mcmc_list_combo[[(mcmc-burn)/thin]]$value<-as.vector(pi_0_l)

        pi_0_l_k_mcmc_list_combo[[(mcmc-burn)/thin]]$mcmc_iter<-((mcmc-burn)/thin)

        #----------------------------------------------------------------------------------#

        #b:

        b_mcmc_list[[(mcmc-burn)/thin]]$value<-matrix(b_mat,nrow=(q*n),ncol=1)

        b_mcmc_list[[(mcmc-burn)/thin]]$mcmc_iter<-((mcmc-burn)/thin)

        #----------------------------------------------------------------------------------#

        #beta (pJ x1):

        beta_mcmc_list_combo[[(mcmc-burn)/thin]]$value<-
          lapply(beta_partition_k,as.data.frame)%>%
          dplyr::bind_rows()%>%
          unlist%>%as.vector

        beta_mcmc_list_combo[[(mcmc-burn)/thin]]$mcmc_iter<-((mcmc-burn)/thin)


      }#end of Store MCMC samples if statement


      #######################################################
      #End of MCMC iterations for loop
      #######################################################

    }



    #Collapse relevant MCMC storage lists:


    Sigma_mcmc_df<-dplyr::bind_rows(Sigma_mcmc_list)

    G_mcmc_df<-dplyr::bind_rows(G_mcmc_list)

    Y_transpose_mcmc_df<-dplyr::bind_rows(Y_transpose_mcmc_list)

    #----------------------------------------------------------------------------------#

    pi_0_l_k_mcmc_df<-dplyr::bind_rows(pi_0_l_k_mcmc_list_combo)

    tau_2_l_k_mcmc_df<-dplyr::bind_rows(tau_2_l_k_mcmc_list_combo)

    #----------------------------------------------------------------------------------#


    pi_0_k_mcmc_df<-dplyr::bind_rows(pi_0_k_mcmc_list)


    #----------------------------------------------------------------------------------#

    b_mcmc_df<-dplyr::bind_rows(b_mcmc_list)

    beta_mcmc_df<-dplyr::bind_rows(beta_mcmc_list_combo)

    #----------------------------------------------------------------------------------#

    #----------------------------------------------------------------------------------#

    MCMC_output_df<-rbind(beta_mcmc_df,
                          pi_0_l_k_mcmc_df,
                          tau_2_l_k_mcmc_df,
                          s_2_mcmc_df,
                          b_mcmc_df,
                          pi_0_k_mcmc_df,
                          G_mcmc_df,
                          Y_transpose_mcmc_df,
                          Sigma_mcmc_df,
                          sigma_2_mcmc_df,
                          theta_beta_mcmc_df,
                          theta_tau_2_mcmc_df,
                          MSE_mcmc_df,
                          deviance_mcmc_df)

    #----------------------------------------------------------------------------------#

    #Computation time:

    end_comp_time<-Sys.time()


    #Summary MCMC measures:

    #Function for calculating MCMC autocorrelation using the coda package:

    auto_corr_tidy_df_summ_fun<-function(mcmc_value,lag){

      auto_corr<-coda::mcmc(mcmc_value)%>%coda::autocorr(.,lags=lag,relative=F)

      auto_corr_df<-auto_corr[,,1]%>%as.data.frame

      row_names_autocorr<-rownames(auto_corr_df)

      names(auto_corr_df)<-'autocorrelation'

      auto_corr_df$lag<-row_names_autocorr

      return(auto_corr_df$autocorrelation)

    }


    #Summarized MCMC results:

    MCMC_summary_df<-MCMC_output_df%>%
      dplyr::group_by(feature,occasion,feature_group,
                      subject,parameter,
                      row_occasion,q,row_q)%>%
      dplyr::summarise(MCMC_mean=mean(value),
                       MCMC_median=median(value),
                       MCMC_sd=sd(value),
                       MCMC_LCL=quantile(value,probs = .05/2),
                       MCMC_UCL=quantile(value,probs=1-(.05/2)),
                       Geweke_diag=coda::geweke.diag(coda::mcmc(value))[[1]]%>%as.vector,
                       # lag_0=auto_corr_tidy_df_summ_fun(mcmc_value=value,lag=0),
                       lag_5=auto_corr_tidy_df_summ_fun(mcmc_value=value,lag=5),
                       lag_10=auto_corr_tidy_df_summ_fun(mcmc_value=value,lag=10),
                       ESS=coda::mcmc(value)%>%coda::effectiveSize(.))



    # MCMC_summary_df$feature<-as.character(MCMC_summary_df$feature)

    #Modify notation a bit:

    MCMC_summary_df<-MCMC_summary_df%>%
      dplyr::mutate(parameter=
                      ifelse(parameter=='pi_0_l_k','pi_0_l',
                             ifelse(parameter=='tau_2_l_k','tau_2_l',parameter)))


    #Format output a bit:

    MCMC_summary_df<-as.data.frame(MCMC_summary_df)

    MCMC_summary_df$comp_time<-(end_comp_time-start_comp_time)%>%as.numeric

    MCMC_summary_df<-MCMC_summary_df%>%
      dplyr::mutate(subject=as.numeric(subject),
                    q=as.numeric(q))

    #----------------------------------------------------------------------------------#

    #Raw MCMC results:

    MCMC_output_df$comp_time<-(end_comp_time-start_comp_time)%>%as.numeric

    MCMC_output_df<-MCMC_output_df%>%
      dplyr::select(-feature_group_size)


    #Modify notation a bit:

    MCMC_output_df<-MCMC_output_df%>%
      dplyr::mutate(parameter=
                      ifelse(parameter=='pi_0_l_k','pi_0_l',
                             ifelse(parameter=='tau_2_l_k','tau_2_l',parameter)))

    #Format output a bit:

    MCMC_output_df<-MCMC_output_df%>%
      dplyr::mutate(subject=as.numeric(subject),
                    q=as.numeric(q))


    #----------------------------------------------------------------------------------#

    #end of Gibbs sampler##########

    #----------------------------------------------------------------------------------#

    #----------------------------------------------------------------------------------------#

    if(raw_MCMC_output==T){

      return(list(

        'MCMC_output'=MCMC_output_df,

        'MCMC_summary'=MCMC_summary_df

      ))

    }else{

      return(list(

        'MCMC_summary'=MCMC_summary_df

      ))

    }



  }

  #Run MCMC algorithm function:

  MCMC_output<-MV_LMM_MCMC_fun(nsim, burn, thin)

  #End of simulation/MCMC function:

  return(MCMC_output)

}



