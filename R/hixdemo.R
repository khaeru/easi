hixdemo <- function(use_var_soc = TRUE) {
  data(hixdata)

  shares_HIX=hixdata[,2:10]
  log.price_HIX=hixdata[,11:19]

  if (use_var_soc) {
    var.soc_HIX <- hixdata[,21:25]
  } else {
    var.soc_HIX <- NULL
  }

  log.exp_HIX=hixdata[,20]

  est <- easi(shares=shares_HIX, log.price=log.price_HIX,
              var.soc=var.soc_HIX, log.exp=log.exp_HIX)

  return(est)
}
