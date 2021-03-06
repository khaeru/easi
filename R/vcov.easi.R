vcov.easi <- function(object = object, ...) {
  neq <- object$neq
  VARS <- object$VARS
  VARS2 <- c()
  for (i in 1:neq) {
    VARS2 <- c(VARS2, paste(paste0("eq", i), VARS, sep = "_"))
  }
  tp <- object$CoefCov
  colnames(tp) <- rownames(tp) <- VARS2
  return(tp)
}
