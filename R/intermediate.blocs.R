intermediate.blocs <- function(object = object, log.price = NULL,
  var.soc = NULL, log.exp = NULL) {

  shares <- object$shares
  fit3sls <- object$fit3sls
  neq <- object$neq
  y.power <- object$y.power
  nsoc <- object$nsoc
  interact <- object$interact
  py.inter <- object$py.inter
  zy.inter <- object$zy.inter
  pz.inter <- object$pz.inter
  interpz <- object$interpz

  # Use values from object by default
  if (is.null(log.price)) log.price <- object$log.price
  if (is.null(var.soc)) var.soc <- object$var.soc
  if (is.null(log.exp)) log.exp <- object$log.exp

  n <- length(log.exp)

  # ** Recovery of the estimated coefficients***
  Estimates <- summary(fit3sls)$coefficients

  coef = Estimates[, 1]
  n <- length(log.exp)


  # Labels or names of the equations:
  noms <- c()
  for (i in 1:neq) noms <- c(noms, paste0("eq", i))

  # ***** new price Matrix *******
  P = log.price


  # **** new sociodemographic matrix *****
  Z = cbind(rep(1, n), var.soc)

  # **** Budget shares matrix *******
  w = matrix(0, n, neq + 1)
  for (i in 1:(neq + 1)) w[, i] <- shares[, i]


  # Recovery of coefficients for the variables p and p * z
  # Note: the first element of Z is a constant to capture the direct price
  # effects

  my.array <- array(0, dim = c(nsoc + 1, (neq + 1), (neq + 1)))

  # Step 1: Recovery of coefficients of p variables
  a0 <- matrix(0, neq, neq)
  for (i in 1:neq) {
    for (j in 1:neq) {
      a0[i, j] <- coef[paste0("eq", i, "_np", j)]
      my.array[1, j, i] <- a0[i, j]
    }
  }
  my.array[1, 1:neq, neq + 1] <- 0 - apply(a0, 2, sum)

  for (i in 1:neq) {
    my.array[1, neq + 1, i] <- my.array[1, i, neq + 1]
  }
  my.array[1, neq + 1, neq + 1] <- 0 - sum(my.array[1, 1:neq, neq + 1])

  # Step 2: Recovery of coefficients of p*z variables only if required
  if (pz.inter) {
    for (i in 1:neq) {
      for (j in interpz) {
        for (k in 1:neq) {
          my.array[j + 1, k, i] <- coef[paste0("eq", i, "_np", k, "z",
          j)]
        }
      }
    }
  }

  for (t in interpz) {
    for (i in 1:neq) {
      for (j in 1:neq) {
        my.array[t + 1, i, neq + 1] <- my.array[t + 1, i, neq + 1] -
          my.array[t + 1, i, j]
      }
    }
  }

  for (t in interpz) {
    for (i in 1:neq) {
      my.array[t + 1, neq + 1, i] <- my.array[t + 1, i, neq + 1]
    }
  }

  for (t in interpz) {
    for (i in 1:neq) {
      my.array[t + 1, neq + 1, neq + 1] <- my.array[t + 1, neq + 1, neq + 1]
        - my.array[t + 1, neq + 1, i]
    }
  }

  # construction of the sum 'sum_j sum_k sum_t a_jkt z_t p_j p_k'
  # (calculation of y)
  # 'EASI made EASIER' (PENDAKUR 2008 - page 11 formula 22)
  a <- my.array
  tot = 0
  for (j in 1:neq) {
    for (k in 1:neq) {
      for (t in c(1:(nsoc + 1))) {
        tempo <- a[t, k, j] * P[, k] * P[, j] * Z[, t]
        tot <- tot + tempo
      }
    }
  }

  # Recovery of coefficients of p*y variables only if required
  bjk = matrix(0, neq + 1, neq + 1)
  tot2 = 0
  if (py.inter) {
    for (i in 1:neq) {
      for (j in 1:neq) {
        bjk[j, i] <- coef[paste0("eq", i, "_ynp", j)]
      }
    }

    for (j in 1:neq) {
      bjk[j, neq + 1] <- 0 - sum(bjk[j, 1:neq])
    }

    for (j in 1:neq) {
      bjk[neq + 1, j] <- bjk[j, neq + 1]
    }

    bjk[neq + 1, neq + 1] <- 0 - sum(bjk[neq + 1, 1:neq])

    colnames(bjk) <- c(noms, "Others")

    # construction of the sum 'sum_j sum_k b_jk p_j p_k' (calculation of y)
    # 'EASI made EASIER' (PENDAKUR 2008 - page 11 formula 22)
    for (j in 1:neq) {
      for (k in 1:neq) {
        tempo <- bjk[j, k] * P[, j] * P[, k]
        tot2 <- tot2 + tempo
      }
    }
  }

  # construction of the sum 'sum_j w_j p_j' (calculation of y)
  # 'EASI made EASIER' (PENDAKUR 2008 - page 11 formula 22)
  tot0 = 0
  for (j in 1:neq) {
    tempo <- w[, j] * P[, j]
    tot0 <- tot0 + tempo
  }

  # Recovery of coefficients of y^r variables (calculation of w_j)
  bjr = matrix(0, y.power, neq + 1)
  for (i in 1:neq) {
    for (j in 1:y.power) {
      bjr[j, i] <- coef[paste0("eq", i, "_y", j)]
    }
  }

  for (j in 1:y.power) {
    bjr[j, neq + 1] <- 0 - sum(bjr[j, 1:neq])
  }

  colnames(bjr) <- c(noms, "others")

  # Recovery of coefficients of z variables (calculation of w_j)
  gjt = matrix(0, nsoc, neq + 1)
  if (nsoc) {
    for (i in 1:neq) {
      for (j in 1:nsoc) {
        gjt[j, i] <- coef[paste0("eq", i, "_z", j)]
      }
    }

    for (j in 1:nsoc) {
      gjt[j, neq + 1] <- 0 - sum(gjt[j, 1:neq])
    }
  }

  colnames(gjt) <- c(noms, "others")

  # Calculation of y 'EASI made EASIER' (PENDAKUR 2008 - page 11 formula 22)
  y <- (log.exp - tot0 + 1/2 * tot)
  if (interact) {
    y <- y/(1 - 1/2 * tot2)
  }

  # Recovery of coefficients of y*z variables (calculation of w_j) only if
  # required
  hjt = matrix(0, nsoc, neq + 1)
  if (zy.inter) {
    for (i in 1:neq) {
      for (j in 1:nsoc) {
        hjt[j, i] <- coef[paste0("eq", i, "_yz", j)]
      }
    }

    for (j in 1:nsoc) {
      hjt[j, neq + 1] <- 0 - sum(hjt[j, 1:neq])
    }
  }
  colnames(hjt) <- c(noms, "others")

  # Recovery of the constants (calculation of w_j)
  cc = c()
  for (i in 1:neq) {
    cc = cbind(cc, coef[paste0("eq", i, "_(Intercept)")])
  }

  cc = cbind(cc, 1 - sum(cc))

  colnames(cc) <- c(noms, "others")

  result <- list(
    bjk = bjk,
    bjr = bjr,
    cc = cc,
    gjt = gjt,
    hjt = hjt,
    noms = noms,
    my.array = my.array,
    P = P,
    tot = tot,
    tot0 = tot0,
    tot2 = tot2,
    w = w,
    y = y,
    Z = Z
    )

  return(result)
}
