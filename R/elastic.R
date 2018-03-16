elastic <- function(object, type = c("price", "income", "demographics"),
                    sd = FALSE) {
  # Calculate elasticities from an EASI result
  #
  # Args:
  #   object: an EASI results object, returned by easi()
  #   type: a vector including any of the values "price", "income" and/or
  #         "demographics". The corresponding elasticities are computed and
  #         returned.
  #   sd: if TRUE (default: FALSE), the standard deviations of the
  #       quantities are also returned.
  #
  # Returns:
  #   A list with some or all of the following names:
  #
  #   Price
  #
  #   EP: semi-elasticities of budget shares in respect to log.prices.
  #   EP_SE: standard deviations of semi-elasticities of budget shares in
  #          respect to log.prices.
  #   EPS: matrix of compensated quantity derivatives with respect to unlogged
  #        log.prices.
  #   EPQ: compensated (good-specific) expenditures with respect to log.prices.
  #   ELASTPRICE: elasticities of quantities in respect to log.prices.
  #   ELASTPRICE_SE: standard deviations of elasticities of quantities in
  #                  respect to log.prices.
  #
  #   Income
  #
  #   ELASTINCOME: elasticities of quantities in respect to income, mean across
  #                all observations.
  #   ELASTINCOME_SE: standard deviations of elasticities of quantities in
  #                   respect to income.
  #   qty.income: elasticities of quantities in respect to income, all
  #               observations.
  #   qty.income.pctile: median of qty.income within each percentile of log
  #                      total expenditure.
  #   ER: semi-elasticities of budget shares in respect to real expenditures,
  #       mean across all observations.
  #   share.income: semi-elasticities of budget shares in respect to real
  #                 expenditures, all observations.
  #   share.income.pctile: median of share.income within each percentile of log
  #                        total expenditure.
  #
  #   Demographics
  #
  #   ER_SE: standard deviations of semi-elasticities of budget shares in
  #          respect to real expenditures.
  #   EZ: semi-elasticities of budget shares in respect to demographics.
  #   EZ_SE: standard deviations of semi-elasticities of budget shares in
  #          respect to demographics.

  type <- type[type %in% c("price", "income", "demographics")]

  if (length(type) == 0) {
    return(list())
  }

  result <- list()

  fit3sls <- object$fit3sls
  varlist <- object$varlist
  var.soc <- object$var.soc
  shares <- object$shares
  log.price <- object$log.price
  neq <- object$neq
  y.power <- object$y.power
  nsoc <- object$nsoc
  interact <- object$interact
  py.inter <- object$py.inter
  zy.inter <- object$zy.inter
  pz.inter <- object$pz.inter
  interpz <- object$interpz
  log.exp <- object$log.exp
  labels.price <- object$labels.price
  labels.soc <- object$labels.soc
  labels.share <- object$labels.share

  n <- length(log.exp)

  temp <- intermediate.blocs(object)
  my.array <- temp$my.array
  tot <- temp$tot
  tot2 <- temp$tot2
  tot0 <- temp$tot0
  bjk <- temp$bjk
  P <- temp$P
  w <- temp$w
  Z <- temp$Z
  bjr <- temp$bjr
  gjt <- temp$gjt
  hjt <- temp$hjt
  cc <- temp$cc
  noms <- object$noms
  lnx <- object$log.exp
  y <- object$y

  if ("price" %in% type) {
    # Calculation of log.price elasticities
    # *** semi-elasticities with respect to log.prices
    # page 13 formula 23 EASI made EASIER (Pendakur 2008)
    EP <- matrix(0, neq + 1, neq + 1)
    a <- my.array
    for (i in 1:(neq + 1)) {
      for (k in 1:(neq + 1)) {
        tot10 <- 0
        for (t in (1:(nsoc + 1))) {
          tempo <- a[t, k, i] * Z[, t]
          tot10 <- tot10 + tempo
        }
        tot10 <- tot10 + bjk[k, i] * y
        EP[k, i] <- mean(tot10)
      }
    }

    colnames(EP) <- rownames(EP) <- c(labels.price[1:neq], "pothers")
    result$EP <- EP

    # Calculation of standard deviations of log.price elasticities (Delta
    # method) if EPDELTA=TRUE
    if (sd) {
      ttt <- colnames(summary(fit3sls)$coefCov)
      EP_SE <- matrix(0, neq + 1, neq + 1)
      ELASTPRICE_SE <- matrix(0, neq + 1, neq + 1)
      for (i in 1:neq) {
        for (j in 1:neq) {
          tt <- paste0("eq", i, "_np", j)

          if (pz.inter) {
          for (t in interpz) {
            tt <- c(tt, paste0("eq", i, "_np", j, "z", t))
          }
          }

          if (py.inter) {
          tt <- c(tt, paste0("eq", i, "_ynp", j))
          }

          tnum <- match(tt, ttt)

          DD <- summary(fit3sls)$coefCov[tnum, tnum]

          MAT <- Z[, 1]
          if (pz.inter)
          MAT <- cbind(MAT, Z[, interpz + 1])
          if (py.inter)
          MAT <- cbind(MAT, y)

          # TODO apply the same optimization as in engel.R
          EP_SE[i, j] <- median(sqrt(diag(as.matrix(MAT) %*% as.matrix(DD) %*%
          t(as.matrix(MAT)))))
          ELASTPRICE_SE[i, j] <- EP_SE[i, j]/mean(shares[, j])
          rm(DD)
        }
      }

      EP_SE[neq + 1, 1:neq] <- sqrt(apply(EP_SE[1:neq, 1:neq]^2, 1, sum))
      EP_SE[, neq + 1] <- EP_SE[neq + 1, ]
      EP_SE[neq + 1, neq + 1] <- sqrt(sum(EP_SE[neq + 1, 1:neq]))

      for (i in (1:(neq + 1))) {
        ELASTPRICE_SE[i, neq + 1] <- EP_SE[i, neq + 1]/mean(shares[, neq +
          1])
      }

      ELASTPRICE_SE[neq + 1, 1:neq] <- ELASTPRICE_SE[1:neq, neq + 1]
      ELASTPRICE_SE[neq + 1, neq + 1] <- sqrt(sum(ELASTPRICE_SE[1:neq, neq +
        1]))


      colnames(ELASTPRICE_SE) <- rownames(ELASTPRICE_SE) <- colnames(EP_SE) <-
        rownames(EP_SE) <- c(labels.price[1:neq], "pothers")

      result$EP_SE <- EP_SE
      result$ELASTPRICE_SE <- ELASTPRICE_SE
    }

    # Normalised Slutsky matrix
    # matrix of compensated quantity derivatives with respect to unlogged
    # log.prices)
    # own-log.price Slutsky terms in Pendakur page 849 'Tricks with Hicks : The
    # EASI demand system' (Lewbel & Pendakur 2008)
    EPS <- EP + apply(shares[, 1:(neq + 1)], 2, mean) %*%
      t(apply(shares[, 1:(neq + 1)], 2, mean)) -
      matrix(diag(apply(shares[, 1:(neq + 1)], 2, mean)),
      neq + 1, neq + 1)
    colnames(EPS) <- rownames(EPS) <- c(labels.price[1:neq], "pothers")
    result$EPS <- EPS

    # Compensated (good-specific) expenditures elasticities with respect to
    # log.prices
    # own-log.price Quant elast in Pendakur page 849 'Tricks with Hicks
    # : The EASI demand system' (Lewbel & Pendakur 2008)
    # TODO apply the same optimization as in engel.R
    EPQ <- solve(diag(apply(shares[, 1:(neq + 1)], 2, mean))) %*%
      (EP + apply(shares[, 1:(neq + 1)], 2, mean) %*%
      t(apply(shares[, 1:(neq + 1)], 2, mean)))
    colnames(EPQ) <- rownames(EPQ) <- c(labels.price[1:neq], "pothers")
    result$EPQ <- EPQ

    # calculation of elasticity of good j with respect to the log.price of good
    # i
    # calculation of elasticity of good j with respect to income
    ajk <- my.array
    ELASTPRICE <- matrix(0, neq + 1, neq + 1)
    for (i in 1:(neq + 1)) {
      for (q in 1:(neq + 1)) {

        C <- 0
        for (j in 1:y.power) {
          tempo <- j * bjr[j, i] * y^{
          j - 1
          }
          C <- C + tempo
        }

        D <- 0
        if (zy.inter) {
          for (j in 1:nsoc) {
          tempo <- hjt[j, i] * Z[, j + 1]
          D <- D + tempo
          }
        }

        E <- 0
        for (t in (1:(nsoc + 1))) {
          tempo <- ajk[t, q, i] * Z[, t]
          E <- E + tempo
        }

        G <- 0
        if (py.inter) {
          for (k in 1:(neq + 1)) {
          tempo <- bjk[k, i] * P[, k]
          G <- G + tempo
          }

          F <- bjk[q, i] * y
        }

        U <- 0
        for (d in 1:(neq + 1)) {
          for (t in (1:(nsoc + 1))) {
          tempo <- ajk[t, d, i] * Z[, t] * P[, d]
          U <- U + tempo
          }
        }

        B <- -mean(shares[, q] + U)/mean(1 - 1/2 * tot2) - mean(y)/mean(1 -
          1/2 * tot2) * mean(G)

        H <- B * (C + D + G) + E + F

        ELASTPRICE[q, i] <- mean(H)/mean(shares[, i]) - as.numeric((i ==
          q))

      }
    }

    colnames(ELASTPRICE) <- rownames(ELASTPRICE) <- c(labels.price[1:neq],
      "pothers")

    result$ELASTPRICE <- ELASTPRICE
  }

  if ("income" %in% type) {
    # calculation of elasticity of good j with respect to income
    ajk <- my.array
    ei.full <- matrix(0, n, neq + 1)
    ELASTINCOME <- matrix(0, 1, neq + 1)
    for (i in 1:(neq + 1)) {
      for (q in 1:(neq + 1)) {
        C <- rowSums(sapply(1:y.power, function(j) j * bjr[j, i] * y ^ (j - 1)))

        D <- G <- F <- 0
        if (zy.inter) {
          D <- rowSums(sapply(1:nsoc, function(j) hjt[j, i] * Z[, j + 1]))
        }

        E <- rowSums(sapply(1:(nsoc + 1), function(t) ajk[t, q, i] * Z[, t]))

        if (py.inter) {
          G <- rowSums(sapply(1:(neq + 1), function(k) bjk[k, i] * P[, k]))
          F <- bjk[q, i] * y
        }

        U <- 9
        for (d in 1:(neq + 1)) {
          U <- U + rowSums(sapply(1:(nsoc + 1),
                                  function(t) ajk[t, d, i] * Z[, t] * P[, d]))
        }

        B <- -mean(shares[, q] + U) / mean(1 - 1/2 * tot2) - mean(y) /
             mean(1 - 1/2 * tot2) * mean(G)

        H <- B * (C + D + G) + E + F


        ELASTINCOME[1, i] <- 1 + (mean(C + D + G)) / mean(shares[, i])
        ei.full[, i] <- 1 + (C + D + G) / shares[, i]
      }
    }

    # Similar to code for Engel curves
    qtile <- cut(lnx, breaks = quantile(lnx, seq(0, 1, 0.01)),
                 include.lowest = TRUE, labels = 1:100)

    # Compute percentile median
    ei.pctile <- matrix(0, 100, neq + 1)
    for (i in 1:100) {
      for (j in 1:(neq + 1)) {
        ei.pctile[i, j] <- median(ei.full[qtile == i, j])
      }
    }

    colnames(ELASTINCOME) <- colnames(ei.full) <- colnames(ei.pctile) <-
      labels.share
    result$ELASTINCOME <- ELASTINCOME
    result$qty.income <- ei.full
    result$qty.income.pctile <- ei.pctile

    # Calculation of income elasticities of budget shares
    # page 13 formula 23 'EASI made EASIER' (Pendakur 2008)
    ER.full <- matrix(NA, n, neq + 1)
    ER <- matrix(NA, 1, neq + 1)
    for (i in 1:(neq + 1)) {
      tempo1 <- tempo3 <- 0

      tempo2 <- rowSums(sapply(1:y.power,
                               function(t) bjr[t, i] * t * y ^ (t - 1)))

      if (zy.inter) {
        tempo3 <- rowSums(sapply(1:nsoc, function(t) hjt[t, i] * Z[, t + 1]))
      }

      if (py.inter) {
        tempo1 <- rowSums(sapply(1:(neq + 1), function(k) bjk[k, i] * P[, k]))
      }

      ER.full[, i] <- tempo1 + tempo2 + tempo3
      ER[i] <- mean(ER.full[, i])
    }

    # Compute percentile means
    ER.quantile <- matrix(0, 100, neq + 1)
    for (i in 1:100) {
      for (j in 1:(neq + 1)) {
        ER.quantile[i, j] <- median(ER.full[qtile == i, j])
      }
    }

    colnames(ER) <- colnames(ER.full) <- colnames(ER.quantile) <- labels.share

    result$ER <- ER
    result$share.income <- ER.full
    result$share.income.pctile <- ER.quantile

    # Calculation of standard deviations of income elasticities (delta method)
    if (sd) {
      ttt <- colnames(summary(fit3sls)$coefCov)
      ER_SE <- matrix(0, 1, neq + 1)
      ELASTINCOME_SE <- matrix(0, 1, neq + 1)
      for (i in (1:neq)) {
        tt <- c()
        for (j in (1:y.power)) {
          tt <- c(tt, paste0("eq", i, "_y", j))
        }

        if (zy.inter) {
          for (t in (1:nsoc)) {
          tt <- c(tt, paste0("eq", i, "_yz", t))
          }
        }
        if (py.inter) {
          for (j in 1:neq) {
          tt <- c(tt, paste0("eq", i, "_ynp", j))
          }
        }

        tnum <- match(tt, ttt)

        DD <- summary(fit3sls)$coefCov[c(tnum), c(tnum)]

        MAT <- c()
        for (r in 1:y.power) {
          MAT <- cbind(MAT, r * y^{
          r - 1
          })
        }
        if (zy.inter)
          MAT <- cbind(MAT, Z[, -1])
        if (py.inter)
          MAT <- cbind(MAT, P[, (1:neq)])

        # TODO apply the same optimization as in engel.R
        ER_SE[1, i] <- median(sqrt(diag(as.matrix(MAT) %*% as.matrix(DD) %*%
          t(as.matrix(MAT)))))
        ELASTINCOME_SE[1, i] <- ER_SE[1, i]/mean(shares[, i])
      }

      ER_SE[1, neq + 1] <- -sqrt(sum(ER_SE[1, 1:neq]^2))
      ELASTINCOME_SE[1, neq + 1] <- ER_SE[1, neq + 1]/mean(shares[, neq + 1])

      ER_SE <- as.matrix(ER_SE)
      ELASTINCOME_SE <- as.matrix(ELASTINCOME_SE)

      colnames(ER_SE) <- colnames(ELASTINCOME_SE) <- c(labels.share[1:neq],
        "others")

      result$ER_SE <- ER_SE
      result$ELASTINCOME_SE <- ELASTINCOME_SE
    }
  }

  if ("demographics" %in% type) {
    # Calculation of sociodemographic elasticities of budget shares
    # page 13 formula 23 'EASI made EASIER' (Pendakur 2008)
    EZ <- matrix(0, nsoc, (neq + 1))
    if (nsoc) {
      a <- my.array
      for (i in 1:(neq + 1)) {
        tempo4 <- tempoo <- 0
        for (t in 1:nsoc) {
          if (interact) {
            for (k in 1:(neq + 1)) {
              if (t %in% interpz) {
                tempoo <- a[t, k, i] * P[, k]
                tempo4 <- tempo4 + tempoo
              }
            }
          }

          if (nsoc) {
            tot12 <- gjt[t, i] + hjt[t, i] * y
          } else {
            tot12 <- 0
          }

          tot12 <- tot12 + tempo4
          EZ[t, i] <- mean(tot12)
        }
      }

      rownames(EZ) <- labels.soc
    }
    colnames(EZ) <- c(labels.share[1:neq], "others")
    result$EZ <- EZ

    # Calculation of standard deviations of sociodemographic elasticities (delta
    # method) if 'EZDELTA=TRUE'
    if (sd) {
      ttt <- colnames(summary(fit3sls)$coefCov)
      EZ_SE <- matrix(0, nsoc, neq + 1)
      for (i in 1:neq) {
        for (j in 1:nsoc) {
          tt <- c()
          tt <- c(tt, paste0("eq", i, "_z", j))

          if (zy.inter) {
            tt <- c(tt, paste0("eq", i, "_yz", j))
          }

          if (pz.inter & j %in% interpz) {
            for (t in 1:neq) {
              tt <- c(tt, paste0("eq", i, "_np", t, "z", j))
            }
          }

          tnum <- match(tt, ttt)

          DD <- summary(fit3sls)$coefCov[tnum, tnum]

          MAT <- Z[, 1]

          if (zy.inter) MAT <- cbind(MAT, y)

          if (pz.inter & j %in% interpz) {
            MAT <- cbind(MAT, P[, (1:neq)])
          }

          # TODO apply the same optimization as in engel.R
          EZ_SE[j, i] <- median(sqrt(diag(as.matrix(MAT) %*% as.matrix(DD) %*%
          t(as.matrix(MAT)))))
        }
      }
      for (j in 1:nsoc) EZ_SE[j, neq + 1] <- sqrt(sum(EZ_SE[j, 1:neq]^2))

      colnames(EZ_SE) <- c(labels.share[1:neq], "others")
      rownames(EZ_SE) <- labels.soc
      result$EZ_SE <- EZ_SE
    }
  }

  return(result)
}
