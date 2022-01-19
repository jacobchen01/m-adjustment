# The M-adjustment Criterion

This repository implements the M-Adjustment Criterion proposed by M. Sadaati and J. Tian in their paper Adjustment Criteria for Recovering Causal Effects from Missing Data.

Missing data is a common problem encountered in many empirical sciences in which we may not be able to observe all of the data for a certain variable. For example, if researchers were trying to determine the causal effects of age and gender on obesity rates in schools, it could be the case that students choose not to report their weights. This could be for a variety of reasons; if students are rebelling as a result of being teenagers, then the missingness of obesity would be caused by age. If students who are obese tend not to report that they are, then the missingness of obesity would be caused by whether a student was obese. When this is the case, the data are missing not at random (MNAR). When we are unable to observe all of our data, it may lead to biased analyses. Fortunately, Saadati and Tian at Iowa State University have developed "a covariate adjustment formulation for controlling confounding bias in the presence of MNAR data." Their M-adjustment criterion for missingness graphs, or m-graphs, is implemented in this repository. M-graphs are a variant of directed acyclic graphs in which missingness mechanisms for each partially observed variables are augmented onto the graph.

The basic formulation of the M-adjustment criterion is as follows. Let us assume that X is the treatment variable, and Y is the outcome variable. Z is a proposed adjustment set, and R_W is the set of all missingness mechanisms for variables in X, Y, and Z that are partially observed.
1. No element of Z is either X, Y, an element on the proper causal path from X to Y, or a descendant of a variable on the proper causal path from X to Y.
2. Y and X are d-separated given Z and R_W in the graph where the first edge of every proper causal path from X to Y is removed.
3. Y and R_W are d-separated given X in the graph where all incoming edges to X are deleted.
4. X is either not an ancestor of R_W or is d-separated from Y in the graph where all outgoing edges from X are deleted.

The simplicity of this criterion allows it to be implemented in such a manner that those who use the function do not need to anything about how it is implemented; the M-adjustment criterion can be implemented completely in a black box manner. Researchers that already have m-graphs can find proper M-adjustment sets simply by calling the function listMAdj() without any preliminary knowledge of missing data or missingness mechanisms.

On the other hand, this M-adjustment criterion trades its simplicity for a fairly limiting set of admissible adjustment sets. Because of requirement 3 in the M-adjustment criterion, it would appear that any m-graph that has any missingness indicators that are descendants of an element on a backdoor path or proper causal path between the treatment and outcome would not fufill the M-adjustment criterion. However, just because an adjustment set does not fulfill the M-adjustment criterion does not mean that the causal effect is not recoverable in that m-graph. For example, the graph in datatestingIPWMethod.py is recoverable with the inverse probability weighting (IPW) method in which we reweight the distribution of the missing data set with propensity scores for each missingness indicator. The new kernel we obtain is as if the edges from the partially observed variables to each missingness indicator were deleted.

It is easy to forget to take into account missing data during data analysis; however, failing to do so can lead to biased results. Considering how results from data analyses are often used to advise public policy, it is of paramount importance that researchers use everything in their analytical toolkits to find estimates for unbiased results. Making missingness a more common consideration would contribute to that mission. Although the M-adjustment criterion discussed in this repository is somewhat limiting, it is a step in the right direction in making accounting for missing data more accessible to those outside of the causal inference community.

Saadati, Mojdeh, and Jin Tian. "Adjustment Criteria for Recovering Causal Effects from Missing Data."
https://faculty.sites.iastate.edu/jtian/files/inline-files/ecml-19-cmr.pdf

This repository was written by Jacob Chen and Rohit Bhattacharya as a part of a Winter Study project at Williams College.
January, 2022
