# Data Analysis Report - Text Format

This report contains comprehensive text-based analysis suitable for AI analysis.

================================================================================
BASIC DATASET INFORMATION
================================================================================
Total Rows: 652
Total Columns: 74
Memory Usage: 0.37 MB
Duplicate Rows: 0

--- Column Types ---
float64              :  61 columns
int64                :  13 columns

================================================================================
MISSING DATA ANALYSIS
================================================================================
Total Missing Cells: 3297 out of 48248 (6.83%)

Columns with Missing Data: 51
Columns with Complete Data: 23

--- Top 20 Columns by Missing Data ---
  Col total                      :   348 ( 53.37%)
  Amilasa                        :   300 ( 46.01%)
  Exceso Bases                   :   279 ( 42.79%)
  PCO2                           :   279 ( 42.79%)
   CO3 H                         :   278 ( 42.64%)
  Gasometría: Ph                 :   277 ( 42.48%)
  PCR 72h                        :   126 ( 19.33%)
  Creat 72h                      :   125 ( 19.17%)
  Hct 72h                        :   125 ( 19.17%)
  Lymph 72h                      :   124 ( 19.02%)
  Mono 72h                       :   124 ( 19.02%)
  PMN/Lymph 72h                  :   124 ( 19.02%)
  Leukocytes 72h                 :   124 ( 19.02%)
  Eosinophils 72h                :   124 ( 19.02%)
  PMN 72h                        :   124 ( 19.02%)
  Urea 72h                       :   123 ( 18.87%)
  Waist circ                     :    30 (  4.60%)
  Phosphate                      :    29 (  4.45%)
  Abdominal Exam                 :    28 (  4.29%)
  Height                         :    22 (  3.37%)

--- Missing Data Ranges ---
  Min Missing: 0.15%
  Max Missing: 53.37%
  Median Missing: 1.23%

================================================================================
NUMERICAL FEATURES ANALYSIS
================================================================================
Total Numerical Columns: 74

--- Summary Statistics ---
Feature                           Count         Mean          Std          Min          Max
-----------------------------------------------------------------------------------------------
Age                                 652       68.684       17.395       18.000       99.000
Sex                                 652        0.534        0.499        0.000        1.000
Drinker                             651        0.490        0.829        0.000        2.000
Grams/Day                           651       15.754       33.087        0.000      230.000
Smoker                              651        0.593        0.772        0.000        2.000
Packs/Year                          650       12.715       20.396        0.000      120.000
Etiology                            652        2.141        1.651        1.000        5.000
Previous AP                         652        1.181        0.484        0.000        3.000
CCI                                 652        1.276        1.605        0.000        9.000
HBD                                 651        0.522        0.500        0.000        1.000
DM                                  651        0.160        0.367        0.000        1.000
Dyslipidemias                       650        0.380        0.548        0.000        3.000
Weight                              630       87.417      336.264       38.000     8505.000
Height                              630      162.477       13.558        1.580      195.000
BMI                                 630      814.676    13885.591       15.046   256369.172
Abdominal Exam                      624      100.519       14.721       44.000      174.000
Onset pain                          650       19.449       20.097        1.000      144.000
Length of stay                      652        9.210       13.404        1.000      140.000
Pain Intensity                      650        9.323        1.229        0.000       10.000
PCR Adm                             652       40.495       69.797        0.000      542.000
PCR 48h                             643      123.357      111.517        0.000      573.000
PCR 72h                             526      141.432      123.535        0.000      622.000
Ransom Adm                          652        1.617        1.133        0.000        5.000
Ransom 48h                          652        1.988        1.620        0.000       13.000
BISAP                               652        1.224        0.993        0.000        4.000
SIRS                                652        0.402        0.700        0.000        4.000
Urea Adm                            652       45.225       29.332        6.000      272.000
Urea 48h                            644       42.112       31.662        4.000      269.000
Urea 72h                            529       39.785       31.340        0.000      228.000
Creat Adm                           652        1.053        0.679        0.300        6.550
Creat 48h                           644        1.148        1.954        0.170       43.000
Creat 72h                           527        1.115        2.296        0.200       51.000
Hct Adm                             652       41.873        5.833       10.900       62.600
Hct 48h                             646       39.257       14.574        0.000      373.200
Hct 72h                             527       37.335        4.983       24.000       53.200
Leukocytes Adm                      652    12730.876     5336.432     3030.000    79100.000
PMN Adm                             652    10590.985     6188.121       11.010   100400.000
Lymphocytes Adm                     652     1373.778     1010.271       20.000    10610.000
Mono Adm                            652      571.647      438.355        2.000     8310.000
Eosinophils Adm                     652       63.626      294.157        0.000     6320.000
Leukocytes 48h                      644    11089.146     7825.300     1000.000   158700.000
PMN 48h                             644     8653.601     5124.752      360.000    34360.000
Lymph 48h                           644     1357.694      927.477       90.000    11800.000
Mono 48h                            644      635.927      393.992        4.000     5020.000
Eosinophils 48h                     643      108.694      205.807        0.000     3190.000
Leukocytes 72h                      528     9876.924     4827.568     1470.000    29480.000
PMN 72h                             528     7500.544     4684.430       20.000    26930.000
Lymph 72h                           528     1412.150      812.922       70.000     9250.000
Mono 72h                            528      682.170      455.121       10.000     6240.000
Eosinophils 72h                     528      157.178      187.081        0.000     2100.000
Platelets                           651   229205.280    77888.045       99.000   566000.000
Amilasa                             352     1520.622     1454.724       33.000    14040.000
Lipase                              646     4314.554    19169.768        5.000   443150.000
GOT                                 652      196.494      311.885        1.000     3566.000
GPT                                 651      178.983      233.427        5.000     2697.000
F alc                               648      148.616      126.823        6.000     1061.000
Albumin                             649        3.634        0.541        0.200        7.300
Ca                                  649        8.787        2.826        3.800       79.000
Phosphate                           623        2.940        0.834        0.700        6.300
Glu                                 652      132.619       54.175        7.200      450.000
TG                                  648      150.968      363.110       22.000     6747.000
Col total                           304      155.204       67.000       56.000      880.000
Gasometría: Ph                      375        7.383        0.057        7.130        7.520
 CO3 H                              374       23.667        7.781        9.800      157.000
PCO2                                373       38.210        7.951        8.000       78.100
Exceso Bases                        373       -3.113       31.733     -610.000       23.300
Chest X-Ray                         631        0.233        0.423        0.000        1.000
Petrov                              652        0.299        0.684        0.000        3.000
Waist circ                          622        1.078        0.575        0.012        5.682
PCR/Alb ratio                       649       12.416       23.484        0.000      216.800
PMN/Lymph Admin                     652       13.950       26.061        0.052      532.500
PMN/Lymph 48h                       644        9.451        9.337        0.272       73.106
PMN/Lymph 72h                       528        7.858       11.952        0.010      176.286
Inflammatory Index                  651  3094301.045  6916012.633      483.067 156555000.000

--- Distribution Shape (Skewness & Kurtosis) ---
Feature                            Skewness     Kurtosis Interpretation                
-------------------------------------------------------------------------------------
Age                                  -0.670       -0.202  Left Skewed                   
Sex                                  -0.136       -1.988  Fairly Symmetrical            
Drinker                               1.186       -0.489  Right Skewed                  
Grams/Day                             2.654        8.047  Right Skewed                  
Smoker                                0.845       -0.816  Right Skewed                  
Packs/Year                            1.967        4.225  Right Skewed                  
Etiology                              0.940       -0.932  Right Skewed                  
Previous AP                           2.623        6.218  Right Skewed                  
CCI                                   1.377        1.540  Right Skewed                  
HBD                                  -0.089       -1.998  Fairly Symmetrical            
DM                                    1.862        1.470  Right Skewed                  
Dyslipidemias                         1.302        1.997  Right Skewed                  
Weight                               25.018      627.247  Right Skewed                  
Height                               -5.207       61.518  Left Skewed                   
BMI                                  17.747      314.493  Right Skewed                  
Abdominal Exam                       -0.150        1.547  Fairly Symmetrical            
Onset pain                            1.837        4.131  Right Skewed                  
Length of stay                        5.778       40.949  Right Skewed                  
Pain Intensity                       -2.388        8.354  Left Skewed                   
PCR Adm                               3.269       13.236  Right Skewed                  
PCR 48h                               1.006        0.334  Right Skewed                  
PCR 72h                               0.890        0.195  Right Skewed                  
Ransom Adm                            0.761        0.323  Right Skewed                  
Ransom 48h                            1.549        4.481  Right Skewed                  
BISAP                                 0.750        0.253  Right Skewed                  
SIRS                                  1.662        2.240  Right Skewed                  
Urea Adm                              3.085       14.095  Right Skewed                  
Urea 48h                              2.549        9.529  Right Skewed                  
Urea 72h                              2.739        9.986  Right Skewed                  
Creat Adm                             4.159       22.883  Right Skewed                  
Creat 48h                            17.015      343.265  Right Skewed                  
Creat 72h                            19.700      426.071  Right Skewed                  
Hct Adm                              -0.445        2.123  Fairly Symmetrical            
Hct 48h                              18.616      428.636  Right Skewed                  
Hct 72h                               0.117        0.140  Fairly Symmetrical            
Leukocytes Adm                        3.581       37.009  Right Skewed                  
PMN Adm                               5.870       74.201  Right Skewed                  
Lymphocytes Adm                       2.775       15.600  Right Skewed                  
Mono Adm                              9.148      150.561  Right Skewed                  
Eosinophils Adm                      16.695      332.025  Right Skewed                  
Leukocytes 48h                       10.788      196.743  Right Skewed                  
PMN 48h                               1.103        1.778  Right Skewed                  
Lymph 48h                             4.482       38.536  Right Skewed                  
Mono 48h                              3.822       32.073  Right Skewed                  
Eosinophils 48h                       7.012       87.000  Right Skewed                  
Leukocytes 72h                        1.141        1.212  Right Skewed                  
PMN 72h                               1.142        1.317  Right Skewed                  
Lymph 72h                             3.044       21.447  Right Skewed                  
Mono 72h                              5.102       50.554  Right Skewed                  
Eosinophils 72h                       3.132       23.190  Right Skewed                  
Platelets                             0.692        2.209  Right Skewed                  
Amilasa                               2.981       17.026  Right Skewed                  
Lipase                               19.397      430.638  Right Skewed                  
GOT                                   4.966       40.351  Right Skewed                  
GPT                                   3.441       23.495  Right Skewed                  
F alc                                 2.987       12.345  Right Skewed                  
Albumin                              -0.047        5.489  Fairly Symmetrical            
Ca                                   23.728      590.586  Right Skewed                  
Phosphate                             0.478        0.869  Fairly Symmetrical            
Glu                                   1.643        4.867  Right Skewed                  
TG                                   13.160      204.198  Right Skewed                  
Col total                             5.531       50.912  Right Skewed                  
Gasometría: Ph                       -0.862        2.146  Left Skewed                   
 CO3 H                               13.616      232.352  Right Skewed                  
PCO2                                  0.005        2.106  Fairly Symmetrical            
Exceso Bases                        -18.903      362.449  Left Skewed                   
Chest X-Ray                           1.266       -0.397  Right Skewed                  
Petrov                                2.571        6.323  Right Skewed                  
Waist circ                            1.928        9.222  Right Skewed                  
PCR/Alb ratio                         4.015       21.400  Right Skewed                  
PMN/Lymph Admin                      13.371      246.831  Right Skewed                  
PMN/Lymph 48h                         2.532        9.512  Right Skewed                  
PMN/Lymph 72h                        10.108      135.081  Right Skewed                  
Inflammatory Index                   17.308      374.864  Right Skewed                  

================================================================================
CATEGORICAL FEATURES ANALYSIS
================================================================================
Total Categorical Columns: 17


--- Sex ---
Unique Values: 2
Missing: 0

Value Distribution:
  1                    :   348 ( 53.37%) ██████████████████████████
  0                    :   304 ( 46.63%) ███████████████████████

--- Drinker ---
Unique Values: 3
Missing: 1

Value Distribution:
  0.0                  :   474 ( 72.70%) ████████████████████████████████████
  2.0                  :   142 ( 21.78%) ██████████
  1.0                  :    35 (  5.37%) ██
  nan                  :     1 (  0.15%) 

--- Grams/Day ---
Unique Values: 19
Missing: 1

Value Distribution:
  0.0                  :   474 ( 72.70%) ████████████████████████████████████
  40.0                 :    35 (  5.37%) ██
  30.0                 :    29 (  4.45%) ██
  20.0                 :    23 (  3.53%) █
  100.0                :    18 (  2.76%) █
  60.0                 :    14 (  2.15%) █
  80.0                 :    12 (  1.84%) 
  120.0                :    12 (  1.84%) 
  50.0                 :    11 (  1.69%) 
  10.0                 :     8 (  1.23%) 

--- Smoker ---
Unique Values: 3
Missing: 1

Value Distribution:
  0.0                  :   380 ( 58.28%) █████████████████████████████
  1.0                  :   156 ( 23.93%) ███████████
  2.0                  :   115 ( 17.64%) ████████
  nan                  :     1 (  0.15%) 

--- Etiology ---
Unique Values: 5
Missing: 0

Value Distribution:
  1                    :   411 ( 63.04%) ███████████████████████████████
  5                    :   131 ( 20.09%) ██████████
  4                    :    44 (  6.75%) ███
  2                    :    44 (  6.75%) ███
  3                    :    22 (  3.37%) █

--- Previous AP ---
Unique Values: 4
Missing: 0

Value Distribution:
  1                    :   559 ( 85.74%) ██████████████████████████████████████████
  2                    :    65 (  9.97%) ████
  3                    :    27 (  4.14%) ██
  0                    :     1 (  0.15%) 

--- CCI ---
Unique Values: 9
Missing: 0

Value Distribution:
  0                    :   297 ( 45.55%) ██████████████████████
  1                    :   139 ( 21.32%) ██████████
  2                    :    85 ( 13.04%) ██████
  3                    :    56 (  8.59%) ████
  4                    :    41 (  6.29%) ███
  5                    :    19 (  2.91%) █
  6                    :    11 (  1.69%) 
  7                    :     3 (  0.46%) 
  9                    :     1 (  0.15%) 

--- HBD ---
Unique Values: 2
Missing: 1

Value Distribution:
  1.0                  :   340 ( 52.15%) ██████████████████████████
  0.0                  :   311 ( 47.70%) ███████████████████████
  nan                  :     1 (  0.15%) 

--- DM ---
Unique Values: 2
Missing: 1

Value Distribution:
  0.0                  :   547 ( 83.90%) █████████████████████████████████████████
  1.0                  :   104 ( 15.95%) ███████
  nan                  :     1 (  0.15%) 

--- Dyslipidemias ---
Unique Values: 4
Missing: 2

Value Distribution:
  0.0                  :   420 ( 64.42%) ████████████████████████████████
  1.0                  :   217 ( 33.28%) ████████████████
  2.0                  :     9 (  1.38%) 
  3.0                  :     4 (  0.61%) 
  nan                  :     2 (  0.31%) 

--- Pain Intensity ---
Unique Values: 8
Missing: 2

Value Distribution:
  10.0                 :   452 ( 69.33%) ██████████████████████████████████
  8.0                  :   103 ( 15.80%) ███████
  9.0                  :    47 (  7.21%) ███
  7.0                  :    27 (  4.14%) ██
  5.0                  :    12 (  1.84%) 
  6.0                  :     7 (  1.07%) 
  nan                  :     2 (  0.31%) 
  0.0                  :     1 (  0.15%) 
  2.0                  :     1 (  0.15%) 

--- Ransom Adm ---
Unique Values: 6
Missing: 0

Value Distribution:
  1.0                  :   273 ( 41.87%) ████████████████████
  2.0                  :   157 ( 24.08%) ████████████
  3.0                  :    98 ( 15.03%) ███████
  0.0                  :    84 ( 12.88%) ██████
  4.0                  :    27 (  4.14%) ██
  5.0                  :    13 (  1.99%) 

--- Ransom 48h ---
Unique Values: 11
Missing: 0

Value Distribution:
  1.0                  :   239 ( 36.66%) ██████████████████
  2.0                  :   132 ( 20.25%) ██████████
  3.0                  :   103 ( 15.80%) ███████
  0.0                  :    80 ( 12.27%) ██████
  4.0                  :    51 (  7.82%) ███
  5.0                  :    25 (  3.83%) █
  6.0                  :    11 (  1.69%) 
  7.0                  :     6 (  0.92%) 
  9.0                  :     2 (  0.31%) 
  8.0                  :     2 (  0.31%) 

--- BISAP ---
Unique Values: 5
Missing: 0

Value Distribution:
  1                    :   286 ( 43.87%) █████████████████████
  0                    :   155 ( 23.77%) ███████████
  2                    :   140 ( 21.47%) ██████████
  3                    :    52 (  7.98%) ███
  4                    :    19 (  2.91%) █

--- SIRS ---
Unique Values: 4
Missing: 0

Value Distribution:
  0                    :   465 ( 71.32%) ███████████████████████████████████
  1                    :   116 ( 17.79%) ████████
  2                    :    69 ( 10.58%) █████
  4                    :     2 (  0.31%) 

--- Chest X-Ray ---
Unique Values: 2
Missing: 21

Value Distribution:
  0.0                  :   484 ( 74.23%) █████████████████████████████████████
  1.0                  :   147 ( 22.55%) ███████████
  nan                  :    21 (  3.22%) █

--- Petrov ---
Unique Values: 4
Missing: 0

Value Distribution:
  0                    :   520 ( 79.75%) ███████████████████████████████████████
  1                    :    90 ( 13.80%) ██████
  3                    :    21 (  3.22%) █
  2                    :    21 (  3.22%) █

================================================================================
TARGET VARIABLE ANALYSIS: 'Length of stay'
================================================================================

Basic Statistics:
  Count       : 652
  Missing     : 0
  Mean        : 9.210
  Median      : 6.000
  Std Dev     : 13.404
  Min         : 1.000
  Max         : 140.000
  Q1 (25%)    : 4.000
  Q3 (75%)    : 9.000
  IQR         : 5.000

Distribution Characteristics:
  Skewness    : 5.778
  Kurtosis    : 40.949
  Range       : 139.000
  Coefficient of Variation : 1.455

Log Transformation Effect (log1p):
  Original Skewness : 5.778
  Log Skewness      : 1.505
  Skewness Reduction: 4.274

Percentile Breakdown:
  10th percentile :    3.000
  20th percentile :    4.000
  30th percentile :    4.000
  40th percentile :    5.000
  50th percentile :    6.000
  60th percentile :    6.000
  70th percentile :    8.000
  80th percentile :   10.000
  90th percentile :   16.000
  95th percentile :   22.450
  99th percentile :   75.270

================================================================================
CORRELATION ANALYSIS
================================================================================

--- Correlation with Target Variable: 'Length of stay' ---
Feature                         Correlation Strength       
------------------------------------------------------------
Petrov                               0.5872  Strong         
SIRS                                 0.4819  Moderate       
PCR 72h                              0.4484  Moderate       
Ransom 48h                           0.3743  Moderate       
PCR 48h                              0.3337  Moderate       
Chest X-Ray                          0.3133  Moderate       
BISAP                                0.3064  Moderate       
PMN 72h                              0.2975  Weak           
Leukocytes 72h                       0.2602  Weak           
PMN 48h                              0.2512  Weak           
PMN/Lymph 48h                        0.2428  Weak           
Urea 72h                             0.2372  Weak           
PMN/Lymph 72h                        0.2309  Weak           
Urea 48h                             0.1901  Weak           
Ransom Adm                           0.1901  Weak           
Leukocytes 48h                       0.1554  Weak           
Mono 48h                             0.1434  Weak           
Glu                                  0.1277  Weak           
Dyslipidemias                        0.1266  Weak           
Packs/Year                           0.1176  Weak           
Hct Adm                              0.1116  Weak           
Leukocytes Adm                       0.1055  Weak           
Abdominal Exam                       0.1044  Weak           
Urea Adm                             0.1016  Weak           
PCR/Alb ratio                        0.0999  Very Weak      
PCR Adm                              0.0947  Very Weak      
Mono 72h                             0.0941  Very Weak      
Pain Intensity                       0.0842  Very Weak      
Mono Adm                             0.0838  Very Weak      
Grams/Day                            0.0809  Very Weak      
PMN Adm                              0.0801  Very Weak      
HBD                                  0.0778  Very Weak      
Col total                            0.0751  Very Weak      
TG                                   0.0736  Very Weak      
Hct 48h                              0.0655  Very Weak      
Creat 48h                            0.0651  Very Weak      
Lymphocytes Adm                      0.0648  Very Weak      
Amilasa                              0.0641  Very Weak      
Creat Adm                            0.0601  Very Weak      
Creat 72h                            0.0580  Very Weak      
DM                                   0.0537  Very Weak      
Etiology                             0.0507  Very Weak      
Drinker                              0.0500  Very Weak      
Waist circ                           0.0497  Very Weak      
F alc                                0.0455  Very Weak      
Height                               0.0443  Very Weak      
Age                                  0.0343  Very Weak      
Smoker                               0.0117  Very Weak      
Weight                               0.0067  Very Weak      
CCI                                  0.0065  Very Weak      
Lipase                               0.0051  Very Weak      
PMN/Lymph Admin                      0.0007  Very Weak      
BMI                                 -0.0007  Very Weak      
Platelets                           -0.0007  Very Weak      
Inflammatory Index                  -0.0053  Very Weak      
Hct 72h                             -0.0132  Very Weak      
Eosinophils Adm                     -0.0256  Very Weak      
Exceso Bases                        -0.0273  Very Weak      
GOT                                 -0.0449  Very Weak      
GPT                                 -0.0487  Very Weak      
Ca                                  -0.0530  Very Weak      
Sex                                 -0.0590  Very Weak      
Phosphate                           -0.0603  Very Weak      
Previous AP                         -0.0679  Very Weak      
PCO2                                -0.0855  Very Weak      
Onset pain                          -0.0948  Very Weak      
 CO3 H                              -0.1079  Weak           
Gasometría: Ph                      -0.1081  Weak           
Eosinophils 48h                     -0.1101  Weak           
Albumin                             -0.1155  Weak           
Lymph 48h                           -0.1343  Weak           
Eosinophils 72h                     -0.1445  Weak           
Lymph 72h                           -0.1785  Weak           

--- Top 10 Positive Correlations ---
  Petrov                         :   0.5872
  SIRS                           :   0.4819
  PCR 72h                        :   0.4484
  Ransom 48h                     :   0.3743
  PCR 48h                        :   0.3337
  Chest X-Ray                    :   0.3133
  BISAP                          :   0.3064
  PMN 72h                        :   0.2975
  Leukocytes 72h                 :   0.2602
  PMN 48h                        :   0.2512

--- Top 10 Negative Correlations ---
  Previous AP                    :  -0.0679
  PCO2                           :  -0.0855
  Onset pain                     :  -0.0948
   CO3 H                         :  -0.1079
  Gasometría: Ph                 :  -0.1081
  Eosinophils 48h                :  -0.1101
  Albumin                        :  -0.1155
  Lymph 48h                      :  -0.1343
  Eosinophils 72h                :  -0.1445
  Lymph 72h                      :  -0.1785

--- Highly Correlated Feature Pairs (|r| > 0.8) ---
  PCR Adm                        <-> PCR/Alb ratio                  :   0.9781
  PMN/Lymph Admin                <-> Inflammatory Index             :   0.9588
  Leukocytes 72h                 <-> PMN 72h                        :   0.9521
  Weight                         <-> Hct 48h                        :   0.9096
  Urea 48h                       <-> Urea 72h                       :   0.8503
  Ransom Adm                     <-> Ransom 48h                     :   0.8393
  Urea Adm                       <-> Urea 48h                       :   0.8216
  GOT                            <-> GPT                            :   0.8079

================================================================================
OUTLIER ANALYSIS (IQR Method)
================================================================================
Feature                          Outliers        %  Min_Outlier  Max_Outlier
-------------------------------------------------------------------------------------
Age                                     2    0.31%       18.000       18.000
Grams/Day                              71   10.89%       60.000      230.000
Packs/Year                             30    4.60%       52.000      120.000
Previous AP                            93   14.26%        0.000        3.000
CCI                                    15    2.30%        6.000        9.000
DM                                    104   15.95%        1.000        1.000
Dyslipidemias                           4    0.61%        3.000        3.000
Weight                                 10    1.53%      116.000     8505.000
Height                                  7    1.07%        1.580      195.000
BMI                                    18    2.76%       15.046   256369.172
Abdominal Exam                         11    1.69%       44.000      174.000
Onset pain                             36    5.52%       72.000      144.000
Length of stay                         64    9.82%       17.000      140.000
Pain Intensity                         48    7.36%        0.000        7.000
PCR Adm                                64    9.82%      113.800      542.000
PCR 48h                                 7    1.07%      430.000      573.000
PCR 72h                                 3    0.46%      574.000      622.000
Ransom Adm                             40    6.13%        4.000        5.000
Ransom 48h                             11    1.69%        7.000       13.000
BISAP                                  19    2.91%        4.000        4.000
SIRS                                    2    0.31%        4.000        4.000
Urea Adm                               38    5.83%       90.000      272.000
Urea 48h                               43    6.60%       93.000      269.000
Urea 72h                               43    6.60%       83.000      228.000
Creat Adm                              48    7.36%        1.770        6.550
Creat 48h                              73   11.20%        1.790       43.000
Creat 72h                              37    5.67%        1.830       51.000
Hct Adm                                10    1.53%       10.900       62.600
Hct 48h                                10    1.53%        0.000      373.200
Hct 72h                                 6    0.92%       24.000       53.200
Leukocytes Adm                         12    1.84%    24460.000    79100.000
PMN Adm                                 8    1.23%    24040.000   100400.000
Lymphocytes Adm                        32    4.91%     3350.000    10610.000
Mono Adm                               20    3.07%     1240.000     8310.000
Eosinophils Adm                       140   21.47%       80.000     6320.000
Leukocytes 48h                         11    1.69%    25380.000   158700.000
PMN 48h                                 9    1.38%    22860.000    34360.000
Lymph 48h                              23    3.53%     3020.000    11800.000
Mono 48h                               11    1.69%     1490.000     5020.000
Eosinophils 48h                        37    5.67%      400.000     3190.000
Leukocytes 72h                         14    2.15%    22340.000    29480.000
PMN 72h                                11    1.69%    19600.000    26930.000
Lymph 72h                              11    1.69%     3180.000     9250.000
Mono 72h                               29    4.45%     1390.000     6240.000
Eosinophils 72h                        11    1.69%      620.000     2100.000
Platelets                              28    4.29%       99.000   566000.000
Amilasa                                15    2.30%     4395.000    14040.000
Lipase                                 62    9.51%     9281.000   443150.000
GOT                                    47    7.21%      606.000     3566.000
GPT                                    33    5.06%      617.000     2697.000
F alc                                  47    7.21%      327.000     1061.000
Albumin                                 8    1.23%        0.200        7.300
Ca                                     40    6.13%        3.800       79.000
Phosphate                              12    1.84%        0.700        6.300
Glu                                    38    5.83%        7.200      450.000
TG                                     43    6.60%      243.000     6747.000
Col total                               7    1.07%      270.000      880.000
Gasometría: Ph                         18    2.76%        7.130        7.520
 CO3 H                                 23    3.53%        9.800      157.000
PCO2                                   12    1.84%        8.000       78.100
Exceso Bases                           17    2.61%     -610.000       23.300
Chest X-Ray                           147   22.55%        1.000        1.000
Petrov                                132   20.25%        1.000        3.000
Waist circ                             16    2.45%        2.340        5.682
PCR/Alb ratio                          65    9.97%       33.636      216.800
PMN/Lymph Admin                        52    7.98%       32.432      532.500
PMN/Lymph 48h                          33    5.06%       27.147       73.106
PMN/Lymph 72h                          23    3.53%       21.436      176.286
Inflammatory Index                     46    7.06%  7151841.270 156555000.000
-------------------------------------------------------------------------------------
Total Outlier Instances: 2310

================================================================================
FEATURE GROUP ANALYSIS
================================================================================

--- Demographic (5 features) ---
  Age                            : Mean=     68.68, Std=   17.39, Missing=  0.00%
  Sex                            : Mean=      0.53, Std=    0.50, Missing=  0.00%
  Weight                         : Mean=     87.42, Std=  336.26, Missing=  3.37%
  Height                         : Mean=    162.48, Std=   13.56, Missing=  3.37%
  BMI                            : Mean=    814.68, Std=13885.59, Missing=  3.37%

--- Lifestyle (4 features) ---
  Drinker                        : Mean=      0.49, Std=    0.83, Missing=  0.15%
  Grams/Day                      : Mean=     15.75, Std=   33.09, Missing=  0.15%
  Smoker                         : Mean=      0.59, Std=    0.77, Missing=  0.15%
  Packs/Year                     : Mean=     12.72, Std=   20.40, Missing=  0.31%

--- Medical History (6 features) ---
  Etiology                       : Mean=      2.14, Std=    1.65, Missing=  0.00%
  Previous AP                    : Mean=      1.18, Std=    0.48, Missing=  0.00%
  CCI                            : Mean=      1.28, Std=    1.60, Missing=  0.00%
  HBD                            : Mean=      0.52, Std=    0.50, Missing=  0.15%
  DM                             : Mean=      0.16, Std=    0.37, Missing=  0.15%
  Dyslipidemias                  : Mean=      0.38, Std=    0.55, Missing=  0.31%

--- Clinical Exam (3 features) ---
  Abdominal Exam                 : Mean=    100.52, Std=   14.72, Missing=  4.29%
  Onset pain                     : Mean=     19.45, Std=   20.10, Missing=  0.31%
  Pain Intensity                 : Mean=      9.32, Std=    1.23, Missing=  0.31%

--- PCR (Procalcitonin) (4 features) ---
  PCR Adm                        : Mean=     40.49, Std=   69.80, Missing=  0.00%
  PCR 48h                        : Mean=    123.36, Std=  111.52, Missing=  1.38%
  PCR 72h                        : Mean=    141.43, Std=  123.54, Missing= 19.33%
  PCR/Alb ratio                  : Mean=     12.42, Std=   23.48, Missing=  0.46%

--- Ransom Score (2 features) ---
  Ransom Adm                     : Mean=      1.62, Std=    1.13, Missing=  0.00%
  Ransom 48h                     : Mean=      1.99, Std=    1.62, Missing=  0.00%

--- BISAP Score (1 features) ---
  BISAP                          : Mean=      1.22, Std=    0.99, Missing=  0.00%

--- Kidney Function (6 features) ---
  Urea Adm                       : Mean=     45.22, Std=   29.33, Missing=  0.00%
  Urea 48h                       : Mean=     42.11, Std=   31.66, Missing=  1.23%
  Urea 72h                       : Mean=     39.78, Std=   31.34, Missing= 18.87%
  Creat Adm                      : Mean=      1.05, Std=    0.68, Missing=  0.00%
  Creat 48h                      : Mean=      1.15, Std=    1.95, Missing=  1.23%
  Creat 72h                      : Mean=      1.11, Std=    2.30, Missing= 19.17%

--- Blood Cells (6 features) ---
  Hct Adm                        : Mean=     41.87, Std=    5.83, Missing=  0.00%
  Hct 48h                        : Mean=     39.26, Std=   14.57, Missing=  0.92%
  Hct 72h                        : Mean=     37.34, Std=    4.98, Missing= 19.17%
  Leukocytes Adm                 : Mean=  12730.88, Std= 5336.43, Missing=  0.00%
  Leukocytes 48h                 : Mean=  11089.15, Std= 7825.30, Missing=  1.23%
  Leukocytes 72h                 : Mean=   9876.92, Std= 4827.57, Missing= 19.02%

--- White Blood Cells (8 features) ---
  PMN Adm                        : Mean=  10590.98, Std= 6188.12, Missing=  0.00%
  Lymphocytes Adm                : Mean=   1373.78, Std= 1010.27, Missing=  0.00%
  PMN 48h                        : Mean=   8653.60, Std= 5124.75, Missing=  1.23%
  Lymph 48h                      : Mean=   1357.69, Std=  927.48, Missing=  1.23%
  PMN 72h                        : Mean=   7500.54, Std= 4684.43, Missing= 19.02%
  Lymph 72h                      : Mean=   1412.15, Std=  812.92, Missing= 19.02%
  PMN/Lymph 48h                  : Mean=      9.45, Std=    9.34, Missing=  1.23%
  PMN/Lymph 72h                  : Mean=      7.86, Std=   11.95, Missing= 19.02%

--- Liver Function (4 features) ---
  GOT                            : Mean=    196.49, Std=  311.88, Missing=  0.00%
  GPT                            : Mean=    178.98, Std=  233.43, Missing=  0.15%
  F alc                          : Mean=    148.62, Std=  126.82, Missing=  0.61%
  Albumin                        : Mean=      3.63, Std=    0.54, Missing=  0.46%

--- Enzymes & Markers (2 features) ---
  Amilasa                        : Mean=   1520.62, Std= 1454.72, Missing= 46.01%
  Lipase                         : Mean=   4314.55, Std=19169.77, Missing=  0.92%

--- Time Measurements (32 features) ---
  PCR Adm                        : Mean=     40.49, Std=   69.80, Missing=  0.00%
  PCR 48h                        : Mean=    123.36, Std=  111.52, Missing=  1.38%
  PCR 72h                        : Mean=    141.43, Std=  123.54, Missing= 19.33%
  Ransom Adm                     : Mean=      1.62, Std=    1.13, Missing=  0.00%
  Ransom 48h                     : Mean=      1.99, Std=    1.62, Missing=  0.00%
  Urea Adm                       : Mean=     45.22, Std=   29.33, Missing=  0.00%
  Urea 48h                       : Mean=     42.11, Std=   31.66, Missing=  1.23%
  Urea 72h                       : Mean=     39.78, Std=   31.34, Missing= 18.87%
  Creat Adm                      : Mean=      1.05, Std=    0.68, Missing=  0.00%
  Creat 48h                      : Mean=      1.15, Std=    1.95, Missing=  1.23%
  Creat 72h                      : Mean=      1.11, Std=    2.30, Missing= 19.17%
  Hct Adm                        : Mean=     41.87, Std=    5.83, Missing=  0.00%
  Hct 48h                        : Mean=     39.26, Std=   14.57, Missing=  0.92%
  Hct 72h                        : Mean=     37.34, Std=    4.98, Missing= 19.17%
  Leukocytes Adm                 : Mean=  12730.88, Std= 5336.43, Missing=  0.00%
  PMN Adm                        : Mean=  10590.98, Std= 6188.12, Missing=  0.00%
  Lymphocytes Adm                : Mean=   1373.78, Std= 1010.27, Missing=  0.00%
  Mono Adm                       : Mean=    571.65, Std=  438.36, Missing=  0.00%
  Eosinophils Adm                : Mean=     63.63, Std=  294.16, Missing=  0.00%
  Leukocytes 48h                 : Mean=  11089.15, Std= 7825.30, Missing=  1.23%
  PMN 48h                        : Mean=   8653.60, Std= 5124.75, Missing=  1.23%
  Lymph 48h                      : Mean=   1357.69, Std=  927.48, Missing=  1.23%
  Mono 48h                       : Mean=    635.93, Std=  393.99, Missing=  1.23%
  Eosinophils 48h                : Mean=    108.69, Std=  205.81, Missing=  1.38%
  Leukocytes 72h                 : Mean=   9876.92, Std= 4827.57, Missing= 19.02%
  PMN 72h                        : Mean=   7500.54, Std= 4684.43, Missing= 19.02%
  Lymph 72h                      : Mean=   1412.15, Std=  812.92, Missing= 19.02%
  Mono 72h                       : Mean=    682.17, Std=  455.12, Missing= 19.02%
  Eosinophils 72h                : Mean=    157.18, Std=  187.08, Missing= 19.02%
  PMN/Lymph Admin                : Mean=     13.95, Std=   26.06, Missing=  0.00%
  PMN/Lymph 48h                  : Mean=      9.45, Std=    9.34, Missing=  1.23%
  PMN/Lymph 72h                  : Mean=      7.86, Std=   11.95, Missing= 19.02%

================================================================================
DATA QUALITY ASSESSMENT
================================================================================

1. COMPLETENESS
   Overall Data Completeness: 93.17%
   Complete Rows (no missing): 28
   Rows with at least 1 missing: 624

2. CONSISTENCY
   Duplicate Rows: 0
   Columns with Unexpected Negative Values:
     - Exceso Bases: 244 values

3. VALIDITY
   Numeric Columns: 74
   Features with Extreme Ranges (max/min > 1000):
     - BMI: min=15.05, max=256369.17, ratio=17039
     - PMN Adm: min=11.01, max=100400.00, ratio=9119
     - Mono Adm: min=2.00, max=8310.00, ratio=4155
     - Mono 48h: min=4.00, max=5020.00, ratio=1255
     - PMN 72h: min=20.00, max=26930.00, ratio=1346
     - Platelets: min=99.00, max=566000.00, ratio=5717
     - Lipase: min=5.00, max=443150.00, ratio=88630
     - GOT: min=1.00, max=3566.00, ratio=3566
     - PMN/Lymph Admin: min=0.05, max=532.50, ratio=10157
     - PMN/Lymph 72h: min=0.01, max=176.29, ratio=18510
     - Inflammatory Index: min=483.07, max=156555000.00, ratio=324085

4. ACCURACY / DISTRIBUTION QUALITY
   Highly Skewed Features (|skewness| > 2):
     - Weight: 25.02
     - Ca: 23.73
     - Creat 72h: 19.70
     - Lipase: 19.40
     - Exceso Bases: -18.90
     - Hct 48h: 18.62
     - BMI: 17.75
     - Inflammatory Index: 17.31
     - Creat 48h: 17.02
     - Eosinophils Adm: 16.69

5. OVERALL QUALITY SCORE
   Completeness Score: 93.2/100
   Rating: EXCELLENT

================================================================================
TRAIN vs TEST SET COMPARISON
================================================================================

Dataset Sizes:
  Train: 652 rows × 74 columns
  Test:  163 rows × 73 columns

Common Columns: 73
Train-only Columns: Length of stay

--- Numerical Feature Distribution Comparison ---
Feature                          Train Mean    Test Mean         Diff    Train Std     Test Std
-----------------------------------------------------------------------------------------------
 CO3 H                               23.667       49.646       25.979        7.781      248.929
Abdominal Exam                      100.519      100.939        0.421       14.721       13.564
Age                                  68.684       68.742        0.058       17.395       16.419
Albumin                               3.634        3.675        0.042        0.541        0.524
Amilasa                            1520.622     1306.216     -214.406     1454.724      906.131
BISAP                                 1.224        1.141       -0.083        0.993        0.895
BMI                                 814.676     1717.772      903.096    13885.591    21292.602
CCI                                   1.276        1.564        0.288        1.605        2.006
Ca                                    8.787        8.632       -0.155        2.826        0.684
Chest X-Ray                           0.233        0.217       -0.016        0.423        0.414
Col total                           155.204      171.543       16.339       67.000      150.601
Creat 48h                             1.148        1.008       -0.140        1.954        0.780
Creat 72h                             1.115        0.997       -0.118        2.296        0.868
Creat Adm                             1.053        0.967       -0.086        0.679        0.582
DM                                    0.160        0.185        0.025        0.367        0.390
Drinker                               0.490        0.491        0.001        0.829        0.849
Dyslipidemias                         0.380        0.377       -0.003        0.548        0.590
Eosinophils 48h                     108.694       99.534       -9.160      205.807      164.112
Eosinophils 72h                     157.178      191.379       34.201      187.081      244.182
Eosinophils Adm                      63.626       41.184      -22.442      294.157      103.552
Etiology                              2.141        2.110       -0.031        1.651        1.644
Exceso Bases                         -3.113       -0.792        2.321       31.733        2.829
F alc                               148.616      158.393        9.777      126.823      166.590
GOT                                 196.494      193.172       -3.322      311.885      256.314
GPT                                 178.983      192.540       13.557      233.427      246.631
Gasometría: Ph                        7.383        7.377       -0.007        0.057        0.056
Glu                                 132.619      139.926        7.308       54.175       58.810
Grams/Day                            15.754       12.822       -2.932       33.087       29.086
HBD                                   0.522        0.549        0.027        0.500        0.499
Hct 48h                              39.257       38.734       -0.523       14.574        5.564
Hct 72h                              37.335       40.123        2.788        4.983       30.935
Hct Adm                              41.873       42.514        0.641        5.833        5.001
Height                              162.477      162.159       -0.318       13.558       16.516
Inflammatory Index              3094301.045  3163938.903    69637.858  6916012.633  5585242.634
Leukocytes 48h                    11089.146    11148.957       59.811     7825.300     5519.367
Leukocytes 72h                     9876.924     9762.477     -114.447     4827.568     4384.148
Leukocytes Adm                    12730.876    12465.393     -265.483     5336.432     4724.268
Lipase                             4314.554     3256.907    -1057.647    19169.768     4687.881
Lymph 48h                          1357.694     1315.693      -42.001      927.477      730.507
Lymph 72h                          1412.150     1457.008       44.858      812.922      895.295
Lymphocytes Adm                    1373.778     1308.601      -65.176     1010.271      933.826
Mono 48h                            635.927      646.319       10.392      393.992      339.893
Mono 72h                            682.170      704.000       21.830      455.121      407.532
Mono Adm                            571.647      536.847      -34.801      438.355      272.545
Onset pain                           19.449       17.233       -2.216       20.097       23.848
PCO2                                 38.210       41.114        2.904        7.951        8.678
PCR 48h                             123.357      120.954       -2.402      111.517      118.179
PCR 72h                             141.432      139.620       -1.812      123.535      114.869
PCR Adm                              40.495       40.016       -0.479       69.797       67.034
PCR/Alb ratio                        12.416       11.427       -0.988       23.484       20.032
PMN 48h                            8653.601     9615.006      961.405     5124.752     9643.575
PMN 72h                            7500.544     7189.939     -310.604     4684.430     4154.111
PMN Adm                           10590.985    10441.012     -149.972     6188.121     4578.111
PMN/Lymph 48h                         9.451        9.583        0.132        9.337       11.134
PMN/Lymph 72h                         7.858        6.916       -0.942       11.952        7.779
PMN/Lymph Admin                      13.950       14.055        0.105       26.061       21.379
Packs/Year                           12.715       10.614       -2.102       20.396       18.296
Pain Intensity                        9.323        9.296       -0.027        1.229        1.200
Petrov                                0.299        0.301        0.002        0.684        0.686
Phosphate                             2.940        2.923       -0.017        0.834        0.731
Platelets                        229205.280   223244.043    -5961.237    77888.045    86427.423
Previous AP                           1.181        1.166       -0.015        0.484        0.448
Ransom 48h                            1.988        1.907       -0.080        1.620        1.409
Ransom Adm                            1.617        1.574       -0.042        1.133        1.102
SIRS                                  0.402        0.362       -0.040        0.700        0.656
Sex                                   0.534        0.534        0.000        0.499        0.500
Smoker                                0.593        0.509       -0.084        0.772        0.723
TG                                  150.968      110.199      -40.769      363.110       81.455
Urea 48h                             42.112       39.626       -2.486       31.662       24.882
Urea 72h                             39.785       36.776       -3.009       31.340       25.247
Urea Adm                             45.225       42.524       -2.701       29.332       20.235
Waist circ                            1.078        1.151        0.073        0.575        0.517
Weight                               87.417       78.336       -9.081      336.264       65.701