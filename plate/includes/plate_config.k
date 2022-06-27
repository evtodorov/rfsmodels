$# LS-DYNA Keyword file created by LS-PrePost(R) V4.7.17 - 08Jul2020
$# Created on Jun-27-2022 (12:45:40)
*KEYWORD
*CONTROL_TERMINATION
$#  endtim    endcyc     dtmin    endeng    endmas     nosol     
       5.0         0       0.0       0.01.000000E8         0
*DATABASE_ELOUT
$#      dt    binary      lcur     ioopt   option1   option2   option3   option4
      0.01         2         0         1         0         0         0         0
*DATABASE_NODFOR
$#      dt    binary      lcur     ioopt     
      0.01         2         0         1
*DATABASE_NODOUT
$#      dt    binary      lcur     ioopt   option1   option2       
      0.01         2         0         1       0.0         2
*DATABASE_EXTENT_BINARY
$#   neiph     neips    maxint    strflg    sigflg    epsflg    rltflg    engflg
         0         0         3         0         1         1         1         1
$#  cmpflg    ieverp    beamip     dcomp      shge     stssz    n3thdt   ialemat
         0         0         0         1         1         1         2         1
$# nintsld   pkp_sen      sclp     hydro     msscl     therm    intout    nodout
         0         0       1.0         0         0         0          ALL_GL
$#    dtdt    resplt     neipb     quadr     cubic     
         0         0         0         0         0
*DATABASE_MASSOUT
$#   setid     ndflg     rbflg     
         0         4         0
*DATABASE_NODAL_FORCE_GROUP
$#    nsid       cid   
         1         0
*DATABASE_HISTORY_NODE_SET
$#     id1       id2       id3       id4       id5       id6       id7       id8
         1         0         0         0         0         0         0         0
*DATABASE_HISTORY_SHELL_SET
$#     id1       id2       id3       id4       id5       id6       id7       id8
         2         0         0         0         0         0         0         0
*SET_NODE_GENERAL_TITLE
ALL_NODES
$#     sid       da1       da2       da3       da4    solver      
         1       0.0       0.0       0.0       0.0MECH
$#  option        e1        e2        e3        e4        e5        e6        e7
ALL                0         0         0         0         0         0         0
*SET_SHELL_GENERAL_TITLE
ALL_SHELLS
$#     sid       da1       da2       da3       da4   
         2       0.0       0.0       0.0       0.0
$#  option        e1        e2        e3        e4        e5        e6        e7
ALL                0         0         0         0         0         0         0
*COMMENT  *DATABASE_EXTENT_BINARY
         0         0         3         0         1         1         1         1
         0         0         0         1         1         1         1         1
         0         0       1.0         0         0         0          ALL
         0         0         0         0         0
*END
