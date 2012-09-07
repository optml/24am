/*******************************************************************************
!   Copyright(C) 2010-2012 Intel Corporation. All Rights Reserved.
!   
!   The source code, information  and  material ("Material") contained herein is
!   owned  by Intel Corporation or its suppliers or licensors, and title to such
!   Material remains  with Intel Corporation  or its suppliers or licensors. The
!   Material  contains proprietary information  of  Intel or  its  suppliers and
!   licensors. The  Material is protected by worldwide copyright laws and treaty
!   provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
!   modified, published, uploaded, posted, transmitted, distributed or disclosed
!   in any way  without Intel's  prior  express written  permission. No  license
!   under  any patent, copyright  or  other intellectual property rights  in the
!   Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
!   implication, inducement,  estoppel or  otherwise.  Any  license  under  such
!   intellectual  property  rights must  be express  and  approved  by  Intel in
!   writing.
!   
!   *Third Party trademarks are the property of their respective owners.
!   
!   Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
!   this  notice or  any other notice embedded  in Materials by Intel or Intel's
!   suppliers or licensors in any way.
!
!*******************************************************************************
!  Content:
!      Intel(R) Math Kernel Library PBLAS C example's definitions file
!
!******************************************************************************/
#include <mkl_scalapack.h>

#ifndef  mkl_constants_and_headers_h
#define  mkl_constants_and_headers_h


#ifdef _WIN_

/* Definitions for proper work of examples on Windows */
#define blacs_pinfo_ BLACS_PINFO
#define blacs_get_ BLACS_GET
#define blacs_gridinit_ BLACS_GRIDINIT
#define blacs_gridinfo_ BLACS_GRIDINFO
#define blacs_barrier_ BLACS_BARRIER
#define blacs_gridexit_ BLACS_GRIDEXIT
#define blacs_exit_ BLACS_EXIT
#define igebs2d_ IGEBS2D
#define igebr2d_ IGEBR2D
#define sgebs2d_ SGEBS2D
#define sgebr2d_ SGEBR2D
#define dgebs2d_ DGEBS2D
#define dgebr2d_ DGEBR2D
#define sgesd2d_ SGESD2D
#define sgerv2d_ SGERV2D
#define dgesd2d_ DGESD2D
#define dgerv2d_ DGERV2D
#define numroc_ NUMROC
#define descinit_ DESCINIT
#define psnrm2_ PSNRM2
#define pdnrm2_ PDNRM2
#define psscal_ PSSCAL
#define pdscal_ PDSCAL
#define psdot_ PSDOT
#define pddot_ PDDOT
#define pslamch_ PSLAMCH
#define pdlamch_ PDLAMCH
#define indxg2l_ INDXG2L
#define pscopy_ PSCOPY
#define pdcopy_ PDCOPY
#define pstrsv_ PSTRSV
#define pdtrsv_ PDTRSV
#define pstrmv_ PSTRMV
#define pdtrmv_ PDTRMV
#define pslange_ PSLANGE
#define pdlange_ PDLANGE
#define psgemm_ PSGEMM
#define pdgemm_ PDGEMM
#define psgeadd_ PSGEADD
#define pdgeadd_ PDGEADD

#endif

/* Pi-number */
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

/* Definition of MIN and MAX functions */
#define MAX(a,b)((a)<(b)?(b):(a))
#define MIN(a,b)((a)>(b)?(b):(a))

/* Definition of matrix descriptor */
typedef MKL_INT MDESC[ 9 ];



/* Parameters */
extern const double zero  , one  , two  , negone ;
extern const MKL_INT i_zero, i_one, i_four, i_negone;
extern MKL_INT i_tmp1, i_tmp2, i_tmp3;
extern const char trans;
extern const char transNo;
extern const char C_CHAR_SCOPE_ALL;
extern const char C_CHAR_SCOPE_ROWS;
extern const char C_CHAR_SCOPE_COLS;
extern const char C_CHAR_GENERAL_TREE_CATHER;

#endif
