#pragma once
#include "device.h"
#include <stdio.h>
struct BSSN_EVAR_DERIVS{
	DEVICE_REAL *grad_0_alpha;
	DEVICE_REAL *grad_1_alpha;
	DEVICE_REAL *grad_2_alpha;
	DEVICE_REAL *grad_0_chi;
	DEVICE_REAL *grad_1_chi;
	DEVICE_REAL *grad_2_chi;
	DEVICE_REAL *grad_0_K;
	DEVICE_REAL *grad_1_K;
	DEVICE_REAL *grad_2_K;
	DEVICE_REAL *grad_0_Gt0;
	DEVICE_REAL *grad_1_Gt0;
	DEVICE_REAL *grad_2_Gt0;
	DEVICE_REAL *grad_0_Gt1;
	DEVICE_REAL *grad_1_Gt1;
	DEVICE_REAL *grad_2_Gt1;
	DEVICE_REAL *grad_0_Gt2;
	DEVICE_REAL *grad_1_Gt2;
	DEVICE_REAL *grad_2_Gt2;
	DEVICE_REAL *grad_0_beta0;
	DEVICE_REAL *grad_1_beta0;
	DEVICE_REAL *grad_2_beta0;
	DEVICE_REAL *grad_0_beta1;
	DEVICE_REAL *grad_1_beta1;
	DEVICE_REAL *grad_2_beta1;
	DEVICE_REAL *grad_0_beta2;
	DEVICE_REAL *grad_1_beta2;
	DEVICE_REAL *grad_2_beta2;
	DEVICE_REAL *grad_0_B0;
	DEVICE_REAL *grad_1_B0;
	DEVICE_REAL *grad_2_B0;
	DEVICE_REAL *grad_0_B1;
	DEVICE_REAL *grad_1_B1;
	DEVICE_REAL *grad_2_B1;
	DEVICE_REAL *grad_0_B2;
	DEVICE_REAL *grad_1_B2;
	DEVICE_REAL *grad_2_B2;
	DEVICE_REAL *grad_0_gt0;
	DEVICE_REAL *grad_1_gt0;
	DEVICE_REAL *grad_2_gt0;
	DEVICE_REAL *grad_0_gt1;
	DEVICE_REAL *grad_1_gt1;
	DEVICE_REAL *grad_2_gt1;
	DEVICE_REAL *grad_0_gt2;
	DEVICE_REAL *grad_1_gt2;
	DEVICE_REAL *grad_2_gt2;
	DEVICE_REAL *grad_0_gt3;
	DEVICE_REAL *grad_1_gt3;
	DEVICE_REAL *grad_2_gt3;
	DEVICE_REAL *grad_0_gt4;
	DEVICE_REAL *grad_1_gt4;
	DEVICE_REAL *grad_2_gt4;
	DEVICE_REAL *grad_0_gt5;
	DEVICE_REAL *grad_1_gt5;
	DEVICE_REAL *grad_2_gt5;
	DEVICE_REAL *grad_0_At0;
	DEVICE_REAL *grad_1_At0;
	DEVICE_REAL *grad_2_At0;
	DEVICE_REAL *grad_0_At1;
	DEVICE_REAL *grad_1_At1;
	DEVICE_REAL *grad_2_At1;
	DEVICE_REAL *grad_0_At2;
	DEVICE_REAL *grad_1_At2;
	DEVICE_REAL *grad_2_At2;
	DEVICE_REAL *grad_0_At3;
	DEVICE_REAL *grad_1_At3;
	DEVICE_REAL *grad_2_At3;
	DEVICE_REAL *grad_0_At4;
	DEVICE_REAL *grad_1_At4;
	DEVICE_REAL *grad_2_At4;
	DEVICE_REAL *grad_0_At5;
	DEVICE_REAL *grad_1_At5;
	DEVICE_REAL *grad_2_At5;
	DEVICE_REAL *grad2_0_0_gt0;
	DEVICE_REAL *grad2_0_1_gt0;
	DEVICE_REAL *grad2_0_2_gt0;
	DEVICE_REAL *grad2_1_1_gt0;
	DEVICE_REAL *grad2_1_2_gt0;
	DEVICE_REAL *grad2_2_2_gt0;
	DEVICE_REAL *grad2_0_0_gt1;
	DEVICE_REAL *grad2_0_1_gt1;
	DEVICE_REAL *grad2_0_2_gt1;
	DEVICE_REAL *grad2_1_1_gt1;
	DEVICE_REAL *grad2_1_2_gt1;
	DEVICE_REAL *grad2_2_2_gt1;
	DEVICE_REAL *grad2_0_0_gt2;
	DEVICE_REAL *grad2_0_1_gt2;
	DEVICE_REAL *grad2_0_2_gt2;
	DEVICE_REAL *grad2_1_1_gt2;
	DEVICE_REAL *grad2_1_2_gt2;
	DEVICE_REAL *grad2_2_2_gt2;
	DEVICE_REAL *grad2_0_0_gt3;
	DEVICE_REAL *grad2_0_1_gt3;
	DEVICE_REAL *grad2_0_2_gt3;
	DEVICE_REAL *grad2_1_1_gt3;
	DEVICE_REAL *grad2_1_2_gt3;
	DEVICE_REAL *grad2_2_2_gt3;
	DEVICE_REAL *grad2_0_0_gt4;
	DEVICE_REAL *grad2_0_1_gt4;
	DEVICE_REAL *grad2_0_2_gt4;
	DEVICE_REAL *grad2_1_1_gt4;
	DEVICE_REAL *grad2_1_2_gt4;
	DEVICE_REAL *grad2_2_2_gt4;
	DEVICE_REAL *grad2_0_0_gt5;
	DEVICE_REAL *grad2_0_1_gt5;
	DEVICE_REAL *grad2_0_2_gt5;
	DEVICE_REAL *grad2_1_1_gt5;
	DEVICE_REAL *grad2_1_2_gt5;
	DEVICE_REAL *grad2_2_2_gt5;
	DEVICE_REAL *grad2_0_0_chi;
	DEVICE_REAL *grad2_0_1_chi;
	DEVICE_REAL *grad2_0_2_chi;
	DEVICE_REAL *grad2_1_1_chi;
	DEVICE_REAL *grad2_1_2_chi;
	DEVICE_REAL *grad2_2_2_chi;
	DEVICE_REAL *grad2_0_0_alpha;
	DEVICE_REAL *grad2_0_1_alpha;
	DEVICE_REAL *grad2_0_2_alpha;
	DEVICE_REAL *grad2_1_1_alpha;
	DEVICE_REAL *grad2_1_2_alpha;
	DEVICE_REAL *grad2_2_2_alpha;
	DEVICE_REAL *grad2_0_0_beta0;
	DEVICE_REAL *grad2_0_1_beta0;
	DEVICE_REAL *grad2_0_2_beta0;
	DEVICE_REAL *grad2_1_1_beta0;
	DEVICE_REAL *grad2_1_2_beta0;
	DEVICE_REAL *grad2_2_2_beta0;
	DEVICE_REAL *grad2_0_0_beta1;
	DEVICE_REAL *grad2_0_1_beta1;
	DEVICE_REAL *grad2_0_2_beta1;
	DEVICE_REAL *grad2_1_1_beta1;
	DEVICE_REAL *grad2_1_2_beta1;
	DEVICE_REAL *grad2_2_2_beta1;
	DEVICE_REAL *grad2_0_0_beta2;
	DEVICE_REAL *grad2_0_1_beta2;
	DEVICE_REAL *grad2_0_2_beta2;
	DEVICE_REAL *grad2_1_1_beta2;
	DEVICE_REAL *grad2_1_2_beta2;
	DEVICE_REAL *grad2_2_2_beta2;
	DEVICE_REAL *kograd_0_alpha;
	DEVICE_REAL *kograd_1_alpha;
	DEVICE_REAL *kograd_2_alpha;
	DEVICE_REAL *kograd_0_chi;
	DEVICE_REAL *kograd_1_chi;
	DEVICE_REAL *kograd_2_chi;
	DEVICE_REAL *kograd_0_K;
	DEVICE_REAL *kograd_1_K;
	DEVICE_REAL *kograd_2_K;
	DEVICE_REAL *kograd_0_Gt0;
	DEVICE_REAL *kograd_1_Gt0;
	DEVICE_REAL *kograd_2_Gt0;
	DEVICE_REAL *kograd_0_Gt1;
	DEVICE_REAL *kograd_1_Gt1;
	DEVICE_REAL *kograd_2_Gt1;
	DEVICE_REAL *kograd_0_Gt2;
	DEVICE_REAL *kograd_1_Gt2;
	DEVICE_REAL *kograd_2_Gt2;
	DEVICE_REAL *kograd_0_beta0;
	DEVICE_REAL *kograd_1_beta0;
	DEVICE_REAL *kograd_2_beta0;
	DEVICE_REAL *kograd_0_beta1;
	DEVICE_REAL *kograd_1_beta1;
	DEVICE_REAL *kograd_2_beta1;
	DEVICE_REAL *kograd_0_beta2;
	DEVICE_REAL *kograd_1_beta2;
	DEVICE_REAL *kograd_2_beta2;
	DEVICE_REAL *kograd_0_B0;
	DEVICE_REAL *kograd_1_B0;
	DEVICE_REAL *kograd_2_B0;
	DEVICE_REAL *kograd_0_B1;
	DEVICE_REAL *kograd_1_B1;
	DEVICE_REAL *kograd_2_B1;
	DEVICE_REAL *kograd_0_B2;
	DEVICE_REAL *kograd_1_B2;
	DEVICE_REAL *kograd_2_B2;
	DEVICE_REAL *kograd_0_gt0;
	DEVICE_REAL *kograd_1_gt0;
	DEVICE_REAL *kograd_2_gt0;
	DEVICE_REAL *kograd_0_gt1;
	DEVICE_REAL *kograd_1_gt1;
	DEVICE_REAL *kograd_2_gt1;
	DEVICE_REAL *kograd_0_gt2;
	DEVICE_REAL *kograd_1_gt2;
	DEVICE_REAL *kograd_2_gt2;
	DEVICE_REAL *kograd_0_gt3;
	DEVICE_REAL *kograd_1_gt3;
	DEVICE_REAL *kograd_2_gt3;
	DEVICE_REAL *kograd_0_gt4;
	DEVICE_REAL *kograd_1_gt4;
	DEVICE_REAL *kograd_2_gt4;
	DEVICE_REAL *kograd_0_gt5;
	DEVICE_REAL *kograd_1_gt5;
	DEVICE_REAL *kograd_2_gt5;
	DEVICE_REAL *kograd_0_At0;
	DEVICE_REAL *kograd_1_At0;
	DEVICE_REAL *kograd_2_At0;
	DEVICE_REAL *kograd_0_At1;
	DEVICE_REAL *kograd_1_At1;
	DEVICE_REAL *kograd_2_At1;
	DEVICE_REAL *kograd_0_At2;
	DEVICE_REAL *kograd_1_At2;
	DEVICE_REAL *kograd_2_At2;
	DEVICE_REAL *kograd_0_At3;
	DEVICE_REAL *kograd_1_At3;
	DEVICE_REAL *kograd_2_At3;
	DEVICE_REAL *kograd_0_At4;
	DEVICE_REAL *kograd_1_At4;
	DEVICE_REAL *kograd_2_At4;
	DEVICE_REAL *kograd_0_At5;
	DEVICE_REAL *kograd_1_At5;
	DEVICE_REAL *kograd_2_At5;
};