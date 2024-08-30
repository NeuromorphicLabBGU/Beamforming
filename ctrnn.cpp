// ctrnn.cpp : Defines the entry point for the console application.
//
//**********************************************************************************************************************
//********                                                                                                     *********
//********  Program: ctrnn (Continuous Time Recurrent Neural Network)                                          *********
//********  Description:                                                                                       *********
//********  This program is an example of how the CTN neurons can be used to design feature-extraction         *********
//********  blocks to provide the pre-processing operation for deep-learning neural networks. In this          *********
//********  example the PDM bit-streams from two identical digital microphones directly feed CTRNN neurons     *********
//********  in a pulse cycle that is based on the PDM clock-rate (1.536MHz). The neural network in this        *********
//********  example performs the Generalized Cross Correlation (GCC) pre-process that is needed in every       *********
//********  implementation of acoustic beam-formers. The GCC network uses 97 neurons and appropriate delay     *********
//********  lines of 1-bit samples. This design is planed to be implemented with ALTERA FPGA                   *********
//********                                                                                                     *********
//**********************************************************************************************************************

#include "stdafx.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>

#define DLY_BUFF_LEN16   16
#define MAX_LOOPS   50000000



//**********************************************************************************************************************
//********                                                                                                     *********
//********                              CTN Integrate and Fire Neuron                                          *********
//********                                                                                                     *********
//**********************************************************************************************************************


#define IDENTITY 0			// 0: Identity (biased up) 1: Binary step  2: Sigmoid (Logistic)
#define BINARY_STEP 1		// 0: Identity (biased up) 1: Binary step  2: Sigmoid (Logistic)
#define SIGMOID 2			// 0: Identity (biased up) 1: Binary step  2: Sigmoid (Logistic)
#define P_VALUE_2POW19 524287 // Maximum neuron value. The CTN accumulator is 20-bit signed integer
#define N_VALUE_2POW19 -524288 // Minimum neuron value. The CTN accumulator is 20-bit signed integer
#define GAUSSIAN_RAND_ORDER 8 // Number of unified distributed r.v components for gaussian approximation
//
struct CTN {
	//
	// Neuron Parameters:
	int activation_type;	// 0: Identity (biased up) 1: Binary step  2: Sigmoid (Logistic)
	int numb_synapses;		// 10-bit unsigned number of input synapses to the neuron ranges 1-1023
	short *w;				// 9-bit signed integers array of input synapse weights signed integers
	short teta;				// 10-bit signed integer bias
	int leakage_factor;		// 3-bit unsigned integer, defines the decay factor 1-2**(-leak_factor) based on shift right
	int leakage_period;		// 16-bit unsigned integer, defines the number of pulse cycles to perform a leakage step
	//
	// Neuron Variables:
	int acc;				// 20-bit signed integer
	bool *f;				// 1-bit boolean array of input synapse pulses 
	int leakage_timer;		// 16-bit unsigned integer, counts the number of pulse cycles following latest leakage step
	int pn_generator;		// 15-bit unsigned integer, used for producing pseudo-random numbers
	int rand_gauss_var;		// 16-bit integer, used for pseudo-random number generation or produce output pulses
};




// This function performs the (Continous Time Neuron) CTN neuron behavior per pulse
// cycle. It is called once every pulse cycle, and may or may not produce a pulsed
// output.There are several parameters that direct the operation mode and time
// constants of the firing neuron, and these are kept constant during operation.
//
// Neuron activation_type (parameter):
// There are 3 different types of activation functions:
// IDENTITY: This activation function is linear to the input and is based on a linearly distributed random number.
// BINARY: This activation function produces an output pulse if and only if the neuron integrator value is positive..
// SIGMOID: This activation function is close to Sigmoid() and is based on a sum of 8 linearly distributed random
//          numbers. The probability density function of the sum of random numbers is close to a Gaussian function.
//
// Neuron numb_synapses (parameter):
// This is a 10-bit unsigned number that defines the number of input synapses of the neuron (ranges 1-1023).
//
// Neuron *w (parameter array):
// This is a pointer to an array of 9-bit signed numbers that define the synapses input weights. The array size must
// be equal to the parameter numb_synapses, and each weight ranges between -256 to +255.
//
// Neuron teta:
// This is a 10-bit signed number that is added to the neuron integrator once in every pulse cycle, and it ranges
// between -512 to +511.
//
// Neuron leakage_factor (parameter):
// This is a 3-bit unsigned number that defines the integrator leakage rate and has an effect on the neuron gain and
// delay. The leakage_factor parameter defines the number of shifts to the right that are applied to the integrator
// value to be subtracted from the integrator value. This happens every time that a leakage operation is executed,
// and is equivalent to multiplying the integrator by the factor alpha = 1.0 - 2**(-leakage_factor).
// Expected leakage time constant (or delay) is:	pulse_cycle * 2**(leakage_factor) * (1 + leakage_period)
// Expected gain in IDENTITY and SIGMOID modes:	leakage_factor > 2:   2**(2*leakage_factor-3) * (1 + leakage_period)
//                                          	leakage_factor < 3:   2**(leakage_factor) * (1 + leakage_period)
//
// Neuron leakage_period (parameter):
// This is a 16-bit unsigned number that defines the rate of integrator leakage operation and has an effect on the
// neuron gain and delay. A value 0 defines a full rate, that's to say that a leakage operation is performed
// per every pulse cycle. A value n defines a leakage operation once every n+1 pulse cycles.
// Expected leakage time constant (or delay) is:	pulse_cycle * 2**(leakage_factor) * (1 + leakage_period)
// Expected gain in IDENTITY and SIGMOID modes:	leakage_factor > 2:   2**(2*leakage_factor-3) * (1 + leakage_period)
//                                          	leakage_factor < 3:   2**(leakage_factor) * (1 + leakage_period)
//
// Note that in the IDENTITY and SIGMOID modes, the neuron has a unity gain (for W=255, teta=-128) if leakage_factor=5
// and leakage_period=0 (with typical delay of 32 pulse-cycles).
//
//
// Neuron pn_generator:
// This is a 15-bit unsigned variable that is used for generating pseudo-randon numbers in the range 1-32767. The
// pseudo-random generator is used for producing homogeneous pulses at the neuron output over the time-axis, and
// is only used for neurons that have either IDENTITY or SIGMOID activation types.
//
// The CTN simulator operation is devided into 5 sequencial steps: Accumulate, Bias, Output, Leakage, and Idle.
// If the enable input is FALSE then only steps 3 and 5 are performed (in order to continue the output
// pulse pattern)
//
//
bool ctn_cycle(CTN *neuron, bool enable) {

	int i;				// General temporary integer
	int decay_delta;	// Temporary 20-bit signed integer
	bool out_pulse;		// output is true="pulse" or false="no_pulse"

	if(enable) {	// Enable steps 1,2, and 4
		//
		// Step 1: Accumulate: Accumulate all contributions from all input synapses:
		for(i=0; i<neuron->numb_synapses; i++) {
			if(neuron->f[i]) {						// If input is '1'
				if(neuron->leakage_factor >= 3) {	// 3,4,5,6, or 7
					neuron->acc += neuron->w[i] << (neuron->leakage_factor - 3); // Max. number of bits (including sign-bit): 13 = 9+(7-3)
				}
				else {	// 0,1, or 2
					neuron->acc += neuron->w[i];	// Max. number of bits (including sign-bit): 9
				}
				// Truncate the result to 20-bit (including sign bit):
				if(neuron->acc > P_VALUE_2POW19) neuron->acc = P_VALUE_2POW19;
				else if(neuron->acc < N_VALUE_2POW19) neuron->acc = N_VALUE_2POW19;
			}
		}

		// Step 2: Bias: Add the neuron Bias value teta:
		if(neuron->leakage_factor >= 3) {	// 3,4,5,6, or 7
			neuron->acc += neuron->teta << (neuron->leakage_factor - 3);	// Max. number of bits (including sign-bit): 14 = 10+(7-3)
		}
		else {	// 0,1, or 2
			neuron->acc += neuron->teta;	// Max. number of bits (including sign-bit): 10
		}
		// Truncate the result to 20-bit (including sign bit):
		if(neuron->acc > P_VALUE_2POW19) neuron->acc = P_VALUE_2POW19;
		else if(neuron->acc < N_VALUE_2POW19) neuron->acc = N_VALUE_2POW19;
	}	// If enable

	// Step 3: Output: Prepare the neuron output for the current pulse-cycle ("pulse" or "no pulse"):
	switch(neuron->activation_type) {
		case 0:	// "Identity"
			if(neuron->acc > 32767) {	// Positive out of range
				out_pulse = true;
				neuron->rand_gauss_var = 32767;
			}
			else if(neuron->acc < -32767) {	// Negative out of range
				out_pulse = false;
				neuron->rand_gauss_var = 32767;
			}
			else {						// In range. Produce proportional average pulse-rate
				neuron->rand_gauss_var += (neuron->acc + 32768);	// Convert neuron->acc to a positive integer between 1-65535
				if(neuron->rand_gauss_var >= 65536) {	// rand_gauss_var overflow. Produce a pulse and retain the remainder
					neuron->rand_gauss_var -= 65536;
					out_pulse = true;
				}
				else out_pulse = false;
			}
			break;	
		case 1:	// "Binary step"
			out_pulse = (neuron->acc > 0);	// Compare 20-bit integer to 0
			break;
		case 2:	// "Sigmoid"
			// Prepare a closed to gaussian variable around 0:
			neuron->rand_gauss_var = 0;										// Initial condition is 0
			for(i=0; i < GAUSSIAN_RAND_ORDER; i++) {
				neuron->rand_gauss_var += (neuron->pn_generator & 0x1fff);	// Add the 13-bit signed random variables
				neuron->pn_generator = ((neuron->pn_generator) >> 1) | ((neuron->pn_generator & 0x4000) ^ ((neuron->pn_generator & 0x0001) << 14));	// Shift left and feed-back d14 ^ d0
			}
			if((neuron->rand_gauss_var & 0x8000) != 0) neuron->rand_gauss_var = neuron->rand_gauss_var | 0xFFFF0000;	// Extend the sign bit to the left
			out_pulse = (neuron->acc > neuron->rand_gauss_var);	// Compare 20-bit integer to 16-bit integer
	}

	if(enable) {
		// Step 4: Leakage: Decay the accumulated value:
		if(neuron->leakage_timer >= neuron->leakage_period) {
			if(neuron->acc < 0) {
				decay_delta = (-neuron->acc) >> neuron->leakage_factor;
				if((decay_delta == 0) && (neuron->acc != 0)) decay_delta = 1;
			}
			else {	// neuron->acc >= 0
				decay_delta = -(neuron->acc >> neuron->leakage_factor);
				if((decay_delta == 0) && (neuron->acc != 0)) decay_delta = -1;
			}
			neuron->acc += decay_delta;
			//
			neuron->leakage_timer = 0;	// Restart the timer after a leakage operation
		}
		else {
			neuron->leakage_timer = neuron->leakage_timer + 1;	// Increment the timer if no leakage operation
		}
	}	// If enable

	// Step 5: Idle: Return and produce the neuron output:
	return out_pulse;

}



bool ctn_cycle_with_vector_file(CTN *neuron, bool enable, FILE *vector_file) {

	int i;				// General temporary integer
	int j;				// General temporary integer
	int k;				// General temporary integer
	static bool out_pulse;		// output is true="pulse" or false="no_pulse"

	/*
	// HEXADECIMAL mode
	for(i=0; i < neuron->numb_synapses; i++) {
		j = (neuron->f[i])? 1: 0;
		fprintf(vector_file, "%01d ",j);					// Print "1" for TRUE input and "0" for FALSE input 
	}
	j = (enable)? 1: 0;
	fprintf(vector_file, "%01d ",j);					// Print "1" for enable=TRUE and "0" for enable=FALSE 
	fprintf(vector_file, "%05X ",(neuron->acc & 0x0fffff));	// Print 5 hexadecimal digits of ACC 
	fprintf(vector_file, "%04X ",(neuron->pn_generator & 0x07fff));	// Print 4 hexadecimal digits of PN_GENERATOR 
	fprintf(vector_file, "%04X ",(neuron->leakage_timer & 0x0ffff));	// Print 4 hexadecimal digits of LEAKAGE_TIMER 
	j = (out_pulse)? 1: 0;
	fprintf(vector_file, "%01d\n",j);					// Print "1" for out_pulse=TRUE and "0" for out_pulse=FALSE
	//
	*/
	// BINARY mode
	for(i=0; i < neuron->numb_synapses; i++) {
		j = (neuron->f[i])? 1: 0;
		fprintf(vector_file, "%01d",j);				// Print "1" for TRUE input and "0" for FALSE input 
	}
	j = (enable)? 1: 0;
	fprintf(vector_file, "%01d",j);					// Print "1" for enable=TRUE and "0" for enable=FALSE
	//
	k = neuron->acc & 0x0fffff;						// Print 20-bits of ACC 
	for(i=0; i < 20; i++) {
		j = ((k & 0x80000) != 0)? 1: 0;
		fprintf(vector_file, "%01d",j);				// Print "1" for bit=TRUE and "0" for bit=FALSE
		k = k << 1;
	}
	//
	k = neuron->pn_generator & 0x07fff;				// Print 15-bits of PN_GENERATOR 
	for(i=0; i < 15; i++) {
		j = ((k & 0x4000) != 0)? 1: 0;
		fprintf(vector_file, "%01d",j);				// Print "1" for bit=TRUE and "0" for bit=FALSE
		k = k << 1;
	}
	//
	k = neuron->rand_gauss_var & 0x0ffff;			// Print 16-bits of RAND_GAUSS_VAR 
	for(i=0; i < 16; i++) {
		j = ((k & 0x8000) != 0)? 1: 0;
		fprintf(vector_file, "%01d",j);				// Print "1" for bit=TRUE and "0" for bit=FALSE
		k = k << 1;
	}
	//
	k = neuron->leakage_timer & 0x0ffff;			// Print 16-bits of LEAKAGE_TIMER
	for(i=0; i < 16; i++) {
		j = ((k & 0x8000) != 0)? 1: 0;
		fprintf(vector_file, "%01d",j);				// Print "1" for bit=TRUE and "0" for bit=FALSE
		k = k << 1;
	}
	//
	j = (out_pulse)? 1: 0;
	fprintf(vector_file, "%01d\n",j);				// Print "1" for out_pulse=TRUE and "0" for out_pulse=FALSE 

	out_pulse = ctn_cycle(neuron, enable);
	//
	return out_pulse;
}


void push_new_sample_with_shift_right(bool *shift_reg, int reg_length, bool new_samp)
{
	int i;

	for(i=reg_length-1; i>0; i--) {
		shift_reg[i] = shift_reg[i-1];
	}
	shift_reg[0] = new_samp;
}



//**********************************************************************************************************************
//********                                                                                                     *********
//********              Generalized Cross Correlation (GCC) Neural Network for Beam-Forming                    *********
//********                                                                                                     *********
//********     This is a neural network that performs cross correlations process directrly on the PDM inputs   *********
//********     from the two microphones. It uses 101 CTN neurons orgenized in total of 6 layers:               *********
//********     1st Layer contains 2 Neurons, performs DC offset extraction on Mic.1 and Mic2.                  *********
//********     2nd Layer contains 2 Neurons, performs BPF on Mic.1 and Mic2.                                   *********
//********     3rd Layer contains 3 Neurons, performs Amplification and BPF on Mic.1 and Mic2.                 *********
//********     4th Layer contains 62 Neurons and makes phase-detection on 31 different cross-delays            *********
//********     5th Layer contains 31 Neurons and makes Integrate & Compare operation on the 31 correlations    *********
//********     6th Layer contains one Neuron which calculates the 5-bit delay from the outputs of 5th layer    *********
//********                                                                                                     *********
//********     The delay output is taken from the 5 MSbits of the accumulator of the output neuron (layer 6)   *********
//********     and defines the delay between Mic.1 and Mic.2 signals in terms of 10.417 micro-seconds (or      *********
//********     samples at 96 KS/sec). A positive result defines a positive delay of Mic.2 signal after Mic.1   *********
//********     signal, and a negative delay defines a delay of Mic.1 signal after Mic.2 signal. The result     *********
//********     is in the range [-15,+15] and is plotted in the "right" channel of the PCM output file - which  *********
//********     is a raw (intel format) 16-bit stereo PCM at 1,536,000 samp/sec.                                *********
//********                                                                                                     *********
//**********************************************************************************************************************


int main(int argc, char* argv[])
{
	int i,j;				// Temporary integers
	char c;					// Temporary char
	char input_filename1[256];	// Filename. Mic.1 PDM bitsream binary file (MSbit comes first) 
	char input_filename2[256];	// Filename. Mic.2 PDM bitsream binary file (MSbit comes first) 
	char output_filename[256];	// Filename. Output file is a PCM file that demonstrates the resulted output
	FILE *input_file1;		// File handler. Mic.1 PDM bitsream binary file (MSbit comes first)
	FILE *input_file2;		// File handler. Mic.2 PDM bitsream binary file (MSbit comes first)
	FILE *output_file;		// File handler. Output file is a PCM file that demonstrates the resulted output
	FILE *vector_file1;		// Binary test vector file for Verilog code development
	FILE *vector_file2;		// Binary test vector file for Verilog code development
	int samp_freq;			// PCM Sample frequency [Hz]
	int clk_freq;			// PDM Clock frequency [Hz]
	long loop_count = 0;	// Simulation loop counter
	//
	short samp_out;			// 16-bit output sample
	int items_read;			// Number of items read from input file
	int samp_rate_ratio;	// Ratio between PDM sampling rate and PCM sampling rate
	int pdm_bit1;			// Current bit at PDM input
	int pdm_bit2;			// Current bit at PDM input
	char pdm_byte1;			// Current byte at PDM input (MSbit comes first)
	char pdm_byte2;			// Current byte at PDM input (MSbit comes first)
	int pdm_bit_count;		// Counts number of bits in byte
	int pdm_samp_count;		// Counts number of PDM samples
	int inp_samp_counter;	// Counter to identify "one second of time passed"
	int time;				// Time in units of Seconds
	//
	// Neurons output pulses for network inter-connection:
	bool pulse_type_0[2];	// type_0 (Mic1 & Mic2 DCs) Output pulse: true or false
	bool pulse_type_1[2];	// type_1 (Mic1 & Mic2 BPFs) Output pulse: true or false
	bool pulse_type_1A[2];	// type_1A (Mic1 & Mic2 Amplifier & BPFs) Output pulse: true or false
	bool pulse_type_2;		// type_2 (Mic1 VAD) Output pulse: true or false
	bool pulse_type_3[62];	// type_3 (Mic1/Mic2 BPF Phase-Detectors), produce CORP_0 to CORP_+-15 and
							// CORN_0 to CORN_+-15 outputs
	bool pulse_type_4[16];	// type_4 (Integrate & Compare the type_3 outputs), produce all SEL_n+1_n outputs
	bool pulse_type_5[8];	// type_5 (Integrate & Compare the type_3 outputs), produce all SEL_... outputs
	bool pulse_type_6[4];	// type_6 (Integrate & Compare the type_3 outputs), produce all SEL_... outputs
	bool pulse_type_7[2];	// type_7 (Integrate & Compare the type_3 outputs), produce all SEL_... outputs
	bool pulse_type_8;		// type_8 (Integrate & Compare the type_3 outputs), produce all SEL_... outputs
	//
	int inp_buff_pntr = 0;					// Input 4-bit delay-line buffer pointer
	bool pulse_dly_buff1[DLY_BUFF_LEN16];	// Contains 16 delay phases (@ 96 KSps steps) for Mic.1 signal
	bool pulse_dly_buff2[DLY_BUFF_LEN16];	// Contains 16 delay phases (@ 96 KSps steps) for Mic.2 signal
	int pulse_dly_buff_pntr = 0;			// Buff pointer for delay phases buffer
	int dly;				// Temporary index
	bool pulse_beam_dly;	// Network output pulse
	//
	// Temporary variables for the input of the 4th layer neuron (neuron "type_9")
	bool sel_1xxxx;			// Decision weight 15
	//
	bool sel_00xxx;			// Auxiliary
	bool sel_01xxx;			// Decision weight 8
	bool sel_10xxx;			// Auxiliary
	bool sel_11xxx;			// Decision weight 8
	//
	bool sel_000xx;			// Auxiliary
	bool sel_001xx;			// Decision weight 4
	bool sel_010xx;			// Auxiliary
	bool sel_011xx;			// Decision weight 4
	bool sel_100xx;			// Auxiliary
	bool sel_101xx;			// Decision weight 4
	bool sel_110xx;			// Auxiliary
	bool sel_111xx;			// Decision weight 4
	//
	bool sel_0000x;			// Auxiliary
	bool sel_0001x;			// Decision weight 2
	bool sel_0010x;			// Auxiliary
	bool sel_0011x;			// Decision weight 2
	bool sel_0100x;			// Auxiliary
	bool sel_0101x;			// Decision weight 2
	bool sel_0110x;			// Auxiliary
	bool sel_0111x;			// Decision weight 2
	bool sel_1000x;			// Auxiliary
	bool sel_1001x;			// Decision weight 2
	bool sel_1010x;			// Auxiliary
	bool sel_1011x;			// Decision weight 2
	bool sel_1100x;			// Auxiliary
	bool sel_1101x;			// Decision weight 2
	bool sel_1110x;			// Auxiliary
	bool sel_1111x;			// Decision weight 2
	//
	bool sel_00001;			// Decision weight 1
	bool sel_00011;			// Decision weight 1
	bool sel_00101;			// Decision weight 1
	bool sel_00111;			// Decision weight 1
	bool sel_01001;			// Decision weight 1
	bool sel_01011;			// Decision weight 1
	bool sel_01101;			// Decision weight 1
	bool sel_01111;			// Decision weight 1
	bool sel_10001;			// Decision weight 1
	bool sel_10011;			// Decision weight 1
	bool sel_10101;			// Decision weight 1
	bool sel_10111;			// Decision weight 1
	bool sel_11001;			// Decision weight 1
	bool sel_11011;			// Decision weight 1
	bool sel_11101;			// Decision weight 1
	bool sel_11111;			// Decision weight 1



	//***********************************************************************
	//********
	//********   1st Layer contains 2 Neurons and works as DC bias calculator.
	//********   Input: 2 PDM bit-streams from two Mics.
	//********   Pulse rate: 1,536 KHz
	//********   CTN Type 0: 2 neurons of single input DC bias calculator
	//********                    
	//***********************************************************************
	//
	CTN type_0[2];			// Two neurons, the first is for Mic.1 and the second is for Mic.2
	short W_type_0[2];		// Input weights array
	bool F_type_0[2];		// Input pulse array: true or false
	//
	//***********************************************************************
	//********
	//********   2nd Layer contains 2 Neurons and works as BPF.
	//********   Input: 2 PDM bit-streams from two Mics.
	//********   Pulse rate: 1,536 KHz
	//********   CTN Type 1: 2 neurons of 9-inputs Bandpass filter
	//********                    
	//***********************************************************************
	//
	// CTN Type 1 for Mic.1 as a Bandpass filter.
	// This neuron has 9 inputs, inputs 1-8 are driven by 8 consecutive PDM pulses in a sliding window manner.
	// Input 9 is fed by the output of neuron Type 0 for DC bias removal
	// The input pulse rate is 1,536 KHz (equals the PDM clock frequency)
	//
	CTN type_1[2];			// Two neurons, the first is for Mic.1 and the second is for Mic.2
	short W_type_1[2][9];	// Input weights array
	bool F_type_1[2][9];	// Input pulse array: true or false
	//
	//***********************************************************************************
	//********
	//********   3rd Layer contains 3 Neurons and works as an Amplifier & BPF and VAD.
	//********   Input: The output from Type 1.
	//********   Pulse rate: 1,536 KHz
	//********   CTN Type 1A: 2 neurons of single input Amplifier and Bandpass filter
	//********   CTN Type 2: Single inputs Voice Activity Detector (VAD)
	//********                    
	//***********************************************************************************
	//
	CTN type_1A[2];			// Two neurons, the first is for Mic.1 and the second is for Mic.2
	short W_type_1A[2];		// Single Input weight
	bool F_type_1A[2];		// Single Input pulse: true or false
	//
	//
	// CTN Type 2 for Mic.1 as a VAD - Voice Activity Detector.
	// This neuron has 1 input which is driven by Type 1 output 
	// The input pulse rate is 1,536 KHz (equals the PDM clock frequency)
	//
	CTN type_2;				// One neuron, VAD is actually a high-amplitude detector applied on Mic.1
	short W_type_2;			// Input weights array
	bool F_type_2;			// Input pulse array: true or false
	//
	//****************************************************************************
	//********
	//********   4rt Layer contains 62 Neurons.
	//********   Input: Two 16-phases delay buffers from 3rd layer type_1A outputs.
	//********   Pulse rate: 96 KHz
	//********   CTN Type 3: 62 neurons of 2-inputs Phase Detector
	//********                    
	//****************************************************************************
	//
	// CTN Type 3 for Layer 3 outputs as a Phase Detector.
	// This neuron has 2 inputs which are driven by type_1A outputs - one is with 0 delay and the other
	// with a certain delay of the other Mic.
	// The input pulse rate is 96 KHz (equals 1/16 of the PDM clock frequency)
	//
	CTN type_3[62];			// 62 neurons, each one has two inputs: one from type_1A[0] and the other from type_1A[1]
	short W_type_3[62][2];	// Input weights array
	bool F_type_3[62][2];	// Input pulse array: true or false
	//
	//
	//****************************************************************************
	//********
	//********   5th Layer contains 31 Neurons.
	//********   Input: Phase Detector outputs from 4th layer type_3 outputs.
	//********   Pulse rate: 96 KHz
	//********   CTN Type 4: 16 neurons of 4-inputs Integrate & Compare
	//********   CTN Type 5: 8 neurons of 8-inputs Integrate & Compare
	//********   CTN Type 6: 4 neurons of 16-inputs Integrate & Compare
	//********   CTN Type 7: 2 neurons of 32-inputs Integrate & Compare
	//********   CTN Type 8: 1 neuron of 64-inputs Integrate & Compare
	//********                    
	//****************************************************************************
	//
	CTN type_4[16];			// 16 neurons, each one has 4 inputs from four type_3 neighboring phase detectors
	short W_type_4[16][4];	// Input weights array
	bool F_type_4[16][4];	// Input pulse array: true or false
	//
	CTN type_5[8];			// 8 neurons, each one has 8 inputs from four type_3 neighboring phase detectors
	short W_type_5[8][8];	// Input weights array
	bool F_type_5[8][8];	// Input pulse array: true or false
	//
	CTN type_6[4];			// 4 neurons, each one has 16 inputs from four type_3 neighboring phase detectors
	short W_type_6[4][16];	// Input weights array
	bool F_type_6[4][16];	// Input pulse array: true or false
	//
	CTN type_7[2];			// 2 neurons, each one has 32 inputs from four type_3 neighboring phase detectors
	short W_type_7[2][32];	// Input weights array
	bool F_type_7[2][32];	// Input pulse array: true or false
	//
	CTN type_8;				// One neuron of 64 inputs from four type_3 neighboring phase detectors
	short W_type_8[64];		// Input weights array
	bool F_type_8[64];		// Input pulse array: true or false
	//
	//
	//******************************************************************************
	//********
	//********   6th Layer contains one Neuron.
	//********   Input: Integrate & Compare outputs from 5th layer type_4-8 outputs.
	//********   Pulse rate: 96 KHz
	//********   CTN Type 9: 1 neuron of 31-inputs Delay Calculator
	//********                    
	//******************************************************************************
	//
	CTN type_9;				// One neuron of 31 inputs from combinatorial logic on type_3-8 neurons
	short W_type_9[31];		// Input weights array
	bool F_type_9[31];		// Input pulse array: true or false



	//**********************************************************
	//********
	//********   Neural Network Configuration and Initialization
	//********                    
	//**********************************************************
	//
	pdm_byte1 = 0;
	pdm_byte2 = 0;
	pdm_bit1 = 1;
	pdm_bit2 = 1;
	pdm_bit_count = 0;
	pdm_samp_count = 0;
	inp_samp_counter = 0;
	pdm_bit_count = 0;
	time=0;
	//
	// CTN Type 0 for Mic.1 as a DC extractor.
	for(i=0; i<2; i++) {
		W_type_0[i] = 8;
		F_type_0[i] = false;
		type_0[i].activation_type = IDENTITY;
		type_0[i].numb_synapses = 1;
		type_0[i].leakage_factor = 6;
		type_0[i].leakage_period = 127;
		type_0[i].leakage_timer = 0;
		type_0[i].w = &W_type_0[i];
		type_0[i].acc = 0;
		type_0[i].teta = -4;
		type_0[i].f = &F_type_0[i];
		type_0[i].pn_generator = 0x0001;
		type_0[i].rand_gauss_var = 0;
	}
	//
	// CTN Type 1 for Mic.1 as a Bandpass filter.
	for(i=0; i<2; i++) {
		for(j=0; j<8; j++) {
			W_type_1[i][j] = 127;	// 128,128,,,128, -128
			F_type_1[i][j] = (j%2 == 0);
		}
//		W_type_1[i][0] = 127;	// DC input weight is -128
		W_type_1[i][8] = -128;	// DC input weight is -128
		//
		type_1[i].activation_type = IDENTITY;
		type_1[i].numb_synapses = 9;
		type_1[i].leakage_factor = 7;
		type_1[i].leakage_period = 2;
		type_1[i].leakage_timer = 0;
		type_1[i].w = &W_type_1[i][0];
		type_1[i].acc = 0;
		type_1[i].teta = -444;			// Average(inputs 0-7)=4*128=512.  Average(input 8)= -64.  teta = -Average(all inputs) = -(512-64) = -448.
		type_1[i].f = &F_type_1[i][0];
		type_1[i].pn_generator = 0x0001;
		type_1[i].rand_gauss_var = 0;
	}
	//
	// CTN Type 1A for Mic.1 and Mic2.
	for(i=0; i<2; i++) {
		W_type_1A[i] = 255;	// 255
		F_type_1A[i] = false;
		type_1A[i].activation_type = BINARY_STEP;
		type_1A[i].numb_synapses = 1;
		type_1A[i].leakage_factor = 7;
		type_1A[i].leakage_period = 2;
		type_1A[i].leakage_timer = 0;
		type_1A[i].w = &W_type_1A[i];
		type_1A[i].acc = 0;
		type_1A[i].teta = -128;	// -127
		type_1A[i].f = &F_type_1A[i];
		type_1A[i].pn_generator = 0x0001;
		type_1A[i].rand_gauss_var = 0;
	}
//	type_1A[0].teta = -133;				// Additional amplified DC compensation (specific Mic calibration)
//	type_1A[1].teta = -135;				// Additional amplified DC compensation (specific Mic calibration)
	//
	// CTN Type 2 for Mic.1 as VAD.
	W_type_2 = 255;		// 255
	F_type_2 = false;
	type_2.activation_type = BINARY_STEP;
	type_2.numb_synapses = 1;
	type_2.leakage_factor = 7;
	type_2.leakage_period = 2;
	type_2.leakage_timer = 0;
	type_2.w = &W_type_2;
	type_2.acc = 0;
	type_2.teta = -135;					// -138 Negative DC offset to eliminated VAD false-detection due to background noise
	type_2.f = &F_type_2;
	type_2.pn_generator = 0x0001;
	type_2.rand_gauss_var = 0;
	//
	// CTN Type 3 for Type_1 outputs as Phase Detectors.
	for(i=0; i<62; i++) {
		for(j=0; j<2; j++) {
			W_type_3[i][j] = 127 - j*254;	// 127,-127
			F_type_3[i][j] = (j%2 == 0);
		}
		type_3[i].activation_type = BINARY_STEP;
		type_3[i].numb_synapses = 2;
		type_3[i].leakage_factor = 3;
		type_3[i].leakage_period = 0;
		type_3[i].leakage_timer = 0;
		type_3[i].w = &W_type_3[i][0];
		type_3[i].acc = 0;
		type_3[i].teta = -1;
		type_3[i].f = &F_type_3[i][0];
		type_3[i].pn_generator = 0x0001;
		type_3[i].rand_gauss_var = 0;
	}
	//
	// Initialize delay lines of Layer 2 outputs:
	for(i=0; i<DLY_BUFF_LEN16; i++) {
		pulse_dly_buff1[i] = (i%2 == 0);
		pulse_dly_buff2[i] = (i%2 == 0);
	}
	//
	// CTN Type 4 for Type_3 outputs as Integrate & Compare:
	for(i=0; i<16; i++) {
		for(j=0; j<4; j++) {
			W_type_4[i][j] = 1 - 2*(j/2);	// 1,1,-1,-1
			F_type_4[i][j] = (j%2 == 0);
		}
		type_4[i].activation_type = BINARY_STEP;
		type_4[i].numb_synapses = 4;
		type_4[i].leakage_factor = 6;
		type_4[i].leakage_period = 32;
		type_4[i].leakage_timer = 0;
		type_4[i].w = &W_type_4[i][0];
		type_4[i].acc = 0;
		type_4[i].teta = 0;
		type_4[i].f = &F_type_4[i][0];
		type_4[i].pn_generator = 0x0001;
		type_4[i].rand_gauss_var = 0;
	}
	//
	// CTN Type 5 for Type_3 outputs as Integrate & Compare:
	for(i=0; i<8; i++) {
		for(j=0; j<8; j++) {
			W_type_5[i][j] = 1 - 2*(j/4);	// 1,1,1,1,-1,-1,-1,-1
			F_type_5[i][j] = (j%2 == 0);
		}
		type_5[i].activation_type = BINARY_STEP;
		type_5[i].numb_synapses = 8;
		type_5[i].leakage_factor = 6;
		type_5[i].leakage_period = 32;
		type_5[i].leakage_timer = 0;
		type_5[i].w = &W_type_5[i][0];
		type_5[i].acc = 0;
		type_5[i].teta = 0;
		type_5[i].f = &F_type_5[i][0];
		type_5[i].pn_generator = 0x0001;
		type_5[i].rand_gauss_var = 0;
	}
	//
	// CTN Type 6 for Type_3 outputs as Integrate & Compare:
	for(i=0; i<4; i++) {
		for(j=0; j<16; j++) {
			W_type_6[i][j] = 1 - 2*(j/8);	// 1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1
			F_type_6[i][j] = (j%2 == 0);
		}
		type_6[i].activation_type = BINARY_STEP;
		type_6[i].numb_synapses = 16;
		type_6[i].leakage_factor = 6;
		type_6[i].leakage_period = 32;
		type_6[i].leakage_timer = 0;
		type_6[i].w = &W_type_6[i][0];
		type_6[i].acc = 0;
		type_6[i].teta = 0;
		type_6[i].f = &F_type_6[i][0];
		type_6[i].pn_generator = 0x0001;
		type_6[i].rand_gauss_var = 0;
	}
	//
	// CTN Type 7 for Type_3 outputs as Integrate & Compare:
	for(i=0; i<2; i++) {
		for(j=0; j<32; j++) {
			W_type_7[i][j] = 1 - 2*(j/16);	// 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
			F_type_7[i][j] = (j%2 == 0);
		}
		type_7[i].activation_type = BINARY_STEP;
		type_7[i].numb_synapses = 32;
		type_7[i].leakage_factor = 6;
		type_7[i].leakage_period = 32;
		type_7[i].leakage_timer = 0;
		type_7[i].w = &W_type_7[i][0];
		type_7[i].acc = 0;
		type_7[i].teta = 0;
		type_7[i].f = &F_type_7[i][0];
		type_7[i].pn_generator = 0x0001;
		type_7[i].rand_gauss_var = 0;
	}
	//
	// CTN Type 8 for Type_3 outputs as Integrate & Compare:
	for(j=0; j<64; j++) {
		W_type_8[j] = 1 - 2*(j/32);	// 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
		F_type_8[j] = (j%2 == 0);
	}
	type_8.activation_type = BINARY_STEP;
	type_8.numb_synapses = 64;
	type_8.leakage_factor = 6;
	type_8.leakage_period = 32;
	type_8.leakage_timer = 0;
	type_8.w = W_type_8;
	type_8.acc = 0;
	type_8.teta = 0;
	type_8.f = F_type_8;
	type_8.pn_generator = 0x0001;
	type_8.rand_gauss_var = 0;
	//
	// CTN Type 9 for Type_3-8 outputs as Delay Calculator:
	for(j=0; j<31; j++) {
		W_type_9[j] = 1 + ((30-j)/16) + 2*((30-j)/24) + 4*((30-j)/28) + 7*((30-j)/30);	// 15,8,8,4,4,4,4,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
		F_type_9[j] = false;
	}
	type_9.activation_type = BINARY_STEP;
	type_9.numb_synapses = 31;
	type_9.leakage_factor = 5;
	type_9.leakage_period = 128;
	type_9.leakage_timer = 0;
	type_9.w = W_type_9;
	type_9.acc = 0;
	type_9.teta = 0;
	type_9.f = F_type_9;
	type_9.pn_generator = 0x0001;
	type_9.rand_gauss_var = 0;


	// Define binary PDM input files for READ operation:
	//
//  // Mic.1 input file:
//	strcpy(input_filename1, "input_file1.pdm");	// Mic.1: file_1, Mic.2: file_3
//	strcpy(input_filename1, "input_file5.pdm");	// Mic.1: file_5, Mic.2: file_6 or 7
//	strcpy(input_filename1, "input_file8.pdm");	// Mic.1: file_8, Mic.2: file_10,11,12, or 13
//	strcpy(input_filename1, "pdm3.pdm");	// Mic.1: pdm3, Mic.2: pdm4
//	strcpy(input_filename1, "pdm3a.pdm");	// Mic.1: pdm3, Mic.2: pdm4
	strcpy(input_filename1, "pdm3b.pdm");	// Mic.1: pdm3, Mic.2: pdm4
	//
	//
	// Mic.2 input file:
//	strcpy(input_filename2, "input_file3.pdm");	// Mic.1: file_1, Mic.2: file_3
//	strcpy(input_filename2, "input_file6.pdm");	// Mic.1: file_5, Mic.2: file_6 or 7
//	strcpy(input_filename2, "input_file7.pdm");	// Mic.1: file_5, Mic.2: file_6 or 7
//	strcpy(input_filename2, "input_file13.pdm"); // Mic.1: file_8, Mic.2: file_10,11,12, or 13
//	strcpy(input_filename2, "pdm4.pdm");	// Mic.1: pdm3, Mic.2: pdm4
//	strcpy(input_filename2, "pdm2a.pdm");	// Mic.1: pdm3, Mic.2: pdm4
	strcpy(input_filename2, "pdm2b.pdm");	// Mic.1: pdm3, Mic.2: pdm4
	//
	//
	clk_freq = 1536000;		// Input PDM Clock frequency [Hz]
	strcpy(output_filename, "output_file.pcm");
	samp_freq = 1536000;	// Output PCM Sample frequency [Hz]
	//
	// Define binary output PCM file for WRITE operation:
	strcpy(output_filename, "output_file.pcm");

	//
	samp_rate_ratio = (int)(((double)clk_freq / (double)samp_freq) + 0.5);
	//

	// Open binary PDM input files for READ operation:
	input_file1 = fopen(input_filename1, "rb");
	input_file2 = fopen(input_filename2, "rb");
	//
	// Open binary output PCM file for WRITE operation:
	output_file = fopen(output_filename, "wb");


	//**********************************************************
	//********
	//********   Vector files Initialization:
	//********                    
	//**********************************************************

	vector_file1 = fopen("test_vector_neuron_type1_0.txt", "wt");
	for(j=0; j<type_1[0].numb_synapses; j++) {
		fprintf(vector_file1, "w[%d] = %d\n", j, W_type_1[0][j]);
	}
	fprintf(vector_file1, "teta = %d\n", type_1[0].teta);
	fprintf(vector_file1, "activation_type = %d\n", type_1[0].activation_type);
	fprintf(vector_file1, "numb_synapses = %d\n", type_1[0].numb_synapses);
	fprintf(vector_file1, "leakage_factor = %d\n", type_1[0].leakage_factor);
	fprintf(vector_file1, "leakage_period = %d\n\n", type_1[0].leakage_period);
	fprintf(vector_file1, "inputs_0-%d en acc[19:0] pn_gen[14:0] rand_gauss_var[15:0] leakage_timer[15:0] out\n\n", type_1[0].numb_synapses-1);
	//
	vector_file2 = fopen("test_vector_neuron_type9.txt", "wt");
	for(j=0; j<type_9.numb_synapses; j++) {
		fprintf(vector_file2, "w[%d] = %d\n", j, W_type_9[j]);
	}
	fprintf(vector_file2, "teta = %d\n", type_9.teta);
	fprintf(vector_file2, "activation_type = %d\n", type_9.activation_type);
	fprintf(vector_file2, "numb_synapses = %d\n", type_9.numb_synapses);
	fprintf(vector_file2, "leakage_factor = %d\n", type_9.leakage_factor);
	fprintf(vector_file2, "leakage_period = %d\n\n", type_9.leakage_period);
	fprintf(vector_file2, "inputs_0-%d en acc[19:0] pn_gen[14:0] rand_gauss_var[15:0] leakage_timer[15:0] out\n\n", type_9.numb_synapses-1);
	//


	//**********************************************************
	//********
	//********   Main loop runs in the PDM bit rate 1,536 KHz
	//********                    
	//**********************************************************


	// Loop initialization. Read the first 8-bits for each Mic.
	items_read = fread(&pdm_byte1, 1, 1, input_file1);
	items_read = fread(&pdm_byte2, 1, 1, input_file2);


	// Main loop runs in the PDM bit rate 1,536 KHz
	while((items_read > 0) && (loop_count < MAX_LOOPS)){
		//

		// Handle the PDM samples read operation and the PCM samples write operation:
		pdm_samp_count++;
		pdm_bit1 = (pdm_byte1 >= 0)? -1 : 1;	// Check MSbit location (sign bit)
		pdm_byte1 = pdm_byte1 << 1;				// Prepare next bit in the MSbit location
		pdm_bit2 = (pdm_byte2 >= 0)? -1 : 1;	// Check MSbit location (sign bit)
		pdm_byte2 = pdm_byte2 << 1;				// Prepare next bit in the MSbit location
		pdm_bit_count++;
		if(pdm_bit_count == 8) {				// Need to read the next byte from the two input files				
			pdm_bit_count = 0;
			items_read = fread(&pdm_byte1, 1, 1, input_file1);
			items_read = fread(&pdm_byte2, 1, 1, input_file2);
		}
		//
		// Handle the PCM output file:
		if((pdm_samp_count == samp_rate_ratio) && (items_read > 0)) {	// Here the PCM sample rate is also 1,536 KHz
			pdm_samp_count = 0;
			//
			// Define the "Left" channel of the PCM output file as a reference to the Mic.1 signal:
			samp_out = (short)(type_1A[0].acc/16);
//			samp_out = (short)(8000*pulse_type_0[0]);
//			samp_out = (short)(8000*pulse_type_1A[1]);
//			samp_out = (short)(8000*pulse_type_3[0]);
//			samp_out = (short)(8000*pulse_type_8);
//			samp_out = (short)(type_1[0].acc/16);
//			samp_out = (short)(8000*pulse_type_1[0]);
//			samp_out = (short)(type_2.acc/16);
			fwrite(&samp_out, 2, 1, output_file);
			//
			inp_samp_counter++;
			// Update the time for display/log message:
			if(inp_samp_counter >= samp_freq) {
				inp_samp_counter = 0;
				time++;
				printf("time[sec]: %d\n", time);	// Display simulation status
			}

			// Neuron network debug/output:
			// Define the "Right" channel of the PCM output file::
			samp_out = (short)(type_9.acc/32);		// Delay output result
//			samp_out = (short)(type_1A[1].acc/16);
//			samp_out = (short)(type_1[1].acc/16);		// Delay output result
//			samp_out = (short)(8000*pulse_type_2);
//			samp_out = (short)(8000*pulse_type_1[1]);
//			samp_out = (short)(8000*pulse_type_0[0]);
//			samp_out = (short)(8000*pulse_type_1A[0]);
//			samp_out = (short)(8000*pulse_type_3[1]);
//			samp_out = (short)(type_0[0].acc);
//			samp_out = (short)(type_2.acc/16);
			fwrite(&samp_out, 2, 1, output_file);
		}

		// Handle the firing neuron network:
		// First layer: Two neurons get the PDM input and produce DC offset:
		F_type_0[0] = (bool)(pdm_bit1 == 1);	// 1: True, 0: False
		F_type_0[1] = (bool)(pdm_bit2 == 1);	// 1: True, 0: False
		//
		// 2nd layer: Two neurons get 8 consecutive PDM inputs and make BPF:
		push_new_sample_with_shift_right(F_type_1[0], 8, (bool)(pdm_bit1 == 1));
		F_type_1[0][8] = pulse_type_0[0];	// DC compensation
		push_new_sample_with_shift_right(F_type_1[1], 8, (bool)(pdm_bit2 == 1));
		F_type_1[1][8] = pulse_type_0[1];	// DC compensation
		//
		// 3rd layer: Two neurons get inputs from Type 1 and make aplification and more BPF:
		F_type_1A[0] = pulse_type_1[0];
		F_type_1A[1] = pulse_type_1[1];
		// Also one neuron calculates VAD:
		F_type_2 = pulse_type_1[0];
		//
		// Run layers 1-3 with 1.536MHz pulses:
//		pulse_type_1[0] = ctn_cycle_with_vector_file(&type_1[0], true, vector_file1);	// BPF on Mic.1. Also produce vector file
		pulse_type_0[0] = ctn_cycle(&type_0[0], true);	// DC on Mic.1.
		pulse_type_0[1] = ctn_cycle(&type_0[1], true);	// DC on Mic.2
		pulse_type_1[0] = ctn_cycle(&type_1[0], true);	// BPF on Mic.1. Also produce vector file
		pulse_type_1[1] = ctn_cycle(&type_1[1], true);	// BPF on Mic.2
		pulse_type_1A[0] = ctn_cycle(&type_1A[0], true);	// Amp.BPF on Mic.1. Also produce vector file
		pulse_type_1A[1] = ctn_cycle(&type_1A[1], true);	// Amp.BPF on Mic.2
		pulse_type_2 = ctn_cycle(&type_2, true);		// VAD on Mic.1
		//
		inp_buff_pntr = (inp_buff_pntr+1)%16;


		// 4th layer: 62 neurons get the BPFs outputs from delay lines and make Phase Detection outputs on 31 phases.
		// This layer and all coming layers use pulse rate of 96 KHz (layer 1 output is down-sampled by a factor of 16):
		if(inp_buff_pntr == 0) {				// Once every 16 PDM bits (at 96 KHz)
			//
			// Build two 16-phase delay-lines: One for Mic.1 BPF signal and the other for Mic.2 BPF signal:
			pulse_dly_buff1[pulse_dly_buff_pntr] = pulse_type_1A[0];	// Delayed Mic.1 BPF signal, 96 KHz pulse rate
			pulse_dly_buff2[pulse_dly_buff_pntr] = pulse_type_1A[1];	// Delayed Mic.2 BPF signal, 96 KHz pulse rate
			//
			// Build a phase-detection array for 16 delay phases of Mic.1 versus Mic.2:
			for(dly=0; dly<DLY_BUFF_LEN16; dly++) {
				//
				F_type_3[2*dly][0] = pulse_dly_buff1[(DLY_BUFF_LEN16 + pulse_dly_buff_pntr - dly)%DLY_BUFF_LEN16];
				F_type_3[2*dly][1] = pulse_type_1A[1];
				pulse_type_3[2*dly] = ctn_cycle(&type_3[2*dly], true);		// Produce CORP_0,CORP_+1,,,CORP_+15 outputs
				F_type_3[2*dly+1][0] = F_type_3[2*dly][1];
				F_type_3[2*dly+1][1] = F_type_3[2*dly][0];
				pulse_type_3[2*dly+1] = ctn_cycle(&type_3[2*dly+1], true);	// Produce CORN_0,CORN_+1,,,CORN_+15 outputs
			}
			//
			// Build a phase-detection array for 15 delay phases of Mic.2 versus Mic.1 (phase 0 delay always done in neuron type_3[0]):
			for(dly=1; dly<DLY_BUFF_LEN16; dly++) {
				//
				F_type_3[2*dly+30][0] = pulse_dly_buff2[(DLY_BUFF_LEN16 + pulse_dly_buff_pntr - dly)%DLY_BUFF_LEN16];
				F_type_3[2*dly+30][1] = pulse_type_1A[0];
				pulse_type_3[2*dly+30] = ctn_cycle(&type_3[2*dly+30], true);		// Produce CORP_-1,,,CORP_-15 outputs
				F_type_3[2*dly+31][0] = F_type_3[2*dly+30][1];
				F_type_3[2*dly+31][1] = F_type_3[2*dly+30][0];
				pulse_type_3[2*dly+31] = ctn_cycle(&type_3[2*dly+31], true);		// Produce CORN_-1,,,CORN_-15 outputs
			}
			//
			// 5th layer: 32 neurons get the 31 Phase Detection outputs from layer 2 and make Integrate & Compare operations at 96 KHz.
			// Integrate and Compare the 16 delay phases of Mic.1 versus Mic.2:
			for(dly=0; dly<DLY_BUFF_LEN16; dly++) {							// Make the inputs interconnections first:
				//
				F_type_8[2*dly] = pulse_type_3[2*dly];						// CORP inputs to produce the -0:15 part of SEL_+0:15_-0:15
				F_type_8[2*dly+1] = pulse_type_3[2*dly+1];					// CORN inputs to produce the -0:15 part of SEL_+0:15_-0:15
				//
				F_type_7[0][2*dly] = pulse_type_3[2*dly];					// CORP inputs to produce SEL_+8:17_+0:7
				F_type_7[0][2*dly+1] = pulse_type_3[2*dly+1];				// CORN inputs to produce SEL_+8:17_+0:7
				//
				F_type_6[dly/8][(2*dly)%16] = pulse_type_3[2*dly];			// CORP inputs to produce SEL_+4:7_+0:3 and +12:5_+8:11
				F_type_6[dly/8][1+(2*dly)%16] = pulse_type_3[2*dly+1];		// CORN inputs to produce SEL_+4:7_+0:3 and +12:5_+8:11
				//
				F_type_5[dly/4][(2*dly)%8] = pulse_type_3[2*dly];			// CORP inputs to produce SEL_+2:3_+0:1,,,+14:15_+12:13
				F_type_5[dly/4][1+(2*dly)%8] = pulse_type_3[2*dly+1];		// CORN inputs to produce SEL_+2:3_+0:1,,,+14:15_+12:13
				//
				F_type_4[dly/2][(2*dly)%4] = pulse_type_3[2*dly];			// CORP inputs to produce SEL_+1_0,,,SEL_+15_+14
				F_type_4[dly/2][1+(2*dly)%4] = pulse_type_3[2*dly+1];		// CORN inputs to produce SEL_+1_0,,,SEL_+15_+14
			}
			//
			// Integrate and Compare the 16 delay phases of Mic.2 versus Mic.1:
			for(dly=0; dly<DLY_BUFF_LEN16; dly++) {							// Continue the inputs interconnections:
				//
				F_type_8[2*dly+32] = pulse_type_3[2*dly+30];				// CORP inputs to produce the +0:15 part of SEL_+0:15_-0:15
				F_type_8[2*dly+33] = pulse_type_3[2*dly+31];				// CORN inputs to produce the +0:15 part of SEL_+0:15_-0:15
				//
				F_type_7[1][2*dly] = pulse_type_3[2*dly+30];				// CORP inputs to produce SEL_-8:17_-0:7
				F_type_7[1][2*dly+1] = pulse_type_3[2*dly+31];				// CORN inputs to produce SEL_-8:17_-0:7
				//
				F_type_6[dly/8+2][(2*dly)%16] = pulse_type_3[2*dly+30];		// CORP inputs to produce SEL_-4:7_-0:3 and -12:5_-8:11
				F_type_6[dly/8+2][1+(2*dly)%16] = pulse_type_3[2*dly+31];	// CORN inputs to produce SEL_-4:7_-0:3 and -12:5_-8:11
				//
				F_type_5[dly/4+4][(2*dly)%8] = pulse_type_3[2*dly+30];		// CORP inputs to produce SEL_-2:3_-0:1,,,-14:15_-12:13
				F_type_5[dly/4+4][1+(2*dly)%8] = pulse_type_3[2*dly+31];	// CORN inputs to produce SEL_-2:3_-0:1,,,-14:15_-12:13
				//
				F_type_4[dly/2+8][(2*dly)%4] = pulse_type_3[2*dly+30];		// CORP inputs to produce SEL_-1_0,,,SEL_-15_-14
				F_type_4[dly/2+8][1+(2*dly)%4] = pulse_type_3[2*dly+31];	// CORN inputs to produce SEL_-1_0,,,SEL_-15_-14
			}
			// Correct (override) the phase detection sources for 0 delay:
			F_type_8[32] = pulse_type_3[0];									// Override CORP input[32] to produce the -0 part of SEL_+0:15_-0:15
			F_type_8[33] = pulse_type_3[1];									// Override CORN input[33] to produce the -0 part of SEL_+0:15_-0:15
			//
			F_type_7[1][0] = pulse_type_3[0];								// Override input[0] of neuron 1 to produce SEL_-8:17_-0:7
			F_type_7[1][1] = pulse_type_3[1];								// Override input[1] of neuron 1 to produce SEL_-8:17_-0:7
			//
			F_type_6[2][0] = pulse_type_3[0];								// Override input[0] of neuron 2 to produce SEL_-4:7_-0:3 and -12:5_-8:11
			F_type_6[2][1] = pulse_type_3[1];								// Override input[1] of neuron 2 to produce SEL_-4:7_-0:3 and -12:5_-8:11
			//
			F_type_5[4][0] = pulse_type_3[0];								// Override input[0] of neuron 4 to produce SEL_-2:3_-0:1,,,-14:15_-12:13
			F_type_5[4][1] = pulse_type_3[1];								// Override input[1] of neuron 4 to produce SEL_-2:3_-0:1,,,-14:15_-12:13
			//
			F_type_4[8][0] = pulse_type_3[0];								// Override input[0] of neuron 8 to produce SEL_-1_0
			F_type_4[8][1] = pulse_type_3[1];								// Override input[1] of neuron 8 to produce SEL_-1_0
			//

			// Now run all neurons of layer 3 for a single pulse:
			pulse_type_8 = ctn_cycle(&type_8, true);
			//
			for(i=0; i<2; i++) {
				pulse_type_7[i] = ctn_cycle(&type_7[i], true);
			}
			//
			for(i=0; i<4; i++) {
				pulse_type_6[i] = ctn_cycle(&type_6[i], true);
			}
			//
			for(i=0; i<8; i++) {
				pulse_type_5[i] = ctn_cycle(&type_5[i], true);
			}
			//
			for(i=0; i<16; i++) {
				pulse_type_4[i] = ctn_cycle(&type_4[i], true);
			}
			//

			// 6th layer: One neuron gets 31 inputs composed from the 32 neurons of layer 5.
			// A combinatorial logic array produce these 31 signals out of the 32 signals of layer 5.
			// The 4th layer runs at pulse rate of 96 KHz.
			sel_1xxxx = !pulse_type_8;							// Decision weight 15
			//
			sel_00xxx = !sel_1xxxx & pulse_type_7[1];			// Auxiliary
			sel_01xxx = !sel_1xxxx & !pulse_type_7[1];			// Decision weight 8
			sel_10xxx = sel_1xxxx & !pulse_type_7[0];			// Auxiliary
			sel_11xxx = sel_1xxxx & pulse_type_7[0];			// Decision weight 8
			//
			sel_000xx = sel_00xxx & pulse_type_6[3];			// Auxiliary
			sel_001xx = sel_00xxx & !pulse_type_6[3];			// Decision weight 4
			sel_010xx = sel_01xxx & pulse_type_6[2];			// Auxiliary
			sel_011xx = sel_01xxx & !pulse_type_6[2];			// Decision weight 4
			sel_100xx = sel_10xxx & !pulse_type_6[0];			// Auxiliary
			sel_101xx = sel_10xxx & pulse_type_6[0];			// Decision weight 4
			sel_110xx = sel_11xxx & !pulse_type_6[1];			// Auxiliary
			sel_111xx = sel_11xxx & pulse_type_6[1];			// Decision weight 4
			//
			sel_0000x = sel_000xx & pulse_type_5[7];			// Auxiliary
			sel_0001x = sel_000xx & !pulse_type_5[7];			// Decision weight 2
			sel_0010x = sel_001xx & pulse_type_5[6];			// Auxiliary
			sel_0011x = sel_001xx & !pulse_type_5[6];			// Decision weight 2
			sel_0100x = sel_010xx & pulse_type_5[5];			// Auxiliary
			sel_0101x = sel_010xx & !pulse_type_5[5];			// Decision weight 2
			sel_0110x = sel_011xx & pulse_type_5[4];			// Auxiliary
			sel_0111x = sel_011xx & !pulse_type_5[4];			// Decision weight 2
			sel_1000x = sel_100xx & !pulse_type_5[0];			// Auxiliary
			sel_1001x = sel_100xx & pulse_type_5[0];			// Decision weight 2
			sel_1010x = sel_101xx & !pulse_type_5[1];			// Auxiliary
			sel_1011x = sel_101xx & pulse_type_5[1];			// Decision weight 2
			sel_1100x = sel_110xx & !pulse_type_5[2];			// Auxiliary
			sel_1101x = sel_110xx & pulse_type_5[2];			// Decision weight 2
			sel_1110x = sel_111xx & !pulse_type_5[3];			// Auxiliary
			sel_1111x = sel_111xx & pulse_type_5[3];			// Decision weight 2
			//
			sel_00001 = sel_0000x & !pulse_type_4[15];			// Decision weight 1
			sel_00011 = sel_0001x & !pulse_type_4[14];			// Decision weight 1
			sel_00101 = sel_0010x & !pulse_type_4[13];			// Decision weight 1
			sel_00111 = sel_0011x & !pulse_type_4[12];			// Decision weight 1
			sel_01001 = sel_0100x & !pulse_type_4[11];			// Decision weight 1
			sel_01011 = sel_0101x & !pulse_type_4[10];			// Decision weight 1
			sel_01101 = sel_0110x & !pulse_type_4[9];			// Decision weight 1
			sel_01111 = sel_0111x & !pulse_type_4[8];			// Decision weight 1
			sel_10001 = sel_1000x & pulse_type_4[0];			// Decision weight 1
			sel_10011 = sel_1001x & pulse_type_4[1];			// Decision weight 1
			sel_10101 = sel_1010x & pulse_type_4[2];			// Decision weight 1
			sel_10111 = sel_1011x & pulse_type_4[3];			// Decision weight 1
			sel_11001 = sel_1100x & pulse_type_4[4];			// Decision weight 1
			sel_11011 = sel_1101x & pulse_type_4[5];			// Decision weight 1
			sel_11101 = sel_1110x & pulse_type_4[6];			// Decision weight 1
			sel_11111 = sel_1111x & pulse_type_4[7];			// Decision weight 1
			//
			// Only the following signals are used for the type_9 neuron's inputs:
			F_type_9[0] = sel_1xxxx;			// Decision weight 15
			F_type_9[1] = sel_01xxx;			// Decision weight 8
			F_type_9[2] = sel_11xxx;			// Decision weight 8
			F_type_9[3] = sel_001xx;			// Decision weight 4
			F_type_9[4] = sel_011xx;			// Decision weight 4
			F_type_9[5] = sel_101xx;			// Decision weight 4
			F_type_9[6] = sel_111xx;			// Decision weight 4
			F_type_9[7] = sel_0001x;			// Decision weight 2
			F_type_9[8] = sel_0011x;			// Decision weight 2
			F_type_9[9] = sel_0101x;			// Decision weight 2
			F_type_9[10] = sel_0111x;			// Decision weight 2
			F_type_9[11] = sel_1001x;			// Decision weight 2
			F_type_9[12] = sel_1011x;			// Decision weight 2
			F_type_9[13] = sel_1101x;			// Decision weight 2
			F_type_9[14] = sel_1111x;			// Decision weight 2
			F_type_9[15] = sel_00001;			// Decision weight 1
			F_type_9[16] = sel_00011;			// Decision weight 1
			F_type_9[17] = sel_00101;			// Decision weight 1
			F_type_9[18] = sel_00111;			// Decision weight 1
			F_type_9[19] = sel_01001;			// Decision weight 1
			F_type_9[20] = sel_01011;			// Decision weight 1
			F_type_9[21] = sel_01101;			// Decision weight 1
			F_type_9[22] = sel_01111;			// Decision weight 1
			F_type_9[23] = sel_10001;			// Decision weight 1
			F_type_9[24] = sel_10011;			// Decision weight 1
			F_type_9[25] = sel_10101;			// Decision weight 1
			F_type_9[26] = sel_10111;			// Decision weight 1
			F_type_9[27] = sel_11001;			// Decision weight 1
			F_type_9[28] = sel_11011;			// Decision weight 1
			F_type_9[29] = sel_11101;			// Decision weight 1
			F_type_9[30] = sel_11111;			// Decision weight 1
			//
			pulse_beam_dly = ctn_cycle(&type_9, pulse_type_2);
//			pulse_beam_dly = ctn_cycle_with_vector_file(&type_9, pulse_type_2, vector_file2);

			// Handle the delay-laines pointer (at 96 KHz)
			pulse_dly_buff_pntr = (pulse_dly_buff_pntr+1)%DLY_BUFF_LEN16;

		}	// ... Once every 16 PDM bits (at 96 KHz)

	loop_count++;	// Simulation loop counter

	}	// while()... Main loop runs in the PDM bit rate 1,536 KHz


	// Terminate the program and close all files:
	fclose(input_file1);
	fclose(input_file2);
	fclose(output_file);

	fclose(vector_file1);
	fclose(vector_file2);

	// Wait for user to exit the program:
	printf("\nPress any key...");
	c=0;
	while(c == 0) {
		if(kbhit()) {
			c = -1;
		}
	}

	return 0;
}

