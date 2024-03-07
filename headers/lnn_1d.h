#ifndef __MYSTRUCT_H__
#define __MYSTRUCT_H__

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lnn_t {
    unsigned int sensory_size;
    unsigned int reservoir_size;
    unsigned int output_size;
    unsigned int ode_unfolds;

    float **sensory_w_activation;
    float **sensory_rev_activation;
    float *cm_time;   
    float **w_activation;
    float **rev_activation;
    float *w_numerator_sensory;
    float *w_denominator_sensory;
    float *state;
    float *w_numerator;
    float *w_denominator;
    float *numerator;
    float *denominator;
    float *input_placeholder;
    float *output_placeholder;
} lnn_t;

typedef struct lnn_t* LNN;

float fSigmoid(float val) {
    return 1. / (1.+exp(-val));
}

float* fallocate_1d(unsigned int n_elems) {
    float* array = (float*)calloc(n_elems, sizeof(float));
    return array;
}

float** fallocate_2D(unsigned int rows, unsigned int cols) {
    float **array = (float**)malloc(rows*sizeof(float*));
    for(unsigned int i = 0; i < rows; i++) {
        array[i] = (float*)calloc(cols, sizeof(float));
    }
    return array;
}

void reset_state(LNN lnn) {
    memset(lnn->state, 0., sizeof(float)*lnn->reservoir_size);
}

void new_sample_reset(LNN lnn) {
    memset(lnn->w_numerator_sensory,0,sizeof(float)*(lnn->reservoir_size));
    memset(lnn->w_denominator_sensory,0,sizeof(float)*(lnn->reservoir_size));
    memset(lnn->w_numerator,0,sizeof(float)*(lnn->reservoir_size));
    memset(lnn->w_denominator,0,sizeof(float)*(lnn->reservoir_size));   
    memset(lnn->numerator,0,sizeof(float)*(lnn->reservoir_size));
    memset(lnn->denominator,0,sizeof(float)*(lnn->reservoir_size));
}

LNN init_lnn(unsigned int sensory_size, unsigned int reservoir_size, unsigned int output_size, unsigned int ode_unfolds) {
    LNN lnn = (LNN)malloc(sizeof(lnn_t));
    lnn->sensory_size = sensory_size;
    lnn->reservoir_size = reservoir_size;
    lnn->output_size = output_size;
    lnn->ode_unfolds = ode_unfolds;

    lnn->sensory_w_activation = fallocate_2D(sensory_size, reservoir_size);
    lnn->sensory_rev_activation = fallocate_2D(sensory_size, reservoir_size);

    lnn->cm_time = fallocate_1d(reservoir_size);

    lnn->w_activation = fallocate_2D(reservoir_size, reservoir_size);
    lnn->rev_activation = fallocate_2D(reservoir_size,reservoir_size);

    lnn->w_numerator_sensory = fallocate_1d(reservoir_size);
    lnn->w_denominator_sensory = fallocate_1d(reservoir_size);
    lnn->w_numerator = fallocate_1d(reservoir_size);
    lnn->w_denominator = fallocate_1d(reservoir_size);
    lnn->numerator = fallocate_1d(reservoir_size);
    lnn->denominator = fallocate_1d(reservoir_size);

    lnn->state = fallocate_1d(reservoir_size);

    lnn->input_placeholder = fallocate_1d(sensory_size);
    lnn->output_placeholder = fallocate_1d(output_size);

    return lnn;
}

void map_inputs(LNN lnn, float* weights, float* biases) {
    for(unsigned int i = 0; i < lnn->sensory_size; i++) {
        lnn->input_placeholder[i] *= weights[i];
        lnn->input_placeholder[i] += biases[i];
    }
}

void ode_solver(
    LNN lnn,
    float* lnn_sensory_mu_,
    float* lnn_sensory_sigma_,
    float* lnn_sensory_w_,
    float* lnn_sensory_sparsity_mask_,
    float* lnn_sensory_erev_,
    float* lnn_cm_,
    float* lnn_mu_,
    float* lnn_sigma_,
    float* lnn_w_,
    float* lnn_sparsity_mask_,
    float* lnn_erev_,
    float* lnn_gleak_,
    float* lnn_vleak_
) {
    for(unsigned int j = 0; j < lnn->sensory_size; j++) {
        for(unsigned int i = 0; i < lnn->reservoir_size; i++) {
            lnn->sensory_w_activation[j][i] = lnn->input_placeholder[j] - lnn_sensory_mu_[i+j*lnn->reservoir_size];
            lnn->sensory_w_activation[j][i] = lnn_sensory_sigma_[i+j*lnn->reservoir_size] * lnn->sensory_w_activation[j][i];
            lnn->sensory_w_activation[j][i] = fSigmoid(lnn->sensory_w_activation[j][i]);
            lnn->sensory_w_activation[j][i] *= lnn_sensory_w_[i+j*lnn->reservoir_size];
            lnn->sensory_w_activation[j][i] *= lnn_sensory_sparsity_mask_[i+j*lnn->reservoir_size];             
            lnn->sensory_rev_activation[j][i] = lnn->sensory_w_activation[j][i] * lnn_sensory_erev_[i+j*lnn->reservoir_size];
            lnn->w_numerator_sensory[i] += lnn->sensory_rev_activation[j][i];
            lnn->w_denominator_sensory[i] += lnn->sensory_w_activation[j][i];
        }
    }
    
    for(unsigned int i = 0; i < lnn->reservoir_size; i++) {
        lnn->cm_time[i] = lnn_cm_[i] * (float)(lnn->ode_unfolds);
    }

    for(unsigned t = 0; t < lnn->ode_unfolds; t++) {
        memset(lnn->w_numerator,0,sizeof(float)*lnn->reservoir_size);
        memset(lnn->w_denominator,0,sizeof(float)*lnn->reservoir_size); 
        for(unsigned int j = 0; j < lnn->reservoir_size; j++) {
            for(unsigned int i = 0; i < lnn->reservoir_size; i++) {
                lnn->w_activation[j][i] = lnn->state[j] - lnn_mu_[i+j*lnn->reservoir_size];
                lnn->w_activation[j][i] = lnn_sigma_[i+j*lnn->reservoir_size] * lnn->w_activation[j][i];
                lnn->w_activation[j][i] = fSigmoid(lnn->w_activation[j][i]);
                lnn->w_activation[j][i] *= lnn_w_[i+j*lnn->reservoir_size];
                lnn->w_activation[j][i] *= lnn_sparsity_mask_[i+j*lnn->reservoir_size];             
                lnn->rev_activation[j][i] = lnn->w_activation[j][i] * lnn_erev_[i+j*lnn->reservoir_size];
                lnn->w_numerator[i] += lnn->rev_activation[j][i];
                lnn->w_denominator[i] += lnn->w_activation[j][i];
            }
        }
        
        for(unsigned int j = 0; j < lnn->reservoir_size; j++) {
            lnn->w_numerator[j] += lnn->w_numerator_sensory[j];
            lnn->w_denominator[j] += lnn->w_denominator_sensory[j];
            lnn->numerator[j] = lnn->cm_time[j] * lnn->state[j] + lnn_gleak_[j] * lnn_vleak_[j] + lnn->w_numerator[j];
            lnn->denominator[j] = lnn->cm_time[j] + lnn_gleak_[j] + lnn->w_denominator[j];
            lnn->state[j] = lnn->numerator[j] / (lnn->denominator[j] + 1e-6F);             
        }
    }
}

void map_outputs(LNN lnn, float* weights, float* biases) {
    memcpy(lnn->output_placeholder, lnn->state, sizeof(float)*lnn->output_size);
    for(unsigned int i = 0; i < lnn->sensory_size; i++) {
        lnn->output_placeholder[i] *= weights[i];
        lnn->output_placeholder[i] += biases[i];
    }
}

void lnn_forward(
    LNN lnn,
    float* sample,
    float* lnn_input_w_,
    float* lnn_input_b_,
    float* lnn_sensory_mu_,
    float* lnn_sensory_sigma_,
    float* lnn_sensory_w_,
    float* lnn_sensory_sparsity_mask_,
    float* lnn_sensory_erev_,
    float* lnn_cm_,
    float* lnn_mu_,
    float* lnn_sigma_,
    float* lnn_w_,
    float* lnn_sparsity_mask_,
    float* lnn_erev_,
    float* lnn_gleak_,
    float* lnn_vleak_,
    float* lnn_output_w_,
    float* lnn_output_b_
) {
    new_sample_reset(lnn);
    memcpy(lnn->input_placeholder, sample, lnn->sensory_size*sizeof(float));
    map_inputs(lnn, lnn_input_w_, lnn_input_b_);
    ode_solver(
        lnn,
        lnn_sensory_mu_,
        lnn_sensory_sigma_,
        lnn_sensory_w_,
        lnn_sensory_sparsity_mask_,
        lnn_sensory_erev_,
        lnn_cm_,
        lnn_mu_,
        lnn_sigma_,
        lnn_w_,
        lnn_sparsity_mask_,
        lnn_erev_,
        lnn_gleak_,
        lnn_vleak_
    );
    map_outputs(lnn, lnn_output_w_, lnn_output_b_);
}


#ifdef __cplusplus
}
#endif

#endif