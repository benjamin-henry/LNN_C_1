#include "./headers/lnn_1d.h"
#include "./headers/lnn_params.h"
#include <stdio.h>

#define N_SAMPLES 48

float samples[N_SAMPLES][LNN_SENSORY_SIZE] = {
    { 0.0000000e+00F,  1.0000000e+00F },
    { 1.9918598e-01F,  9.7996169e-01F },
    { 3.9038926e-01F,  9.2064989e-01F },
    { 5.6594712e-01F,  8.2444155e-01F },
    { 7.1882367e-01F,  6.9519246e-01F },
    { 8.4289229e-01F,  5.3808236e-01F },
    { 9.3318063e-01F,  3.5940778e-01F },
    { 9.8607028e-01F,  1.6632935e-01F },
    { 9.9944156e-01F, -3.3414979e-02F },
    { 9.7275865e-01F, -2.3182015e-01F },
    { 9.0709090e-01F, -4.2093477e-01F },
    { 8.0507004e-01F, -5.9317976e-01F },
    { 6.7078471e-01F, -7.4165213e-01F },
    { 5.0961661e-01F, -8.6040157e-01F },
    { 3.2802486e-01F, -9.4466907e-01F },
    { 1.3328695e-01F, -9.9107748e-01F },
    {-6.6792637e-02F, -9.9776685e-01F },
    {-2.6419541e-01F, -9.6446919e-01F },
    {-4.5101011e-01F, -8.9251882e-01F },
    {-6.1974990e-01F, -7.8479940e-01F },
    {-7.6365221e-01F, -6.4562786e-01F },
    {-8.7694991e-01F, -4.8058176e-01F },
    {-9.5510250e-01F, -2.9627559e-01F },
    {-9.9497783e-01F, -1.0009569e-01F },
    {-9.9497783e-01F,  1.0009569e-01F },
    {-9.5510250e-01F,  2.9627559e-01F },
    {-8.7694991e-01F,  4.8058176e-01F },
    {-7.6365221e-01F,  6.4562786e-01F },
    {-6.1974990e-01F,  7.8479940e-01F },
    {-4.5101011e-01F,  8.9251882e-01F },
    {-2.6419541e-01F,  9.6446919e-01F },
    {-6.6792637e-02F,  9.9776685e-01F },
    { 1.3328695e-01F,  9.9107748e-01F },
    { 3.2802486e-01F,  9.4466907e-01F },
    { 5.0961661e-01F,  8.6040157e-01F },
    { 6.7078471e-01F,  7.4165213e-01F },
    { 8.0507004e-01F,  5.9317976e-01F },
    { 9.0709090e-01F,  4.2093477e-01F },
    { 9.7275865e-01F,  2.3182015e-01F },
    { 9.9944156e-01F,  3.3414979e-02F },
    { 9.8607028e-01F, -1.6632935e-01F },
    { 9.3318063e-01F, -3.5940778e-01F },
    { 8.4289229e-01F, -5.3808236e-01F },
    { 7.1882367e-01F, -6.9519246e-01F },
    { 5.6594712e-01F, -8.2444155e-01F },
    { 3.9038926e-01F, -9.2064989e-01F },
    { 1.9918598e-01F, -9.7996169e-01F },
    { 3.6739403e-16F, -1.0000000e+00F }
};

int main() {
    LNN lnn = init_lnn(LNN_SENSORY_SIZE, LNN_RESERVOIR_SIZE, LNN_OUTPUT_SIZE, LNN_ODE_UNFOLDS);   
    
    for(unsigned int sample=0; sample < N_SAMPLES; sample++) {
        lnn_forward(
            lnn,
            (float*)samples[sample],
            (float*)lnn_input_w, (float*)lnn_input_b,
            (float*)lnn_sensory_mu,(float*)lnn_sensory_sigma,(float*)lnn_sensory_w,(float*)lnn_sensory_sparsity_mask,(float*)lnn_sensory_erev,
            (float*)lnn_cm,
            (float*)lnn_mu,(float*)lnn_sigma,(float*)lnn_w,(float*)lnn_sparsity_mask,(float*)lnn_erev,
            (float*)lnn_gleak,(float*)lnn_vleak,
            (float*)lnn_output_w,(float*)lnn_output_b
        );
        
        for(unsigned int i = 0; i < LNN_OUTPUT_SIZE; i++) {
            printf("%f ", lnn->output_placeholder[i]);
        }
        printf("\n");
    }
    exit(EXIT_SUCCESS);
}
