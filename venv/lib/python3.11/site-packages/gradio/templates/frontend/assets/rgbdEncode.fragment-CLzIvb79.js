import{j as r}from"./index-GP0Wrvz2.js";import"./helperFunctions-Bj0DbmZ2.js";import"./index-BPZXbhyd.js";import"./svelte/svelte.js";const e="rgbdEncodePixelShader",t=`varying vUV: vec2f;var textureSamplerSampler: sampler;var textureSampler: texture_2d<f32>;
#include<helperFunctions>
#define CUSTOM_FRAGMENT_DEFINITIONS
@fragment
fn main(input: FragmentInputs)->FragmentOutputs {fragmentOutputs.color=toRGBD(textureSample(textureSampler,textureSamplerSampler,input.vUV).rgb);}`;r.ShadersStoreWGSL[e]||(r.ShadersStoreWGSL[e]=t);const p={name:e,shader:t};export{p as rgbdEncodePixelShaderWGSL};
//# sourceMappingURL=rgbdEncode.fragment-CLzIvb79.js.map
