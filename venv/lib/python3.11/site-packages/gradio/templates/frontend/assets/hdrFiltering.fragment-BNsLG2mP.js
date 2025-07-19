import{j as i}from"./index-GP0Wrvz2.js";import"./helperFunctions-DlAeMh6z.js";import"./hdrFilteringFunctions-CVG-nAAH.js";import"./index-BPZXbhyd.js";import"./svelte/svelte.js";const r="hdrFilteringPixelShader",e=`#include<helperFunctions>
#include<importanceSampling>
#include<pbrBRDFFunctions>
#include<hdrFilteringFunctions>
uniform float alphaG;uniform samplerCube inputTexture;uniform vec2 vFilteringInfo;uniform float hdrScale;varying vec3 direction;void main() {vec3 color=radiance(alphaG,inputTexture,direction,vFilteringInfo);gl_FragColor=vec4(color*hdrScale,1.0);}`;i.ShadersStore[r]||(i.ShadersStore[r]=e);const c={name:r,shader:e};export{c as hdrFilteringPixelShader};
//# sourceMappingURL=hdrFiltering.fragment-BNsLG2mP.js.map
