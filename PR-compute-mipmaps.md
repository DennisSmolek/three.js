# WebGPURenderer: Add compute mipmap generation

Related issue: none yet (gpuweb#386 and #5253 discuss mipmaps in WebGPU core; this is the userland answer for three.js).

**Description**

Adds an opt-in `mipmapGeneration` parameter to `WebGPURenderer`:

```js
const renderer = new THREE.WebGPURenderer( { mipmapGeneration: 'compute' } ); // default: 'render'
```

With `'compute'`, mipmaps are generated with compute shaders instead of a render pass per level and layer. One dispatch writes up to six levels (four at the default `maxStorageTexturesPerShaderStage`), so a full chain takes 2 dispatches at powers of two up to 4096², and 3–6 at typical render sizes (1920×1080: 4). Cube maps and array textures take the same number of dispatches as a 2D texture, since workgroup z selects the layer.

**Results match render-pass mipmaps.** Each dispatch takes its first level as one bilinear sample at each texel centre in normalized coordinates, as the render pass does, so odd sizes give the same result. Further levels of a dispatch are exact 2:1 reductions in workgroup memory, so a dispatch ends where a non-unit dimension is odd; once a level fits one workgroup, a tail kernel finishes the chain with the same rule. Every level is quantized as it is stored before the next one reads it, so a chain equals level-by-level generation. float16 levels are rounded to nearest even in the shader, because the device's own f32 → f16 conversion truncates on D3D12.

**What changes**

- `WebGPUTextureMipmapUtils` (new, `src/renderers/webgpu/utils/`): the WGSL generator of the mip-chain kernels, the batch planner and the per-device defaults, vendored from [pmndrs/downsampler](https://github.com/pmndrs/downsampler) at `da3e609`. The source is the library's single-file `dist/mipmap.js` (`createMipmapKernel`, `planMipmapBatches`, `kernelDefaults`, `mipmapFormats`, `mipmapAccess`; in-place `'average'` chains only), adapted to three's code style. The class around it adds the pipeline, bind group layout and per-texture bind group caches. No runtime dependency.
- `WebGPUTextureUtils`: with `'compute'`, textures whose mipmaps are generated and whose format compute can write get `STORAGE_BINDING` usage at creation (stripped again from their MSAA attachments). `_generateMipmaps()` uses compute when the texture qualifies and render passes otherwise. `WebGPUTexturePassUtils` is unchanged.
- `WebGPUBackend`: with `'compute'`, the device requests `maxStorageTexturesPerShaderStage: min( 6, adapter limit )` unless `requiredLimits` already sets it (and only if that is above the default of 4). A device passed in by the app is used as is.
- Docs for the new parameter on `WebGPURenderer` and `WebGPUBackend`.

**What stays on render passes, and why**

| Textures | Why |
|---|---|
| sRGB formats (`rgba8unorm-srgb`, `bgra8unorm-srgb`), so most color maps | WebGPU never allows sRGB storage textures, not even through a view. A linear-reduce + encode + copy path is being built in the library; it can follow. |
| Compressed textures | Mipmaps come with the data or are generated before compression. |
| Multisampled textures | Can't be storage textures. Their resolve textures (sample count 1) do use compute. |
| 3D textures, depth textures | Not supported by the kernel (3D needs a different footprint; depth can't be a storage texture). |
| `rgb9e5ufloat`, `rg11b10ufloat`, `rgb10a2unorm`, integer formats | Not storage formats (or not supported by the vendored kernel). The library has a buffer output path for the packed float formats; it isn't vendored. |
| Formats needing a missing feature | `r8unorm`, `rg8unorm`, `r16float`, `rg16float`, the 16-bit normalized and 8-bit snorm one- and two-channel formats need `texture-formats-tier1`; `bgra8unorm` needs `bgra8unorm-storage`. |
| Every texture in compatibility mode | Compatibility mode allows one view of a texture per dispatch, so an in-place chain (read level n, write n+1 …) fails validation there. The library has a scratch-texture path for it; it isn't vendored. |
| Textures with uploaded mipmaps, `generateMipmaps = false` | Nothing to generate. |

Supported formats: `rgba8unorm`, `rgba8snorm`, `rgba16float`, `rgba32float`, `r32float`, `rg32float`, plus the feature-gated ones above. float32 formats sample with `float32-filterable` and load otherwise; 16-bit normalized formats always load (they sample as unfilterable on Dawn).

**Cube maps.** three creates cube textures with `textureBindingViewDimension: 'cube'`. That attribute constrains bindings only on devices without `'core-features-and-limits'`; on core devices the compute path binds the cube as a 6-layer `2d-array` (sampled and storage views), which validates and was tested (`CubeRenderTarget` + `CubeCamera`, rgba8 and rgba16float). In compatibility mode a `'cube'` texture can't be viewed as `2d-array` and WGSL has no cube storage textures, so cube maps fall back with everything else there.

**Cost and why it is opt-in**

- Speed (from the library's benchmark, not re-timed in three: Apple M5 Pro, Chrome, GPU timestamps, medians of 11 rounds; library SPEC §8 and §2.1): render passes ÷ compute was 2.4× at 1920×1080 rgba8, 2.0× in rgba16float, 7.6× at 1024² rgba8, 1.3–1.5× at 4096² (bandwidth-bound), and 26.8× for a 512² six-layer (cube-like) texture. Absolute times at 1080p are tens of µs. Not measured on other vendors through three.
- `STORAGE_BINDING` on sampled textures may have a sampling or compression cost on some GPUs (raised in gpuweb#5253); it hasn't been measured yet. That, and the extra device limit, is why `'render'` stays the default.
- First use of each format / layer layout / level count compiles a compute pipeline (synchronously, as the render path does).

**Testing**

- `npm run lint`, `npm run test-unit` (1982 pass, 0 fail).
- New: `npm run test-webgpu-mipmaps` (`test/webgpu/mipmaps-compute.js`). Headless Chrome on the local GPU, two renderers (`'render'` and `'compute'`) fed the same level 0, every level and layer read back and compared. 24 cases × 3 device configurations (default limits, `maxStorageTexturesPerShaderStage: 4`, compatibility mode): render targets at 1920×1080, 960×540, 1129×635, 1477×831, 1024², 2048², 255×257, 31×31, 333×1, 1×300, 97×1000 (rgba8unorm and rgba16float), 4× MSAA, cube render targets (rgba8, rgba16float), data textures (rgba8, rgba16float, rgba32float, r8unorm, a 3-layer array), and sRGB textures (must fall back). It asserts the expected path (and that only compute textures get `STORAGE_BINDING`), no GPU validation errors, identical level 0, and:
  - **each compute level within one 8-bit step of a render pass from the same input level** (float formats: steps of their content's range);
  - **the whole chain within two steps** of the render-pass chain.

  Results on Apple M5 Pro (Metal), Chrome 155: 72 of 72 cases pass. Per level: at most 1 step for 8-bit formats, 0.125 (half-float rounding) to 0.934 (sampler weights at odd sizes) for rgba16float, 0 for rgba32float. Chains: at most 2 steps. At even sizes every 8-bit difference falls on an exact tie (a 2×2 sum of 2 mod 4). The kernel rounds ties to even, as the library's CPU oracle does. Metal render passes follow no fixed tie rule: a round-half-up kernel differed from them at 46 % of ties, round-to-even at 37 %. A one-step difference at one level can carry into the next.
- e2e (`test/e2e/puppeteer.js --webgpu`, 209 WebGPU examples, same machine): screenshots regenerated locally with the default render path, then compared against a build where `'compute'` was the default (a local patch, not part of this PR). 95 examples generated compute mipmaps (rgba8unorm and rgba16float; 66 six-layer cube textures, e.g. `webgpu_cubemap_dynamic`, `webgpu_cubemap_adjustments`, plus 2D render targets, equirect HDR and normal maps); sRGB textures fell back as expected. 208 of 209 match. The exception is `webgpu_refraction` at 0.1 % of pixels (both runs), which renders identically with render passes (0.0 %, two control runs). Its 512² rgba8 normal map's mipmaps become screen-UV offsets (× 0.1) across hard wall edges, so one-step differences in the mips move a few edge pixels. Two other failures in the first pass (`webgpu_compile_async`, which generates no mipmaps, and `webgpu_loader_gltf`) were load timeouts and passed on rerun. With the default `'render'`, e2e is unaffected: that path is unchanged.
- Example: `webgpu_textures_mipmaps_compute` shows render-pass and compute mipmaps of an animated 1920×1080 render target side by side, with a mip level slider and an sRGB option that shows the fallback.

**Keeping the vendored copy in sync.** The header of `WebGPUTextureMipmapUtils.js` names the library commit. The generated WGSL is byte-identical to the library's. The library keeps golden kernels for 56 option sets (`test/golden/mipmap/`), and `node scripts/check-vendored.mjs <path to WebGPUTextureMipmapUtils.js>`, run in the library, compares this file's `createMipmapKernel` (a named export for that purpose) against them. Result for this PR: 54 of 56 match, 0 differ. The other 2 are not covered: they are `f16Rounding: 'device'` variants, which three doesn't use and the copy rejects. A kernel change should land in the library first, with its GPU suite (every format, odd and prime sizes, 1D tails, layers, 1/4/5/6 levels per dispatch, against a CPU oracle), and then be re-vendored and re-checked. The parity test above checks the integration on a real GPU.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
