import DataMap from '../../common/DataMap.js';
import { GPUFeatureName, GPUTextureViewDimension } from './WebGPUConstants.js';

// Compute mipmap generation, used when the renderer is created with
// `mipmapGeneration: 'compute'`.
//
// Vendored from the downsampler library (https://github.com/pmndrs/downsampler,
// commit 488eff6): the WGSL generator of its 'exact' kernel and tail kernel
// (src/webgpu/kernel.ts), the batch planner (src/core/plan.ts) and the pipeline
// and bind group caching of its Downsampler (src/webgpu/downsampler.ts), reduced
// to in-place 'average' chains. Keep it in sync with the library: its GPU suite
// is the oracle for these kernels.
//
// Results match render-pass mipmaps ('exact' semantics): a batch's first level is
// one bilinear sample at each texel centre in normalized coordinates, as a render
// pass takes it, so odd sizes give the same result. Further levels of a batch are
// exact 2:1 reductions in workgroup memory, so a batch ends where a non-unit
// dimension is odd. Once a level fits one workgroup, a tail kernel finishes the
// chain with the same rule, odd sizes included. Every level is quantized as
// stored: normalized formats in integer units with ties to even, float16 rounded
// to nearest even in the shader (the device's own conversion truncates on D3D12).

const _formats = {
	'rgba8unorm': { quantization: 'unorm8' },
	'rgba8snorm': { quantization: 'snorm8' },
	'bgra8unorm': { quantization: 'unorm8', feature: GPUFeatureName.BGRA8UNormStorage },
	'r8unorm': { quantization: 'unorm8', feature: GPUFeatureName.TextureFormatsTier1 },
	'rg8unorm': { quantization: 'unorm8', feature: GPUFeatureName.TextureFormatsTier1 },
	'r8snorm': { quantization: 'snorm8', feature: GPUFeatureName.TextureFormatsTier1 },
	'rg8snorm': { quantization: 'snorm8', feature: GPUFeatureName.TextureFormatsTier1 },
	'r16unorm': { quantization: 'unorm16', feature: GPUFeatureName.TextureFormatsTier1 },
	'rg16unorm': { quantization: 'unorm16', feature: GPUFeatureName.TextureFormatsTier1 },
	'rgba16unorm': { quantization: 'unorm16', feature: GPUFeatureName.TextureFormatsTier1 },
	'r16snorm': { quantization: 'snorm16', feature: GPUFeatureName.TextureFormatsTier1 },
	'rg16snorm': { quantization: 'snorm16', feature: GPUFeatureName.TextureFormatsTier1 },
	'rgba16snorm': { quantization: 'snorm16', feature: GPUFeatureName.TextureFormatsTier1 },
	'rgba16float': { quantization: 'float16' },
	'r16float': { quantization: 'float16', feature: GPUFeatureName.TextureFormatsTier1 },
	'rg16float': { quantization: 'float16', feature: GPUFeatureName.TextureFormatsTier1 },
	'rgba32float': { quantization: 'float32' },
	'r32float': { quantization: 'float32' },
	'rg32float': { quantization: 'float32' }
};

// Normalized values are reduced in integer units, so 2:1 averages are exact and ties round to even.
const _integerDomains = {
	unorm8: { scale: 255, min: 0, max: 255 },
	snorm8: { scale: 127, min: - 127, max: 127 },
	unorm16: { scale: 65535, min: 0, max: 65535 },
	snorm16: { scale: 32767, min: - 32767, max: 32767 }
};

// Largest first output of a tail batch: one 32×32 workgroup tile.
const TAIL_TILE = 32;

const mipSize = ( size, level ) => Math.max( 1, size >> level );

// First-output tile of a batch of `count` levels: 8×8 up to four levels, 16×16 for five, 32×32 for six.
const tileSize = ( count ) => Math.max( 8, 1 << ( count - 1 ) );

// Invocations per workgroup edge; each writes ( tile / side )² first-output texels.
const workgroupSide = ( tile ) => Math.min( tile, 16 );

// Tail tile: the first output's larger side, rounded up to 8, 16 or 32.
const tailTileFor = ( width, height ) => Math.min( TAIL_TILE, Math.max( 8, 2 ** Math.ceil( Math.log2( Math.max( width, height ) ) ) ) );

/**
 * Splits the mip chain of a `width` × `height` texture into dispatches. A batch
 * reads one level and writes up to `levelsPerDispatch` levels; it ends early where
 * a non-unit dimension of its last level is odd. Once a first output fits
 * `tailMaxTile`², one tail batch finishes the chain.
 *
 * @private
 * @param {number} width - The width of level 0.
 * @param {number} height - The height of level 0.
 * @param {number} mipLevelCount - The levels of the texture.
 * @param {number} levelsPerDispatch - The most levels one dispatch writes (1–6).
 * @param {number} tailMaxTile - The largest tail tile (8 or 32).
 * @param {number} exactMinTile - The smallest tile of regular batches (8, 16 or 32).
 * @return {Array<Object>} The batches.
 */
function planBatches( width, height, mipLevelCount, levelsPerDispatch, tailMaxTile, exactMinTile ) {

	const batches = [];
	const lastLevel = mipLevelCount - 1;

	for ( let level = 1; level <= lastLevel; ) {

		const outputWidth = mipSize( width, level ), outputHeight = mipSize( height, level );
		let count = 1, w = outputWidth, h = outputHeight;

		// Continue in workgroup memory only while each non-unit dimension halves exactly.
		while ( count < levelsPerDispatch && level + count <= lastLevel && ( w === 1 || w % 2 === 0 ) && ( h === 1 || h % 2 === 0 ) ) {

			w = Math.max( 1, w / 2 );
			h = Math.max( 1, h / 2 );
			count ++;

		}

		// The tail: one workgroup finishes what would otherwise take more batches.
		const tailCount = Math.min( levelsPerDispatch, lastLevel - level + 1 );

		if ( count < tailCount && outputWidth <= tailMaxTile && outputHeight <= tailMaxTile ) {

			batches.push( { tail: true, sourceMipLevel: level - 1, count: tailCount, tile: tailTileFor( outputWidth, outputHeight ), groupsX: 1, groupsY: 1 } );
			level += tailCount;
			continue;

		}

		const tile = Math.max( exactMinTile, tileSize( count ) );
		batches.push( { tail: false, sourceMipLevel: level - 1, count, tile, groupsX: Math.ceil( outputWidth / tile ), groupsY: Math.ceil( outputHeight / tile ) } );
		level += count;

	}

	return batches;

}

// float16: round each channel to the nearest even half, as f32. In the half normal range the f32
// bits are rounded to 10 mantissa bits with integer math (a carry into the exponent is correct);
// below it, round( x · 2^24 ) · 2^-24. Not ( x + c ) - c, which fast math folds to x on Metal.
const float16Wgsl = /* wgsl */`
fn ds_round_f16(v: vec4f) -> vec4f {
	let bits = bitcast<vec4u>(v);
	let a = bits & vec4u(0x7fffffffu);
	let normal = (a + vec4u(0xfffu) + ((a >> vec4u(13u)) & vec4u(1u))) & vec4u(0xffffe000u);
	let subnormal = bitcast<vec4u>(round(bitcast<vec4f>(a) * 16777216.0) * (1.0 / 16777216.0));
	let r = select(subnormal, normal, a >= vec4u(0x38800000u));
	let q = select(select(r, vec4u(0x7f800000u), a >= vec4u(0x477ff000u)), vec4u(0x7fc00000u), a > vec4u(0x7f800000u));
	return bitcast<vec4f>(q | (bits & vec4u(0x80000000u)));
}
`;

/**
 * Returns the WGSL of one batch. Bindings: 0 sampler ('sample' access only),
 * 1 the source level, 2… one write-only storage texture per output level.
 * Workgroup z selects the array layer.
 *
 * Each workgroup writes a tile×tile block of the first output. Adjacent invocations
 * read and store adjacent texels (a 2×2 block per invocation measured 30–45 %
 * slower). Later levels are 2:1 reductions of the tile in workgroup memory, with
 * clamped reads for unit dimensions and partial tiles. A tail kernel is one
 * workgroup: the first output as above, then each further level from the previous
 * one by the same rule at any parity, with bilinear weights computed in the shader.
 *
 * @private
 * @param {string} format - The texture format.
 * @param {string} dimension - The view dimension, '2d' or '2d-array'.
 * @param {number} count - The levels written by the dispatch.
 * @param {string} access - 'sample' (filterable formats) or 'load'.
 * @param {boolean} tail - Whether this is a tail kernel.
 * @param {number} tile - The first-output tile edge (8, 16 or 32).
 * @return {string} The WGSL code.
 */
function createKernel( format, dimension, count, access, tail, tile ) {

	const quantization = _formats[ format ].quantization;
	const domain = _integerDomains[ quantization ];
	const array = dimension === GPUTextureViewDimension.TwoDArray;
	const suffix = array ? '_array' : '';
	const layer = array ? ', i32(group.z)' : '';
	const side = workgroupSide( tile );
	const samples = tile / side;

	let code = access === 'sample' ? '@group(0) @binding(0) var imageSampler: sampler;\n' : '';

	code += `@group(0) @binding(1) var source: texture_2d${ suffix }<f32>;
var<workgroup> tile: array<vec4f, ${ tile * tile }>;
`;

	for ( let i = 0; i < count; i ++ ) {

		code += `@group(0) @binding(${ i + 2 }) var mip${ i }: texture_storage_2d${ suffix }<${ format }, write>;\n`;

	}

	// Intermediate levels are quantized exactly as they are stored. Normalized values stay in
	// integer units, so 2:1 averages are exact and ties round to even (with k/255 floats, float
	// error turns ties into round-ups that drift along 1D tails). The parity test uses i32:
	// converting a negative float to u32 is undefined (snorm).
	if ( domain !== undefined ) {

		code += `
const SCALE = ${ domain.scale }.0;
fn quantize(value: vec4f) -> vec4f {
	let scaled = clamp(value, vec4f(${ domain.min }.0), vec4f(${ domain.max }.0));
	let lower = floor(scaled);
	let fraction = scaled - lower;
	let nearest = select(lower, lower + vec4f(1), fraction > vec4f(0.5));
	return select(nearest, lower + vec4f(vec4i(lower) & vec4i(1)), fraction == vec4f(0.5));
}
`;

	} else if ( quantization === 'float16' ) {

		code += float16Wgsl + `
const SCALE = 1.0;
fn quantize(value: vec4f) -> vec4f { return ds_round_f16(value); }
`;

	} else {

		code += `
const SCALE = 1.0;
fn quantize(value: vec4f) -> vec4f { return value; }
`;

	}

	if ( access === 'sample' ) {

		code += `
fn first(pos: vec2u, size: vec2u, layer: u32) -> vec4f {
	let uv = (vec2f(pos) + 0.5) / vec2f(size);
	return quantize(textureSampleLevel(source, imageSampler, uv${ array ? ', i32(layer)' : '' }, 0.0) * SCALE);
}
`;

	} else {

		// Bilinear weights at the output texel centre, as the sampler applies them but without its
		// fixed-point weight rounding. The sample position (x + 0.5) * S / n - 0.5 is
		// ((2x + 1) * S - n) / 2n: integer math keeps the base texel exact (GPU float division isn't
		// correctly rounded), and gives weights of exactly 0 or 0.5 at even sizes. Normalized texels
		// are rounded back to exact integer units on load, so later sums are exact.
		code += `
fn load(p: vec2u, layer: u32) -> vec4f {
	let v = textureLoad(source, vec2i(p)${ array ? ', i32(layer)' : '' }, 0) * SCALE;
	return ${ domain !== undefined ? 'round(v)' : 'v' };
}

fn first(pos: vec2u, size: vec2u, layer: u32) -> vec4f {
	let sourceSize = textureDimensions(source);
	let last = sourceSize - vec2u(1);
	let numerator = (2u * pos + vec2u(1)) * sourceSize - size;
	let denominator = 2u * size;
	let lo = min(numerator / denominator, last);
	let hi = min(lo + vec2u(1), last);
	let f = vec2f(numerator % denominator) / vec2f(denominator);
	let top = mix(load(lo, layer), load(vec2u(hi.x, lo.y), layer), f.x);
	let bottom = mix(load(vec2u(lo.x, hi.y), layer), load(hi, layer), f.x);
	return quantize(mix(top, bottom, f.y));
}
`;

	}

	// Later sizes are derived from the first (floored halves, at least 1), not queried:
	// textureDimensions on a storage texture is a resource query per call.
	const dims = ( i ) => i === 0 ? 'textureDimensions(mip0)' : `max(size0 >> vec2u(${ i }u), vec2u(1u))`;
	const store = ( i, pos ) => `textureStore(mip${ i }, vec2i(${ pos })${ layer }, value / SCALE);`;

	if ( tail ) {

		// The same rule from a level held in the tile (scaled, quantized units).
		code += `
fn tileAt(p: vec2u) -> vec4f { return tile[p.y * ${ tile }u + p.x]; }
fn tailStep(pos: vec2u, sourceSize: vec2u, size: vec2u) -> vec4f {
	let last = sourceSize - vec2u(1);
	let numerator = (2u * pos + vec2u(1)) * sourceSize - size;
	let denominator = 2u * size;
	let lo = min(numerator / denominator, last);
	let hi = min(lo + vec2u(1), last);
	let f = vec2f(numerator % denominator) / vec2f(denominator);
	let top = mix(tileAt(lo), tileAt(vec2u(hi.x, lo.y)), f.x);
	let bottom = mix(tileAt(vec2u(lo.x, hi.y)), tileAt(hi), f.x);
	return quantize(mix(top, bottom, f.y));
}
`;

	}

	code += `
@compute @workgroup_size(${ side }, ${ side })
fn main(@builtin(local_invocation_id) local: vec3u, @builtin(workgroup_id) group: vec3u) {
	let size0 = ${ dims( 0 ) };
	var value: vec4f;
`;

	for ( let i = 0; i < samples * samples; i ++ ) {

		code += `
	{
		let p = local.xy + vec2u(${ ( i % samples ) * side }u, ${ Math.floor( i / samples ) * side }u);
		let pos0 = group.xy * ${ tile }u + p;
		value = first(min(pos0, size0 - vec2u(1)), size0, group.z);
		tile[p.y * ${ tile }u + p.x] = value;
		if (all(pos0 < size0)) { ${ store( 0, 'pos0' ) } }
	}
`;

	}

	for ( let i = 1; i < count; i ++ ) {

		if ( tail ) {

			// The first output fits the tile, so every later level fits half of it, at most
			// the workgroup's side: one texel per invocation.
			code += `
	workgroupBarrier();
	{
		let size = ${ dims( i ) };
		if (all(local.xy < size)) { value = tailStep(local.xy, ${ dims( i - 1 ) }, size); }
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (all(local.xy < ${ dims( i ) })) {
		tile[local.y * ${ tile }u + local.x] = value;
		${ store( i, 'local.xy' ) }
	}
`;

		} else {

			const extent = tile >> i;
			const at = ( x, y ) => `tile[${ y }.y * ${ tile }u + ${ x }.x]`;

			code += `
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		let last = ${ dims( i - 1 ) } - group.xy * ${ extent * 2 }u - vec2u(1);
		let lo = min(local.xy * 2u, last);
		let hi = min(lo + vec2u(1), last);
		value = quantize((${ at( 'lo', 'lo' ) } + ${ at( 'hi', 'lo' ) } + ${ at( 'lo', 'hi' ) } + ${ at( 'hi', 'hi' ) }) * 0.25);
	}
	// Every reader must finish before the shared tile is overwritten.
	workgroupBarrier();
	if (local.x < ${ extent }u && local.y < ${ extent }u) {
		tile[local.y * ${ tile }u + local.x] = value;
		let pos = group.xy * ${ extent }u + local.xy;
		if (all(pos < ${ dims( i ) })) { ${ store( i, 'pos' ) } }
	}
`;

		}

	}

	return code + '}\n';

}

/**
 * A WebGPU backend utility module that generates mipmaps with compute shaders,
 * used by {@link WebGPUTextureUtils} when the renderer is created with
 * `mipmapGeneration: 'compute'`. A dispatch writes up to six levels (four at the
 * default `maxStorageTexturesPerShaderStage`), so a mip chain takes a few
 * dispatches instead of a render pass per level and layer. Results match
 * render-pass mipmaps, odd sizes included.
 *
 * Textures it can't write (sRGB, compressed, multisampled, 3D or packed formats,
 * and every texture in compatibility mode) keep render-pass generation.
 *
 * @private
 * @augments DataMap
 */
class WebGPUTextureMipmapUtils extends DataMap {

	/**
	 * Constructs a new utility object.
	 *
	 * @param {GPUDevice} device - The WebGPU device.
	 */
	constructor( device ) {

		super();

		/**
		 * The WebGPU device.
		 *
		 * @type {GPUDevice}
		 */
		this.device = device;

		/**
		 * Whether the device is in compatibility mode. Compatibility mode allows one
		 * view of a texture per dispatch, so an in-place chain can't be written there.
		 *
		 * @type {boolean}
		 */
		this.compatibilityMode = ! device.features.has( GPUFeatureName.CoreFeaturesAndLimits );

		// 5 and 6 levels take 16 × 16 invocations.
		const wide = device.limits.maxComputeInvocationsPerWorkgroup >= 256;

		/**
		 * The most levels one dispatch writes: one storage texture binding per level.
		 *
		 * @type {number}
		 */
		this.levelsPerDispatch = Math.max( 1, Math.min( wide ? 6 : 4, device.limits.maxStorageTexturesPerShaderStage ) );

		// Regular batches: 32×32 tiles (16 × 16 invocations, 2 × 2 texels each) were fastest on
		// NVIDIA and Intel, 16×16 on Apple GPUs. Software adapters compile 32×32 tiles slowly.
		const info = device.adapterInfo;

		this.exactMinTile = wide && info?.isFallbackAdapter !== true ? ( info?.vendor === 'apple' ? 16 : 32 ) : 8;
		this.tailMaxTile = wide ? TAIL_TILE : 8;

		this.sampler = device.createSampler( { label: 'mipmapComputeSampler', minFilter: 'linear', magFilter: 'linear' } );
		this.pipelines = new Map();
		this.layouts = new Map();

	}

	/**
	 * Returns whether compute mipmaps can write a texture of the given format on this device.
	 *
	 * @param {string} format - The GPU texture format.
	 * @return {boolean} Whether the format is supported.
	 */
	isFormatSupported( format ) {

		const info = _formats[ format ];

		return this.compatibilityMode === false && info !== undefined && ( info.feature === undefined || this.device.features.has( info.feature ) );

	}

	/**
	 * Returns whether compute mipmaps can generate the levels of the given texture.
	 *
	 * @param {GPUTexture} textureGPU - The GPU texture.
	 * @return {boolean} Whether the texture is supported.
	 */
	canGenerateMipmaps( textureGPU ) {

		return this.isFormatSupported( textureGPU.format ) &&
			textureGPU.dimension === '2d' && textureGPU.sampleCount === 1 && textureGPU.mipLevelCount > 1 &&
			( textureGPU.usage & GPUTextureUsage.STORAGE_BINDING ) !== 0 &&
			( textureGPU.usage & GPUTextureUsage.TEXTURE_BINDING ) !== 0;

	}

	/**
	 * Generates the mipmaps of the given GPU texture from its level 0. The texture
	 * must be supported, see {@link WebGPUTextureMipmapUtils#canGenerateMipmaps}.
	 *
	 * @param {GPUTexture} textureGPU - The GPU texture.
	 * @param {?GPUCommandEncoder} [encoder=null] - An optional command encoder used to generate mipmaps.
	 */
	generateMipmaps( textureGPU, encoder = null ) {

		const textureData = this.get( textureGPU );

		if ( textureData.batches === undefined ) textureData.batches = this._createBatches( textureGPU );

		const commandEncoder = encoder || this.device.createCommandEncoder( { label: 'mipmapComputeEncoder' } );

		// Dispatches in a compute pass are separate usage scopes, so each batch reads what the previous one wrote.
		const pass = commandEncoder.beginComputePass( { label: 'mipmapCompute' } );

		for ( const batch of textureData.batches ) {

			pass.setPipeline( batch.pipeline );
			pass.setBindGroup( 0, batch.bindGroup );
			pass.dispatchWorkgroups( batch.groupsX, batch.groupsY, textureGPU.depthOrArrayLayers );

		}

		pass.end();

		if ( encoder === null ) this.device.queue.submit( [ commandEncoder.finish() ] );

	}

	/**
	 * Plans the dispatches of a texture's mip chain and creates their bind groups.
	 *
	 * @private
	 * @param {GPUTexture} textureGPU - The GPU texture.
	 * @return {Array<Object>} The dispatches.
	 */
	_createBatches( textureGPU ) {

		const { format } = textureGPU;

		// Layers and cube faces are views of a 2d-array (not available in compatibility mode).
		const dimension = textureGPU.depthOrArrayLayers > 1 ? GPUTextureViewDimension.TwoDArray : GPUTextureViewDimension.TwoD;
		const access = this._getAccess( format );
		const view = ( level ) => textureGPU.createView( { label: `mipmapCompute ${ level }`, dimension, baseMipLevel: level, mipLevelCount: 1 } );

		const batches = planBatches( textureGPU.width, textureGPU.height, textureGPU.mipLevelCount, this.levelsPerDispatch, this.tailMaxTile, this.exactMinTile );

		return batches.map( ( batch ) => {

			const { pipeline, layout } = this._getPipeline( format, dimension, batch.count, access, batch.tail, batch.tile );

			const entries = access === 'sample' ? [ { binding: 0, resource: this.sampler } ] : [];
			entries.push( { binding: 1, resource: view( batch.sourceMipLevel ) } );

			for ( let i = 0; i < batch.count; i ++ ) {

				entries.push( { binding: i + 2, resource: view( batch.sourceMipLevel + 1 + i ) } );

			}

			const bindGroup = this.device.createBindGroup( { label: 'mipmapCompute', layout, entries } );

			return { pipeline, bindGroup, groupsX: batch.groupsX, groupsY: batch.groupsY };

		} );

	}

	/**
	 * Returns how a batch reads its source level: a filtering sampler where the
	 * format allows it, else `textureLoad` with weights computed in the shader.
	 *
	 * @private
	 * @param {string} format - The GPU texture format.
	 * @return {string} 'sample' or 'load'.
	 */
	_getAccess( format ) {

		const quantization = _formats[ format ].quantization;

		if ( quantization === 'unorm16' || quantization === 'snorm16' ) return 'load';
		if ( quantization === 'float32' ) return this.device.features.has( GPUFeatureName.Float32Filterable ) ? 'sample' : 'load';

		return 'sample';

	}

	/**
	 * Returns the cached pipeline and bind group layout of a kernel.
	 *
	 * @private
	 * @param {string} format - The GPU texture format.
	 * @param {string} dimension - The view dimension.
	 * @param {number} count - The levels written by the dispatch.
	 * @param {string} access - 'sample' or 'load'.
	 * @param {boolean} tail - Whether this is a tail kernel.
	 * @param {number} tile - The first-output tile edge.
	 * @return {{pipeline: GPUComputePipeline, layout: GPUBindGroupLayout}} The pipeline and its bind group layout.
	 */
	_getPipeline( format, dimension, count, access, tail, tile ) {

		const key = `${ tail ? 'tail' : 'exact' }${ tile }:${ format }:${ dimension }:${ count }:${ access }`;

		let cached = this.pipelines.get( key );

		if ( cached === undefined ) {

			const layout = this._getLayout( format, dimension, count, access );
			const label = `mipmapCompute ${ key }`;
			const module = this.device.createShaderModule( { label, code: createKernel( format, dimension, count, access, tail, tile ) } );

			const pipeline = this.device.createComputePipeline( {
				label,
				layout: this.device.createPipelineLayout( { label, bindGroupLayouts: [ layout ] } ),
				compute: { module, entryPoint: 'main' }
			} );

			cached = { pipeline, layout };

			this.pipelines.set( key, cached );

		}

		return cached;

	}

	/**
	 * Returns the cached bind group layout of a kernel. Regular and tail kernels share layouts.
	 *
	 * @private
	 * @param {string} format - The GPU texture format.
	 * @param {string} dimension - The view dimension.
	 * @param {number} count - The levels written by the dispatch.
	 * @param {string} access - 'sample' or 'load'.
	 * @return {GPUBindGroupLayout} The bind group layout.
	 */
	_getLayout( format, dimension, count, access ) {

		const key = `${ format }:${ dimension }:${ count }:${ access }`;

		let layout = this.layouts.get( key );

		if ( layout === undefined ) {

			const visibility = GPUShaderStage.COMPUTE;
			const entries = access === 'sample' ? [ { binding: 0, visibility, sampler: { type: 'filtering' } } ] : [];

			entries.push( { binding: 1, visibility, texture: { sampleType: access === 'sample' ? 'float' : 'unfilterable-float', viewDimension: dimension } } );

			for ( let i = 0; i < count; i ++ ) {

				entries.push( { binding: i + 2, visibility, storageTexture: { access: 'write-only', format, viewDimension: dimension } } );

			}

			layout = this.device.createBindGroupLayout( { label: `mipmapCompute ${ key }`, entries } );

			this.layouts.set( key, layout );

		}

		return layout;

	}

}

export default WebGPUTextureMipmapUtils;
