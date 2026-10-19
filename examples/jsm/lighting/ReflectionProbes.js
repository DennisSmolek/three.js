import {
	Box3,
	Color,
	CubeArrayTexture,
	CubeReflectionMapping,
	HalfFloatType,
	LinearMipmapLinearFilter,
	Object3D,
	PMREMGenerator,
	RenderTarget,
	Scene,
	Vector3
} from 'three/webgpu';

const _position = /*@__PURE__*/ new Vector3();
const _size = /*@__PURE__*/ new Vector3();
const _region = /*@__PURE__*/ new Box3();
const _destination = /*@__PURE__*/ new Vector3();
const _black = /*@__PURE__*/ new Color( 0x000000 );

/**
 * A local reflection volume. Add it to a {@link ReflectionProbes} container.
 *
 * The volumes are axis-aligned boxes centered on the probe. They follow the
 * probe's world position and scale; rotation is ignored.
 *
 * @augments Object3D
 * @three_import import { ReflectionProbe } from 'three/addons/lighting/ReflectionProbes.js';
 */
class ReflectionProbe extends Object3D {

	/**
	 * Constructs a new reflection probe.
	 *
	 * @param {number} [width=1] - Width of the influence volume along X.
	 * @param {number} [height=1] - Height of the influence volume along Y.
	 * @param {number} [depth=1] - Depth of the influence volume along Z.
	 */
	constructor( width = 1, height = 1, depth = 1 ) {

		super();

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isReflectionProbe = true;

		this.type = 'ReflectionProbe';

		/**
		 * Size of the influence volume. Surfaces inside it use this probe.
		 *
		 * @type {Vector3}
		 */
		this.influenceSize = new Vector3( width, height, depth );

		/**
		 * Size of the box that reflections are projected onto, usually the room's
		 * walls. `null` uses `influenceSize`.
		 *
		 * @type {?Vector3}
		 * @default null
		 */
		this.projectionSize = null;

		/**
		 * Distance over which the influence fades in from the volume's faces.
		 *
		 * @type {number}
		 * @default 0.5
		 */
		this.blendDistance = 0.5;

	}

	copy( source, recursive ) {

		super.copy( source, recursive );

		this.influenceSize.copy( source.influenceSize );
		this.projectionSize = source.projectionSize === null ? null : source.projectionSize.clone();
		this.blendDistance = source.blendDistance;

		return this;

	}

}

/**
 * A set of {@link ReflectionProbe} children baked into one cube-array texture.
 *
 * Assign `reflectionProbes( probes )` to `scene.environmentNode` and every
 * physical material samples the probes with its own roughness and normal.
 * Smaller volumes take priority over larger ones; uncovered surfaces fall
 * back to `scene.environment`. Requires the WebGPU backend.
 *
 * ```js
 * const probes = new ReflectionProbes();
 * const probe = new ReflectionProbe( 8, 5, 8 );
 * probes.add( probe );
 * scene.add( probes );
 *
 * probes.bake( renderer, scene );
 * scene.environmentNode = reflectionProbes( probes ); // from three/addons/tsl/lighting/ReflectionProbesNode.js
 * ```
 *
 * @augments Object3D
 * @three_import import { ReflectionProbes } from 'three/addons/lighting/ReflectionProbes.js';
 */
class ReflectionProbes extends Object3D {

	/**
	 * Constructs a new reflection probe set.
	 */
	constructor() {

		super();

		/**
		 * This flag can be used for type testing.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default true
		 */
		this.isReflectionProbes = true;

		this.type = 'ReflectionProbes';

		/**
		 * Whether the probes use box projection.
		 *
		 * @type {boolean}
		 * @default true
		 */
		this.boxProjection = true;

		/**
		 * The baked cube-array texture: one cube per probe, then the environment.
		 *
		 * @type {?Texture}
		 * @default null
		 */
		this.texture = null;

		/**
		 * The probes in baked order, smallest volume first.
		 *
		 * @type {Array<ReflectionProbe>}
		 */
		this.bakedProbes = [];

		/**
		 * Whether a bake is in progress. Probes then contribute only the environment.
		 *
		 * @type {boolean}
		 * @readonly
		 * @default false
		 */
		this.capturing = false;

		this._renderTarget = null;
		this._pmrem = null;
		this._generator = null;
		this._environment = new Scene();

		this._environmentTexture = undefined;
		this._environmentVersion = - 1;
		this._environmentPMREMVersion = - 1;

	}

	/**
	 * Captures every probe and `scene.environment` into the cube array. Hide
	 * helpers and objects that should not appear in reflections first.
	 *
	 * @param {WebGPURenderer} renderer - The renderer.
	 * @param {Scene} scene - The scene to capture.
	 * @param {Object} [options] - Bake options.
	 * @param {number} [options.resolution=128] - Cube face size, a power of two from 64.
	 * @param {number} [options.near=0.1] - Near plane of the capture camera.
	 * @param {number} [options.far=100] - Far plane of the capture camera.
	 * @param {Array<ReflectionProbe>} [options.probes] - Captures only these probes, for example to
	 * spread a live update over several frames. A bake that changes the layout captures every probe.
	 */
	bake( renderer, scene, options = {} ) {

		// cube arrays can't be emulated on WebGL 2
		if ( renderer.backend.isWebGPUBackend !== true ) throw new Error( 'ReflectionProbes: .bake() requires the WebGPU backend.' );

		const { resolution = 128, near = 0.1, far = 100 } = options;

		this.updateWorldMatrix( true, true );

		const probes = this.children.filter( ( child ) => child.isReflectionProbe ).sort( ( a, b ) => volume( a ) - volume( b ) );
		const maxLod = Math.log2( resolution ) - 3;

		const depth = ( probes.length + 1 ) * 6;

		// Reuse the cube array and the PMREM between bakes of the same layout, so that
		// re-baking every frame neither allocates nor changes the texture materials sample.

		let renderTarget = this._renderTarget;
		let previousRenderTarget = null;

		const layoutChanged = renderTarget === null || renderTarget.width !== resolution || renderTarget.depth !== depth;

		if ( layoutChanged ) {

			// Materials sample the previous cube array while the probes are captured.

			previousRenderTarget = renderTarget;

			if ( this._pmrem !== null ) this._pmrem.dispose();
			this._pmrem = null;

			renderTarget = new RenderTarget( resolution, resolution, {
				depth,
				type: HalfFloatType,
				depthBuffer: false,
				generateMipmaps: false,
				minFilter: LinearMipmapLinearFilter,
				mapping: CubeReflectionMapping
			} );

			renderTarget.texture = new CubeArrayTexture().copy( renderTarget.texture );
			renderTarget.texture.name = 'ReflectionProbes';

			for ( let lod = 0; lod <= maxLod; lod ++ ) {

				renderTarget.texture.mipmaps.push( { width: resolution >> lod, height: resolution >> lod } );

			}

			renderer.initRenderTarget( renderTarget );

			if ( this._generator === null ) this._generator = new PMREMGenerator( renderer );

		}

		// Capturing only some probes keeps the other slots, so it needs the slots of the last bake.

		const sameProbes = probes.length === this.bakedProbes.length && probes.every( ( probe, i ) => probe === this.bakedProbes[ i ] );
		const captured = options.probes !== undefined && layoutChanged === false && sameProbes ? options.probes : probes;

		// The environment is captured again only when it changed. A new cube array starts black,
		// which is what no environment looks like.

		const environmentTexture = scene.environment;

		const environmentChanged = environmentTexture !== this._environmentTexture ||
			( environmentTexture !== null && ( environmentTexture.version !== this._environmentVersion || environmentTexture.pmremVersion !== this._environmentPMREMVersion ) );

		const captureEnvironment = layoutChanged ? environmentTexture !== null : environmentChanged;

		const generator = this._generator;
		const environment = this._environment;
		environment.background = environmentTexture !== null ? environmentTexture : _black;

		const captureSlot = ( source, position, slot ) => {

			this._pmrem = generator.fromScene( source, 0, near, far, { size: resolution, position, renderTarget: this._pmrem } );

			for ( let lod = 0; lod <= maxLod; lod ++ ) {

				const size = resolution >> lod;

				_region.min.set( 0, 0, 0 );
				_region.max.set( size, size, 6 );
				_destination.set( 0, 0, slot * 6 );

				renderer.copyTextureToTexture( this._pmrem.texture, renderTarget.texture, _region, _destination, lod, lod );

			}

		};

		this.capturing = true;

		try {

			// The environment goes first, as the probe captures reflect it.

			if ( captureEnvironment ) captureSlot( environment, _position.set( 0, 0, 0 ), probes.length );

			for ( const probe of captured ) {

				const slot = probes.indexOf( probe );

				if ( slot !== - 1 ) captureSlot( scene, probe.getWorldPosition( _position ), slot );

			}

		} finally {

			this.capturing = false;

		}

		this._environmentTexture = environmentTexture;
		this._environmentVersion = environmentTexture !== null ? environmentTexture.version : - 1;
		this._environmentPMREMVersion = environmentTexture !== null ? environmentTexture.pmremVersion : - 1;

		if ( previousRenderTarget !== null ) previousRenderTarget.dispose();

		this._renderTarget = renderTarget;
		this.texture = renderTarget.texture;
		this.bakedProbes = probes;

	}

	/**
	 * Frees the GPU-related resources allocated by this instance.
	 */
	dispose() {

		if ( this._renderTarget !== null ) this._renderTarget.dispose();
		if ( this._pmrem !== null ) this._pmrem.dispose();
		if ( this._generator !== null ) this._generator.dispose();

		this._renderTarget = null;
		this._pmrem = null;
		this._generator = null;
		this.texture = null;

	}

}

function volume( probe ) {

	probe.getWorldScale( _size ).multiply( probe.influenceSize );

	return _size.x * _size.y * _size.z;

}

export { ReflectionProbe, ReflectionProbes };
