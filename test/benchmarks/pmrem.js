import assert from 'node:assert/strict';
import { execSync } from 'node:child_process';
import { copyFileSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';

// Run from the repository root with: npm run benchmark-pmrem -- [builds=<git ref>,...] [cases=bake,cold,...] [runs=3] [captures=<directory>] [headful]
// Compares PMREMGenerator in the src/ of each build on the same page. A build is a git ref, or "." for the
// working tree, which is also appended when it isn't listed. The first build is the baseline.
// Builds rotate within each run to reduce the effect of browser and driver warm-up.
const args = Object.fromEntries( process.argv.slice( 2 ).map( arg => arg.split( '=' ) ).map( ( [ key, value ] ) => [ key, value ?? true ] ) );
const refs = ( args.builds || 'upstream/dev' ).split( ',' );
if ( refs.includes( '.' ) === false ) refs.push( '.' );
const runs = Number( args.runs || 3 );
const cases = ( args.cases || 'bake,cold,scene,equirect,sampling,readback' ).split( ',' );
const caseSizes = {
	bake: '64,128,256,512,1024',
	cold: '256',
	scene: '64,128,256',
	equirect: '',
	sampling: '256',
	readback: '64,256'
};

const cacheDirectory = 'node_modules/.cache/pmrem-benchmark';
rmSync( cacheDirectory, { recursive: true, force: true } );

// Raw rgba16f cubes to read back in place of the procedural sky, see pmrem.html.
const captures = [];

if ( args.captures ) {

	mkdirSync( `${ cacheDirectory }/captures`, { recursive: true } );

	for ( const file of readdirSync( args.captures ).filter( name => name.endsWith( '.bin' ) ) ) {

		copyFileSync( `${ args.captures }/${ file }`, `${ cacheDirectory }/captures/${ file }` );
		captures.push( `../../../${ cacheDirectory }/captures/${ file }` );

	}

}

const builds = refs.map( ( ref, index ) => {

	if ( ref === '.' ) return { name: 'working tree', src: '../../../src' };

	const directory = `${ cacheDirectory }/${ index }`;
	mkdirSync( directory, { recursive: true } );
	execSync( `git archive ${ ref } src | tar -x -C ${ directory }` );

	return { name: ref, src: `../../../${ directory }/src` };

} );

const server = createServer();
let browser;

try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( {
		headless: args.headful !== true,
		args: [ '--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist', '--use-angle=metal' ]
	} );

	async function runPage( build, testCase ) {

		const page = await browser.newPage();
		const errors = [];
		page.on( 'pageerror', error => errors.push( error.message ) );
		page.on( 'console', message => message.type() === 'error' && errors.push( message.text() ) );

		const query = new URLSearchParams( { src: build.src, case: testCase, sizes: caseSizes[ testCase ] } );
		if ( testCase === 'readback' && captures.length > 0 ) query.set( 'captures', captures.join( ',' ) );
		await page.goto( `http://localhost:${ server.address().port }/test/benchmarks/pmrem/pmrem.html?${ query }` );
		await page.waitForFunction( () => window.benchmarkResults !== undefined, { timeout: 600000 } );
		const result = await page.evaluate( () => window.benchmarkResults );
		await page.close();

		assert.equal( result.error, undefined, `${ build.name }, ${ testCase }: ${ result.error }` );
		assert.deepEqual( errors, [], `${ build.name }, ${ testCase } must run without errors` );
		assert.equal( result.adapter, 'WebGPU', 'The benchmark needs the WebGPU backend' );

		return result;

	}

	for ( const testCase of cases ) {

		if ( testCase === 'readback' ) {

			const results = [];
			for ( const build of builds ) results.push( await runPage( build, testCase ) );

			for ( let index = 1; index < builds.length; index ++ ) {

				const table = [];

				for ( let i = 0; i < results[ index ].rows.length; i ++ ) {

					const a = results[ 0 ].rows[ i ], b = results[ index ].rows[ i ];

					// Builds that size the PMREM differently can't be compared texel by texel.
					if ( a.output !== b.output ) continue;

					b.levels.forEach( ( level, lod ) => table.push( { input: b.input, lod, size: level.size, ...compare( a.levels[ lod ].data, level.data ) } ) );

				}

				console.log( `\nreadback: ${ builds[ index ].name } against ${ builds[ 0 ].name }, relative error per mip level on luminance` );
				console.table( table );

			}

			continue;

		}

		const samples = builds.map( () => [] );

		for ( let run = 0; run < ( testCase === 'cold' ? runs * 2 : runs ); run ++ ) {

			for ( let i = 0; i < builds.length; i ++ ) {

				const index = ( i + run ) % builds.length;
				samples[ index ].push( await runPage( builds[ index ], testCase ) );

			}

		}

		const table = samples[ 0 ][ 0 ].rows.map( ( row, i ) => {

			const best = samples.map( buildSamples => Math.min( ...buildSamples.map( sample => sample.rows[ i ].ms ) ) );
			const cpu = samples.map( buildSamples => Math.min( ...buildSamples.map( sample => sample.rows[ i ].cpuMs ) ) );

			return builds.map( ( build, index ) => ( {
				input: row.input,
				build: build.name,
				output: samples[ index ][ 0 ].rows[ i ].output,
				ms: + best[ index ].toFixed( 2 ),
				speedup: `${ ( best[ 0 ] / best[ index ] ).toFixed( 2 ) }×`,
				'cpu ms': + cpu[ index ].toFixed( 3 )
			} ) );

		} ).flat();

		console.log( `\n${ testCase }: minimum of ${ samples[ 0 ].length } runs, milliseconds per call (${ samples[ 0 ][ 0 ].revision }), speedup against ${ builds[ 0 ].name }` );
		console.table( table );

	}

} finally {

	if ( browser !== undefined ) await browser.close();
	server.close();
	rmSync( cacheDirectory, { recursive: true, force: true } );

}

function decodeHalf( h ) {

	const exponent = ( h >> 10 ) & 0x1f, mantissa = h & 0x3ff;
	const value = exponent === 0 ? mantissa * 2 ** - 24 : exponent === 31 ? Infinity : ( 1 + mantissa / 1024 ) * 2 ** ( exponent - 15 );

	return h & 0x8000 ? - value : value;

}

// Per texel |a - b| / max( a, 5% of the level's mean ) on luminance, which keeps near-black texels from dominating.
function compare( a, b ) {

	const halfsA = new Uint16Array( Buffer.from( a, 'base64' ).buffer.slice( 0 ) );
	const halfsB = new Uint16Array( Buffer.from( b, 'base64' ).buffer.slice( 0 ) );

	const luminance = ( halfs, i ) => 0.2126 * decodeHalf( halfs[ i ] ) + 0.7152 * decodeHalf( halfs[ i + 1 ] ) + 0.0722 * decodeHalf( halfs[ i + 2 ] );

	let mean = 0, identical = 0;
	const texels = halfsA.length / 4;

	for ( let i = 0; i < halfsA.length; i += 4 ) mean += luminance( halfsA, i ) / texels;

	const errors = [];

	for ( let i = 0; i < halfsA.length; i += 4 ) {

		const reference = luminance( halfsA, i );
		errors.push( Math.abs( luminance( halfsB, i ) - reference ) / Math.max( reference, 0.05 * mean ) );

		if ( halfsA[ i ] === halfsB[ i ] && halfsA[ i + 1 ] === halfsB[ i + 1 ] && halfsA[ i + 2 ] === halfsB[ i + 2 ] ) identical ++;

	}

	errors.sort( ( x, y ) => x - y );

	return {
		'mean %': + ( 100 * errors.reduce( ( sum, error ) => sum + error, 0 ) / errors.length ).toFixed( 4 ),
		'p99 %': + ( 100 * errors[ Math.floor( errors.length * 0.99 ) ] ).toFixed( 4 ),
		'max %': + ( 100 * errors[ errors.length - 1 ] ).toFixed( 4 ),
		'identical %': + ( 100 * identical / texels ).toFixed( 2 )
	};

}
