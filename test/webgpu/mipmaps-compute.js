import assert from 'node:assert/strict';
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';

// Checks that `mipmapGeneration: 'compute'` gives the same mipmaps as render passes,
// within one 8-bit step at every level and layer, in headless Chrome on this GPU.
// Run from the repository root with: npm run test-webgpu-mipmaps

const server = createServer();
let browser;

try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', '--ignore-gpu-blocklist' ] } );

	const page = await browser.newPage();
	const pageErrors = [];
	page.on( 'pageerror', error => pageErrors.push( error.message ) );
	await page.goto( `http://localhost:${ server.address().port }/test/webgpu/mipmaps-compute.html` );
	await page.waitForFunction( () => window.testResults !== undefined, { timeout: 300000 } );
	const { adapter, maxStorageTexturesPerShaderStage, levelsPerDispatch, errors, results } = await page.evaluate( () => window.testResults );

	console.log( `Adapter: ${ adapter }; maxStorageTexturesPerShaderStage ${ maxStorageTexturesPerShaderStage }, ${ levelsPerDispatch } levels per dispatch.` );

	const tolerance = 1 + 1e-6;

	for ( const result of results ) {

		const ok = result.path === result.expectedPath && result.stepSteps <= tolerance && result.chainSteps <= 2 * tolerance;
		console.log( `${ ok ? 'ok  ' : 'FAIL' } ${ result.name } [${ result.format }, ${ result.path }]: per level ${ result.stepSteps.toFixed( 3 ) }, chain ${ result.chainSteps.toFixed( 3 ) } steps` );

		for ( const { level, size, step, chain } of result.levels.slice( 1 ) ) {

			if ( process.argv.includes( '--verbose' ) ) console.log( `       L${ level } ${ size }: per level ${ step === null ? '-' : `${ step.maxSteps.toFixed( 3 ) } (${ step.differing } differ, ${ step.ties } at ties)` }; chain ${ chain.maxSteps.toFixed( 3 ) } (${ chain.differing } differ)` );

		}

	}

	assert.deepEqual( pageErrors, [], 'The page must run without errors' );
	assert.deepEqual( errors, [], 'There must be no GPU validation errors' );

	for ( const result of results ) {

		assert.equal( result.path, result.expectedPath, `${ result.name }: expected ${ result.expectedPath } mipmaps` );
		assert.equal( result.storage, result.expectedPath === 'compute', `${ result.name }: STORAGE_BINDING usage only for compute mipmaps` );
		assert.ok( result.levelZeroEqual, `${ result.name }: level 0 must be identical` );
		assert.ok( result.stepSteps <= tolerance, `${ result.name }: a compute level differs by ${ result.stepSteps } 8-bit steps from a render pass of the same input` );
		assert.ok( result.chainSteps <= 2 * tolerance, `${ result.name }: the compute chain differs by ${ result.chainSteps } 8-bit steps from the render-pass chain` );

	}

	console.log( `${ results.length } cases passed.` );

} finally {

	if ( browser ) await browser.close();
	server.close();

}
