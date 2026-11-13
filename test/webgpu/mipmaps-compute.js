import assert from 'node:assert/strict';
import puppeteer from 'puppeteer';
import { createServer } from '../../utils/server.js';

// Checks that `mipmapGeneration: 'compute'` gives the same mipmaps as render passes, in
// headless Chrome on this GPU, at every level and layer: each compute level within one
// 8-bit step of a render pass from the same input level, and the whole chain within two
// (render passes may round exact ties up where compute rounds them to even, and a
// one-step difference can carry into the next level). Float formats count steps of
// their content's range.
// Run from the repository root with: npm run test-webgpu-mipmaps

const server = createServer();
let browser;

try {

	await new Promise( resolve => server.listen( 0, resolve ) );
	browser = await puppeteer.launch( { headless: true, args: [ '--enable-unsafe-webgpu', '--ignore-gpu-blocklist' ] } );

	// The default device (six levels per dispatch where the adapter allows), the default
	// storage texture limit (four), and compatibility mode (render passes only).

	const tolerance = 1 + 1e-6;
	let total = 0;

	for ( const query of [ '', '?limit=4', '?compat' ] ) {

		const page = await browser.newPage();
		const pageErrors = [];
		page.on( 'pageerror', error => pageErrors.push( error.message ) );
		await page.goto( `http://localhost:${ server.address().port }/test/webgpu/mipmaps-compute.html${ query }` );
		await page.waitForFunction( () => window.testResults !== undefined, { timeout: 300000 } );
		const { adapter, compatibilityMode, maxStorageTexturesPerShaderStage, levelsPerDispatch, errors, results } = await page.evaluate( () => window.testResults );
		await page.close();

		console.log( `\n${ query || 'default' }: ${ adapter }${ compatibilityMode ? ', compatibility mode' : '' }; maxStorageTexturesPerShaderStage ${ maxStorageTexturesPerShaderStage }, ${ levelsPerDispatch ?? 'no' } levels per dispatch.` );

		for ( const result of results ) {

			const ok = result.path === result.expectedPath && result.stepSteps <= tolerance && result.chainSteps <= 2 * tolerance;
			console.log( `${ ok ? 'ok  ' : 'FAIL' } ${ result.name } [${ result.format }, ${ result.path }]: per level ${ result.stepSteps.toFixed( 3 ) }, chain ${ result.chainSteps.toFixed( 3 ) } steps` );

			for ( const { level, size, step, chain } of result.levels.slice( 1 ) ) {

				if ( process.argv.includes( '--verbose' ) ) console.log( `       L${ level } ${ size }: per level ${ step === null ? '-' : `${ step.maxSteps.toFixed( 3 ) } (${ step.differing } differ, ${ step.ties } at ties)` }; chain ${ chain.maxSteps.toFixed( 3 ) } (${ chain.differing } differ)` );

			}

		}

		assert.deepEqual( pageErrors, [], `${ query }: the page must run without errors` );
		assert.deepEqual( errors, [], `${ query }: there must be no GPU validation errors` );

		for ( const result of results ) {

			assert.equal( result.path, result.expectedPath, `${ result.name }: expected ${ result.expectedPath } mipmaps` );
			assert.equal( result.storage, result.expectedPath === 'compute', `${ result.name }: STORAGE_BINDING usage only for compute mipmaps` );
			assert.ok( result.levelZeroEqual, `${ result.name }: level 0 must be identical` );
			assert.ok( result.stepSteps <= tolerance, `${ result.name }: a compute level differs by ${ result.stepSteps } 8-bit steps from a render pass of the same input` );
			assert.ok( result.chainSteps <= 2 * tolerance, `${ result.name }: the compute chain differs by ${ result.chainSteps } 8-bit steps from the render-pass chain` );

		}

		console.log( `${ results.length } cases passed.` );
		total += results.length;

	}

	console.log( `\n${ total } cases passed.` );

} finally {

	if ( browser ) await browser.close();
	server.close();

}
