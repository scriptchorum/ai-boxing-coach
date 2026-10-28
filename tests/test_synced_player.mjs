import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const source = await readFile(new URL('../src/boxing_coach/player/player.js', import.meta.url), 'utf8');
const {default: mount} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

function fixture({mirror = false, callback = true} = {}) {
  const listeners = new Map();
  let frame, raf, canceled = 0, draws = 0, cleared = 0, transform;
  globalThis.requestAnimationFrame = fn => {raf = fn; return 17;};
  globalThis.cancelAnimationFrame = () => {raf = null;};
  const video = {
    paused:true, seeking:false, readyState:2, videoWidth:720, videoHeight:1280, currentTime:0,
    addEventListener(e,fn) {listeners.set(e,fn);}, removeEventListener(e) {listeners.delete(e);},
    pause() {this.paused = true;}, load() {}, removeAttribute() {this.src = '';},
  };
  if (callback) {
    video.requestVideoFrameCallback = fn => {frame = fn; return 9;};
    video.cancelVideoFrameCallback = () => {canceled++;};
  }
  const context = {setTransform(...args) {transform=args;}, resetTransform() {}, drawImage() {draws++;}, clearRect() {cleared++;}};
  const canvas = {width:0, height:0, style:{}, getContext:()=>context};
  const status = {textContent:''};
  const cleanup = mount({data:new Uint8Array([mirror ? 1 : 0, 1, 2]),parentElement:{querySelector:s=>({video,canvas,'.status':status}[s])}});
  return {video, canvas, status, cleanup, listeners, emit:e=>listeners.get(e)?.(), present:t=>frame(0,{mediaTime:t}), animate:()=>raf?.(), get draws(){return draws;}, get cleared(){return cleared;}, get canceled(){return canceled;}, get transform(){return transform;}};
}

test('presented frames redraw the canvas with media time, not a fixed FPS timer', () => {
  const f=fixture();
  f.video.paused=false; f.present(1.217); f.present(1.280);
  assert.equal(f.draws,2); assert.match(f.status.textContent,/1\.280/);
  assert.equal(f.canvas.width,720); assert.equal(f.canvas.height,1280);
  f.cleanup();
});
test('seek dims stale frame and redraws after seek while paused', () => {
  const f=fixture(); f.present(0.5);
  f.video.seeking=true; f.emit('seeking'); f.present(1);
  assert.equal(f.draws,1); assert.equal(f.canvas.style.opacity,'.25');
  f.video.seeking=false; f.video.currentTime=3; f.emit('seeked');
  assert.equal(f.draws,2); assert.match(f.status.textContent,/3\.000/);
  f.cleanup();
});
test('mirror affects canvas only and cleanup detaches callbacks', () => {
  const f=fixture({mirror:true}); f.present(2);
  assert.deepEqual(f.transform,[-1,0,0,1,720,0]);
  const count=f.draws; f.cleanup(); f.present(3);
  assert.equal(f.draws,count); assert.equal(f.canceled,1); assert.equal(f.listeners.size,0);
  assert.equal(f.video.src,''); assert.equal(f.cleared,1);
});
test('fallback animation follows playback and labels approximate time', () => {
  const f=fixture({callback:false}); f.video.paused=false; f.video.currentTime=1.5;
  f.emit('play'); f.animate();
  assert.match(f.status.textContent,/1\.500/); assert.match(f.status.textContent,/近似值/);
  f.cleanup();
});
test('decode error clears old pixels and offers precise review', () => {
  const f=fixture(); f.emit('error');
  assert.equal(f.cleared,1); assert.match(f.status.textContent,/手动逐帧/);
  f.cleanup();
});
