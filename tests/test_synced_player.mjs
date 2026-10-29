import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const source = await readFile(new URL('../src/boxing_coach/player/player.js', import.meta.url), 'utf8');
const {default: mount} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

function fixture({mirror = false, callback = true, poses = null} = {}) {
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
  const lines=[], dots=[];
  const context = {setTransform(...args) {transform=args;}, resetTransform() {}, drawImage() {draws++;}, clearRect() {cleared++;},
    beginPath(){}, moveTo(){}, lineTo(...p){lines.push(p);},stroke(){},arc(...p){dots.push(p);},fill(){}};
  const canvas = {width:0, height:0, style:{}, getContext:()=>context};
  const status = {textContent:''};
  let data=new Uint8Array([mirror ? 1 : 0, 1, 2]);
  if (poses) {
    const json=new TextEncoder().encode(JSON.stringify(poses));
    data=new Uint8Array(5+json.length+2); data[0]=2+Number(mirror);
    new DataView(data.buffer).setUint32(1,json.length,true); data.set(json,5);
  }
  const cleanup = mount({data,parentElement:{querySelector:s=>({video,canvas,'.status':status,'.hint':{textContent:''}}[s])}});
  return {video, canvas, status, cleanup, listeners,lines,dots, emit:e=>listeners.get(e)?.(), present:t=>frame(0,{mediaTime:t}), animate:()=>raf?.(), get draws(){return draws;}, get cleared(){return cleared;}, get canceled(){return canceled;}, get transform(){return transform;}};
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

const poseFixture={edges:[[11,13]],frames:[
  {time:1.120,index:7,status:'single_model_pose',points:{11:[.2,.3],13:[.4,.5]}},
  {time:1.177,index:8,status:'multiple_poses',points:{}},
  {time:1.250,index:9,status:'no_pose',points:{}}
]};
test('quality rejection shows its reason and never draws points',()=>{
  const f=fixture({poses:{edges:[[11,13]],frames:[{...poseFixture.frames[0],status:'quality_rejected',reason:'身体关键点落在画面外'}]}});
  f.present(1.120); assert.equal(f.lines.length,0); assert.equal(f.dots.length,0);
  assert.match(f.status.textContent,/身体关键点落在画面外/); f.cleanup();
});
test('partial body display draws supplied limbs and names hidden parts',()=>{
  const f=fixture({poses:{edges:[[11,13],[23,25]],frames:[{...poseFixture.frames[0],hidden_parts:['左腿']}]}});
  f.present(1.120); assert.equal(f.lines.length,1); assert.match(f.status.textContent,/部分显示，未显示：左腿/); f.cleanup();
});
test('pose follows exact VFR presentation timestamps; unmatched and multiple frames hide it',()=>{
  const f=fixture({poses:poseFixture}); f.emit('loadedmetadata'); assert.equal(f.video.currentTime,1.120);
  f.present(1.120); assert.equal(f.lines.length,1); assert.match(f.status.textContent,/源帧 7/);
  f.present(1.150); assert.equal(f.lines.length,1); assert.match(f.status.textContent,/未匹配/);
  f.present(1.177); assert.equal(f.lines.length,1); assert.match(f.status.textContent,/多个候选/);
  f.present(1.250); assert.match(f.status.textContent,/未检出/);
  f.present(2); assert.match(f.status.textContent,/未匹配/); f.cleanup();
});
test('mirrored points are not mirrored twice; pause preserves exact frame',()=>{
  const f=fixture({poses:poseFixture,mirror:true}); f.present(1.120);
  assert.deepEqual(f.lines[0],[288,640]); f.emit('pause'); assert.match(f.status.textContent,/源帧 7/);
  f.video.seeking=true; f.emit('seeking'); f.video.seeking=false;
  f.video.currentTime=1.250; f.emit('seeked'); assert.match(f.status.textContent,/未匹配/);
  f.present(1.250); f.emit('pause'); assert.match(f.status.textContent,/未检出/); f.cleanup();
});
test('fallback browser never guesses an overlay from currentTime',()=>{
  const f=fixture({poses:poseFixture,callback:false}); f.video.currentTime=1.120;
  f.emit('loadeddata'); assert.equal(f.dots.length,0); assert.match(f.status.textContent,/不支持精确帧/); f.cleanup();
});
