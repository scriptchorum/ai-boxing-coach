export default function ({data, parentElement}) {
  const video = parentElement.querySelector('video');
  const canvas = parentElement.querySelector('canvas');
  const status = parentElement.querySelector('.status');
  const context = canvas.getContext('2d');
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const mirror = (bytes[0] & 1) === 1;
  let offset = 1, poses = null;
  if (bytes[0] & 2) {
    const length = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(1,true);
    poses = JSON.parse(new TextDecoder().decode(bytes.subarray(5,5+length)));
    offset = 5+length;
    parentElement.querySelector('.hint').textContent = '点击左侧 ▶ 播放，右侧显示已识别区间的骨架。骨架是模型估计，可能跟到背景人物或器械，不代表动作评价。';
  }
  const url = URL.createObjectURL(new Blob([bytes.subarray(offset)], {type:'video/mp4'}));
  const supportsFrames = typeof video.requestVideoFrameCallback === 'function';
  let active = true, frameCallback = null, animation = null, displayedTime = 0, confirmed = false;
  const events = [];
  function listen(event, handler) {
    video.addEventListener(event, handler);
    events.push([event, handler]);
  }
  function poseAt(seconds) {
    // Match an actual presented PTS, never interpolate or carry the previous pose.
    let lo=0, hi=poses.frames.length;
    while (lo<hi) {
      const mid=(lo+hi)>>1;
      if (poses.frames[mid].time<seconds) lo=mid+1; else hi=mid;
    }
    return [poses.frames[lo],poses.frames[lo-1]].find(f=>f && Math.abs(f.time-seconds)<0.001);
  }
  function draw(seconds, exact=false) {
    if (!active || video.seeking || video.readyState < 2 || !video.videoWidth) return;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.style.aspectRatio = `${canvas.width}/${canvas.height}`;
    }
    context.setTransform(mirror ? -1 : 1, 0, 0, 1, mirror ? canvas.width : 0, 0);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    context.resetTransform();
    let poseStatus = '';
    if (poses) {
      const frame = exact && supportsFrames ? poseAt(seconds) : null;
      if (frame?.status === 'single_model_pose') {
        const points=frame.points;
        const pixel=p=>[p[0]*canvas.width,p[1]*canvas.height];
        // Points already use corrected image coordinates: do not mirror them twice.
        context.strokeStyle='#00dcff'; context.lineWidth=Math.max(2,canvas.width/300);
        for (const [a,b] of poses.edges) if (points[a] && points[b]) {
          context.beginPath(); context.moveTo(...pixel(points[a])); context.lineTo(...pixel(points[b])); context.stroke();
        }
        context.fillStyle='#ffb400';
        for (const p of Object.values(points)) {
          context.beginPath(); context.arc(...pixel(p),Math.max(3,canvas.width/180),0,2*Math.PI); context.fill();
        }
        poseStatus=` · 骨架源帧 ${frame.index}（模型估计）`;
      } else if (frame) {
        poseStatus=frame.status==='multiple_poses' ? ' · 多个候选，骨架已隐藏' : ' · 未检出，骨架已隐藏';
      } else {
        poseStatus=supportsFrames ? ' · 此画面未匹配已识别帧，骨架已隐藏' : ' · 此浏览器不支持精确帧回调，请用下方逐帧骨架核对';
      }
    }
    displayedTime = seconds;
    canvas.style.opacity = '1';
    status.textContent = `${video.paused ? '已暂停' : '播放中'} · 同步画面时间 ${seconds.toFixed(3)} 秒 · ${mirror ? '右侧已纠正镜像' : '保留原画面'}${supportsFrames ? '' : ' · 浏览器兼容模式，时间为近似值'}${poseStatus}`;
  }
  function presented(now, metadata) {
    confirmed = !video.seeking;
    draw(metadata.mediaTime, confirmed);
    if (active) frameCallback = video.requestVideoFrameCallback(presented);
  }
  function animate() {
    draw(video.currentTime);
    if (active && !video.paused) animation = requestAnimationFrame(animate);
  }
  listen('loadeddata', () => draw(video.currentTime));
  listen('seeked', () => { if (!confirmed) draw(video.currentTime); });
  listen('loadedmetadata', () => {
    if (poses?.frames.length) video.currentTime = poses.frames[0].time;
  });
  listen('seeking', () => {
    confirmed = false;
    canvas.style.opacity = '.25';
    status.textContent = '正在定位，右侧旧画面暂时变暗…';
  });
  function stopped() {
    // Keep the last frame/pose pair drawn by the presentation callback. Reading
    // video again here could pair a newly decoded image with the previous PTS.
    if (poses && confirmed) status.textContent = status.textContent.replace(/^播放中/,'已暂停');
    else draw(confirmed ? displayedTime : video.currentTime,confirmed);
  }
  listen('pause', stopped);
  listen('ended', stopped);
  listen('play', () => {
    if (!supportsFrames) {
      cancelAnimationFrame(animation);
      animation = requestAnimationFrame(animate);
    }
  });
  listen('error', () => {
    context.clearRect(0, 0, canvas.width, canvas.height);
    status.textContent = '浏览器无法播放此视频，请切换手动逐帧查看。';
  });
  if (supportsFrames) frameCallback = video.requestVideoFrameCallback(presented);
  video.src = url;
  return () => {
    active = false;
    for (const [event, handler] of events) video.removeEventListener(event, handler);
    if (frameCallback !== null) video.cancelVideoFrameCallback(frameCallback);
    if (animation !== null) cancelAnimationFrame(animation);
    video.pause();
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(url);
    context.clearRect(0, 0, canvas.width, canvas.height);
  };
}
