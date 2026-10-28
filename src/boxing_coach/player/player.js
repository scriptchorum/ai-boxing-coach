export default function ({data, parentElement}) {
  const video = parentElement.querySelector('video');
  const canvas = parentElement.querySelector('canvas');
  const status = parentElement.querySelector('.status');
  const context = canvas.getContext('2d');
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const mirror = bytes[0] === 1;
  const url = URL.createObjectURL(new Blob([bytes.subarray(1)], {type:'video/mp4'}));
  const supportsFrames = typeof video.requestVideoFrameCallback === 'function';
  let active = true, frameCallback = null, animation = null, displayedTime = 0;
  const events = [];
  function listen(event, handler) {
    video.addEventListener(event, handler);
    events.push([event, handler]);
  }
  function draw(seconds) {
    if (!active || video.seeking || video.readyState < 2 || !video.videoWidth) return;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.style.aspectRatio = `${canvas.width}/${canvas.height}`;
    }
    context.setTransform(mirror ? -1 : 1, 0, 0, 1, mirror ? canvas.width : 0, 0);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    context.resetTransform();
    displayedTime = seconds;
    canvas.style.opacity = '1';
    status.textContent = `${video.paused ? '已暂停' : '播放中'} · 同步画面时间 ${seconds.toFixed(3)} 秒 · ${mirror ? '右侧已纠正镜像' : '保留原画面'}${supportsFrames ? '' : ' · 浏览器兼容模式，时间为近似值'}`;
  }
  function presented(now, metadata) {
    draw(metadata.mediaTime);
    if (active) frameCallback = video.requestVideoFrameCallback(presented);
  }
  function animate() {
    draw(video.currentTime);
    if (active && !video.paused) animation = requestAnimationFrame(animate);
  }
  listen('loadeddata', () => draw(video.currentTime));
  listen('seeked', () => draw(video.currentTime));
  listen('seeking', () => {
    canvas.style.opacity = '.25';
    status.textContent = '正在定位，右侧旧画面暂时变暗…';
  });
  listen('pause', () => draw(displayedTime));
  listen('ended', () => draw(displayedTime));
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
