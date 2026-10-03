// A scoped browser adapter for native clipboard, drag/drop and IME behavior.
// File contents never leave the browser until the user submits the composer.
const id = await dioxus.recv();
const root = document.getElementById(id);
if (!root) return;
window.__rustinferComposers ??= new Map();
window.__rustinferComposers.get(id)?.();
const controller = new AbortController();
const options = { signal: controller.signal };
const input = root.querySelector('textarea');
const picker = root.querySelector('input[type=file]');
const readers = new Set();
let dragDepth = 0;
let composing = false;
let compositionEnded = -Infinity;
let pendingFiles = 0;
let closed = false;
let queue = Promise.resolve();
const emit = (event) => { if (!closed) dioxus.send(event); };
const error = (message) => emit({ kind: 'error', message });
const number = (name, fallback) => {
    const value = Number(root.dataset[name]);
    return Number.isFinite(value) && value >= 0 ? value : fallback;
};
const cleanup = () => {
    if (closed) return;
    closed = true;
    controller.abort();
    for (const reader of readers) reader.abort();
    readers.clear();
    window.__rustinferComposers.delete(id);
};
window.__rustinferComposers.set(id, cleanup);
const readDataURL = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    readers.add(reader);
    reader.onload = () => { readers.delete(reader); resolve(reader.result); };
    reader.onerror = () => { readers.delete(reader); reject(new Error('无法读取这张图片，请重试。')); };
    reader.onabort = () => { readers.delete(reader); reject(new Error('图片读取已取消。')); };
    reader.readAsDataURL(file);
});
const checkDimensions = (url, maxDimension) => new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
        const valid = image.naturalWidth > 0 && image.naturalHeight > 0
            && image.naturalWidth <= maxDimension && image.naturalHeight <= maxDimension;
        image.src = '';
        valid ? resolve() : reject(new Error(`图片宽高不能超过 ${maxDimension} 像素。`));
    };
    image.onerror = () => reject(new Error('图片无法解码，请重新选择 PNG 或 JPG 图片。'));
    image.src = url;
});
const addFiles = (files) => {
    if (closed || root.dataset.disabled === 'true') return;
    // Snapshot synchronously: clipboard/drop FileList may no longer be readable later.
    const selected = Array.from(files);
    for (const file of selected) {
        const maxImages = Math.min(number('maxImages', 4), 4);
        // Unsupported models still allow a local drop/paste preview; submission is gated in Rust.
        const previewLimit = maxImages || 4;
        if (number('attachmentCount', 0) + pendingFiles >= previewLimit) {
            error(`最多可添加 ${previewLimit} 张图片，请先移除已有图片。`);
            break;
        }
        pendingFiles += 1;
        emit({ kind: 'loading', active: true });
        queue = queue.then(async () => {
            try {
                if (closed) return;
                if (!['image/png', 'image/jpeg'].includes(file.type)) {
                    throw new Error('目前支持 PNG 和 JPG 图片，其他附件类型暂未启用。');
                }
                const maxBytes = Math.min(number('maxImageBytes', 10 * 1024 * 1024) || 10 * 1024 * 1024, 10 * 1024 * 1024);
                if (!file.size || file.size > maxBytes) {
                    throw new Error(`图片不能为空，且每张不超过 ${(maxBytes / (1024 * 1024)).toFixed(1)} MB。`);
                }
                const signature = new Uint8Array(await file.slice(0, 8).arrayBuffer());
                const isPng = [137, 80, 78, 71, 13, 10, 26, 10].every((byte, i) => signature[i] === byte);
                const isJpeg = signature[0] === 255 && signature[1] === 216 && signature[2] === 255;
                if ((file.type === 'image/png' && !isPng) || (file.type === 'image/jpeg' && !isJpeg)) {
                    throw new Error('图片内容与文件格式不符，请选择有效的 PNG 或 JPG 图片。');
                }
                const data_url = await readDataURL(file);
                await checkDimensions(data_url, Math.min(number('maxImageDimension', 8192) || 8192, 8192));
                emit({ kind: 'attachment', attachment: {
                    id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`,
                    name: file.name || `粘贴的图片.${isPng ? 'png' : 'jpg'}`,
                    mime_type: file.type,
                    size: file.size,
                    data_url,
                } });
            } catch (reason) {
                error(reason instanceof Error ? reason.message : '图片读取失败，请重试。');
            } finally {
                pendingFiles -= 1;
                if (!pendingFiles) emit({ kind: 'loading', active: false });
            }
        });
    }
};
picker.addEventListener('change', () => {
    addFiles(picker.files);
    picker.value = '';
}, options);
root.addEventListener('paste', (event) => {
    const files = event.clipboardData?.files;
    if (files?.length) {
        event.preventDefault();
        addFiles(files);
    }
}, options);
root.addEventListener('dragenter', (event) => {
    if (!Array.from(event.dataTransfer?.types || []).includes('Files')) return;
    event.preventDefault();
    dragDepth += 1;
    emit({ kind: 'drag', active: true });
}, options);
root.addEventListener('dragover', (event) => {
    if (!Array.from(event.dataTransfer?.types || []).includes('Files')) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
}, options);
root.addEventListener('dragleave', (event) => {
    event.preventDefault();
    dragDepth = Math.max(dragDepth - 1, 0);
    if (!dragDepth) emit({ kind: 'drag', active: false });
}, options);
root.addEventListener('drop', (event) => {
    if (!event.dataTransfer?.files.length) return;
    event.preventDefault();
    dragDepth = 0;
    emit({ kind: 'drag', active: false });
    addFiles(event.dataTransfer.files);
}, options);
input.addEventListener('compositionstart', () => { composing = true; }, options);
input.addEventListener('compositionend', () => {
    composing = false;
    compositionEnded = performance.now();
}, options);
input.addEventListener('keydown', (event) => {
    // Safari may report isComposing=false for the Enter that confirms IME input.
    if (event.key !== 'Enter' || event.shiftKey || event.isComposing || composing
        || event.keyCode === 229 || performance.now() - compositionEnded < 60) return;
    event.preventDefault();
    if (!event.repeat) root.querySelector('[data-composer-send]')?.click();
}, options);
input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 200) + 'px';
}, options);
document.addEventListener('pointerdown', (event) => {
    if (!root.querySelector('.composer-attach-control')?.contains(event.target)) {
        emit({ kind: 'dismiss_menu' });
    }
}, options);
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') emit({ kind: 'dismiss_menu' });
}, options);
// Keep this eval's message channel alive until the component is unmounted.
await new Promise((resolve) => controller.signal.addEventListener('abort', resolve, { once: true }));
