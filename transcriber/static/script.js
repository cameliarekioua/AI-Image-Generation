let mediaRecorder;
let audioChunks = [];

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');
const transcript = document.getElementById('transcript');
const imageContainer = document.getElementById('imageContainer');

startBtn.onclick = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    audioChunks = [];
    mediaRecorder.ondataavailable = event => audioChunks.push(event.data);

    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
        const extension = mediaRecorder.mimeType.split('/')[1] || 'webm';
        const formData = new FormData();
        formData.append('file', audioBlob, `recording.${extension}`);

        status.textContent = 'Transcribing and generating image...';
        transcript.textContent = '';
        imageContainer.innerHTML = '';

        try {
            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData
            });

            const rawText = await response.text();
            console.log('Raw response:', rawText);

            if (!response.ok) {
                transcript.textContent = `Error: ${response.status} ${response.statusText}`;
                status.textContent = '';
                return;
            }

            const data = JSON.parse(rawText);
            transcript.textContent = data.text || 'No transcription found';

            if (data.image_base64) {
                const img = document.createElement('img');
                img.src = 'data:image/png;base64,' + data.image_base64;
                imageContainer.appendChild(img);
            } else {
                imageContainer.innerHTML = '<p>No image generated.</p>';
            }
        } catch (err) {
            console.error('Fetch error:', err);
            transcript.textContent = 'Failed to fetch or parse response';
        } finally {
            status.textContent = '';
        }
    };

    mediaRecorder.start();
    status.textContent = 'Recording...';
    startBtn.disabled = true;
    stopBtn.disabled = false;
};

stopBtn.onclick = () => {
    mediaRecorder.stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    status.textContent = 'Processing...';
};
