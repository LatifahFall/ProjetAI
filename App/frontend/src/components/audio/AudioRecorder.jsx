// src/components/audio/AudioRecorder.jsx
import { useEffect, useRef, useState } from "react";

export default function AudioRecorder({ onRecorded }) {
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);

  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState("");
  const [audioUrl, setAudioUrl] = useState("");

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    };
  }, [audioUrl]);

  const start = async () => {
    setError("");
    setAudioUrl("");
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mediaRecorderRef.current = mr;

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        onRecorded?.(blob, url);
      };

      mr.start();
      setIsRecording(true);
    } catch (e) {
      setError("Micro non autorisé ou indisponible.");
    }
  };

  const stop = () => {
    const mr = mediaRecorderRef.current;
    if (!mr) return;

    mr.stop();
    setIsRecording(false);

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  };

  return (
    <div className="bg-white rounded-3xl border border-gray-100 p-8">
      <h2 className="font-black text-lg text-slate-900 mb-2">🎙️ Enregistrement vocal</h2>
      <p className="text-sm text-slate-500 mb-6">
        Parlez naturellement. À l’arrêt, on lance l’analyse IA.
      </p>

      {error && (
        <div className="mb-4 text-sm font-semibold text-red-600 bg-red-50 border border-red-100 rounded-xl p-3">
          {error}
        </div>
      )}

      <div className="flex flex-wrap gap-3">
        {!isRecording ? (
          <button
            onClick={start}
            className="bg-[#646cff] text-white px-6 py-3 rounded-2xl font-black hover:bg-[#535bf2] transition"
          >
            Démarrer
          </button>
        ) : (
          <button
            onClick={stop}
            className="bg-gray-900 text-white px-6 py-3 rounded-2xl font-black hover:opacity-90 transition"
          >
            Stop
          </button>
        )}

        {audioUrl && (
          <a
            href={audioUrl}
            download="recording.webm"
            className="px-6 py-3 rounded-2xl font-black border border-gray-200 hover:bg-gray-50 transition"
          >
            Télécharger
          </a>
        )}
      </div>

      {audioUrl && (
        <div className="mt-6">
          <audio controls src={audioUrl} className="w-full" />
        </div>
      )}
    </div>
  );
}
