import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Upload, Trash2, Send, FileAudio, Sparkles, Activity, UserPlus, Search, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { supabase } from '../lib/supabaseClient';
import { useNavigate } from "react-router-dom";

const CapturePage = () => {
  const agentId = localStorage.getItem('agent_id');
  const navigate = useNavigate();

  // --- ÉTATS ---
  const [isRecording, setIsRecording] = useState(false);
  const [audioSource, setAudioSource] = useState(null);
  const [fileToUpload, setFileToUpload] = useState(null);
  const [timer, setTimer] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // --- ÉTATS CLIENTS (SUPABASE) ---
  const [showClientModal, setShowClientModal] = useState(false);
  const [searchClient, setSearchClient] = useState("");
  const [selectedClient, setSelectedClient] = useState(null);
  const [clients, setClients] = useState([]);
  const [newClient, setNewClient] = useState({ nom: '', email: '', telephone: '' });


  useEffect(() => {
    if (!agentId) {
      window.location.href = "/login";
    }
  }, [agentId]);

  // 1. Charger les clients au démarrage
  useEffect(() => {
    fetchClients();
  }, [agentId]);

  const fetchClients = async () => {
    if (!agentId) return;
    const { data, error } = await supabase
      .from('clients')
      .select('*')
      .eq('agent_id', agentId)
      .order('nom', { ascending: true });
    
    if (!error) setClients(data);
  };

  const filteredClients = clients.filter(c => 
    c.nom?.toLowerCase().includes(searchClient.toLowerCase())
  );

  // --- RÉFÉRENCES & CHRONO ---
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => setTimer(t => t + 1), 1000);
    } else {
      clearInterval(timerRef.current);
      setTimer(0);
    }
    return () => clearInterval(timerRef.current);
  }, [isRecording]);

  const formatTime = (s) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;

  // --- LOGIQUE AUDIO ---
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaRecorderRef.current.ondataavailable = (e) => audioChunksRef.current.push(e.data);
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setFileToUpload(blob);
        setAudioSource(URL.createObjectURL(blob));
      };
      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) { alert("Accès micro refusé"); }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
      setIsRecording(false);
    }
  };

  // --- LOGIQUE CLIENT SUPABASE ---
  const handleCreateClient = async (e) => {
    e.preventDefault();
    if (!newClient.nom || !agentId) return;

    try {
      const { data, error } = await supabase
        .from('clients')
        .insert([{
          nom: newClient.nom,
          email: newClient.email,
          telephone: newClient.telephone,
          agent_id: agentId
        }])
        .select()
        .single();

      if (error) throw error;
      setClients([...clients, data]);
      setSelectedClient(data);
      setSearchClient(data.nom);
      setShowClientModal(false);
      setNewClient({ nom: '', email: '', telephone: '' });
    } catch (error) {
      alert("Erreur Supabase : " + error.message);
    }
  };

  // --- ENVOI AU BACKEND (FLASK) ---
  const sendToBackend = async () => {
  if (!selectedClient) {
    alert("Veuillez sélectionner un client");
    return;
  }

  if (!fileToUpload) {
    alert("Veuillez capturer ou importer un audio");
    return;
  }

  if (!agentId) {
    alert("Agent non connecté");
    return;
  }

  setIsAnalyzing(true);

  const formData = new FormData();
  formData.append("audio_file", fileToUpload, "capture_audio.wav");

  try {
    // 1️⃣ Appel IA (Flask)
    const response = await axios.post(
      "http://127.0.0.1:5000/predict",
      formData,
      { headers: { "Content-Type": "multipart/form-data" } }
    );

    console.log("Réponse Backend:", response.data);

    // 2️⃣ ENREGISTREMENT DANS SUPABASE ✅
   const { error } = await supabase.from("analyses").insert([
  {
    client_id: selectedClient.id,
    transcription: response.data.transcription,

    score_o: response.data.traits.Openness,
    score_c: response.data.traits.Conscientiousness,
    score_e: response.data.traits.Extraversion,
    score_a: response.data.traits.Agreeableness,
    score_n: response.data.traits.Neuroticism,

    commentaire_ia: "Analyse automatique"
  }
]);
    if (error) {
      console.error("❌ Erreur insertion analyses :", error.message);
      alert("Erreur lors de l'enregistrement de l'analyse");
      return;
    }

    if (error) {
      console.error("Erreur Supabase :", error);
      alert("❌ Supabase: " + error.message);
      return;
    }


    alert("✅ Analyse enregistrée avec succès !");
    navigate("/stats");
    } catch (error) {
      console.error("Erreur globale :", error);
      alert("❌ Erreur lors de l'analyse");
    } finally {
      setIsAnalyzing(false);
    }
};


  return (
    <div className="min-h-screen w-full flex items-center justify-center p-6 font-sans relative overflow-hidden bg-[#fdfeff]">
      
      {/* BACKGROUND DESIGN */}
      <div className="absolute inset-0 z-0">
        <motion.div animate={{ x: [0, 30, 0], y: [0, 50, 0] }} transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
          className="absolute top-[-10%] left-[-5%] w-[40%] h-[40%] bg-indigo-200/40 rounded-full blur-[120px]" />
        <motion.div animate={{ x: [0, -40, 0], y: [0, -60, 0] }} transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
          className="absolute bottom-[-10%] right-[-5%] w-[45%] h-[45%] bg-blue-100/60 rounded-full blur-[120px]" />
      </div>

      <div className="absolute inset-0 z-[1] opacity-[0.03] pointer-events-none bg-[url('https://grainy-gradients.vercel.app/noise.svg')]" />

      <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
        className="relative z-10 bg-white/80 backdrop-blur-2xl w-full max-w-xl rounded-[3.5rem] shadow-[0_32px_64px_-16px_rgba(0,0,0,0.08)] p-12 border border-white/40"
      >
        <header className="text-center mb-12 relative">
          <div className="inline-flex p-3 bg-indigo-50/50 rounded-2xl mb-4 text-indigo-600 border border-indigo-100/50">
            <Sparkles size={24} />
          </div>
          <h1 className="text-4xl font-black text-slate-900 tracking-tight">Analyse IA</h1>
          <p className="text-slate-500 mt-2 font-medium">Capturez et analysez la voix du client</p>
        </header>

        <AnimatePresence mode="wait">
          {!audioSource ? (
            <motion.div key="input" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-10">
              <div className="flex flex-col items-center justify-center">
                <div className="relative">
                  {isRecording && <motion.div animate={{ scale: [1, 2], opacity: [0.4, 0] }} transition={{ repeat: Infinity, duration: 1.5 }} className="absolute inset-0 bg-red-200 rounded-full" />}
                  <button onClick={isRecording ? stopRecording : startRecording}
                    className={`relative z-10 w-24 h-24 rounded-full flex items-center justify-center transition-all shadow-lg ${
                      isRecording ? 'bg-red-500 text-white shadow-red-200' : 'bg-indigo-600 text-white shadow-indigo-200 hover:scale-105'
                    }`}>
                    {isRecording ? <Square fill="currentColor" size={28} /> : <Mic size={36} />}
                  </button>
                </div>
                <div className="mt-6 text-center">
                  <span className={`text-3xl font-black tabular-nums tracking-tighter ${isRecording ? 'text-red-500' : 'text-slate-300'}`}>
                    {isRecording ? formatTime(timer) : "0:00"}
                  </span>
                </div>
              </div>

              <label className="flex flex-col items-center justify-center h-28 border-2 border-dashed border-slate-200 rounded-[2.5rem] cursor-pointer hover:bg-white/50 hover:border-indigo-300 transition-all group">
                <Upload size={24} className="text-slate-300 group-hover:text-indigo-500 mb-2 transition-colors" />
                <span className="text-slate-400 font-bold text-xs uppercase tracking-widest">Importer un fichier</span>
                <input type="file" className="hidden" accept="audio/*" onChange={(e) => {
                  const f = e.target.files[0];
                  if(f){ setFileToUpload(f); setAudioSource(URL.createObjectURL(f)); }
                }} />
              </label>
            </motion.div>
          ) : (
            <motion.div key="preview" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="space-y-6">
              
              <div className="bg-slate-50/50 p-6 rounded-[2.5rem] border border-slate-100 space-y-4 shadow-inner">
                <div className="flex items-center justify-between px-2">
                  <label className="text-[10px] font-black uppercase tracking-widest text-slate-400">Client</label>
                  <button onClick={() => setShowClientModal(true)} className="flex items-center gap-1 text-[10px] font-black text-indigo-600 uppercase hover:underline">
                    <UserPlus size={12} /> Nouveau
                  </button>
                </div>

                <div className="relative group">
                  <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-300 group-focus-within:text-indigo-500 pointer-events-none">
                    <Search size={18} />
                  </div>
                  <input type="text" placeholder="Chercher un client..." value={searchClient}
                    onChange={(e) => {setSearchClient(e.target.value); setSelectedClient(null);}}
                    className="w-full pl-12 pr-4 py-4 bg-white rounded-2xl border border-slate-100 focus:border-indigo-300 outline-none transition-all font-medium text-slate-700 shadow-sm"
                  />
                  
                  {searchClient && !selectedClient && (
                    <div className="absolute z-50 w-full mt-2 bg-white rounded-2xl shadow-xl border border-slate-100 max-h-40 overflow-y-auto p-2">
                      {filteredClients.length > 0 ? filteredClients.map(c => (
                        <button key={c.id} onClick={() => {setSelectedClient(c); setSearchClient(c.nom);}}
                          className="w-full text-left px-4 py-3 hover:bg-indigo-50 rounded-xl transition-colors font-bold text-slate-700 text-sm flex justify-between">
                          {c.nom}
                        </button>
                      )) : (
                        <p className="p-4 text-[10px] text-slate-400 text-center font-bold">Aucun résultat</p>
                      )}
                    </div>
                  )}
                </div>
              </div>

              <div className="bg-white/50 p-6 rounded-[2.5rem] border border-white shadow-inner">
                <div className="flex items-center justify-between mb-4">
                   <div className="flex items-center gap-3">
                     <div className="bg-indigo-600 p-2 rounded-xl text-white shadow-lg"><FileAudio size={18} /></div>
                     <span className="text-xs font-black text-slate-900 uppercase">Aperçu Audio</span>
                   </div>
                   <button onClick={() => {setAudioSource(null); setFileToUpload(null);}} className="text-slate-300 hover:text-red-500"><Trash2 size={18} /></button>
                </div>
                <audio src={audioSource} controls className="w-full mb-2" />
              </div>

              <button disabled={isAnalyzing || !selectedClient} onClick={sendToBackend}
                className="w-full py-5 bg-slate-900 text-white rounded-2xl font-black flex items-center justify-center gap-3 hover:bg-black transition-all shadow-xl disabled:bg-slate-200"
              >
                {isAnalyzing ? (
                  <div className="flex items-center gap-2">
                    <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1 }} className="h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                    Traitement...
                  </div>
                ) : (
                  <><Send size={18} /> Lancer l'analyse pour {selectedClient?.nom}</>
                )}
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* --- MODALE AJOUT CLIENT --- */}
      <AnimatePresence>
        {showClientModal && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-6">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setShowClientModal(false)} className="absolute inset-0 bg-slate-900/20 backdrop-blur-sm" />
            <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }}
              className="relative bg-white w-full max-w-md rounded-[3rem] p-10 shadow-2xl"
            >
              <h2 className="text-2xl font-black text-slate-900 mb-8">Nouveau Client</h2>
              <form onSubmit={handleCreateClient} className="space-y-4">
                <input required type="text" value={newClient.nom} onChange={(e) => setNewClient({...newClient, nom: e.target.value})} placeholder="Nom complet"
                  className="w-full px-6 py-4 bg-slate-50 rounded-2xl border border-slate-100 outline-none focus:border-indigo-300 font-bold" />
                <input type="email" value={newClient.email} onChange={(e) => setNewClient({...newClient, email: e.target.value})} placeholder="Email"
                  className="w-full px-6 py-4 bg-slate-50 rounded-2xl border border-slate-100 outline-none focus:border-indigo-300 font-bold" />
                <input type="tel" value={newClient.telephone} onChange={(e) => setNewClient({...newClient, telephone: e.target.value})} placeholder="Téléphone"
                  className="w-full px-6 py-4 bg-slate-50 rounded-2xl border border-slate-100 outline-none focus:border-indigo-300 font-bold" />
                <button type="submit" className="w-full py-5 bg-indigo-600 text-white rounded-2xl font-black hover:bg-indigo-700 transition-all">
                  Créer et Sélectionner
                </button>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default CapturePage;