import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, Upload, Trash2, Send, FileAudio, Sparkles, UserPlus, Search, ArrowLeft, Building2, MapPin, Briefcase } from 'lucide-react';
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

  // --- ÉTATS CLIENTS ---
  const [showClientModal, setShowClientModal] = useState(false);
  const [searchClient, setSearchClient] = useState("");
  const [selectedClient, setSelectedClient] = useState(null);
  const [clients, setClients] = useState([]);
  const [newClient, setNewClient] = useState({ 
    nom: '', 
    email: '', 
    telephone: '',
    company_name: '',
    industry: '',
    location: ''
  });

  useEffect(() => {
    if (!agentId) navigate("/login");
  }, [agentId, navigate]);

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
    } catch (err) { console.error("Micro refusé"); }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
      setIsRecording(false);
    }
  };

  // --- LOGIQUE CRÉATION CLIENT ---
  const handleCreateClient = async (e) => {
    e.preventDefault();
    if (!newClient.nom || !agentId) return;

    try {
      const { data, error } = await supabase
        .from('clients')
        .insert([{ ...newClient, agent_id: agentId }])
        .select().single();

      if (error) throw error;
      setClients([...clients, data]);
      setSelectedClient(data);
      setSearchClient(data.nom);
      setShowClientModal(false);
      setNewClient({ nom: '', email: '', telephone: '', company_name: '', industry: '', location: '' });
    } catch (error) {
      console.error(error.message);
    }
  };

  // --- ENVOI AU BACKEND (FLASK) ---
  const sendToBackend = async () => {
    if (!selectedClient || !fileToUpload) return;

    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append("audio_file", fileToUpload, "capture_audio.wav");

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData);

      const { error } = await supabase.from("analyses").insert([
        {
          client_id: selectedClient.id,
          transcription: response.data.transcription,
          score_o: response.data.traits.Openness,
          score_c: response.data.traits.Conscientiousness,
          score_e: response.data.traits.Extraversion,
          score_a: response.data.traits.Agreeableness,
          score_n: response.data.traits.Neuroticism,
          commentaire_ia: "Analyse multimodale automatique"
        }
      ]);

      if (error) throw error;
      navigate("/stats");
    } catch (error) {
      console.error("Analyse échouée", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center p-6 relative overflow-hidden bg-gray-50">
      
      {/* Background decoration */}
      <div className="absolute top-8 left-8 z-20">
        <button onClick={() => navigate('/dashboard')} className="flex items-center gap-2 text-sm font-bold text-gray-400 hover:text-indigo-600 transition-colors">
          <ArrowLeft size={18} /> Dashboard
        </button>
      </div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        className="relative z-10 bg-white w-full max-w-xl rounded-[3rem] shadow-2xl p-10 border border-gray-100"
      >
        <header className="text-center mb-10">
          <div className="inline-flex p-3 bg-indigo-50 rounded-2xl mb-4 text-indigo-600">
            <Sparkles size={24} />
          </div>
          <h1 className="text-3xl font-black text-gray-900">Nouvelle Analyse</h1>
          <p className="text-gray-400 font-medium">Enregistrez ou importez la voix du client</p>
        </header>

        <AnimatePresence mode="wait">
          {!audioSource ? (
            <motion.div key="input" className="space-y-8">
              <div className="flex flex-col items-center">
                <div className="relative">
                  {isRecording && <motion.div animate={{ scale: [1, 1.8], opacity: [0.3, 0] }} transition={{ repeat: Infinity, duration: 1.5 }} className="absolute inset-0 bg-red-100 rounded-full" />}
                  <button onClick={isRecording ? stopRecording : startRecording}
                    className={`relative z-10 w-24 h-24 rounded-full flex items-center justify-center transition-all ${
                      isRecording ? 'bg-red-500 text-white shadow-xl shadow-red-200' : 'bg-indigo-600 text-white shadow-xl shadow-indigo-100 hover:scale-105'
                    }`}>
                    {isRecording ? <Square fill="currentColor" size={24} /> : <Mic size={32} />}
                  </button>
                </div>
                <span className={`mt-6 text-3xl font-black tabular-nums ${isRecording ? 'text-red-500' : 'text-gray-200'}`}>
                  {isRecording ? formatTime(timer) : "0:00"}
                </span>
              </div>

              <label className="flex flex-col items-center justify-center h-24 border-2 border-dashed border-gray-200 rounded-2xl cursor-pointer hover:bg-gray-50 transition-all">
                <Upload size={20} className="text-gray-300 mb-1" />
                <span className="text-gray-400 font-bold text-[10px] uppercase tracking-widest">Importer Audio</span>
                <input type="file" className="hidden" accept="audio/*" onChange={(e) => {
                  const f = e.target.files[0];
                  if(f){ setFileToUpload(f); setAudioSource(URL.createObjectURL(f)); }
                }} />
              </label>
            </motion.div>
          ) : (
            <motion.div key="preview" className="space-y-6">
              <div className="bg-gray-50 p-6 rounded-[2rem] border border-gray-100 space-y-4">
                <div className="flex items-center justify-between">
                  <label className="text-[10px] font-black uppercase tracking-widest text-gray-400">Sélection Client</label>
                  <button onClick={() => setShowClientModal(true)} className="text-[10px] font-black text-indigo-600 flex items-center gap-1 uppercase hover:underline">
                    <UserPlus size={12} /> Nouveau
                  </button>
                </div>
                <div className="relative">
                  <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-300" />
                  <input type="text" placeholder="Rechercher..." value={searchClient}
                    onChange={(e) => {setSearchClient(e.target.value); setSelectedClient(null);}}
                    className="w-full pl-10 pr-4 py-4 bg-white rounded-xl border border-gray-100 focus:ring-2 focus:ring-indigo-500 outline-none font-bold text-gray-700"
                  />
                  {searchClient && !selectedClient && (
                    <div className="absolute z-50 w-full mt-2 bg-white rounded-xl shadow-xl border border-gray-100 max-h-40 overflow-y-auto">
                      {filteredClients.map(c => (
                        <button key={c.id} onClick={() => {setSelectedClient(c); setSearchClient(c.nom);}}
                          className="w-full text-left px-4 py-3 hover:bg-indigo-50 font-bold text-gray-700 text-sm">
                          {c.nom}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="bg-gray-50 p-4 rounded-[2rem] border border-gray-100">
                <div className="flex justify-between items-center mb-2 px-2">
                    <span className="text-[10px] font-black text-gray-400 uppercase tracking-widest">Lecteur Audio</span>
                    <button onClick={() => {setAudioSource(null); setFileToUpload(null);}} className="text-red-400 hover:text-red-600"><Trash2 size={16}/></button>
                </div>
                <audio src={audioSource} controls className="w-full" />
              </div>

              <button disabled={isAnalyzing || !selectedClient} onClick={sendToBackend}
                className="w-full py-5 bg-indigo-600 text-white rounded-2xl font-black flex items-center justify-center gap-3 hover:bg-indigo-700 transition-all disabled:bg-gray-200"
              >
                {isAnalyzing ? "Traitement IA en cours..." : "Lancer l'analyse"}
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* --- MODALE CLIENT AVEC LES NOUVEAUX CHAMPS --- */}
      <AnimatePresence>
        {showClientModal && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-6 bg-black/20 backdrop-blur-sm">
            <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white w-full max-w-lg rounded-[2.5rem] p-8 shadow-2xl overflow-y-auto max-h-[90vh]"
            >
              <h2 className="text-2xl font-black text-gray-900 mb-6">Nouveau Profil Client</h2>
              <form onSubmit={handleCreateClient} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-[10px] font-black text-gray-400 uppercase ml-1">Nom complet *</label>
                    <input required type="text" value={newClient.nom} onChange={(e) => setNewClient({...newClient, nom: e.target.value})} className="w-full px-5 py-3 bg-gray-50 rounded-xl border-none outline-none focus:ring-2 focus:ring-indigo-500 font-bold" />
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] font-black text-gray-400 uppercase ml-1">Email</label>
                    <input type="email" value={newClient.email} onChange={(e) => setNewClient({...newClient, email: e.target.value})} className="w-full px-5 py-3 bg-gray-50 rounded-xl border-none outline-none focus:ring-2 focus:ring-indigo-500 font-bold" />
                  </div>
                </div>

                <div className="space-y-1">
                    <label className="text-[10px] font-black text-gray-400 uppercase ml-1">Téléphone</label>
                    <input type="tel" value={newClient.telephone} onChange={(e) => setNewClient({...newClient, telephone: e.target.value})} className="w-full px-5 py-3 bg-gray-50 rounded-xl border-none outline-none focus:ring-2 focus:ring-indigo-500 font-bold" />
                </div>

                <div className="pt-2 border-t border-gray-100 space-y-4">
                    <div className="relative">
                        <Building2 className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-300" size={18} />
                        <input type="text" placeholder="Nom de l'entreprise" value={newClient.company_name} onChange={(e) => setNewClient({...newClient, company_name: e.target.value})} className="w-full pl-12 pr-5 py-4 bg-gray-50 rounded-xl outline-none focus:ring-2 focus:ring-indigo-500 font-bold" />
                    </div>
                    <div className="relative">
                        <Briefcase className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-300" size={18} />
                        <input type="text" placeholder="Secteur d'activité" value={newClient.industry} onChange={(e) => setNewClient({...newClient, industry: e.target.value})} className="w-full pl-12 pr-5 py-4 bg-gray-50 rounded-xl outline-none focus:ring-2 focus:ring-indigo-500 font-bold" />
                    </div>
                    <div className="relative">
                        <MapPin className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-300" size={18} />
                        <input type="text" placeholder="Localisation" value={newClient.location} onChange={(e) => setNewClient({...newClient, location: e.target.value})} className="w-full pl-12 pr-5 py-4 bg-gray-50 rounded-xl outline-none focus:ring-2 focus:ring-indigo-500 font-bold" />
                    </div>
                </div>

                <div className="flex gap-3 pt-4">
                  <button type="button" onClick={() => setShowClientModal(false)} className="flex-1 py-4 bg-gray-100 text-gray-500 rounded-xl font-black">Annuler</button>
                  <button type="submit" className="flex-1 py-4 bg-indigo-600 text-white rounded-xl font-black shadow-lg">Créer Profil</button>
                </div>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default CapturePage;