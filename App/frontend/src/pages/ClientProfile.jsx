import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabaseClient';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { TrendingUp, User, Calendar, BarChart3, ArrowLeft } from 'lucide-react';
import { motion } from 'framer-motion';

const ClientProfile = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const agentId = localStorage.getItem('agent_id');
  
  const [clients, setClients] = useState([]);
  const [selectedClientId, setSelectedClientId] = useState(searchParams.get('clientId') || '');
  const [clientInfo, setClientInfo] = useState(null);
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [conversionScore, setConversionScore] = useState(null);

  useEffect(() => {
    if (!agentId) {
      navigate('/login');
      return;
    }
    fetchClients();
  }, [agentId, navigate]);

  useEffect(() => {
    if (selectedClientId) {
      fetchClientInfo();
      fetchClientHistory();
    } else {
      setClientInfo(null);
      setAnalyses([]);
      setConversionScore(null);
    }
  }, [selectedClientId]);

  const fetchClients = async () => {
    try {
      const { data, error } = await supabase
        .from('clients')
        .select('id, nom, company_name')
        .eq('agent_id', agentId)
        .order('nom', { ascending: true });

      if (error) throw error;
      setClients(data || []);
    } catch (error) {
      console.error('Erreur lors du chargement des clients:', error);
    }
  };

  const fetchClientInfo = async () => {
    try {
      const { data, error } = await supabase
        .from('clients')
        .select('*')
        .eq('id', selectedClientId)
        .single();

      if (error) throw error;
      setClientInfo(data);
    } catch (error) {
      console.error('Erreur lors du chargement des infos client:', error);
    }
  };

  const fetchClientHistory = async () => {
    setLoading(true);
    try {
      const { data, error } = await supabase
        .from('analyses')
        .select('id, date_analyse, score_o, score_c, score_e, score_a, score_n, transcription, commentaire_ia')
        .eq('client_id', selectedClientId)
        .order('date_analyse', { ascending: true });

      if (error) throw error;
      
      setAnalyses(data || []);
      calculateConversionScore(data || []);
    } catch (error) {
      console.error('Erreur lors du chargement de l\'historique:', error);
      setAnalyses([]);
    } finally {
      setLoading(false);
    }
  };

  const calculateConversionScore = (analysesData) => {
    if (!analysesData || analysesData.length === 0) {
      setConversionScore(null);
      return;
    }

    // Prendre la dernière analyse (ou moyenne si plusieurs)
    const latest = analysesData[analysesData.length - 1];
    const score_e = latest.score_e || 0;
    const score_a = latest.score_a || 0;
    const score_c = latest.score_c || 0;
    const score_o = latest.score_o || 0;
    const score_n = latest.score_n || 0;

    // Formule de conversion
    let score = (
      (score_e * 0.35) +      // Extraversion
      (score_a * 0.30) +      // Agreeableness
      (score_c * 0.25) +      // Conscientiousness
      (score_o * 0.10)        // Openness
    ) * 100;

    // Pénalité pour Neuroticism
    score = score * (1 - score_n * 0.2);

    // S'assurer que le score est entre 0 et 100
    score = Math.max(0, Math.min(100, score));

    setConversionScore(Math.round(score));
  };

  // Préparer les données pour le Radar Chart (dernière analyse ou moyenne)
  const getRadarData = () => {
    if (analyses.length === 0) return [];

    const latest = analyses[analyses.length - 1];
    return [
      { trait: 'Openness', value: latest.score_o || 0 },
      { trait: 'Conscientiousness', value: latest.score_c || 0 },
      { trait: 'Extraversion', value: latest.score_e || 0 },
      { trait: 'Agreeableness', value: latest.score_a || 0 },
      { trait: 'Neuroticism', value: latest.score_n || 0 }
    ];
  };

  // Préparer les données pour le Line Chart (évolution temporelle)
  const getLineChartData = () => {
    return analyses.map(analysis => ({
      date: new Date(analysis.date_analyse).toLocaleDateString('fr-FR', { 
        month: 'short', 
        day: 'numeric' 
      }),
      fullDate: analysis.date_analyse,
      Openness: analysis.score_o || 0,
      Conscientiousness: analysis.score_c || 0,
      Extraversion: analysis.score_e || 0,
      Agreeableness: analysis.score_a || 0,
      Neuroticism: analysis.score_n || 0
    }));
  };

  const getConversionScoreColor = () => {
    if (!conversionScore) return 'gray';
    if (conversionScore >= 80) return 'green';
    if (conversionScore >= 60) return 'orange';
    return 'red';
  };

  const getConversionScoreMessage = () => {
    if (!conversionScore) return '';
    if (conversionScore >= 80) return 'Client très réceptif';
    if (conversionScore >= 60) return 'Client modérément réceptif';
    return 'Client peu réceptif';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-100 p-6 py-12">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate('/clients')}
            className="flex items-center gap-2 text-indigo-600 hover:text-indigo-700 font-bold mb-4"
          >
            <ArrowLeft size={20} />
            Retour aux clients
          </button>
          <h1 className="text-4xl font-black text-gray-900 mb-2">
            Profil Client & Évolution
          </h1>
          <p className="text-gray-500">Analyse approfondie de la personnalité et évolution temporelle</p>
        </div>

        {/* Sélection du client */}
        <div className="bg-white rounded-2xl p-6 mb-6 shadow-sm border border-gray-100">
          <label className="block text-sm font-bold text-gray-700 mb-3">
            Sélectionner un client
          </label>
          <select
            value={selectedClientId}
            onChange={(e) => {
              setSelectedClientId(e.target.value);
              navigate(`/client-profile?clientId=${e.target.value}`, { replace: true });
            }}
            className="w-full max-w-md px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 font-bold"
          >
            <option value="">— Choisir un client —</option>
            {clients.map((client) => (
              <option key={client.id} value={client.id}>
                {client.nom} {client.company_name ? `(${client.company_name})` : ''}
              </option>
            ))}
          </select>
        </div>

        {loading && (
          <div className="text-center py-12">
            <p className="text-gray-500 font-bold">Chargement des données...</p>
          </div>
        )}

        {!loading && selectedClientId && analyses.length === 0 && (
          <div className="bg-white rounded-2xl p-12 text-center shadow-sm border border-gray-100">
            <BarChart3 className="mx-auto text-gray-300 mb-4" size={48} />
            <p className="text-gray-500 font-bold text-lg">
              Aucune analyse disponible pour ce client
            </p>
          </div>
        )}

        {!loading && selectedClientId && analyses.length > 0 && (
          <>
            {/* Informations client */}
            {clientInfo && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-2xl p-6 mb-6 shadow-sm border border-gray-100"
              >
                <div className="flex items-center gap-4">
                  <div className="bg-indigo-600 text-white p-3 rounded-xl">
                    <User size={24} />
                  </div>
                  <div>
                    <h2 className="text-2xl font-black text-gray-900">{clientInfo.nom}</h2>
                    {clientInfo.company_name && (
                      <p className="text-gray-500 font-bold">{clientInfo.company_name}</p>
                    )}
                    <div className="flex items-center gap-4 mt-2 text-sm text-gray-500">
                      <div className="flex items-center gap-1">
                        <Calendar size={14} />
                        {analyses.length} analyse{analyses.length > 1 ? 's' : ''}
                      </div>
                      {analyses.length > 0 && (
                        <>
                          <span>•</span>
                          <span>Première: {new Date(analyses[0].date_analyse).toLocaleDateString('fr-FR')}</span>
                          <span>•</span>
                          <span>Dernière: {new Date(analyses[analyses.length - 1].date_analyse).toLocaleDateString('fr-FR')}</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Score de conversion */}
            {conversionScore !== null && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-2xl p-6 mb-6 shadow-sm border border-gray-100"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-black text-gray-900 mb-1">Score de Conversion</h3>
                    <p className="text-sm text-gray-500">{getConversionScoreMessage()}</p>
                  </div>
                  <div className="text-right">
                    <div className={`text-4xl font-black ${
                      getConversionScoreColor() === 'green' ? 'text-green-600' :
                      getConversionScoreColor() === 'orange' ? 'text-orange-600' :
                      'text-red-600'
                    }`}>
                      {conversionScore}%
                    </div>
                    <p className="text-xs text-gray-400 mt-1">de chance d'acceptation</p>
                  </div>
                </div>
                <div className="mt-4">
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full transition-all ${
                        getConversionScoreColor() === 'green' ? 'bg-green-600' :
                        getConversionScoreColor() === 'orange' ? 'bg-orange-600' :
                        'bg-red-600'
                      }`}
                      style={{ width: `${conversionScore}%` }}
                    />
                  </div>
                </div>
              </motion.div>
            )}

            {/* Radar Chart - Scores actuels */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-2xl p-6 mb-6 shadow-sm border border-gray-100"
            >
              <h3 className="text-xl font-black text-gray-900 mb-6">Scores OCEAN Actuels</h3>
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={getRadarData()}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="trait" />
                    <PolarRadiusAxis domain={[0, 1]} />
                    <Radar
                      dataKey="value"
                      stroke="#4f46e5"
                      fill="#6366f1"
                      fillOpacity={0.6}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </motion.div>

            {/* Line Chart - Évolution temporelle */}
            {analyses.length > 1 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-2xl p-6 mb-6 shadow-sm border border-gray-100"
              >
                <div className="flex items-center gap-2 mb-6">
                  <TrendingUp className="text-indigo-600" size={24} />
                  <h3 className="text-xl font-black text-gray-900">Évolution Temporelle</h3>
                </div>
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={getLineChartData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="Openness"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        dot={{ r: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="Conscientiousness"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={{ r: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="Extraversion"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={{ r: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="Agreeableness"
                        stroke="#f59e0b"
                        strokeWidth={2}
                        dot={{ r: 4 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="Neuroticism"
                        stroke="#ef4444"
                        strokeWidth={2}
                        dot={{ r: 4 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </motion.div>
            )}

            {analyses.length === 1 && (
              <div className="bg-blue-50 rounded-2xl p-6 border border-blue-200 text-center">
                <p className="text-blue-700 font-bold">
                  Une seule analyse disponible. L'évolution temporelle nécessite au moins 2 analyses.
                </p>
              </div>
            )}
          </>
        )}

        {!selectedClientId && (
          <div className="bg-white rounded-2xl p-12 text-center shadow-sm border border-gray-100">
            <User className="mx-auto text-gray-300 mb-4" size={48} />
            <p className="text-gray-500 font-bold text-lg">
              Sélectionnez un client pour voir son profil et son évolution
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ClientProfile;

