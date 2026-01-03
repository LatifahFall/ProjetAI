import React, { useState, useEffect } from 'react';
import { Users, Plus, Search, Edit2, Trash2, X, Save, TrendingUp, ArrowLeft } from 'lucide-react';
import { supabase } from '../lib/supabaseClient';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

const Clients = () => {
  const navigate = useNavigate();
  const agentId = localStorage.getItem('agent_id');
  
  const [clients, setClients] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  
  // États de la Modal
  const [showModal, setShowModal] = useState(false);
  const [editingClient, setEditingClient] = useState(null);
  const [formData, setFormData] = useState({
    nom: '',
    email: '',
    telephone: '',
    company_name: '',
    industry: '',
    location: ''
  });

  useEffect(() => {
    if (!agentId) {
      navigate('/login');
      return;
    }
    fetchClients();
  }, [agentId, navigate]);

  const fetchClients = async () => {
    try {
      const { data, error } = await supabase
        .from('clients')
        .select('*')
        .eq('agent_id', agentId)
        .order('nom', { ascending: true });

      if (error) throw error;
      setClients(data || []);
    } catch (error) {
      console.error('Erreur:', error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenModal = (client = null) => {
    if (client) {
      setEditingClient(client);
      setFormData({
        nom: client.nom || '',
        email: client.email || '',
        telephone: client.telephone || '',
        company_name: client.company_name || '',
        industry: client.industry || '',
        location: client.location || ''
      });
    } else {
      setEditingClient(null);
      setFormData({ nom: '', email: '', telephone: '', company_name: '', industry: '', location: '' });
    }
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setEditingClient(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.nom.trim()) return;

    try {
      if (editingClient) {
        // MISE À JOUR
        const { data, error } = await supabase
          .from('clients')
          .update({
            nom: formData.nom,
            email: formData.email || null,
            telephone: formData.telephone || null,
            company_name: formData.company_name || null,
            industry: formData.industry || null,
            location: formData.location || null
          })
          .eq('id', editingClient.id)
          .select()
          .single();

        if (error) throw error;
        setClients(clients.map(c => c.id === editingClient.id ? data : c));
      } else {
        // CRÉATION
        const { data, error } = await supabase
          .from('clients')
          .insert([{
            agent_id: agentId,
            nom: formData.nom,
            email: formData.email || null,
            telephone: formData.telephone || null,
            company_name: formData.company_name || null,
            industry: formData.industry || null,
            location: formData.location || null
          }])
          .select()
          .single();

        if (error) throw error;
        setClients([...clients, data]);
      }
      handleCloseModal();
    } catch (error) {
      console.error('Erreur:', error.message);
    }
  };

  const handleDelete = async (clientId) => {
    if (!window.confirm(`Supprimer définitivement ce client ?`)) return;

    try {
      const { error } = await supabase
        .from('clients')
        .delete()
        .eq('id', clientId);

      if (error) throw error;
      setClients(clients.filter(c => c.id !== clientId));
    } catch (error) {
      console.error('Erreur:', error.message);
    }
  };

  const filteredClients = clients.filter(client =>
    client.nom?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    client.email?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    client.company_name?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
            <div className="w-12 h-12 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
            <p className="font-black text-gray-400 uppercase tracking-widest text-xs">Chargement Clients...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 px-6 py-12 lg:px-20">
      <div className="max-w-7xl mx-auto">
        
        {/* Navigation de retour */}
        <button
          onClick={() => navigate('/dashboard')}
          className="group flex items-center gap-2 text-gray-400 hover:text-indigo-600 font-bold mb-8 transition-colors"
        >
          <ArrowLeft size={20} className="group-hover:-translate-x-1 transition-transform" />
          Dashboard
        </button>

        {/* En-tête */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-10">
          <div>
            <h1 className="text-4xl font-black text-gray-900 mb-2">Vos Clients</h1>
            <p className="text-gray-500 font-medium text-lg">Gérez votre portefeuille client</p>
          </div>
          <button
            onClick={() => handleOpenModal()}
            className="flex items-center gap-2 bg-indigo-600 text-white px-8 py-4 rounded-2xl font-black hover:bg-indigo-700 transition-all shadow-lg hover:shadow-indigo-200"
          >
            <Plus size={20} />
            Nouveau Client
          </button>
        </div>

        {/* Barre de recherche */}
        <div className="bg-white rounded-[2rem] p-4 mb-8 shadow-sm border border-gray-100">
          <div className="relative">
            <Search className="absolute left-6 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Rechercher par nom, email ou entreprise..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-14 pr-6 py-4 bg-gray-50 rounded-2xl border-none focus:ring-2 focus:ring-indigo-500 font-medium"
            />
          </div>
        </div>

        {/* Grille des Clients */}
        {filteredClients.length === 0 ? (
          <div className="bg-white rounded-[3rem] p-20 text-center border border-dashed border-gray-200">
            <Users className="mx-auto text-gray-200 mb-6" size={64} />
            <p className="text-gray-400 font-bold text-xl">Aucun client trouvé</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence>
              {filteredClients.map((client) => (
                <motion.div
                  key={client.id}
                  layout
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="bg-white rounded-[2.5rem] p-8 shadow-sm border border-gray-50 hover:shadow-xl transition-all group"
                >
                  <div className="flex justify-between items-start mb-6">
                    <div className="w-12 h-12 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600 font-black text-xl">
                        {client.nom.charAt(0)}
                    </div>
                    <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button onClick={() => navigate(`/client-profile?clientId=${client.id}`)} className="p-2 text-green-600 hover:bg-green-50 rounded-xl transition-colors"><TrendingUp size={18} /></button>
                      <button onClick={() => handleOpenModal(client)} className="p-2 text-indigo-600 hover:bg-indigo-50 rounded-xl transition-colors"><Edit2 size={18} /></button>
                      <button onClick={() => handleDelete(client.id)} className="p-2 text-red-600 hover:bg-red-50 rounded-xl transition-colors"><Trash2 size={18} /></button>
                    </div>
                  </div>

                  <h3 className="text-xl font-black text-gray-900 mb-1">{client.nom}</h3>
                  <p className="text-indigo-600 text-sm font-black uppercase tracking-widest mb-4">{client.company_name || "Particulier"}</p>

                  <div className="space-y-3 pt-4 border-t border-gray-50">
                    <InfoRow label="Email" value={client.email} />
                    <InfoRow label="Secteur" value={client.industry} />
                    <InfoRow label="Ville" value={client.location} />
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}

        {/* Modal avec design moderne */}
        <AnimatePresence>
          {showModal && (
            <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-gray-900/40 backdrop-blur-md">
              <motion.div
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 50 }}
                className="bg-white rounded-[3rem] p-10 w-full max-w-lg shadow-2xl relative overflow-hidden"
              >
                <div className="flex justify-between items-center mb-8">
                  <h2 className="text-2xl font-black text-gray-900">{editingClient ? 'Modifier Profil' : 'Nouveau Client'}</h2>
                  <button onClick={handleCloseModal} className="p-2 hover:bg-gray-100 rounded-full transition-colors"><X size={24} /></button>
                </div>

                <form onSubmit={handleSubmit} className="space-y-5">
                  <div className="grid grid-cols-2 gap-4">
                    <ModalInput label="Nom complet *" value={formData.nom} onChange={(v) => setFormData({...formData, nom: v})} required />
                    <ModalInput label="Email" type="email" value={formData.email} onChange={(v) => setFormData({...formData, email: v})} />
                  </div>
                  <ModalInput label="Téléphone" value={formData.telephone} onChange={(v) => setFormData({...formData, telephone: v})} />
                  <ModalInput label="Entreprise" value={formData.company_name} onChange={(v) => setFormData({...formData, company_name: v})} />
                  <div className="grid grid-cols-2 gap-4">
                    <ModalInput label="Secteur" value={formData.industry} onChange={(v) => setFormData({...formData, industry: v})} />
                    <ModalInput label="Ville" value={formData.location} onChange={(v) => setFormData({...formData, location: v})} />
                  </div>

                  <div className="flex gap-4 pt-6">
                    <button type="button" onClick={handleCloseModal} className="flex-1 py-4 bg-gray-100 text-gray-500 rounded-2xl font-black hover:bg-gray-200 transition-colors">Annuler</button>
                    <button type="submit" className="flex-1 py-4 bg-indigo-600 text-white rounded-2xl font-black hover:bg-indigo-700 transition-all shadow-lg flex items-center justify-center gap-2">
                      <Save size={18} /> {editingClient ? 'Enregistrer' : 'Créer'}
                    </button>
                  </div>
                </form>
              </motion.div>
            </div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

const InfoRow = ({ label, value }) => (
  <div className="flex justify-between text-sm">
    <span className="text-gray-400 font-bold">{label}</span>
    <span className="text-gray-700 font-medium truncate ml-4">{value || "-"}</span>
  </div>
);

const ModalInput = ({ label, value, onChange, type = "text", required = false }) => (
  <div>
    <label className="block text-[10px] font-black uppercase tracking-widest text-gray-400 mb-2 ml-1">{label}</label>
    <input
      type={type}
      required={required}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full px-5 py-4 bg-gray-50 border-none rounded-2xl focus:ring-2 focus:ring-indigo-500 font-medium text-sm"
    />
  </div>
);

export default Clients;
