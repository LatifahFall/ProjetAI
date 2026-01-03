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
  
  // Modal states
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
      console.error('Erreur lors du chargement des clients:', error);
      alert('Erreur: ' + error.message);
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
      setFormData({
        nom: '',
        email: '',
        telephone: '',
        company_name: '',
        industry: '',
        location: ''
      });
    }
    setShowModal(true);
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setEditingClient(null);
    setFormData({
      nom: '',
      email: '',
      telephone: '',
      company_name: '',
      industry: '',
      location: ''
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.nom.trim()) {
      alert('Le nom est requis');
      return;
    }

    try {
      if (editingClient) {
        // UPDATE
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
        alert('Client modifié avec succès');
      } else {
        // CREATE
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
        alert('Client créé avec succès');
      }
      
      handleCloseModal();
    } catch (error) {
      console.error('Erreur:', error);
      alert('Erreur: ' + error.message);
    }
  };

  const handleDelete = async (clientId, clientName) => {
    if (!confirm(`Êtes-vous sûr de vouloir supprimer ${clientName} ?`)) {
      return;
    }

    try {
      const { error } = await supabase
        .from('clients')
        .delete()
        .eq('id', clientId);

      if (error) throw error;
      
      setClients(clients.filter(c => c.id !== clientId));
      alert('Client supprimé avec succès');
    } catch (error) {
      console.error('Erreur:', error);
      alert('Erreur: ' + error.message);
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
        <div className="text-xl font-bold text-gray-600">Chargement...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 px-6 py-12">
      <div className="max-w-7xl mx-auto">
        {/* Back Button */}
        <div className="mb-6">
          <button
            onClick={() => navigate('/dashboard')}
            className="flex items-center gap-2 text-indigo-600 hover:text-indigo-700 font-bold"
          >
            <ArrowLeft size={20} />
            Retour au dashboard
          </button>
        </div>

        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-black text-gray-900 mb-2">Gestion des Clients</h1>
            <p className="text-gray-500">Créez et gérez vos clients</p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={() => handleOpenModal()}
              className="flex items-center gap-2 bg-indigo-600 text-white px-6 py-3 rounded-xl font-bold hover:bg-indigo-700 transition-colors shadow-lg"
            >
              <Plus size={20} />
              Nouveau Client
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="bg-white rounded-2xl p-4 mb-6 shadow-sm border border-gray-100">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Rechercher un client (nom, email, entreprise)..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-12 pr-4 py-3 bg-gray-50 rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>
        </div>

        {/* Clients Grid */}
        {filteredClients.length === 0 ? (
          <div className="bg-white rounded-2xl p-12 text-center shadow-sm border border-gray-100">
            <Users className="mx-auto text-gray-300 mb-4" size={48} />
            <p className="text-gray-500 font-bold text-lg">
              {searchTerm ? 'Aucun client trouvé' : 'Aucun client pour le moment'}
            </p>
            {!searchTerm && (
              <button
                onClick={() => handleOpenModal()}
                className="mt-4 text-indigo-600 font-bold hover:underline"
              >
                Créer votre premier client
              </button>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredClients.map((client) => (
              <motion.div
                key={client.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100 hover:shadow-lg transition-shadow"
              >
                <div className="flex justify-between items-start mb-4">
                  <div className="flex-1">
                    <h3 className="text-xl font-black text-gray-900 mb-1">{client.nom}</h3>
                    {client.company_name && (
                      <p className="text-sm text-gray-500 font-bold">{client.company_name}</p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => navigate(`/client-profile?clientId=${client.id}`)}
                      className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                      title="Voir le profil"
                    >
                      <TrendingUp size={18} />
                    </button>
                    <button
                      onClick={() => handleOpenModal(client)}
                      className="p-2 text-indigo-600 hover:bg-indigo-50 rounded-lg transition-colors"
                      title="Modifier"
                    >
                      <Edit2 size={18} />
                    </button>
                    <button
                      onClick={() => handleDelete(client.id, client.nom)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      title="Supprimer"
                    >
                      <Trash2 size={18} />
                    </button>
                  </div>
                </div>

                <div className="space-y-2 text-sm">
                  {client.email && (
                    <p className="text-gray-600">
                      <span className="font-bold">Email:</span> {client.email}
                    </p>
                  )}
                  {client.telephone && (
                    <p className="text-gray-600">
                      <span className="font-bold">Téléphone:</span> {client.telephone}
                    </p>
                  )}
                  {client.industry && (
                    <p className="text-gray-600">
                      <span className="font-bold">Secteur:</span> {client.industry}
                    </p>
                  )}
                  {client.location && (
                    <p className="text-gray-600">
                      <span className="font-bold">Localisation:</span> {client.location}
                    </p>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        )}

        {/* Modal */}
        <AnimatePresence>
          {showModal && (
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="bg-white rounded-3xl p-8 w-full max-w-md shadow-2xl"
              >
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-2xl font-black text-gray-900">
                    {editingClient ? 'Modifier le client' : 'Nouveau client'}
                  </h2>
                  <button
                    onClick={handleCloseModal}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    <X size={24} />
                  </button>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2">
                      Nom complet *
                    </label>
                    <input
                      type="text"
                      required
                      value={formData.nom}
                      onChange={(e) => setFormData({ ...formData, nom: e.target.value })}
                      className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2">Email</label>
                    <input
                      type="email"
                      value={formData.email}
                      onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                      className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2">Téléphone</label>
                    <input
                      type="tel"
                      value={formData.telephone}
                      onChange={(e) => setFormData({ ...formData, telephone: e.target.value })}
                      className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2">
                      Nom de l'entreprise
                    </label>
                    <input
                      type="text"
                      value={formData.company_name}
                      onChange={(e) => setFormData({ ...formData, company_name: e.target.value })}
                      className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2">
                      Secteur d'activité
                    </label>
                    <input
                      type="text"
                      value={formData.industry}
                      onChange={(e) => setFormData({ ...formData, industry: e.target.value })}
                      className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-bold text-gray-700 mb-2">
                      Localisation
                    </label>
                    <input
                      type="text"
                      value={formData.location}
                      onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                      className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>

                  <div className="flex gap-3 pt-4">
                    <button
                      type="button"
                      onClick={handleCloseModal}
                      className="flex-1 px-4 py-3 bg-gray-100 text-gray-700 rounded-xl font-bold hover:bg-gray-200 transition-colors"
                    >
                      Annuler
                    </button>
                    <button
                      type="submit"
                      className="flex-1 px-4 py-3 bg-indigo-600 text-white rounded-xl font-bold hover:bg-indigo-700 transition-colors flex items-center justify-center gap-2"
                    >
                      <Save size={18} />
                      {editingClient ? 'Modifier' : 'Créer'}
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

export default Clients;

