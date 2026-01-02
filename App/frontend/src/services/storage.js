// src/services/storage.js

const KEY = "personaquest_data_v1";

function defaultData() {
  return {
    clients: [],
    analyses: [], // {id, clientId, createdAt, score, ocean, audioUrl, advice}
  };
}

export function loadData() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : defaultData();
  } catch {
    return defaultData();
  }
}

export function saveData(data) {
  localStorage.setItem(KEY, JSON.stringify(data));
}

export function addClient({ name, company }) {
  const data = loadData();
  const id = crypto.randomUUID();
  const client = { id, name, company, createdAt: new Date().toISOString() };
  data.clients.unshift(client);
  saveData(data);
  return client;
}

export function getClients() {
  return loadData().clients;
}

export function getClientById(id) {
  return loadData().clients.find((c) => c.id === id) || null;
}

export function addAnalysis(analysis) {
  const data = loadData();
  data.analyses.unshift(analysis);
  saveData(data);
  return analysis;
}

export function getAnalysesByClient(clientId) {
  return loadData().analyses.filter((a) => a.clientId === clientId);
}

export function getLastAnalysisDate(clientId) {
  const list = getAnalysesByClient(clientId);
  if (!list.length) return null;
  return list[0].createdAt;
}
