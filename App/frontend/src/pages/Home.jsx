import React from 'react';
import { 
  Brain, Users, BarChart3, ShieldCheck, Zap, CheckCircle2, 
  Target, Headphones, TrendingUp, Lock, Rocket, ArrowRight 
} from 'lucide-react';

const Home = () => {
  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
      const offset = 80; // Pour ne pas être caché par la navbar
      const bodyRect = document.body.getBoundingClientRect().top;
      const elementRect = element.getBoundingClientRect().top;
      const elementPosition = elementRect - bodyRect;
      const offsetPosition = elementPosition - offset;

      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      });
    }
  };

  return (
    <div className="min-h-screen bg-white text-gray-900 font-sans scroll-smooth">
      
      {/* 1. NAVBAR FIXE */}
      <nav className="fixed top-0 w-full bg-white/90 backdrop-blur-md z-50 flex justify-between items-center px-6 md:px-12 py-4 border-b border-gray-50">
        <div className="flex items-center gap-2 text-xl font-bold tracking-tighter text-blue-700 cursor-pointer" onClick={() => window.scrollTo({top: 0, behavior: 'smooth'})}>
          <Brain size={28} />
          <span>OCEAN<span className="text-gray-400 font-light">INSIGHT</span></span>
        </div>
        <div className="hidden lg:flex gap-8 text-sm font-medium text-gray-500">
          <button onClick={() => scrollToSection('solution')} className="hover:text-blue-600 transition">Solution</button>
          <button onClick={() => scrollToSection('ocean')} className="hover:text-blue-600 transition">Modèle OCEAN</button>
          <button onClick={() => scrollToSection('ia')} className="hover:text-blue-600 transition">Technologie IA</button>
          <button onClick={() => scrollToSection('usages')} className="hover:text-blue-600 transition">Cas d'usage</button>
        </div>
        <button onClick={() => window.location.href='/login'} className="bg-gray-900 text-white px-6 py-2 rounded-full text-sm font-bold hover:bg-gradient-ocean transition">
          Espace Agent
        </button>
      </nav>

      {/* 2. SECTION HERO */}
      <header className="pt-32 pb-20 px-6 text-center bg-gradient-to-b from-blue-50/50 to-white">
        <div className="max-w-4xl mx-auto">
          <span className="inline-block px-4 py-1.5 rounded-full bg-blue-100 text-blue-700 text-xs font-bold uppercase mb-6">
            Analyse Vocale en Temps Réel
          </span>
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-gray-900 mb-8 leading-tight">
            Transformez chaque conversation en <span className="text-blue-600">intelligence stratégique</span>
          </h1>
          <p className="text-xl text-gray-500 mb-10 leading-relaxed max-w-3xl mx-auto">
            Découvrez la personnalité de vos clients en temps réel grâce à l'analyse vocale. 
            Optimisez vos relations, affinez vos contrats et maximisez la satisfaction.
          </p>
          <div className="flex flex-wrap justify-center gap-4 mb-16">
            <button className="bg-gradient-ocean text-white px-10 py-4 rounded-xl font-bold text-lg shadow-xl hover:scale-105 transition-all">
              Demander une démo
            </button>
            <button className="bg-white border border-gray-200 px-10 py-4 rounded-xl font-bold text-lg hover:bg-gray-50 transition-all">
              Essayer gratuitement
            </button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 border-t border-gray-100 pt-12">
            <StatItem value="94%" label="Précision d'analyse" />
            <StatItem value="+47%" label="Satisfaction client" />
            <StatItem value="< 30s" label="Analyse temps réel" />
          </div>
        </div>
      </header>

      {/* 3. PROBLÉMATIQUE */}
      <section className="py-24 px-6 bg-white">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-16 italic text-gray-400">"Les défis quotidiens de la relation client"</h2>
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div className="space-y-6">
              <PainPoint text="Difficultés à comprendre les besoins réels lors des appels" />
              <PainPoint text="Contrats standardisés inadaptés aux profils individuels" />
              <PainPoint text="Insatisfaction client difficile à anticiper" />
              <PainPoint text="Perte d'opportunités par manque d'insights" />
            </div>
            <div className="bg-gray-900 p-8 rounded-3xl text-white shadow-2xl">
              <h3 className="text-2xl font-bold mb-4 text-blue-400">Le saviez-vous ?</h3>
              <p className="text-gray-400 leading-relaxed">
                68% des clients quittent une entreprise car ils sentent qu'elle est indifférente à leur égard. 
                L'intuition seule ne suffit plus à grande échelle.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 4. NOTRE SOLUTION */}
      <section id="solution" className="py-24 px-6 bg-gradient-ocean text-white rounded-[3rem] mx-4 shadow-inner">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-8">Votre avantage concurrentiel</h2>
          <p className="text-xl text-blue-100 mb-16 max-w-3xl mx-auto">
            Notre IA s'appuie sur le modèle OCEAN pour transformer chaque appel en données exploitables.
          </p>
          <div className="grid md:grid-cols-4 gap-8">
            <Step num="1" title="Intégration" desc="Connectez votre téléphonie en un clic." />
            <Step num="2" title="Analyse" desc="IA active pendant la conversation." />
            <Step num="3" title="Profil" desc="Réception instantanée du profil OCEAN." />
            <Step num="4" title="Action" desc="Conseils pour adapter votre approche." />
          </div>
        </div>
      </section>

      {/* 5. LE MODÈLE OCEAN (Version Interactive) */}
      <section id="ocean" className="py-24 px-6 bg-gray-50/50">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-6 text-gray-900 uppercase tracking-tight">La science au service de la compréhension humaine</h2>
          <p className="text-gray-500 mb-12 max-w-3xl mx-auto leading-relaxed">
            Le modèle des Big Five représente le consensus scientifique en psychologie. 
            Notre IA analyse cinq dimensions fondamentales qui déterminent le comportement et les préférences de chaque individu.
          </p>
          
          {/* Grille de sélection */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center mb-20">
            <OceanCard trait="Ouverture" icon={<Brain />} id="trait-o" onClick={() => scrollToSection('detail-o')} />
            <OceanCard trait="Conscience" icon={<ShieldCheck />} id="trait-c" onClick={() => scrollToSection('detail-c')} />
            <OceanCard trait="Extraversion" icon={<Users />} id="trait-e" onClick={() => scrollToSection('detail-e')} />
            <OceanCard trait="Agréabilité" icon={<BarChart3 />} id="trait-a" onClick={() => scrollToSection('detail-a')} />
            <OceanCard trait="Névrosisme" icon={<Zap />} id="trait-n" onClick={() => scrollToSection('detail-n')} />
          </div>

          {/* DÉTAILS SCIENTIFIQUES DES TRAITS */}
          <div className="space-y-12 text-left max-w-5xl mx-auto">
            <TraitDetail 
              id="detail-o"
              letter="O"
              name="Ouverture (Openness)"
              reveal="Le degré de curiosité intellectuelle, de créativité et d'ouverture aux nouvelles expériences."
              high={["Apprécie l'innovation et les solutions créatives", "Ouvert aux idées non conventionnelles", "Valorise l'originalité dans les offres"]}
              low={["Préfère les approches éprouvées et traditionnelles", "Recherche la stabilité et la prévisibilité", "Valorise la fiabilité avant l'innovation"]}
              app="Adaptez vos propositions entre solutions innovantes et éprouvées selon le profil."
            />
            <TraitDetail 
              id="detail-c"
              letter="C"
              name="Conscience (Conscientiousness)"
              reveal="Le niveau d'organisation, de discipline et de fiabilité dans les engagements."
              high={["Accorde une importance aux détails", "Respecte scrupuleusement les délais", "Exige des informations structurées"]}
              low={["Privilégie la flexibilité aux processus rigides", "Prend des décisions plus spontanées", "Apprécie les approches adaptatives"]}
              app="Calibrez le niveau de détail et de formalisme de vos présentations."
            />
            <TraitDetail 
              id="detail-e"
              letter="E"
              name="Extraversion"
              reveal="Le niveau d'énergie sociale, d'enthousiasme et de besoin d'interaction."
              high={["Apprécie les échanges dynamiques", "Préfère les communications directes", "Valorise les relations personnelles"]}
              low={["Préfère les communications réfléchies", "Apprécie le temps de réflexion", "Répond mieux aux approches calmes"]}
              app="Ajustez votre rythme et style de communication pour créer une connexion optimale."
            />
            <TraitDetail 
              id="detail-a"
              letter="A"
              name="Agréabilité (Agreeableness)"
              reveal="Le degré de coopération, d'empathie et de considération envers autrui."
              high={["Recherche l'harmonie et le consensus", "Valorise les relations à long terme", "Apprécie la transparence"]}
              low={["Privilégie les résultats objectifs", "Prend des décisions basées sur la logique", "Valorise la compétitivité"]}
              app="Équilibrez vos arguments entre bénéfices relationnels et avantages concrets."
            />
            <TraitDetail 
              id="detail-n"
              letter="N"
              name="Névrosisme (Neuroticism)"
              reveal="Le niveau de stabilité émotionnelle et de gestion du stress."
              high={["Recherche la réassurance et les garanties", "Sensible aux risques potentiels", "Apprécie le suivi proactif"]}
              low={["Confiant dans la prise de décision", "Gère facilement l'incertitude", "Apprécie l'autonomie"]}
              app="Modulez le niveau de réassurance et de support selon le profil émotionnel."
            />
          </div>
        </div>
      </section>

      {/* 6. IA & RÉSULTATS */}
      <section id="ia" className="py-24 px-6 bg-white">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-2 gap-16">
            <div>
              <h2 className="text-4xl font-bold mb-8 text-gray-900">Comment notre IA révolutionne la relation client</h2>
              <div className="space-y-8">
                <IAFeature icon={<TrendingUp />} title="Analyse vocale avancée" desc="Plus de 200 paramètres (ton, rythme, pauses) pour identifier la personnalité." />
                <IAFeature icon={<Rocket />} title="Apprentissage continu" desc="Le système s'améliore à chaque interaction grâce au machine learning." />
                <IAFeature icon={<Lock />} title="Confidentialité totale" desc="Conformité RGPD, chiffrement et anonymisation des données." />
              </div>
            </div>
            <div className="bg-gray-50 p-10 rounded-3xl border border-gray-100 shadow-sm">
              <h3 className="text-2xl font-bold mb-8 text-gray-900 italic">Des performances mesurables</h3>
              <div className="space-y-6 text-sm">
                <ResultBar label="Précision d'identification" percent="94%" />
                <ResultBar label="Augmentation Conversion" percent="+35%" />
                <ResultBar label="Réduction du Churn" percent="-52%" />
                <ResultBar label="Productivité équipe" percent="+28%" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 7. CAS D'USAGE */}
      <section id="usages" className="py-24 px-6 bg-gray-900 text-white">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-4xl font-bold mb-16">Une solution adaptée à chaque besoin</h2>
          <div className="grid md:grid-cols-3 gap-8 text-left">
            <UsageCard icon={<Target />} title="Optimisation Ventes" list={["Adaptation du discours", "Identification arguments", "Amélioration closing"]} />
            <UsageCard icon={<BarChart3 />} title="Personnalisation Contrats" list={["Clauses adaptées", "Ajustement conditions", "Structuration sur-mesure"]} />
            <UsageCard icon={<Headphones />} title="Gestion Satisfaction" list={["Détection précoce risques", "Routage intelligent", "Réduction des escalades"]} />
          </div>
        </div>
      </section>

      {/* 8. CTA FINAL */}
      <section className="py-24 px-6 text-center">
        <h2 className="text-4xl font-bold mb-8 text-gray-900">Prêt à transformer vos relations clients ?</h2>
        <p className="text-gray-500 mb-12 font-medium italic">Installation en 48h • Support dédié • Sans engagement</p>
        <div className="flex flex-wrap justify-center gap-6">
          <button className="bg-gradient-ocean text-white px-10 py-4 rounded-xl font-bold text-lg hover:shadow-2xl transition">Réserver une démo</button>
          <button className="bg-white border border-gray-200 px-10 py-4 rounded-xl font-bold text-lg hover:bg-gray-50 transition">Essai gratuit 14 jours</button>
        </div>
      </section>

      <footer className="py-12 px-6 border-t border-gray-100 flex justify-between items-center flex-wrap gap-4 text-gray-400 text-sm">
        <div className="flex items-center gap-2 font-bold text-gray-900 uppercase tracking-tighter">
           <Brain size={20} className="text-blue-600" /> Ocean Insight
        </div>
        <p>© 2026 OCEAN INSIGHT - Technologie d'Analyse Comportementale</p>
        <div className="flex gap-6 font-medium text-gray-600">
          <button className="hover:text-blue-600">Confidentialité</button>
          <button className="hover:text-blue-600">Conditions</button>
        </div>
      </footer>
    </div>
  );
};

// COMPOSANTS INTERNES
const StatItem = ({ value, label }) => (
  <div className="p-4 rounded-2xl bg-white shadow-sm border border-gray-50">
    <div className="text-3xl font-black text-blue-600 mb-1">{value}</div>
    <div className="text-sm text-gray-500 font-semibold uppercase tracking-widest">{label}</div>
  </div>
);

const PainPoint = ({ text }) => (
  <div className="flex items-center gap-4 p-5 bg-white rounded-2xl border border-gray-100 hover:border-red-200 transition-colors shadow-sm">
    <div className="w-6 h-6 bg-red-100 text-red-500 rounded-full flex items-center justify-center font-bold text-xs shrink-0">✕</div>
    <p className="text-gray-700 font-medium text-sm leading-relaxed">{text}</p>
  </div>
);

const Step = ({ num, title, desc }) => (
  <div className="relative p-8 bg-blue-700/40 rounded-3xl border border-blue-400/20 text-center backdrop-blur-sm">
    <div className="mx-auto w-12 h-12 bg-white text-blue-600 rounded-2xl flex items-center justify-center font-black shadow-lg mb-6 rotate-3">0{num}</div>
    <h4 className="font-bold text-xl mb-3">{title}</h4>
    <p className="text-sm text-blue-100 leading-relaxed">{desc}</p>
  </div>
);

const OceanCard = ({ trait, icon, onClick }) => (
  <button 
    onClick={onClick}
    className="p-6 bg-white border border-gray-100 rounded-3xl hover:shadow-2xl hover:border-blue-200 transition-all group flex flex-col items-center"
  >
    <div className="text-gray-400 group-hover:text-blue-600 mb-4 transition-colors">{React.cloneElement(icon, {size: 36})}</div>
    <h3 className="font-black text-sm uppercase tracking-tighter text-gray-600 group-hover:text-blue-900">{trait}</h3>
  </button>
);

const TraitDetail = ({ id, letter, name, reveal, high, low, app }) => (
  <div id={id} className="p-10 bg-white border border-gray-100 rounded-[2.5rem] shadow-sm hover:shadow-md transition-shadow">
    <div className="flex items-center gap-4 mb-8">
      <div className="w-16 h-16 bg-gradient-ocean text-white rounded-2xl flex items-center justify-center text-3xl font-black shadow-lg shadow-blue-200">{letter}</div>
      <h3 className="text-3xl font-black text-gray-900">{name}</h3>
    </div>
    
    <div className="grid md:grid-cols-2 gap-12 mb-8">
      <div>
        <p className="text-blue-600 font-bold text-xs uppercase mb-4 tracking-widest italic tracking-widest underline decoration-2 underline-offset-4">Ce que cela révèle</p>
        <p className="text-gray-600 font-medium text-lg leading-relaxed mb-8">{reveal}</p>
        
        <div className="p-6 bg-blue-50 rounded-2xl border-l-4 border-blue-600">
          <p className="text-blue-700 font-black text-xs uppercase mb-4 tracking-widest flex items-center gap-2 italic">
            <Target size={14} /> Application commerciale
          </p>
          <p className="text-blue-900 font-bold leading-relaxed">{app}</p>
        </div>
      </div>

      <div className="space-y-8 text-sm">
        <div className="bg-green-50/50 p-6 rounded-2xl border border-green-100">
          <p className="text-green-700 font-black mb-4 uppercase text-xs tracking-widest flex items-center gap-2 italic underline decoration-green-400">Score Élevé</p>
          <ul className="space-y-3">
            {high.map((item, i) => <li key={i} className="flex gap-3 text-gray-600 font-medium"><CheckCircle2 size={16} className="text-green-500 shrink-0" /> {item}</li>)}
          </ul>
        </div>
        <div className="bg-orange-50/50 p-6 rounded-2xl border border-orange-100">
          <p className="text-orange-700 font-black mb-4 uppercase text-xs tracking-widest flex items-center gap-2 italic underline decoration-orange-400">Score Faible</p>
          <ul className="space-y-3">
            {low.map((item, i) => <li key={i} className="flex gap-3 text-gray-600 font-medium"><div className="w-1.5 h-1.5 rounded-full bg-orange-400 mt-2 shrink-0" /> {item}</li>)}
          </ul>
        </div>
      </div>
    </div>
  </div>
);

const IAFeature = ({ icon, title, desc }) => (
  <div className="flex gap-5 group">
    <div className="p-4 bg-gray-50 text-blue-600 rounded-2xl h-fit group-hover:bg-gradient-ocean group-hover:text-white transition-all shadow-sm">{React.cloneElement(icon, {size: 24})}</div>
    <div>
      <h4 className="font-black text-xl text-gray-900 mb-1">{title}</h4>
      <p className="text-gray-500 text-sm leading-relaxed">{desc}</p>
    </div>
  </div>
);

const ResultBar = ({ label, percent }) => (
  <div>
    <div className="flex justify-between mb-3 text-sm font-bold text-gray-700 tracking-tight">
      <span className="uppercase">{label}</span>
      <span className="text-blue-600">{percent}</span>
    </div>
    <div className="w-full bg-gray-200 h-2.5 rounded-full overflow-hidden">
      <div className="bg-gradient-ocean h-full rounded-full transition-all duration-1000" style={{width: percent.includes('+') ? '85%' : percent}}></div>
    </div>
  </div>
);

const UsageCard = ({ icon, title, list }) => (
  <div className="p-8 bg-gray-800 rounded-[2.5rem] hover:bg-gray-700 transition-all border border-gray-700 shadow-xl group">
    <div className="text-blue-400 mb-8 group-hover:scale-110 transition-transform">{React.cloneElement(icon, {size: 40})}</div>
    <h3 className="text-2xl font-black mb-6 tracking-tight text-white">{title}</h3>
    <ul className="space-y-4 text-sm font-medium">
      {list.map((item, i) => (
        <li key={i} className="flex gap-3 text-gray-400 items-center">
          <ArrowRight size={14} className="text-blue-500" /> {item}
        </li>
      ))}
    </ul>
  </div>
);

export default Home;