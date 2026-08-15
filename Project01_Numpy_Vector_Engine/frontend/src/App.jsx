import React, { useState, useEffect, useCallback } from 'react';
import { 
  Sliders, 
  Cpu, 
  RotateCcw, 
  Compass, 
  Activity, 
  Info, 
  CheckCircle2, 
  Database,
  ArrowRight,
  Sparkles,
  Zap
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

const PRESETS = [
  { name: 'Identical (Parallel)', a: [2.0, 3.0, 1.0], b: [2.0, 3.0, 1.0], desc: 'Max similarity (1.0)' },
  { name: 'Orthogonal (90°)', a: [1.0, 0.0, 0.0], b: [0.0, 1.0, 0.0], desc: 'Zero similarity (0.0)' },
  { name: 'Opposite (180°)', a: [1.0, 2.0, 3.0], b: [-1.0, -2.0, -3.0], desc: 'Inverse similarity (-1.0)' },
  { name: 'Semantic Query & Doc', a: [0.8, 2.4, -1.2], b: [1.1, 2.1, -0.9], desc: 'Typical RAG match' },
];

export default function App() {
  const [vecA, setVecA] = useState([1.5, 2.0, 3.0]);
  const [vecB, setVecB] = useState([2.0, 1.0, -1.0]);
  const [metrics, setMetrics] = useState(null);
  const [isServerUp, setIsServerUp] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  const fetchMetrics = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/calculate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vector_a: vecA, vector_b: vecB }),
      });

      if (!res.ok) throw new Error('API error');
      const data = await res.json();
      setMetrics(data);
      setIsServerUp(true);
    } catch {
      // Local fallback calculation when backend isn't actively running
      setIsServerUp(false);
      const dot = vecA.reduce((acc, val, i) => acc + val * vecB[i], 0);
      const normA = Math.hypot(...vecA);
      const normB = Math.hypot(...vecB);
      const cosSim = normA && normB ? dot / (normA * normB) : 0;
      const eucDist = Math.hypot(...vecA.map((val, i) => val - vecB[i]));

      setMetrics({
        cosine_similarity: cosSim,
        euclidean_distance: eucDist,
        dot_product: dot,
        norm_a: normA,
        norm_b: normB,
      });
    }
  }, [vecA, vecB]);

  useEffect(() => {
    fetchMetrics();
  }, [fetchMetrics]);

  const handleSliderChange = (setter, index, value) => {
    setter(prev => {
      const updated = [...prev];
      updated[index] = parseFloat(value);
      return updated;
    });
  };

  const applyPreset = (preset) => {
    setVecA([...preset.a]);
    setVecB([...preset.b]);
  };

  // Helper for Cosine alignment badge
  const getAlignmentStatus = (cos) => {
    if (cos > 0.85) return { label: 'Strong Semantic Match', color: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10' };
    if (cos > 0.3) return { label: 'Moderate Similarity', color: 'text-cyan-400 border-cyan-500/30 bg-cyan-500/10' };
    if (cos > -0.3) return { label: 'Orthogonal / Unrelated', color: 'text-amber-400 border-amber-500/30 bg-amber-500/10' };
    return { label: 'Opposite Semantic Meaning', color: 'text-rose-400 border-rose-500/30 bg-rose-500/10' };
  };

  return (
    <div className="min-h-screen bg-[#07090e] text-slate-100 p-4 sm:p-8 font-sans selection:bg-indigo-500/30">
      <div className="max-w-6xl mx-auto space-y-8">

        {/* Top Developer Bar */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-indigo-600/20 border border-indigo-500/30 text-indigo-400">
                <Cpu className="w-6 h-6" />
              </div>
              <h1 className="text-2xl font-bold tracking-tight text-white">
                NumPy Vector Math & Similarity Engine
              </h1>
            </div>
            <p className="text-sm text-slate-400">
              Interactive high-dimensional linear algebra workbench for Vector DBs and RAG pipelines.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono">
              <span className={`w-2.5 h-2.5 rounded-full animate-pulse ${isServerUp ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-amber-400'}`} />
              <span className="text-slate-300">
                {isServerUp ? 'FastAPI Microservice Active' : 'Client-Side Pure Math Mode'}
              </span>
            </div>
          </div>
        </header>

        {/* Quick Presets Section */}
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Zap className="w-4 h-4 text-indigo-400" /> Test Common Vector Scenarios
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {PRESETS.map((p, idx) => (
              <button
                key={idx}
                onClick={() => applyPreset(p)}
                className="text-left p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-indigo-500/50 hover:bg-slate-850 transition duration-150 group"
              >
                <div className="text-xs font-semibold text-slate-200 group-hover:text-indigo-300">{p.name}</div>
                <div className="text-[11px] text-slate-500 mt-0.5">{p.desc}</div>
              </button>
            ))}
          </div>
        </section>

        {/* Main Vector Sliders Interface */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Vector A Box */}
          <div className="relative rounded-2xl bg-gradient-to-b from-slate-900/90 to-slate-950 border border-slate-800 p-6 shadow-xl space-y-5">
            <div className="flex items-center justify-between border-b border-slate-800/60 pb-3">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 shadow-[0_0_8px_#6366f1]" />
                <h2 className="font-semibold text-sm tracking-wider uppercase text-indigo-300">
                  Vector A (Query Embedding)
                </h2>
              </div>
              <span className="text-xs font-mono bg-indigo-950/60 border border-indigo-500/20 px-2.5 py-1 rounded-md text-indigo-300">
                Norm ||A||: {metrics ? metrics.norm_a.toFixed(2) : '0.00'}
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              Represents the high-dimensional query vector generated from user input text chunk before similarity lookup.
            </p>

            <div className="space-y-4 pt-1">
              {vecA.map((val, idx) => (
                <div key={idx} className="space-y-1.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-400">Dim {idx + 1} (Feature Axis)</span>
                    <span className="text-indigo-300 font-bold bg-slate-800/80 px-2 py-0.5 rounded">{val.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="-5"
                    max="5"
                    step="0.1"
                    value={val}
                    onChange={e => handleSliderChange(setVecA, idx, e.target.value)}
                    className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Vector B Box */}
          <div className="relative rounded-2xl bg-gradient-to-b from-slate-900/90 to-slate-950 border border-slate-800 p-6 shadow-xl space-y-5">
            <div className="flex items-center justify-between border-b border-slate-800/60 pb-3">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]" />
                <h2 className="font-semibold text-sm tracking-wider uppercase text-emerald-300">
                  Vector B (Document Chunk)
                </h2>
              </div>
              <span className="text-xs font-mono bg-emerald-950/60 border border-emerald-500/20 px-2.5 py-1 rounded-md text-emerald-300">
                Norm ||B||: {metrics ? metrics.norm_b.toFixed(2) : '0.00'}
              </span>
            </div>

            <p className="text-xs text-slate-400 leading-relaxed">
              Represents the indexed document chunk vector stored inside ChromaDB, Pinecone, or FAISS index.
            </p>

            <div className="space-y-4 pt-1">
              {vecB.map((val, idx) => (
                <div key={idx} className="space-y-1.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-400">Dim {idx + 1} (Feature Axis)</span>
                    <span className="text-emerald-300 font-bold bg-slate-800/80 px-2 py-0.5 rounded">{val.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="-5"
                    max="5"
                    step="0.1"
                    value={val}
                    onChange={e => handleSliderChange(setVecB, idx, e.target.value)}
                    className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>
              ))}
            </div>
          </div>

        </section>

        {/* Calculated Linear Algebra Metrics Section */}
        {metrics && (
          <section className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Activity className="w-4 h-4 text-indigo-400" />
                NumPy Computed Similarity Metrics
              </h3>
              {(() => {
                const status = getAlignmentStatus(metrics.cosine_similarity);
                return (
                  <span className={`text-xs font-medium px-3 py-1 rounded-full border ${status.color}`}>
                    {status.label}
                  </span>
                );
              })()}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              
              {/* Card 1: Cosine Similarity */}
              <div className="rounded-2xl bg-slate-900/90 border border-slate-800 p-6 flex flex-col justify-between space-y-4 hover:border-indigo-500/40 transition">
                <div>
                  <div className="flex justify-between items-start">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Cosine Similarity
                    </span>
                    <span className="text-[10px] font-mono bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-2 py-0.5 rounded">
                      cos(θ)
                    </span>
                  </div>
                  <div className="text-4xl font-extrabold font-mono text-indigo-400 mt-3">
                    {metrics.cosine_similarity.toFixed(4)}
                  </div>
                </div>

                <div className="space-y-2 border-t border-slate-800/80 pt-3">
                  <div className="text-xs text-slate-400">
                    <strong className="text-slate-200">How it works:</strong> Measures the angular cosine between two vectors, ignoring magnitude.
                  </div>
                  <div className="text-[11px] text-slate-500 font-mono bg-slate-950 p-2 rounded border border-slate-800/60">
                    Formula: (A · B) / (||A|| * ||B||)
                  </div>
                  <div className="text-[11px] text-emerald-400">
                    ✓ Core metric used in RAG & Semantic Retrieval.
                  </div>
                </div>
              </div>

              {/* Card 2: Euclidean Distance (L2) */}
              <div className="rounded-2xl bg-slate-900/90 border border-slate-800 p-6 flex flex-col justify-between space-y-4 hover:border-emerald-500/40 transition">
                <div>
                  <div className="flex justify-between items-start">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Euclidean Distance (L2)
                    </span>
                    <span className="text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded">
                      ||A - B||
                    </span>
                  </div>
                  <div className="text-4xl font-extrabold font-mono text-emerald-400 mt-3">
                    {metrics.euclidean_distance.toFixed(4)}
                  </div>
                </div>

                <div className="space-y-2 border-t border-slate-800/80 pt-3">
                  <div className="text-xs text-slate-400">
                    <strong className="text-slate-200">How it works:</strong> Straight-line spatial distance between points in multi-dimensional space.
                  </div>
                  <div className="text-[11px] text-slate-500 font-mono bg-slate-950 p-2 rounded border border-slate-800/60">
                    Formula: √ Σ(A_i - B_i)²
                  </div>
                  <div className="text-[11px] text-emerald-400">
                    ✓ Lower value = Closer vectors in space.
                  </div>
                </div>
              </div>

              {/* Card 3: Dot Product */}
              <div className="rounded-2xl bg-slate-900/90 border border-slate-800 p-6 flex flex-col justify-between space-y-4 hover:border-amber-500/40 transition">
                <div>
                  <div className="flex justify-between items-start">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Dot Product (Inner Product)
                    </span>
                    <span className="text-[10px] font-mono bg-amber-500/10 text-amber-400 border border-amber-500/20 px-2 py-0.5 rounded">
                      A · B
                    </span>
                  </div>
                  <div className="text-4xl font-extrabold font-mono text-amber-400 mt-3">
                    {metrics.dot_product.toFixed(4)}
                  </div>
                </div>

                <div className="space-y-2 border-t border-slate-800/80 pt-3">
                  <div className="text-xs text-slate-400">
                    <strong className="text-slate-200">How it works:</strong> Combines vector direction and length; identical to cosine when normalized.
                  </div>
                  <div className="text-[11px] text-slate-500 font-mono bg-slate-950 p-2 rounded border border-slate-800/60">
                    Formula: Σ(A_i * B_i)
                  </div>
                  <div className="text-[11px] text-amber-400">
                    ✓ Used in transformer attention & Matrix factorization.
                  </div>
                </div>
              </div>

            </div>
          </section>
        )}

        {/* Footer Technical Note */}
        <footer className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
          <span>Week 01 Project: Pure NumPy Linear Algebra & Microservice Architecture</span>
          <span className="font-mono text-slate-400">Python 3.11 • NumPy • FastAPI • React 18</span>
        </footer>

      </div>
    </div>
  );
}