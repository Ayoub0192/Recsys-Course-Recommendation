"use client";

import { useState } from "react";
import {
  Brain,
  Target,
  BarChart3,
  Compass,
  Sparkles,
  Activity,
} from "lucide-react";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

const API_BASE = "http://127.0.0.1:8000";

type CoursePathStep = {
  step: number;
  question: string;
  lecture: string;
  concept: string;
  predicted_correct_probability: number;
};

type MasteryPoint = {
  concept: string;
  mastery: number;
  count: number;
};

type TabKey = "questions" | "lessons" | "path" | "mastery";

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<TabKey>("questions");

  const [userId, setUserId] = useState<number>(123);

  // Text inputs (CSV style)
  const [questionsCsv, setQuestionsCsv] = useState("101, 102, 103, 104, 105, 201, 202, 203, 204, 205, 301, 302, 303, 304, 305, 401, 402, 403, 404, 405");
  const [conceptsCsv, setConceptsCsv] = useState("5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 9, 10, 10, 11, 11, 11, 12, 12");
  const [lecturesCsv, setLecturesCsv] = useState("10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19");
  const [elapsedCsv, setElapsedCsv] = useState("30, 40, 35, 45, 50, 55, 60, 62, 58, 70, 75, 80, 77, 85, 90, 95, 92, 100, 110, 120");
  const [timeCsv, setTimeCsv] = useState("1000, 2000, 3000, 4000, 5000, 6000, 7200, 8300, 9400, 10500, 11600, 13000, 14200, 15400, 16800, 18000, 19500, 21000, 22500, 24000");
  const [correctCsv, setCorrectCsv] = useState("1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1");

  const [topK, setTopK] = useState(5);
  const [steps, setSteps] = useState(6);

  // API data
  const [recommendedQuestions, setRecommendedQuestions] = useState<string[]>(
    []
  );
  const [recommendedLessons, setRecommendedLessons] = useState<string[]>([]);
  const [coursePath, setCoursePath] = useState<CoursePathStep[]>([]);
  const [mastery, setMastery] = useState<MasteryPoint[]>([]);

  const [loading, setLoading] = useState<{
    questions: boolean;
    lessons: boolean;
    path: boolean;
    mastery: boolean;
  }>({
    questions: false,
    lessons: false,
    path: false,
    mastery: false,
  });

  const [error, setError] = useState<string | null>(null);

  // -------- Utils --------

  const parseCsvToNumberArray = (csv: string): number[] =>
    csv
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
      .map((s) => Number(s))
      .filter((n) => !Number.isNaN(n));

  const buildPayload = () => {
    const question = parseCsvToNumberArray(questionsCsv);
    const concept = parseCsvToNumberArray(conceptsCsv);
    const lecture = parseCsvToNumberArray(lecturesCsv);
    const elapsed = parseCsvToNumberArray(elapsedCsv);
    const time = parseCsvToNumberArray(timeCsv);
    const correct = parseCsvToNumberArray(correctCsv);

    const lengths = [
      question.length,
      concept.length,
      lecture.length,
      elapsed.length,
      time.length,
      correct.length,
    ];
    const allEqual = lengths.every((l) => l === lengths[0] && l > 0);

    if (!allEqual) {
      throw new Error(
        "All history fields must have the same non-zero length (questions, concepts, lectures, elapsed, time, correct)."
      );
    }

    return {
      user_id: userId,
      question,
      concept,
      lecture,
      elapsed,
      time,
      correct,
      topk: topK,
      steps,
    };
  };

  // -------- API Calls --------

  const handleRecommendQuestions = async () => {
    try {
      setError(null);
      setLoading((p) => ({ ...p, questions: true }));
      const payload = buildPayload();

      const res = await fetch(`${API_BASE}/recommend_questions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Backend error while fetching questions");
      const data = await res.json();
      setRecommendedQuestions(data.recommended_questions || []);
      setActiveTab("questions");
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Unexpected error");
    } finally {
      setLoading((p) => ({ ...p, questions: false }));
    }
  };

  const handleRecommendLessons = async () => {
    try {
      setError(null);
      setLoading((p) => ({ ...p, lessons: true }));
      const payload = buildPayload();

      const res = await fetch(`${API_BASE}/recommend_lessons`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Backend error while fetching lessons");
      const data = await res.json();
      setRecommendedLessons(data.recommended_lessons || []);
      setActiveTab("lessons");
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Unexpected error");
    } finally {
      setLoading((p) => ({ ...p, lessons: false }));
    }
  };

  const handleCoursePath = async () => {
    try {
      setError(null);
      setLoading((p) => ({ ...p, path: true }));
      const payload = buildPayload();

      const res = await fetch(`${API_BASE}/course_path`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Backend error while fetching course path");
      const data = await res.json();
      setCoursePath(data.course_path || []);
      setActiveTab("path");
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Unexpected error");
    } finally {
      setLoading((p) => ({ ...p, path: false }));
    }
  };

  const handleMastery = async () => {
    try {
      setError(null);
      setLoading((p) => ({ ...p, mastery: true }));
      const payload = buildPayload();

      const res = await fetch(`${API_BASE}/mastery_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Backend error while fetching mastery");
      const data = await res.json();
      setMastery(data.mastery_graph || []);
      setActiveTab("mastery");
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Unexpected error");
    } finally {
      setLoading((p) => ({ ...p, mastery: false }));
    }
  };

  const avgMastery =
    mastery.length > 0
      ? mastery.reduce((acc, m) => acc + m.mastery, 0) / mastery.length
      : 0;

  return (
    <div className="flex min-h-screen">
      {/* ---------------- SIDEBAR ---------------- */}
      <aside className="w-80 border-r border-slate-800 bg-slate-950/80 backdrop-blur-xl p-6 flex flex-col gap-6">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-brand-500 to-cyan-400 flex items-center justify-center shadow-soft">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <div className="font-semibold text-sm uppercase tracking-[0.2em] text-slate-50">
              LearnWise Inc
            </div>
            <div className="font-semibold text-slate-50">
            
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-800/80 card-surface p-4 gradient-border">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs uppercase tracking-wide text-slate-400">
              Étudiant
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-1 text-[11px] font-medium text-emerald-400 border border-emerald-500/40">
              <Sparkles className="h-3 w-3" /> Profil simulé
            </span>
          </div>
          <div className="text-2xl font-semibold text-slate-50">
            ID #{userId}
          </div>
          <p className="mt-1 text-xs text-slate-400">
            Modifie l’historique pour simuler différents parcours.
          </p>
        </div>

        {/* User ID + TopK + steps */}
        <div className="space-y-3 text-sm">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">
              👤 User ID
            </label>
            <input
              type="number"
              value={userId}
              onChange={(e) => setUserId(Number(e.target.value) || 0)}
              className="w-full rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-sm text-slate-50 focus:outline-none focus:ring-2 focus:ring-brand-500/70"
            />
          </div>

          <div className="flex gap-3">
            <div className="flex-1">
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                🎯 Top-K
              </label>
              <input
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value) || 1)}
                className="w-full rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-sm text-slate-50 focus:outline-none focus:ring-2 focus:ring-brand-500/70"
              />
            </div>
            <div className="flex-1">
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                🧭 Étapes Path
              </label>
              <input
                type="number"
                min={1}
                max={20}
                value={steps}
                onChange={(e) => setSteps(Number(e.target.value) || 1)}
                className="w-full rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-sm text-slate-50 focus:outline-none focus:ring-2 focus:ring-brand-500/70"
              />
            </div>
          </div>
        </div>

        {/* History fields */}
        <div className="space-y-3 text-xs">
          <div>
            <label className="block font-medium text-slate-400 mb-1.5">
              🧩 Questions (IDs)
            </label>
            <textarea
              value={questionsCsv}
              onChange={(e) => setQuestionsCsv(e.target.value)}
              rows={2}
              className="w-full rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-xs text-slate-50 focus:outline-none focus:ring-2 focus:ring-brand-500/70"
            />
          </div>

          <div>
            <label className="block font-medium text-slate-400 mb-1.5">
              🧠 Concepts (IDs)
            </label>
            <textarea
              value={conceptsCsv}
              onChange={(e) => setConceptsCsv(e.target.value)}
              rows={2}
              className="w-full rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-xs text-slate-50 focus:outline-none focus:ring-2 focus:ring-brand-500/70"
            />
          </div>

          <div>
            <label className="block font-medium text-slate-400 mb-1.5">
              🎥 Leçons (IDs)
            </label>
            <textarea
              value={lecturesCsv}
              onChange={(e) => setLecturesCsv(e.target.value)}
              rows={2}
              className="w-full rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-xs text-slate-50 focus:outline-none focus:ring-2 focus:ring-brand-500/70"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block font-medium text-slate-400 mb-1.5">
                ⏱ Elapsed (s)
              </label>
              <textarea
                value={elapsedCsv}
                onChange={(e) => setElapsedCsv(e.target.value)}
                rows={2}
                className="w-full rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-xs text-slate-50 focus:outline-none focus:ring-2 focus:ring-brand-500/70"
              />
            </div>
            <div>
              <label className="block font-medium text-slate-400 mb-1.5">
                🕒 Time
              </label>
              <textarea
                value={timeCsv}
                onChange={(e) => setTimeCsv(e.target.value)}
                rows={2}
                className="w-full rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-xs text-slate-50 focus:outline-none focus:ring-2 focus:ring-brand-500/70"
              />
            </div>
          </div>

          <div>
            <label className="block font-medium text-slate-400 mb-1.5">
              ✅ Correct (0/1)
            </label>
            <textarea
              value={correctCsv}
              onChange={(e) => setCorrectCsv(e.target.value)}
              rows={2}
              className="w-full rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-xs text-slate-50 focus:outline-none focus:ring-2 focus:ring-brand-500/70"
            />
          </div>
        </div>

        {/* Action buttons */}
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
          <button
            onClick={handleRecommendQuestions}
            disabled={loading.questions}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-brand-600 px-3 py-2 font-medium text-white shadow-soft hover:bg-brand-500 disabled:opacity-60"
          >
            <Target className="h-4 w-4" />
            Questions
          </button>
          <button
            onClick={handleRecommendLessons}
            disabled={loading.lessons}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-slate-800 px-3 py-2 font-medium text-slate-100 shadow-soft hover:bg-slate-700 disabled:opacity-60"
          >
            <Activity className="h-4 w-4" />
            Leçons
          </button>
          <button
            onClick={handleCoursePath}
            disabled={loading.path}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-slate-800 px-3 py-2 font-medium text-slate-100 shadow-soft hover:bg-slate-700 disabled:opacity-60 col-span-1"
          >
            <Compass className="h-4 w-4" />
            Path
          </button>
          <button
            onClick={handleMastery}
            disabled={loading.mastery}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-slate-800 px-3 py-2 font-medium text-slate-100 shadow-soft hover:bg-slate-700 disabled:opacity-60 col-span-1"
          >
            <BarChart3 className="h-4 w-4" />
            Mastery
          </button>
        </div>

        <p className="mt-auto text-[11px] text-slate-500">
          💡 Utilise ce panneau pour simuler différents profils étudiants
          (rapides, lents, forts, faibles…). C3Rec s’adapte automatiquement.
        </p>
      </aside>

      {/* ---------------- MAIN AREA ---------------- */}
      <main className="flex-1 p-8 space-y-6">
        {/* HEADER */}
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-slate-50 flex items-center gap-2">
               C3Rec Dashboard
            </h1>
            <p className="text-sm text-slate-400">
              Plateforme de recommandations intelligentes pour un apprentissage
              personnalisé.
            </p>
          </div>
          <div className="rounded-full border border-brand-500/40 bg-brand-500/10 px-4 py-1.5 text-xs text-brand-100 flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            Backend connecté sur <code>127.0.0.1:8000</code>
          </div>
        </header>

        {/* STATS ROW */}
        <section className="grid grid-cols-4 gap-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4 flex flex-col gap-2">
            <div className="flex items-center justify-between text-xs text-slate-400">
              Profil
              <span className="rounded-full border border-slate-700 px-2 py-0.5">
                Simulé
              </span>
            </div>
            <div className="text-lg font-semibold text-slate-50">
              Étudiant #{userId}
            </div>
            <p className="text-xs text-slate-500">
              {parseCsvToNumberArray(questionsCsv).length} interactions
              historisées.
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="text-xs text-slate-400 flex items-center gap-2">
              <Target className="h-3 w-3" />
              Reco Questions
            </div>
            <div className="mt-1 text-2xl font-semibold">
              {recommendedQuestions.length || "-"}
            </div>
            <p className="text-[11px] text-slate-500">
              Basées sur ton historique & la dynamique de réussite.
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="text-xs text-slate-400 flex items-center gap-2">
              <Activity className="h-3 w-3" />
              Reco Leçons
            </div>
            <div className="mt-1 text-2xl font-semibold">
              {recommendedLessons.length || "-"}
            </div>
            <p className="text-[11px] text-slate-500">
              Vidéos & contenus pour débloquer les difficultés.
            </p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
            <div className="text-xs text-slate-400 flex items-center gap-2">
              <BarChart3 className="h-3 w-3" />
              Mastery Moyen
            </div>
            <div className="mt-1 text-2xl font-semibold">
              {mastery.length ? `${Math.round(avgMastery * 100)}%` : "-"}
            </div>
            <p className="text-[11px] text-slate-500">
              Synthèse de la maîtrise par concept.
            </p>
          </div>
        </section>

        {/* TABS */}
        <section className="space-y-4">
          <div className="inline-flex rounded-full border border-slate-800 bg-slate-900/60 p-1 text-xs">
            {[
              { key: "questions", label: "Questions", icon: Target },
              { key: "lessons", label: "Leçons", icon: Activity },
              { key: "path", label: "Course Path", icon: Compass },
              { key: "mastery", label: "Mastery", icon: BarChart3 },
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key as TabKey)}
                  className={`inline-flex items-center gap-2 rounded-full px-4 py-1.5 transition ${
                    activeTab === tab.key
                      ? "bg-brand-500 text-white"
                      : "text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  <Icon className="h-3 w-3" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {error && (
            <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-200">
              ⚠️ {error}
            </div>
          )}

          {/* TAB CONTENT */}
          {activeTab === "questions" && (
            <QuestionsTab
              loading={loading.questions}
              items={recommendedQuestions}
            />
          )}

          {activeTab === "lessons" && (
            <LessonsTab
              loading={loading.lessons}
              items={recommendedLessons}
            />
          )}

          {activeTab === "path" && (
            <CoursePathTab loading={loading.path} steps={coursePath} />
          )}

          {activeTab === "mastery" && (
            <MasteryTab loading={loading.mastery} mastery={mastery} />
          )}
        </section>

        <footer className="pt-4 border-t border-slate-800 mt-4 text-[11px] text-slate-500 flex justify-between">
          <span>C3Rec © 2025 — Recommandations intelligentes pour l’éducation.</span>
          <span>Backend: FastAPI · Frontend: Next.js + Tailwind + Recharts</span>
        </footer>
      </main>
    </div>
  );
}

/* ---------------------- SUB COMPONENTS ---------------------- */

function QuestionsTab({
  loading,
  items,
}: {
  loading: boolean;
  items: string[];
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5 shadow-soft space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-50 flex items-center gap-2">
            ❓ Questions recommandées
          </h2>
          <p className="text-xs text-slate-400">
            Sélection de questions optimales pour la prochaine étape
            d’apprentissage.
          </p>
        </div>
      </div>

      {loading && <p className="text-xs text-slate-400">Chargement…</p>}

      {!loading && items.length === 0 && (
        <p className="text-xs text-slate-500">
          Aucune recommandation encore. Lance une requête depuis la sidebar.
        </p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        {items.map((q, idx) => (
          <div
            key={idx}
            className="rounded-lg border border-slate-800 bg-slate-900/60 p-3 flex flex-col gap-1"
          >
            <span className="text-[11px] text-slate-400">
              Question #{idx + 1}
            </span>
            <span className="text-sm font-medium text-slate-50">{q}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function LessonsTab({
  loading,
  items,
}: {
  loading: boolean;
  items: string[];
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5 shadow-soft space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-50 flex items-center gap-2">
            🎥 Leçons recommandées
          </h2>
          <p className="text-xs text-slate-400">
            Contenus pédagogiques alignés sur les questions et concepts à
            travailler.
          </p>
        </div>
      </div>

      {loading && <p className="text-xs text-slate-400">Chargement…</p>}

      {!loading && items.length === 0 && (
        <p className="text-xs text-slate-500">
          Aucune leçon recommandée pour l’instant. Lance une requête.
        </p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        {items.map((l, idx) => (
          <div
            key={idx}
            className="rounded-lg border border-slate-800 bg-slate-900/60 p-3 flex flex-col gap-2"
          >
            <span className="text-[11px] text-slate-400">
              Leçon #{idx + 1}
            </span>
            <span className="text-sm font-medium text-slate-50">{l}</span>
            <span className="text-[11px] text-slate-500">
              Idéal après les questions recommandées #{idx + 1}.
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CoursePathTab({
  loading,
  steps,
}: {
  loading: boolean;
  steps: CoursePathStep[];
}) {
  const chartData = steps.map((s) => ({
    step: s.step,
    prob: Math.round(s.predicted_correct_probability * 100),
  }));

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5 shadow-soft">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-semibold text-slate-50 flex items-center gap-2">
              🧭 Parcours personnalisé (Course Path)
            </h2>
            <p className="text-xs text-slate-400">
              Séquence recommandée de questions & leçons, simulée pas à pas par
              le modèle.
            </p>
          </div>
        </div>

        {loading && <p className="text-xs text-slate-400">Chargement…</p>}

        {!loading && steps.length === 0 && (
          <p className="text-xs text-slate-500">
            Aucun parcours généré. Clique sur « Path » dans la sidebar.
          </p>
        )}

        {!loading && steps.length > 0 && (
          <div className="space-y-4">
            {/* timeline */}
            <div className="relative border-l border-slate-700/80 pl-4 space-y-4">
              {steps.map((s) => (
                <div key={s.step} className="relative pl-4">
                  <div className="absolute -left-4 top-1.5 h-2.5 w-2.5 rounded-full bg-brand-500 shadow-soft" />
                  <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-3 flex flex-col gap-1">
                    <div className="flex items-center justify-between text-xs text-slate-400">
                      <span>Étape {s.step}</span>
                      <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-300 border border-emerald-500/40">
                        {Math.round(s.predicted_correct_probability * 100)}%
                        réussite attendue
                      </span>
                    </div>
                    <div className="text-sm text-slate-50">
                      <span className="font-medium">Question :</span>{" "}
                      {s.question}
                    </div>
                    <div className="text-xs text-slate-300">
                      <span className="font-medium">Leçon :</span>{" "}
                      {s.lecture}
                    </div>
                    <div className="text-[11px] text-slate-500">
                      Concept suivi : {s.concept}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* line chart */}
            <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-4">
              <h3 className="text-xs font-medium text-slate-200 mb-2 flex items-center gap-2">
                <BarChart3 className="h-3 w-3" />
                Probabilité de réussite le long du parcours
              </h3>
              <div className="h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1f2937"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="step"
                      stroke="#9ca3af"
                      fontSize={11}
                      tickLine={false}
                    />
                    <YAxis
                      domain={[0, 100]}
                      stroke="#9ca3af"
                      fontSize={11}
                      tickFormatter={(v) => `${v}%`}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#020617",
                        border: "1px solid #1f2937",
                        fontSize: 11,
                      }}
                      formatter={(value) => [`${value}%`, "Probabilité"]}
                    />
                    <Line
                      type="monotone"
                      dataKey="prob"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                      activeDot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function MasteryTab({
  loading,
  mastery,
}: {
  loading: boolean;
  mastery: MasteryPoint[];
}) {
  const topConcepts = mastery.slice(0, 8);

  const radarData = topConcepts.map((m) => ({
    concept: m.concept,
    mastery: Math.round(m.mastery * 100),
  }));

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-950/80 p-5 shadow-soft">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-sm font-semibold text-slate-50 flex items-center gap-2">
              📈 Mastery Radar
            </h2>
            <p className="text-xs text-slate-400">
              Vue radar des concepts les plus travaillés et de leur niveau de
              maîtrise.
            </p>
          </div>
        </div>

        {loading && <p className="text-xs text-slate-400">Chargement…</p>}

        {!loading && mastery.length === 0 && (
          <p className="text-xs text-slate-500">
            Aucune donnée de maîtrise. Lance une requête « Mastery ».
          </p>
        )}

        {!loading && mastery.length > 0 && (
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="#1f2937" />
                <PolarAngleAxis
                  dataKey="concept"
                  tick={{ fill: "#cbd5f5", fontSize: 11 }}
                />
                <PolarRadiusAxis
                  angle={30}
                  domain={[0, 100]}
                  tick={{ fill: "#9ca3af", fontSize: 10 }}
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#020617",
                    border: "1px solid #1f2937",
                    fontSize: 11,
                  }}
                  formatter={(value) => [`${value}%`, "Mastery"]}
                />
                <Radar
                  name="Mastery"
                  dataKey="mastery"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.35}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-5 shadow-soft">
        <h3 className="text-sm font-semibold text-slate-50 mb-3">
          Détail par concept
        </h3>

        {loading && <p className="text-xs text-slate-400">Chargement…</p>}

        {!loading && mastery.length > 0 && (
          <div className="space-y-2 max-h-72 overflow-auto pr-1">
            {mastery.map((m, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2"
              >
                <div>
                  <div className="text-xs font-medium text-slate-100">
                    {m.concept}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    {m.count} interactions
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-semibold text-slate-50">
                    {Math.round(m.mastery * 100)}%
                  </div>
                  <div className="mt-1 w-20 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-brand-500"
                      style={{ width: `${Math.round(m.mastery * 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {!loading && mastery.length === 0 && (
          <p className="text-xs text-slate-500">
            En attente de données. Exécute une requête Mastery.
          </p>
        )}
      </div>
    </div>
  );
}
