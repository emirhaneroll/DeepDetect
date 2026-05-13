"use client"

import { motion } from "framer-motion"
import { Brain, Activity, Gauge, TrendingUp } from "lucide-react"
import { cn } from "@/lib/utils"

interface RiskScores {
  cnn: number
  lstm: number
  final: number
}

interface RiskAnalysisProps {
  scores: RiskScores
}

export function RiskAnalysis({ scores }: RiskAnalysisProps) {
  const getRiskLevel = (score: number) => {
    if (score < 30) return { label: "Düşük Risk", color: "text-success", bg: "bg-success" }
    if (score < 60) return { label: "Orta Risk", color: "text-warning", bg: "bg-warning" }
    return { label: "Yüksek Risk", color: "text-destructive", bg: "bg-destructive" }
  }

  const metrics = [
    {
      label: "CNN Sahtecilik Skoru",
      value: scores.cnn,
      icon: Brain,
      description: "Piksel düzeyinde yapaylıkların Evrişimli Sinir Ağı analizi",
      gradient: "from-cyan-500 to-blue-600",
    },
    {
      label: "LSTM Sahtecilik Skoru",
      value: scores.lstm,
      icon: Activity,
      description: "Uzun Kısa Süreli Bellek ağı zamansal örüntü analizi",
      gradient: "from-purple-500 to-pink-600",
    },
    {
      label: "Nihai Risk Skoru",
      value: scores.final,
      icon: Gauge,
      description: "Tüm tespit algoritmalarını birleştiren topluluk modeli",
      gradient: "from-orange-500 to-red-600",
      isMain: true,
    },
  ]

  return (
    <motion.section
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="py-12"
    >
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-foreground mb-2 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-primary" />
          </div>
          Yapay Zeka Risk Analizi
        </h2>
        <p className="text-muted-foreground">
          Derin öğrenme modeli tahminleri ve güven skorları
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {metrics.map((metric, index) => {
          const risk = getRiskLevel(metric.value)
          return (
            <motion.div
              key={metric.label}
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ delay: 0.1 + index * 0.15 }}
              className={cn(
                "glass-card rounded-2xl p-6 border border-border/30 relative overflow-hidden",
                metric.isMain && "ring-2 ring-primary/30"
              )}
            >
              {/* Background Glow */}
              <div className={cn(
                "absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl opacity-20",
                `bg-gradient-to-br ${metric.gradient}`
              )} />

              {/* Header */}
              <div className="flex items-center justify-between mb-6 relative">
                <div className="flex items-center gap-3">
                  <div className={cn(
                    "w-12 h-12 rounded-xl flex items-center justify-center bg-gradient-to-br",
                    metric.gradient
                  )}>
                    <metric.icon className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground">{metric.label}</h3>
                    <span className={cn("text-sm font-medium", risk.color)}>
                      {risk.label}
                    </span>
                  </div>
                </div>
                {metric.isMain && (
                  <span className="px-2 py-1 rounded-full bg-primary/20 text-primary text-xs font-medium">
                    Ana Skor
                  </span>
                )}
              </div>

              {/* Circular Progress */}
              <div className="flex justify-center mb-6">
                <CircularProgress value={metric.value} gradient={metric.gradient} />
              </div>

              {/* Description */}
              <p className="text-sm text-muted-foreground text-center">
                {metric.description}
              </p>

              {/* Risk Indicator Bar */}
              <div className="mt-4 space-y-2">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Güvenli</span>
                  <span>Şüpheli</span>
                  <span>Sahte</span>
                </div>
                <div className="h-2 rounded-full bg-secondary overflow-hidden">
                  <div className="h-full flex">
                    <div className="flex-1 bg-success/30" />
                    <div className="flex-1 bg-warning/30" />
                    <div className="flex-1 bg-destructive/30" />
                  </div>
                </div>
                <motion.div
                  initial={{ left: "0%" }}
                  animate={{ left: `${Math.min(metric.value, 98)}%` }}
                  transition={{ delay: 0.5, duration: 0.8, type: "spring" }}
                  className="relative"
                >
                  <div
                    className={cn(
                      "absolute -top-5 w-2 h-2 rounded-full transform -translate-x-1/2",
                      risk.bg
                    )}
                    style={{ left: `${metric.value}%` }}
                  />
                </motion.div>
              </div>
            </motion.div>
          )
        })}
      </div>
    </motion.section>
  )
}

function CircularProgress({ value, gradient }: { value: number; gradient: string }) {
  const radius = 60
  const stroke = 8
  const normalizedRadius = radius - stroke * 2
  const circumference = normalizedRadius * 2 * Math.PI
  const strokeDashoffset = circumference - (value / 100) * circumference

  return (
    <div className="relative w-32 h-32">
      {/* Glow effect */}
      <div className={cn(
        "absolute inset-0 rounded-full blur-xl opacity-30",
        `bg-gradient-to-br ${gradient}`
      )} />

      <svg height={radius * 2} width={radius * 2} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          stroke="currentColor"
          fill="transparent"
          strokeWidth={stroke}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
          className="text-secondary"
        />
        {/* Progress circle */}
        <motion.circle
          stroke="url(#progressGradient)"
          fill="transparent"
          strokeWidth={stroke}
          strokeLinecap="round"
          r={normalizedRadius}
          cx={radius}
          cy={radius}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ delay: 0.3, duration: 1.2, type: "spring" }}
          style={{
            strokeDasharray: `${circumference} ${circumference}`,
          }}
        />
        <defs>
          <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#00dcff" />
            <stop offset="100%" stopColor="#0088ff" />
          </linearGradient>
        </defs>
      </svg>

      {/* Center value */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
          className="text-3xl font-bold text-foreground"
        >
          {value}
        </motion.span>
        <span className="text-xs text-muted-foreground">%</span>
      </div>
    </div>
  )
}
