"use client"

import { motion } from "framer-motion"
import { Shield, ShieldAlert, ShieldCheck, ShieldQuestion, Sparkles } from "lucide-react"
import { cn } from "@/lib/utils"

type ResultType = "original" | "suspicious" | "fake"

interface FinalResultProps {
  result: ResultType
  confidence: number
  interpretation: string
}

export function FinalResult({ result, confidence, interpretation }: FinalResultProps) {
  const resultConfig = {
    original: {
      icon: ShieldCheck,
      label: "Orijinal",
      description: "Görüntü orijinal görünüyor",
      color: "text-success",
      bgColor: "bg-success",
      borderColor: "border-success",
      glowColor: "shadow-success/30",
      gradient: "from-green-500 to-emerald-600",
    },
    suspicious: {
      icon: ShieldQuestion,
      label: "Şüpheli",
      description: "Potansiyel manipülasyon tespit edildi",
      color: "text-warning",
      bgColor: "bg-warning",
      borderColor: "border-warning",
      glowColor: "shadow-warning/30",
      gradient: "from-yellow-500 to-orange-600",
    },
    fake: {
      icon: ShieldAlert,
      label: "Sahte / Manipüle Edilmiş",
      description: "Yüksek sahtecilik olasılığı",
      color: "text-destructive",
      bgColor: "bg-destructive",
      borderColor: "border-destructive",
      glowColor: "shadow-destructive/30",
      gradient: "from-red-500 to-rose-600",
    },
  }

  const config = resultConfig[result]
  const Icon = config.icon

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
            <Shield className="w-5 h-5 text-primary" />
          </div>
          Nihai Karar
        </h2>
        <p className="text-muted-foreground">
          Yapay zeka destekli özgünlük değerlendirme sonucu
        </p>
      </div>

      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.2, type: "spring" }}
        className={cn(
          "glass-card rounded-3xl p-8 md:p-12 border-2 relative overflow-hidden",
          config.borderColor,
          `shadow-2xl ${config.glowColor}`
        )}
      >
        {/* Background Effects */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className={cn(
            "absolute top-0 right-0 w-96 h-96 rounded-full blur-3xl opacity-10",
            `bg-gradient-to-br ${config.gradient}`
          )} />
          <div className={cn(
            "absolute bottom-0 left-0 w-64 h-64 rounded-full blur-3xl opacity-10",
            `bg-gradient-to-br ${config.gradient}`
          )} />
        </div>

        {/* Animated particles */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {Array.from({ length: 6 }).map((_, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 100 }}
              animate={{
                opacity: [0, 1, 0],
                y: [100, -100],
                x: Math.random() * 50 - 25,
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                delay: i * 0.5,
                ease: "easeOut",
              }}
              className={cn(
                "absolute w-1 h-1 rounded-full",
                config.bgColor
              )}
              style={{ left: `${15 + i * 15}%` }}
            />
          ))}
        </div>

        <div className="relative flex flex-col md:flex-row items-center gap-8">
          {/* Icon Section */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.3, type: "spring", stiffness: 200 }}
            className="relative"
          >
            {/* Pulsing rings */}
            <motion.div
              animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
              className={cn(
                "absolute inset-0 rounded-full",
                config.bgColor,
                "opacity-20"
              )}
            />
            <motion.div
              animate={{ scale: [1, 1.4, 1], opacity: [0.3, 0, 0.3] }}
              transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
              className={cn(
                "absolute inset-0 rounded-full",
                config.bgColor,
                "opacity-10"
              )}
            />

            <div className={cn(
              "w-32 h-32 md:w-40 md:h-40 rounded-full flex items-center justify-center",
              `bg-gradient-to-br ${config.gradient}`,
              "shadow-2xl"
            )}>
              <motion.div
                initial={{ rotate: 0 }}
                animate={{ rotate: result === "original" ? 0 : [0, -5, 5, -5, 0] }}
                transition={{ duration: 0.5, delay: 0.5 }}
              >
                <Icon className="w-16 h-16 md:w-20 md:h-20 text-white" />
              </motion.div>
            </div>

            {/* Sparkle effect for original */}
            {result === "original" && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: [0, 1, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
                className="absolute -top-2 -right-2"
              >
                <Sparkles className="w-8 h-8 text-success" />
              </motion.div>
            )}
          </motion.div>

          {/* Content Section */}
          <div className="flex-1 text-center md:text-left">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <span className={cn(
                "inline-block px-4 py-1 rounded-full text-sm font-medium mb-4",
                `${config.bgColor}/20 ${config.color}`
              )}>
                Analiz Tamamlandı
              </span>
              <h3 className={cn(
                "text-4xl md:text-5xl font-bold mb-3",
                config.color
              )}>
                {config.label}
              </h3>
              <p className="text-xl text-muted-foreground mb-6">
                {config.description}
              </p>
            </motion.div>

            {/* Confidence & Interpretation */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="grid md:grid-cols-2 gap-4"
            >
              <div className="glass-card rounded-xl p-4 border border-border/20">
                <p className="text-sm text-muted-foreground mb-1">Yapay Zeka Güven Seviyesi</p>
                <div className="flex items-center gap-3">
                  <span className={cn("text-3xl font-bold", config.color)}>
                    %{confidence}
                  </span>
                  <div className="flex-1 h-2 rounded-full bg-secondary overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${confidence}%` }}
                      transition={{ delay: 0.6, duration: 0.8 }}
                      className={cn("h-full rounded-full", config.bgColor)}
                    />
                  </div>
                </div>
              </div>

              <div className="glass-card rounded-xl p-4 border border-border/20">
                <p className="text-sm text-muted-foreground mb-1">Risk Yorumu</p>
                <p className="text-foreground font-medium">{interpretation}</p>
              </div>
            </motion.div>
          </div>
        </div>
      </motion.div>
    </motion.section>
  )
}
