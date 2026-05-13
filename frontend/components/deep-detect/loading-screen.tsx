"use client"

import { motion } from "framer-motion"
import { Shield, Scan, Brain, Database, CheckCircle2 } from "lucide-react"
import { useEffect, useState } from "react"

interface LoadingScreenProps {
  isLoading: boolean
  onComplete: () => void
}

const loadingSteps = [
  { icon: Scan, label: "Tarayıcı başlatılıyor...", duration: 800 },
  { icon: Database, label: "Yapay zeka modelleri yükleniyor...", duration: 1000 },
  { icon: Brain, label: "Görüntü analiz ediliyor...", duration: 1500 },
  { icon: CheckCircle2, label: "Rapor oluşturuluyor...", duration: 700 },
]

export function LoadingScreen({ isLoading, onComplete }: LoadingScreenProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    if (!isLoading) return

    let totalTime = 0
    const totalDuration = loadingSteps.reduce((sum, step) => sum + step.duration, 0)

    const stepTimers: NodeJS.Timeout[] = []

    loadingSteps.forEach((step, index) => {
      const timer = setTimeout(() => {
        setCurrentStep(index)
      }, totalTime)
      stepTimers.push(timer)
      totalTime += step.duration
    })

    // Progress animation
    const startTime = Date.now()
    const progressInterval = setInterval(() => {
      const elapsed = Date.now() - startTime
      const newProgress = Math.min((elapsed / totalDuration) * 100, 100)
      setProgress(newProgress)

      if (newProgress >= 100) {
        clearInterval(progressInterval)
        setTimeout(onComplete, 300)
      }
    }, 50)

    return () => {
      stepTimers.forEach(clearTimeout)
      clearInterval(progressInterval)
    }
  }, [isLoading, onComplete])

  if (!isLoading) return null

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/95 backdrop-blur-xl"
    >
      {/* Background grid */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(0,220,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,220,255,0.03)_1px,transparent_1px)] bg-[size:50px_50px]" />

      {/* Radial glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-primary/20 rounded-full blur-3xl" />

      <div className="relative text-center px-4">
        {/* Main Icon */}
        <motion.div
          animate={{
            scale: [1, 1.1, 1],
            rotate: [0, 360],
          }}
          transition={{
            scale: { duration: 2, repeat: Infinity },
            rotate: { duration: 8, repeat: Infinity, ease: "linear" },
          }}
          className="relative mx-auto mb-8 w-32 h-32"
        >
          {/* Outer ring */}
          <div className="absolute inset-0 rounded-full border-4 border-primary/20" />
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
            className="absolute inset-0 rounded-full border-4 border-transparent border-t-primary"
          />

          {/* Inner content */}
          <div className="absolute inset-4 rounded-full bg-gradient-to-br from-primary/20 to-cyan-500/20 flex items-center justify-center">
            <Shield className="w-12 h-12 text-primary" />
          </div>

          {/* Orbiting dots */}
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              animate={{ rotate: 360 }}
              transition={{
                duration: 2 + i * 0.5,
                repeat: Infinity,
                ease: "linear",
              }}
              className="absolute inset-0"
              style={{ transform: `rotate(${i * 120}deg)` }}
            >
              <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-primary" />
            </motion.div>
          ))}
        </motion.div>

        {/* Title */}
        <h2 className="text-2xl font-bold text-foreground mb-2">
          Deep<span className="text-primary">Detect</span>
        </h2>
        <p className="text-muted-foreground mb-8">
          Görüntünüz yapay zeka ile analiz ediliyor
        </p>

        {/* Progress Steps */}
        <div className="max-w-md mx-auto mb-8">
          <div className="space-y-3">
            {loadingSteps.map((step, index) => {
              const StepIcon = step.icon
              const isActive = index === currentStep
              const isComplete = index < currentStep

              return (
                <motion.div
                  key={step.label}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`flex items-center gap-3 p-3 rounded-lg transition-all ${
                    isActive
                      ? "bg-primary/10 border border-primary/30"
                      : isComplete
                      ? "bg-success/5"
                      : "opacity-50"
                  }`}
                >
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                    isActive
                      ? "bg-primary/20"
                      : isComplete
                      ? "bg-success/20"
                      : "bg-secondary/20"
                  }`}>
                    <StepIcon className={`w-4 h-4 ${
                      isActive
                        ? "text-primary animate-pulse"
                        : isComplete
                        ? "text-success"
                        : "text-muted-foreground"
                    }`} />
                  </div>
                  <span className={`text-sm ${
                    isActive ? "text-foreground" : "text-muted-foreground"
                  }`}>
                    {step.label}
                  </span>
                  {isActive && (
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 0.6, repeat: Infinity }}
                      className="ml-auto w-2 h-2 rounded-full bg-primary"
                    />
                  )}
                  {isComplete && (
                    <CheckCircle2 className="ml-auto w-4 h-4 text-success" />
                  )}
                </motion.div>
              )
            })}
          </div>
        </div>

        {/* Progress Bar */}
        <div className="max-w-md mx-auto">
          <div className="flex justify-between text-xs text-muted-foreground mb-2">
            <span>İşleniyor</span>
            <span>%{Math.round(progress)}</span>
          </div>
          <div className="h-2 rounded-full bg-secondary overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-primary to-cyan-500 rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>
    </motion.div>
  )
}
