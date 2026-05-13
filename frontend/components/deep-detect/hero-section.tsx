"use client"

import { useCallback, useState } from "react"
import { motion } from "framer-motion"
import { Upload, Scan, Shield, Zap } from "lucide-react"
import { cn } from "@/lib/utils"

interface HeroSectionProps {
  onImageUpload: (file: File) => void
  isAnalyzing: boolean
}

export function HeroSection({ onImageUpload, isAnalyzing }: HeroSectionProps) {
  const [isDragging, setIsDragging] = useState(false)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file && file.type.startsWith("image/")) {
        onImageUpload(file)
      }
    },
    [onImageUpload]
  )

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        onImageUpload(file)
      }
    },
    [onImageUpload]
  )

  return (
    <section className="relative min-h-[70vh] flex flex-col items-center justify-center px-4 py-16">
      {/* Background Effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse-glow" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse-glow" style={{ animationDelay: "1s" }} />
      </div>

      {/* Title */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="text-center mb-12 relative z-10"
      >
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/30 mb-6"
        >
          <Zap className="w-4 h-4 text-primary" />
          <span className="text-sm text-primary font-medium">Gelişmiş Yapay Zeka Destekli</span>
        </motion.div>

        <h1 className="text-5xl md:text-7xl font-bold mb-4 tracking-tight">
          <span className="text-foreground">Deep</span>
          <span className="text-primary glow-text-cyan">Detect</span>
        </h1>
        <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto text-pretty">
          Yapay Zeka Destekli Görüntü Sahteciliği Tespit Sistemi
        </p>
        <p className="text-sm text-muted-foreground mt-4 max-w-lg mx-auto">
          Görüntüleri manipülasyon, sahtecilik ve özgünlük açısından gelişmiş görüntü işleme ve derin öğrenme modelleriyle analiz edin
        </p>
      </motion.div>

      {/* Upload Area */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.8 }}
        className="w-full max-w-2xl relative z-10"
      >
        <label
          htmlFor="image-upload"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            "relative flex flex-col items-center justify-center w-full min-h-[280px] rounded-2xl cursor-pointer transition-all duration-500",
            "glass border-2 border-dashed",
            isDragging
              ? "border-primary bg-primary/10 scale-[1.02]"
              : "border-border/50 hover:border-primary/50 hover:bg-primary/5",
            isAnalyzing && "pointer-events-none"
          )}
        >
          <input
            id="image-upload"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
            disabled={isAnalyzing}
          />

          {isAnalyzing ? (
            <div className="flex flex-col items-center">
              {/* Scanning Animation */}
              <div className="relative w-24 h-24 mb-6">
                <div className="absolute inset-0 rounded-full border-4 border-primary/30" />
                <div className="absolute inset-0 rounded-full border-4 border-primary border-t-transparent animate-spin" />
                <Scan className="absolute inset-0 m-auto w-10 h-10 text-primary animate-pulse" />
              </div>
              <p className="text-lg font-medium text-foreground">Görüntü Analiz Ediliyor...</p>
              <p className="text-sm text-muted-foreground mt-2">Yapay zeka tespit algoritmaları çalıştırılıyor</p>

              {/* Scan Line Effect */}
              <div className="absolute inset-0 overflow-hidden rounded-2xl">
                <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-primary to-transparent animate-scan-line" />
              </div>
            </div>
          ) : (
            <>
              <div className="relative mb-6">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary/20 to-cyan-500/20 flex items-center justify-center glow-border-cyan">
                  <Upload className="w-10 h-10 text-primary" />
                </div>
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-success flex items-center justify-center"
                >
                  <Shield className="w-3 h-3 text-success-foreground" />
                </motion.div>
              </div>

              <p className="text-lg font-medium text-foreground mb-2">
                {isDragging ? "Görüntüyü buraya bırakın" : "Analiz etmek için bir görüntü yükleyin"}
              </p>
              <p className="text-sm text-muted-foreground mb-4">
                Sürükleyip bırakın veya dosya seçmek için tıklayın
              </p>
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                  PNG, JPG, WEBP
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                  Maksimum 20 MB
                </span>
              </div>
            </>
          )}
        </label>
      </motion.div>

      {/* Features Row */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6, duration: 0.8 }}
        className="flex flex-wrap justify-center gap-6 mt-12 relative z-10"
      >
        {[
          { label: "ORB Analizi", icon: "🔬" },
          { label: "AKAZE Tespiti", icon: "🎯" },
          { label: "CNN Puanlama", icon: "🧠" },
          { label: "LSTM Modelleri", icon: "⚡" },
        ].map((feature, index) => (
          <motion.div
            key={feature.label}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.7 + index * 0.1 }}
            className="flex items-center gap-2 px-4 py-2 rounded-full glass-card border border-border/30"
          >
            <span className="text-lg">{feature.icon}</span>
            <span className="text-sm text-muted-foreground">{feature.label}</span>
          </motion.div>
        ))}
      </motion.div>
    </section>
  )
}
