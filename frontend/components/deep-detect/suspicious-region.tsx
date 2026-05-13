"use client"

import { motion } from "framer-motion"
import { AlertTriangle, MapPin, Scan, ZoomIn } from "lucide-react"

interface SuspiciousRegionProps {
  imageUrl: string
  regions: Array<{
    id: number
    x: number
    y: number
    width: number
    height: number
    confidence: number
  }>
}

export function SuspiciousRegion({ imageUrl, regions }: SuspiciousRegionProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="py-12"
    >
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-2 flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-warning/20 flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-warning" />
            </div>
            Şüpheli Bölge Tespiti
          </h2>
          <p className="text-muted-foreground">
            Yapay zeka tarafından belirlenen potansiyel manipülasyon işaretleri
          </p>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 rounded-full glass-card border border-warning/30">
          <Scan className="w-4 h-4 text-warning animate-pulse" />
          <span className="text-sm text-warning font-medium">
            {regions.length} Bölge Tespit Edildi
          </span>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Heatmap Visualization */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2 glass-card rounded-2xl p-6 border border-border/30"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
              <MapPin className="w-5 h-5 text-primary" />
              Bölge Isı Haritası
            </h3>
            <button className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-secondary/50 text-sm text-muted-foreground hover:text-foreground transition-colors">
              <ZoomIn className="w-4 h-4" />
              Yakınlaştır
            </button>
          </div>

          <div className="relative rounded-xl overflow-hidden bg-secondary/20 aspect-video">
            {/* Base Image */}
            <img
              src={imageUrl}
              alt="Analysis target"
              className="w-full h-full object-contain opacity-60"
              crossOrigin="anonymous"
            />

            {/* Heatmap Overlay */}
            <div className="absolute inset-0 bg-gradient-to-br from-transparent via-destructive/10 to-warning/20 pointer-events-none" />

            {/* Suspicious Regions */}
            {regions.map((region, index) => (
              <motion.div
                key={region.id}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 + index * 0.1 }}
                className="absolute border-2 border-destructive/80 rounded-lg"
                style={{
                  left: `${region.x}%`,
                  top: `${region.y}%`,
                  width: `${region.width}%`,
                  height: `${region.height}%`,
                }}
              >
                {/* Pulsing border effect */}
                <div className="absolute inset-0 border-2 border-destructive/50 rounded-lg animate-ping" />

                {/* Corner markers */}
                <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-destructive" />
                <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-destructive" />
                <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-destructive" />
                <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-destructive" />

                {/* Region label */}
                <div className="absolute -top-6 left-0 px-2 py-0.5 bg-destructive/90 rounded text-xs font-medium text-destructive-foreground">
                  Bölge {index + 1}
                </div>

                {/* Inner glow */}
                <div className="absolute inset-0 bg-destructive/10 rounded-lg" />
              </motion.div>
            ))}

            {/* Scan line effect */}
            <motion.div
              animate={{
                top: ["0%", "100%", "0%"],
              }}
              transition={{
                duration: 4,
                repeat: Infinity,
                ease: "linear",
              }}
              className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent"
            />
          </div>
        </motion.div>

        {/* Region Details */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-card rounded-2xl p-6 border border-border/30"
        >
          <h3 className="text-lg font-semibold text-foreground mb-4">
            Bölge Detayları
          </h3>

          <div className="space-y-4">
            {regions.map((region, index) => (
              <motion.div
                key={region.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 + index * 0.1 }}
                className="p-4 rounded-xl bg-secondary/30 border border-border/20"
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="font-medium text-foreground">
                    Bölge {index + 1}
                  </span>
                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                    region.confidence > 80
                      ? "bg-destructive/20 text-destructive"
                      : region.confidence > 50
                      ? "bg-warning/20 text-warning"
                      : "bg-muted text-muted-foreground"
                  }`}>
                    %{region.confidence} güven
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="text-muted-foreground">
                    Konum: ({region.x}%, {region.y}%)
                  </div>
                  <div className="text-muted-foreground">
                    Boyut: {region.width}% x {region.height}%
                  </div>
                </div>

                {/* Mini confidence bar */}
                <div className="mt-3 h-1.5 rounded-full bg-secondary overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${region.confidence}%` }}
                    transition={{ delay: 0.5 + index * 0.1, duration: 0.6 }}
                    className={`h-full rounded-full ${
                      region.confidence > 80
                        ? "bg-destructive"
                        : region.confidence > 50
                        ? "bg-warning"
                        : "bg-muted-foreground"
                    }`}
                  />
                </div>
              </motion.div>
            ))}

            {regions.length === 0 && (
              <div className="text-center py-8">
                <div className="w-16 h-16 rounded-full bg-success/10 flex items-center justify-center mx-auto mb-4">
                  <Scan className="w-8 h-8 text-success" />
                </div>
                <p className="text-success font-medium">Şüpheli bölge tespit edilmedi</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Görüntü orijinal görünüyor
                </p>
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </motion.section>
  )
}
