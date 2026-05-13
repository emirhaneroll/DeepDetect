"use client"

import { motion } from "framer-motion"
import { Eye, Target, Layers, Fingerprint } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { cn } from "@/lib/utils"

interface AnalysisData {
  orb: { keypoints: number; description: string }
  akaze: { keypoints: number; description: string }
  sift: { keypoints: number; description: string }
  surf: { keypoints: number; description: string }
}

interface AnalysisSectionProps {
  imageUrl: string
  analysisData: AnalysisData
}

const analysisTypes = [
  {
    id: "orb",
    label: "ORB",
    icon: Eye,
    color: "from-cyan-500 to-blue-500",
    description: "Dönüşe duyarsız özellik tespiti için Oriented FAST ve Rotated BRIEF algoritması",
  },
  {
    id: "akaze",
    label: "AKAZE",
    icon: Target,
    color: "from-green-500 to-emerald-500",
    description: "Doğrusal olmayan ölçek uzayı analizi için hızlandırılmış KAZE algoritması",
  },
  {
    id: "sift",
    label: "SIFT",
    icon: Layers,
    color: "from-purple-500 to-violet-500",
    description: "Güçlü anahtar nokta tespiti için Ölçek Değişmez Özellik Dönüşümü",
  },
  {
    id: "surf",
    label: "SURF",
    icon: Fingerprint,
    color: "from-orange-500 to-amber-500",
    description: "Hızlı çoklu ölçek analizi için Hızlandırılmış Güçlü Özellikler",
  },
]

export function AnalysisSection({ imageUrl, analysisData }: AnalysisSectionProps) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="py-12"
    >
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-foreground mb-2">
          Özellik Analizi
        </h2>
        <p className="text-muted-foreground">
          Çoklu algoritma anahtar nokta tespiti ve özellik çıkarma sonuçları
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Image Preview */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-card rounded-2xl p-6 border border-border/30"
        >
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            Yüklenen Görüntü
          </h3>
          <div className="relative rounded-xl overflow-hidden bg-secondary/30 aspect-video">
            <img
              src={imageUrl}
              alt="Uploaded for analysis"
              className="w-full h-full object-contain"
              crossOrigin="anonymous"
            />
            {/* Scan overlay */}
            <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none" />
            <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-background/80 to-transparent">
              <p className="text-sm text-muted-foreground">
                Çözünürlük: 1920 x 1080 • Format: JPEG
              </p>
            </div>
          </div>
        </motion.div>

        {/* Analysis Tabs */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-card rounded-2xl p-6 border border-border/30"
        >
          <Tabs defaultValue="orb" className="w-full">
            <TabsList className="grid grid-cols-4 bg-secondary/30 p-1 rounded-xl mb-6">
              {analysisTypes.map((type) => (
                <TabsTrigger
                  key={type.id}
                  value={type.id}
                  className="data-[state=active]:bg-primary/20 data-[state=active]:text-primary rounded-lg transition-all"
                >
                  <type.icon className="w-4 h-4 mr-1" />
                  {type.label}
                </TabsTrigger>
              ))}
            </TabsList>

            {analysisTypes.map((type) => {
              const data = analysisData[type.id as keyof AnalysisData]
              return (
                <TabsContent key={type.id} value={type.id} className="mt-0">
                  <div className="space-y-6">
                    {/* Keypoint Visualization Placeholder */}
                    <div className="relative rounded-xl overflow-hidden bg-secondary/20 aspect-video">
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className={cn(
                          "w-32 h-32 rounded-full bg-gradient-to-br opacity-20 blur-2xl",
                          type.color
                        )} />
                      </div>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <type.icon className="w-16 h-16 text-muted-foreground/30" />
                      </div>
                      {/* Simulated keypoints */}
                      <div className="absolute inset-0">
                        {Array.from({ length: 20 }).map((_, i) => (
                          <motion.div
                            key={i}
                            initial={{ scale: 0, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ delay: i * 0.05 }}
                            className="absolute w-2 h-2 rounded-full bg-primary/60"
                            style={{
                              left: `${10 + Math.random() * 80}%`,
                              top: `${10 + Math.random() * 80}%`,
                            }}
                          />
                        ))}
                      </div>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="glass-card rounded-xl p-4 border border-border/20">
                        <p className="text-sm text-muted-foreground mb-1">Tespit Edilen Anahtar Noktalar</p>
                        <motion.p
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="text-3xl font-bold text-foreground"
                        >
                          <AnimatedCounter value={data.keypoints} />
                        </motion.p>
                      </div>
                      <div className="glass-card rounded-xl p-4 border border-border/20">
                        <p className="text-sm text-muted-foreground mb-1">Eşleşme Kalitesi</p>
                        <p className="text-3xl font-bold text-success">Yüksek</p>
                      </div>
                    </div>

                    {/* Description */}
                    <div className="glass-card rounded-xl p-4 border border-border/20">
                      <p className="text-sm text-muted-foreground">{type.description}</p>
                      <p className="text-sm text-foreground mt-2">{data.description}</p>
                    </div>
                  </div>
                </TabsContent>
              )
            })}
          </Tabs>
        </motion.div>
      </div>
    </motion.section>
  )
}

function AnimatedCounter({ value }: { value: number }) {
  return (
    <motion.span
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      key={value}
    >
      {value.toLocaleString()}
    </motion.span>
  )
}
